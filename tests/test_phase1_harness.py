"""Phase 1 tests: config, loader, preprocessing, segmentation, partition.

These tests use synthetic images written to a temporary directory, so they run
without the apple dataset present and verify the harness itself rather than any
result computed from it.

Run from the project root::

    python -m pytest tests -v
    python tests/test_phase1_harness.py    # same checks, without pytest
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import pytest  # noqa: E402

from config import get_config  # noqa: E402
from data import (  # noqa: E402
    DatasetError,
    ImageRecord,
    labels_of,
    list_class_files,
    load_dataset,
    read_image,
)
from features.base import (  # noqa: E402
    FeatureExtractionError,
    FeatureExtractor,
    require_non_empty_mask,
    validate_vector,
)
from evaluate import cross_validate_technique  # noqa: E402
from harness import (  # noqa: E402
    IDENTITY_OP,
    AugmentationOp,
    apply_augmentation,
    build_augmentation_plan,
    build_feature_matrix,
    build_pipeline,
    leakage_safe_folds,
    make_cv,
    prepare_sample,
    preprocess,
    segment_fruit,
    stratified_split,
)

CONFIG = get_config()


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

#: Amplitude of the grain added to every synthetic fixture, in grey levels.
#:
#: A perfectly flat image is not a meaningful input to any segmenter that fits
#: a colour model. GrabCut estimates Gaussian mixtures for foreground and
#: background, and on a zero-variance image those mixtures are degenerate, so
#: the smoothness term dominates and the result stays near whatever region was
#: seeded. Measured on these fixtures, a flat disc segments to 0.417 coverage
#: against a true 0.235, while the same disc carrying +/-3 grey levels of
#: grain segments to 0.247 and becomes independent of the background
#: (intersection over union 0.996 between a light and a dark ground).
#:
#: No photograph is ever flat: sensor noise alone exceeds this. The grain
#: makes the fixtures representative rather than making the test lenient.
TEXTURE_AMPLITUDE = 3


def add_texture(
    image: np.ndarray,
    amplitude: int = TEXTURE_AMPLITUDE,
    seed: int = 0,
) -> np.ndarray:
    """Add reproducible fine grain to a synthetic image."""
    if amplitude <= 0:
        return image
    rng = np.random.default_rng(seed)
    grain = rng.integers(-amplitude, amplitude + 1, size=image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + grain, 0, 255).astype(np.uint8)


def synthetic_apple(
    background: int = 30,
    fruit: tuple = (40, 40, 200),
    radius: int = 70,
    size: int = 256,
    texture: int = TEXTURE_AMPLITUDE,
) -> np.ndarray:
    """Build a synthetic BGR image of a coloured disc on a flat background."""
    image = np.full((size, size, 3), background, dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), radius, fruit, thickness=-1)
    return add_texture(image, texture)


def write_synthetic_dataset(root: Path, classes: List[str], per_class: int = 12) -> None:
    """Write a small synthetic dataset with one folder per class."""
    rng = np.random.default_rng(0)
    for class_index, class_name in enumerate(classes):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            background = 25 + class_index * 5
            colour = (40, 40 + class_index * 60, 200 - class_index * 60)
            image = synthetic_apple(background=background, fruit=colour)
            noise = rng.integers(-6, 7, size=image.shape, dtype=np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(class_dir / f"{class_name}_{index:03d}.png"), image)


class ConstantExtractor(FeatureExtractor):
    """A trivial extractor used to test the harness, not the descriptors."""

    name = "constant test descriptor"
    short_name = "TEST"
    dim = 4

    def extract_features(self, bgr_image: np.ndarray, fruit_mask: np.ndarray) -> np.ndarray:
        selected = require_non_empty_mask(fruit_mask, self.short_name)
        pixels = bgr_image[selected]
        return np.array(
            [
                float(selected.mean()),
                float(pixels[:, 0].mean()),
                float(pixels[:, 1].mean()),
                float(pixels[:, 2].mean()),
            ],
            dtype=np.float64,
        )


class AlternativeExtractor(FeatureExtractor):
    """A second, differently shaped extractor, to prove folds are descriptor-blind."""

    name = "alternative test descriptor"
    short_name = "TEST2"
    dim = 2

    def extract_features(self, bgr_image: np.ndarray, fruit_mask: np.ndarray) -> np.ndarray:
        selected = require_non_empty_mask(fruit_mask, self.short_name)
        pixels = bgr_image[selected].astype(np.float64)
        return np.array([pixels.max(), pixels.min()], dtype=np.float64)


class MutatingExtractor(FeatureExtractor):
    """An extractor that tries to scribble on its inputs."""

    name = "badly behaved descriptor"
    short_name = "BAD"
    dim = 1

    def extract_features(self, bgr_image: np.ndarray, fruit_mask: np.ndarray) -> np.ndarray:
        bgr_image[:] = 0
        fruit_mask[:] = 0
        return np.array([1.0])


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def test_seed_is_fixed_at_42():
    assert CONFIG.seed == 42


def test_shared_parameters_match_the_specification():
    assert CONFIG.preprocess.resize == (224, 224)
    assert CONFIG.preprocess.gaussian_kernel == (5, 5)
    assert CONFIG.segmentation.close_kernel == (5, 5)
    assert CONFIG.partition.test_size == pytest.approx(0.2)
    assert CONFIG.partition.cv_folds == 5
    assert CONFIG.classifier.kernel == "rbf"
    assert CONFIG.classifier.C == pytest.approx(1.0)
    assert CONFIG.classifier.gamma == "scale"
    assert CONFIG.classifier.probability is True


def test_no_absolute_paths_are_hardcoded_in_source():
    """No module may contain a drive-letter or POSIX absolute path literal."""
    offenders = []
    for path in sorted(PROJECT_ROOT.glob("*.py")) + sorted(
        (PROJECT_ROOT / "features").glob("*.py")
    ):
        text = path.read_text(encoding="utf-8")
        for marker in ("C:\\\\Users", "C:/Users", '"/home/', "'/home/"):
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
    assert not offenders, f"absolute paths found: {offenders}"


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

def test_loader_finds_classes_and_orders_deterministically(tmp_path):
    root = tmp_path / "primary"
    write_synthetic_dataset(root, list(CONFIG.primary.classes), per_class=5)

    first = load_dataset(root, CONFIG.primary, "primary", CONFIG.image_extensions)
    second = load_dataset(root, CONFIG.primary, "primary", CONFIG.image_extensions)

    assert len(first) == 15
    assert [r.path for r in first] == [r.path for r in second]
    assert sorted(set(labels_of(first).tolist())) == [0, 1, 2]


def test_loader_reports_a_missing_class_folder(tmp_path):
    root = tmp_path / "primary"
    write_synthetic_dataset(root, list(CONFIG.primary.classes)[:2], per_class=3)
    with pytest.raises(DatasetError, match="Missing class folder"):
        load_dataset(root, CONFIG.primary, "primary", CONFIG.image_extensions)


def test_loader_ignores_non_image_files(tmp_path):
    root = tmp_path / "primary"
    write_synthetic_dataset(root, list(CONFIG.primary.classes), per_class=3)
    (root / CONFIG.primary.classes[0] / "notes.txt").write_text("ignore me", encoding="utf-8")
    files = list_class_files(root / CONFIG.primary.classes[0], CONFIG.image_extensions)
    assert all(f.suffix == ".png" for f in files)
    assert len(files) == 3


def test_read_image_returns_three_channel_uint8(tmp_path):
    path = tmp_path / "apple.png"
    cv2.imwrite(str(path), synthetic_apple())
    image = read_image(path)
    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[2] == 3


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

def test_preprocess_resizes_and_preserves_type():
    image = synthetic_apple(size=300)
    output = preprocess(image, CONFIG)
    assert output.shape == (224, 224, 3)
    assert output.dtype == np.uint8


def test_preprocess_does_not_modify_its_input():
    image = synthetic_apple()
    before = image.copy()
    preprocess(image, CONFIG)
    assert np.array_equal(image, before)


def test_preprocess_is_deterministic():
    image = synthetic_apple()
    assert np.array_equal(preprocess(image, CONFIG), preprocess(image, CONFIG))


def test_preprocess_rejects_a_greyscale_input():
    grey = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="BGR image"):
        preprocess(grey, CONFIG)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

#: The configuration with the brief's Otsu method forced on.
#:
#: ``segmentation.method`` defaults to GrabCut, because Otsu on the value
#: channel does not separate fruit from background on this dataset. The Otsu
#: path is still shipped and still reported in the write-up, so it keeps its
#: own tests; those that concern threshold polarity are meaningless under
#: GrabCut and are pinned to this configuration.
OTSU_CONFIG = replace(CONFIG, segmentation=replace(CONFIG.segmentation, method="otsu_v"))

#: True coverage of the default synthetic disc once resized to 224 x 224.
DISC_COVERAGE = np.pi * (70 * 224 / 256) ** 2 / 224 ** 2


@pytest.mark.parametrize("config", [CONFIG, OTSU_CONFIG], ids=["grabcut", "otsu_v"])
def test_segmentation_finds_a_bright_fruit_on_a_dark_background(config):
    image = preprocess(synthetic_apple(background=25, fruit=(60, 60, 220)), config)
    result = segment_fruit(image, config)
    assert not result.failed, result.reason
    assert result.coverage == pytest.approx(DISC_COVERAGE, abs=0.03)
    assert result.contour.size > 0
    x, y, w, h = result.bbox
    assert w > 0 and h > 0


@pytest.mark.parametrize("config", [CONFIG, OTSU_CONFIG], ids=["grabcut", "otsu_v"])
def test_segmentation_finds_a_dark_fruit_on_a_bright_background(config):
    """An inverted contrast must not invert the mask.

    For Otsu this exercises the automatic polarity choice; for GrabCut it
    checks that the border seed identifies the light ground as background.
    """
    image = preprocess(synthetic_apple(background=235, fruit=(50, 40, 60)), config)
    result = segment_fruit(image, config)
    assert not result.failed, result.reason
    assert result.coverage == pytest.approx(DISC_COVERAGE, abs=0.03)
    # The mask must sit on the fruit, not on the background: the centre of the
    # frame is fruit and the corners are background.
    assert result.mask[112, 112] == 255
    assert result.mask[2, 2] == 0


def test_segmentation_keeps_only_the_largest_component():
    image = synthetic_apple(background=25, fruit=(60, 60, 220), radius=70)
    cv2.circle(image, (20, 20), 8, (60, 60, 220), thickness=-1)  # A small distractor.
    result = segment_fruit(preprocess(image, CONFIG), CONFIG)
    count, _ = cv2.connectedComponents(result.mask)
    assert count == 2  # Background plus exactly one retained component.


def test_otsu_flags_a_mask_that_is_too_small():
    image = preprocess(
        synthetic_apple(background=25, fruit=(60, 60, 220), radius=8), OTSU_CONFIG
    )
    result = segment_fruit(image, OTSU_CONFIG)
    assert result.failed
    assert "below the permitted" in result.reason


def test_otsu_flags_a_mask_that_is_too_large():
    """Under a forced polarity, an almost-full-frame mask must be rejected.

    The two Otsu sides are complements, so with ``polarity="auto"`` the
    over-coverage branch is normally avoided by choosing the other side. The
    guard still has to hold when a fixed polarity is configured.
    """
    forced = replace(
        OTSU_CONFIG, segmentation=replace(OTSU_CONFIG.segmentation, polarity="bright")
    )
    image = np.full((256, 256, 3), 40, dtype=np.uint8)
    cv2.circle(image, (128, 128), 250, (230, 230, 230), thickness=-1)
    result = segment_fruit(preprocess(image, forced), forced)
    assert result.failed
    assert result.coverage > forced.segmentation.max_mask_fraction
    assert "above the permitted" in result.reason


def test_otsu_automatic_polarity_never_returns_an_empty_mask_over_a_full_one():
    """A degenerate uniform frame must not be silently reported as 'fine'."""
    image = preprocess(np.full((256, 256, 3), 200, dtype=np.uint8), OTSU_CONFIG)
    result = segment_fruit(image, OTSU_CONFIG)
    assert result.failed, "a uniform frame contains no fruit and must be flagged"


def test_grabcut_flags_a_frame_that_holds_no_fruit():
    """The GrabCut counterpart of the minimum-coverage guard.

    GrabCut is seeded with a definite-foreground core, so it can never return
    an empty mask and an image holding no fruit still yields a mask the size
    of that core. The degeneracy check exists because coverage alone cannot
    detect this, and a plain ellipse of background would otherwise be handed
    to the feature extractors as a fruit.
    """
    image = preprocess(add_texture(np.full((256, 256, 3), 200, dtype=np.uint8)), CONFIG)
    result = segment_fruit(image, CONFIG)
    assert result.failed, "a uniform frame contains no fruit and must be flagged"
    assert "seeded foreground core" in result.reason


def test_grabcut_flags_a_fruit_smaller_than_its_seed():
    """A fruit smaller than the seeded core cannot be segmented from it."""
    image = preprocess(
        synthetic_apple(background=25, fruit=(60, 60, 220), radius=8), CONFIG
    )
    result = segment_fruit(image, CONFIG)
    assert result.failed


def test_grabcut_is_reproducible_across_calls():
    """GrabCut fits its colour models with k-means, which draws on a global RNG.

    Left unseeded it returns a slightly different mask every call, so two
    techniques handed the same image would be described from different masks -
    the one difference between techniques this study exists to exclude.
    """
    image = preprocess(synthetic_apple(), CONFIG)
    masks = [segment_fruit(image, CONFIG).mask for _ in range(4)]
    assert all(np.array_equal(masks[0], mask) for mask in masks[1:])


def blemished_apple(
    background: int,
    size: int = 256,
    texture: int = TEXTURE_AMPLITUDE,
) -> np.ndarray:
    """A mid-brown disc carrying several dark blemishes, on a flat background."""
    image = np.full((size, size, 3), background, dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), 75, (40, 70, 95), thickness=-1)
    for offset_x, offset_y, radius in ((-30, -20, 10), (18, 12, 9), (5, 35, 8)):
        cv2.circle(
            image,
            (size // 2 + offset_x, size // 2 + offset_y),
            radius,
            (18, 26, 34),
            thickness=-1,
        )
    return add_texture(image, texture)


def test_dark_blemishes_stay_inside_the_fruit_mask():
    """A blemish must not be punched out of the mask it is measured against.

    T3 reports blemish ratio as blemished pixels over mask pixels, so a
    blemish excluded from the mask would be invisible to the descriptor that
    exists to find it.
    """
    image = preprocess(blemished_apple(background=25), CONFIG)
    result = segment_fruit(image, CONFIG)
    assert not result.failed, result.reason
    # The blemish centres must all be inside the mask.
    for offset_x, offset_y in ((-30, -20), (18, 12), (5, 35)):
        row = 112 + int(round(offset_y * 224 / 256))
        column = 112 + int(round(offset_x * 224 / 256))
        assert result.mask[row, column] == 255, f"blemish at ({column}, {row}) fell outside the mask"


@pytest.mark.parametrize("config", [CONFIG, OTSU_CONFIG], ids=["grabcut", "otsu_v"])
def test_the_mask_does_not_depend_on_the_background(config):
    """The same fruit must yield the same silhouette on light and dark grounds.

    Without hole filling the dark-background mask is riddled with blemish
    holes while the light-background one is solid, which would make every
    mask-restricted descriptor a function of the backdrop.
    """
    on_dark = segment_fruit(preprocess(blemished_apple(background=25), config), config)
    on_light = segment_fruit(preprocess(blemished_apple(background=235), config), config)

    assert not on_dark.failed and not on_light.failed
    assert on_dark.coverage == pytest.approx(on_light.coverage, abs=0.02)

    overlap = np.count_nonzero((on_dark.mask > 0) & (on_light.mask > 0))
    union = np.count_nonzero((on_dark.mask > 0) | (on_light.mask > 0))
    assert overlap / union > 0.95, "masks of the same fruit differ by background"


def test_segmentation_is_deterministic():
    image = preprocess(synthetic_apple(), CONFIG)
    first, second = segment_fruit(image, CONFIG), segment_fruit(image, CONFIG)
    assert np.array_equal(first.mask, second.mask)
    assert first.bbox == second.bbox


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #

def test_augmentation_plan_is_identical_across_calls():
    first = build_augmentation_plan(20, CONFIG)
    second = build_augmentation_plan(20, CONFIG)
    assert first == second


def test_augmentation_plan_always_keeps_the_original_first():
    plan = build_augmentation_plan(7, CONFIG)
    assert all(ops[0] is IDENTITY_OP for ops in plan)
    assert all(len(ops) == 1 + CONFIG.augmentation.variants_per_image for ops in plan)


def test_augmentation_stays_within_the_configured_bounds():
    limit = CONFIG.augmentation.rotation_degrees
    jitter = CONFIG.augmentation.brightness_jitter
    for ops in build_augmentation_plan(30, CONFIG):
        for op in ops[1:]:
            assert -limit <= op.angle <= limit
            assert 1.0 - jitter <= op.brightness <= 1.0 + jitter


def test_apply_augmentation_preserves_shape_and_leaves_input_alone():
    image = synthetic_apple()
    before = image.copy()
    op = AugmentationOp(flip=True, angle=12.0, brightness=1.15)
    output = apply_augmentation(image, op)
    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert np.array_equal(image, before)


def test_horizontal_flip_is_a_true_mirror():
    image = synthetic_apple()
    flipped = apply_augmentation(image, AugmentationOp(flip=True, angle=0.0, brightness=1.0))
    assert np.array_equal(flipped, image[:, ::-1])


# --------------------------------------------------------------------------- #
# Partition
# --------------------------------------------------------------------------- #

def records_for(root: Path) -> List[ImageRecord]:
    """Build synthetic records under ``root`` and load them."""
    write_synthetic_dataset(root, list(CONFIG.primary.classes), per_class=25)
    return load_dataset(root, CONFIG.primary, "primary", CONFIG.image_extensions)


def test_split_is_eighty_twenty_stratified_and_disjoint(tmp_path):
    records = records_for(tmp_path / "primary")
    partition = stratified_split(records, CONFIG)

    assert len(partition.test) == pytest.approx(len(records) * 0.2, abs=1)
    assert not set(partition.train_indices) & set(partition.test_indices)
    assert len(partition.train) + len(partition.test) == len(records)

    train_counts = np.bincount(labels_of(partition.train), minlength=3)
    test_counts = np.bincount(labels_of(partition.test), minlength=3)
    assert train_counts.min() > 0 and test_counts.min() > 0
    # Stratification: class proportions preserved to within one image.
    for label in range(3):
        expected = int(round(np.count_nonzero(labels_of(records) == label) * 0.2))
        assert abs(int(test_counts[label]) - expected) <= 1


def test_split_is_identical_across_calls(tmp_path):
    records = records_for(tmp_path / "primary")
    first = stratified_split(records, CONFIG)
    second = stratified_split(records, CONFIG)
    assert np.array_equal(first.test_indices, second.test_indices)
    assert np.array_equal(first.train_indices, second.train_indices)


def test_cross_validation_folds_are_identical_across_calls():
    y = np.repeat([0, 1, 2], 30)
    X = np.zeros((y.size, 3))
    first = [tuple(map(tuple, split)) for split in make_cv(CONFIG).split(X, y)]
    second = [tuple(map(tuple, split)) for split in make_cv(CONFIG).split(X, y)]
    assert first == second
    assert len(first) == CONFIG.partition.cv_folds


def test_cross_validation_folds_do_not_overlap():
    y = np.repeat([0, 1, 2], 30)
    X = np.zeros((y.size, 3))
    seen: set = set()
    for _, validation_index in make_cv(CONFIG).split(X, y):
        assert not seen & set(validation_index.tolist())
        seen |= set(validation_index.tolist())
    assert len(seen) == y.size


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #

def test_pipeline_scales_inside_the_pipeline():
    pipeline = build_pipeline(CONFIG)
    assert [name for name, _ in pipeline.steps] == ["scaler", "svc"]
    svc = pipeline.named_steps["svc"]
    assert svc.kernel == "rbf" and svc.C == 1.0 and svc.gamma == "scale"
    assert svc.random_state == CONFIG.seed


# --------------------------------------------------------------------------- #
# The shared feature interface
# --------------------------------------------------------------------------- #

def test_validate_vector_rejects_the_wrong_length():
    with pytest.raises(FeatureExtractionError, match="declares 4"):
        validate_vector(np.zeros(3), expected_dim=4, technique="TEST")


def test_validate_vector_rejects_nan_and_inf():
    for bad in (np.nan, np.inf):
        vector = np.zeros(4)
        vector[2] = bad
        with pytest.raises(FeatureExtractionError, match="non-finite"):
            validate_vector(vector, expected_dim=4, technique="TEST")


def test_empty_mask_is_refused_rather_than_producing_zeros():
    with pytest.raises(FeatureExtractionError, match="empty fruit mask"):
        require_non_empty_mask(np.zeros((10, 10), dtype=np.uint8), "TEST")


def test_extractor_call_validates_its_own_output():
    image = preprocess(synthetic_apple(), CONFIG)
    mask = segment_fruit(image, CONFIG).mask
    vector = ConstantExtractor()(image, mask)
    assert vector.shape == (4,)
    assert vector.dtype == np.float64
    assert np.all(np.isfinite(vector))


# --------------------------------------------------------------------------- #
# Feature matrix construction and leakage guards
# --------------------------------------------------------------------------- #

def test_feature_matrix_augments_training_only(tmp_path):
    records = records_for(tmp_path / "primary")
    partition = stratified_split(records, CONFIG)
    extractor = ConstantExtractor()

    train = build_feature_matrix(partition.train, extractor, augment=True, config=CONFIG)
    test = build_feature_matrix(partition.test, extractor, augment=False, config=CONFIG)

    expected_train_rows = len(partition.train) * (1 + CONFIG.augmentation.variants_per_image)
    assert train.X.shape == (expected_train_rows, extractor.dim)
    assert train.n_augmented == len(partition.train) * CONFIG.augmentation.variants_per_image
    assert test.X.shape == (len(partition.test), extractor.dim)
    assert test.n_augmented == 0


def test_no_test_image_appears_in_the_training_matrix(tmp_path):
    records = records_for(tmp_path / "primary")
    partition = stratified_split(records, CONFIG)
    extractor = ConstantExtractor()

    train = build_feature_matrix(partition.train, extractor, augment=True, config=CONFIG)
    test = build_feature_matrix(partition.test, extractor, augment=False, config=CONFIG)

    assert not set(train.paths) & set(test.paths)


def test_feature_matrix_is_free_of_nan_and_inf(tmp_path):
    records = records_for(tmp_path / "primary")
    matrix = build_feature_matrix(records, ConstantExtractor(), augment=False, config=CONFIG)
    assert np.all(np.isfinite(matrix.X))


def test_extraction_timing_excludes_the_warm_up_call(tmp_path):
    records = records_for(tmp_path / "primary")[:10]
    matrix = build_feature_matrix(records, ConstantExtractor(), augment=False, config=CONFIG)
    assert matrix.extraction_times.size == matrix.X.shape[0]
    assert np.all(matrix.extraction_times >= 0)


def test_a_technique_cannot_corrupt_the_shared_sample(tmp_path):
    """The harness hands over copies, so a rogue technique harms only itself."""
    records = records_for(tmp_path / "primary")[:4]
    sample = prepare_sample(records[0], config=CONFIG)
    image_before, mask_before = sample.image.copy(), sample.mask.copy()

    build_feature_matrix(records, MutatingExtractor(), augment=False, config=CONFIG)

    fresh = prepare_sample(records[0], config=CONFIG)
    assert np.array_equal(fresh.image, image_before)
    assert np.array_equal(fresh.mask, mask_before)


# --------------------------------------------------------------------------- #
# Cross-validation leakage: augmented variants must not cross a fold boundary
# --------------------------------------------------------------------------- #

def augmented_training_matrix(tmp_path, per_class: int = 10):
    """Build an augmented training matrix from a small synthetic dataset."""
    write_synthetic_dataset(tmp_path / "primary", list(CONFIG.primary.classes), per_class=per_class)
    records = load_dataset(tmp_path / "primary", CONFIG.primary, "primary", CONFIG.image_extensions)
    partition = stratified_split(records, CONFIG)
    return build_feature_matrix(partition.train, ConstantExtractor(), augment=True, config=CONFIG)


def test_feature_matrix_records_group_and_variant_provenance(tmp_path):
    """Each row must know which source image it came from."""
    matrix = augmented_training_matrix(tmp_path)
    assert matrix.groups.shape == (matrix.X.shape[0],)
    assert matrix.variants.shape == (matrix.X.shape[0],)
    # Every source image contributes one original plus its configured variants.
    expected = 1 + CONFIG.augmentation.variants_per_image
    for group in np.unique(matrix.groups):
        rows = matrix.groups == group
        assert int(rows.sum()) == expected
        assert int((matrix.variants[rows] == 0).sum()) == 1, "exactly one original per image"


def test_no_augmented_variant_of_a_validation_image_reaches_a_training_fold(tmp_path):
    """The guarantee this whole design exists to provide.

    For every fold, no row on the training side - original or augmented - may
    come from a source image that appears on the validation side.
    """
    matrix = augmented_training_matrix(tmp_path)

    for fold, (train_rows, validation_rows) in enumerate(leakage_safe_folds(matrix, CONFIG)):
        train_images = set(matrix.groups[train_rows].tolist())
        validation_images = set(matrix.groups[validation_rows].tolist())

        shared = train_images & validation_images
        assert not shared, (
            f"fold {fold}: source image(s) {sorted(shared)} appear on both sides of the "
            f"split, so an augmented variant of a held-out image leaked into training"
        )


def test_validation_folds_contain_only_original_images(tmp_path):
    """Scoring on augmented data would not describe real-world performance."""
    matrix = augmented_training_matrix(tmp_path)
    for fold, (_, validation_rows) in enumerate(leakage_safe_folds(matrix, CONFIG)):
        variants = matrix.variants[validation_rows]
        assert np.all(variants == 0), (
            f"fold {fold}: {int((variants != 0).sum())} validation rows are augmented variants"
        )


def test_training_folds_do_keep_the_augmented_variants(tmp_path):
    """Augmentation must still reach training, or it would be pointless."""
    matrix = augmented_training_matrix(tmp_path)
    for train_rows, _ in leakage_safe_folds(matrix, CONFIG):
        assert int((matrix.variants[train_rows] != 0).sum()) > 0


def test_every_original_is_validated_exactly_once(tmp_path):
    """The five folds must partition the originals, with nothing lost or reused."""
    matrix = augmented_training_matrix(tmp_path)
    validated: List[int] = []
    for _, validation_rows in leakage_safe_folds(matrix, CONFIG):
        validated.extend(matrix.groups[validation_rows].tolist())
    assert sorted(validated) == sorted(np.unique(matrix.groups).tolist())


def test_cross_validation_folds_are_identical_across_techniques(tmp_path):
    """Fold membership must not depend on the descriptor, or the t-tests are void."""
    write_synthetic_dataset(tmp_path / "primary", list(CONFIG.primary.classes), per_class=10)
    records = load_dataset(tmp_path / "primary", CONFIG.primary, "primary", CONFIG.image_extensions)
    partition = stratified_split(records, CONFIG)

    first = build_feature_matrix(partition.train, ConstantExtractor(), augment=True, config=CONFIG)
    second = build_feature_matrix(partition.train, AlternativeExtractor(), augment=True, config=CONFIG)

    for (train_a, val_a), (train_b, val_b) in zip(
        leakage_safe_folds(first, CONFIG), leakage_safe_folds(second, CONFIG)
    ):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(val_a, val_b)


def test_naive_row_wise_cross_validation_leaks(tmp_path):
    """Negative control: prove the obvious approach really is broken.

    Splitting the augmented matrix row-wise - what ``cross_validate(pipeline,
    X, y, cv=make_cv())`` would do - scatters near-duplicate variants of one
    image across both sides of every fold. This test asserts that leak exists,
    so that if someone ever "simplifies" the cross-validator back to a plain
    row-wise split, the leakage-safe tests above stop being a formality.
    """
    matrix = augmented_training_matrix(tmp_path)

    naive_leaks = 0
    for train_rows, validation_rows in make_cv(CONFIG).split(matrix.X, matrix.y):
        naive_leaks += len(
            set(matrix.groups[train_rows].tolist()) & set(matrix.groups[validation_rows].tolist())
        )
    assert naive_leaks > 0, "expected the naive row-wise split to leak"

    safe_leaks = sum(
        len(set(matrix.groups[train].tolist()) & set(matrix.groups[val].tolist()))
        for train, val in leakage_safe_folds(matrix, CONFIG)
    )
    assert safe_leaks == 0
    assert naive_leaks > safe_leaks


def test_cross_validate_technique_uses_the_safe_folds(tmp_path):
    """The public entry point must produce one score per fold, leak-free."""
    matrix = augmented_training_matrix(tmp_path)
    report = cross_validate_technique(build_pipeline(CONFIG), matrix, config=CONFIG)
    assert report.accuracy_folds.size == CONFIG.partition.cv_folds
    assert report.macro_f1_folds.size == CONFIG.partition.cv_folds
    assert np.all((report.accuracy_folds >= 0.0) & (report.accuracy_folds <= 1.0))
    assert report.technique == "TEST"


def test_two_techniques_receive_identical_inputs(tmp_path):
    """The core validity requirement: only the descriptor may differ."""
    records = records_for(tmp_path / "primary")[:8]
    first = build_feature_matrix(records, ConstantExtractor(), augment=False, config=CONFIG)
    second = build_feature_matrix(records, ConstantExtractor(), augment=False, config=CONFIG)
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.y, second.y)
    assert first.paths == second.paths


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__)), "-v"]))
