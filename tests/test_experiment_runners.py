"""Tests for the T1 and T2 sub-experiment runners.

Both drivers exist to answer "which configuration of this descriptor is best",
and that question is only meaningful if every arm is judged under identical
conditions. Three things have to hold, and none of them is visible in the
accuracy numbers if it breaks:

**Every arm sees the same images.** The arms are extracted in one pass and a
row is committed only when every arm has described it, so a difference in
accuracy cannot be a difference in which apples an arm was shown.

**Every arm gets the same folds.** ``leakage_safe_folds`` draws folds from a
matrix's group and variant provenance. If two arms disagreed about that
provenance they would be cross-validated on different splits, and the sweep
would be comparing configurations *and* partitions at once.

**No augmented variant reaches a validation fold.** An augmented image is a
near-duplicate of its original, so scoring one against a model trained on the
other measures memorisation. This is the invariant ``run_t3_experiments.py``
already respects and these two follow.

The runners are also checked never to hand the test partition to the extraction
pass, which is asserted against the parsed source rather than by running them,
so the guarantee holds for the full run and not only for whatever a test
happens to exercise.

Run from the project root::

    python -m pytest tests/test_experiment_runners.py -v
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import pytest  # noqa: E402

from config import get_config  # noqa: E402
from data import ImageRecord  # noqa: E402
from features.t1_dominant_colour import DominantColourExtractor  # noqa: E402
from features.t2_glcm import GLCMExtractor  # noqa: E402
from harness import leakage_safe_folds  # noqa: E402
from scripts import run_e1_experiments as e1  # noqa: E402
from scripts import run_e2_experiments as e2  # noqa: E402

CONFIG = get_config()


# --------------------------------------------------------------------------- #
# A tiny dataset on disk
# --------------------------------------------------------------------------- #

def write_apple(path: Path, hue: int, seed: int) -> None:
    """Write one synthetic apple: a coloured disc on a pale background."""
    rng = np.random.default_rng(seed)
    image = np.full((256, 256, 3), 235, dtype=np.uint8)
    cv2.circle(image, (128, 128), 92, (int(hue), 60 + hue // 3, 200 - hue // 2), -1)
    # Some texture, so the GLCM arms have something to disagree about.
    grain = rng.integers(-18, 19, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


#: Images per class in the fixture. Five is the floor, not a preference: the
#: shared configuration cross-validates in five folds and the folds are drawn
#: over source images, so four would make every fold-drawing test unrunnable.
FIXTURE_PER_CLASS = 5


@pytest.fixture(scope="module")
def records(tmp_path_factory) -> List[ImageRecord]:
    """A small synthetic dataset on disk, five images in each of three classes."""
    root = tmp_path_factory.mktemp("primary")
    made: List[ImageRecord] = []
    for label, (folder, display) in enumerate(
        zip(CONFIG.primary.classes, CONFIG.primary.display_names)
    ):
        directory = root / folder
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(FIXTURE_PER_CLASS):
            path = directory / f"{folder}_{index}.png"
            write_apple(path, hue=40 + 70 * label, seed=label * 100 + index)
            made.append(
                ImageRecord(
                    path=path, label=label, class_folder=folder,
                    display_name=display, source="primary",
                )
            )
    return made


# --------------------------------------------------------------------------- #
# Subset selection
# --------------------------------------------------------------------------- #

def test_both_runners_draw_the_same_subset(records: List[ImageRecord]) -> None:
    """A pilot of either sweep must describe the same images as the other."""
    first = e1.balanced_subset(records, 1, CONFIG)
    second = e2.balanced_subset(records, 1, CONFIG)
    assert [r.path for r in first] == [r.path for r in second]


def test_balanced_subset_takes_the_same_count_from_every_class(
    records: List[ImageRecord],
) -> None:
    chosen = e1.balanced_subset(records, 2, CONFIG)
    assert len(chosen) == 2 * CONFIG.primary.n_classes
    assert sorted(r.label for r in chosen) == sorted(
        list(range(CONFIG.primary.n_classes)) * 2
    )


def test_balanced_subset_is_reproducible(records: List[ImageRecord]) -> None:
    assert (
        [r.path for r in e1.balanced_subset(records, 1, CONFIG)]
        == [r.path for r in e1.balanced_subset(records, 1, CONFIG)]
    )


# --------------------------------------------------------------------------- #
# Building the variants
# --------------------------------------------------------------------------- #

def test_e1_variants_start_from_the_configuration() -> None:
    """A variant is config.json plus an override, never the module defaults."""
    built = e1.build_dcd(CONFIG)
    assert built.n_colours == int(CONFIG.t1_colour["n_colours"])
    assert built.space == str(CONFIG.t1_colour["space"]).upper()
    assert built.seed == CONFIG.seed


def test_e2_variants_start_from_the_configuration() -> None:
    built = e2.build_glcm(CONFIG)
    assert built.levels == int(CONFIG.t2_glcm["levels"])
    assert tuple(built.distances) == tuple(CONFIG.t2_glcm["distances"])


def test_e1_override_applies_and_leaves_the_rest_alone() -> None:
    built = e1.build_dcd(CONFIG, n_colours=6)
    assert built.n_colours == 6
    assert built.space == str(CONFIG.t1_colour["space"]).upper()


def test_e2_override_applies_and_leaves_the_rest_alone() -> None:
    built = e2.build_glcm(CONFIG, levels=64)
    assert built.levels == 64
    assert tuple(built.distances) == tuple(CONFIG.t2_glcm["distances"])


@pytest.mark.parametrize("builder", [e1.build_dcd, e2.build_glcm])
def test_an_unknown_override_is_refused_rather_than_ignored(builder) -> None:
    """A typo in the variant table must not silently produce the baseline."""
    with pytest.raises(KeyError):
        builder(CONFIG, not_a_real_parameter=1)


@pytest.mark.parametrize("name,overrides", sorted(e1.DCD_VARIANTS.items()))
def test_every_e1_variant_builds(name: str, overrides: dict) -> None:
    assert e1.build_dcd(CONFIG, **overrides).dim > 0


@pytest.mark.parametrize("name,overrides", sorted(e2.GLCM_VARIANTS.items()))
def test_every_e2_variant_has_the_length_the_sweep_believes(
    name: str, overrides: dict
) -> None:
    built = e2.build_glcm(CONFIG, **overrides)
    assert built.dim == e2.expected_dim(built)


def test_e1_block_layout_locates_the_decay_dimension() -> None:
    """E1.5 slices by column, so the layout has to be what it assumes."""
    extractor = e1.build_dcd(CONFIG)
    decay = e1.assert_block_layout(extractor)
    assert list(extractor.feature_names)[decay] == "T1_decay_share"


# --------------------------------------------------------------------------- #
# The shared pass, and what every arm must agree about
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def e1_matrices(records: List[ImageRecord]):
    """Two small T1 variants over an augmented pass."""
    extractors = {
        "baseline": DominantColourExtractor(n_colours=3, sample_size=800, seed=CONFIG.seed),
        "n_colours_5": DominantColourExtractor(n_colours=5, sample_size=800, seed=CONFIG.seed),
    }
    matrices, refused = e1.extract_all_variants(records, extractors, True, CONFIG)
    assert refused == 0
    return matrices


@pytest.fixture(scope="module")
def e2_matrices(records: List[ImageRecord]):
    """Two small T2 variants over an augmented pass."""
    extractors = {
        "baseline": GLCMExtractor(distances=(1,), angles_deg=(0.0, 90.0), levels=16),
        "levels_64": GLCMExtractor(distances=(1,), angles_deg=(0.0, 90.0), levels=64),
    }
    matrices, refused = e2.extract_all_variants(records, extractors, True, CONFIG)
    assert refused == 0
    return matrices


@pytest.fixture(params=["e1", "e2"])
def matrices(request, e1_matrices, e2_matrices):
    return e1_matrices if request.param == "e1" else e2_matrices


def test_every_arm_describes_exactly_the_same_rows(matrices) -> None:
    """The single-pass guarantee, stated over the matrices it produces."""
    names = list(matrices)
    reference = matrices[names[0]]
    for name in names[1:]:
        other = matrices[name]
        assert other.X.shape[0] == reference.X.shape[0]
        np.testing.assert_array_equal(other.y, reference.y)
        np.testing.assert_array_equal(other.groups, reference.groups)
        np.testing.assert_array_equal(other.variants, reference.variants)
        assert other.paths == reference.paths


def test_every_arm_is_cross_validated_on_identical_folds(matrices) -> None:
    """Otherwise the sweep varies the descriptor and the partition together."""
    names = list(matrices)
    reference = leakage_safe_folds(matrices[names[0]], CONFIG)
    for name in names[1:]:
        folds = leakage_safe_folds(matrices[name], CONFIG)
        assert len(folds) == len(reference)
        for (train_a, validation_a), (train_b, validation_b) in zip(folds, reference):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(validation_a, validation_b)


def test_augmented_variants_never_reach_a_validation_fold(matrices) -> None:
    """The leakage invariant the whole partition scheme exists to provide."""
    matrix = matrices[list(matrices)[0]]
    assert matrix.n_augmented > 0, "the fixture must actually augment for this to test anything"
    for _, validation_rows in leakage_safe_folds(matrix, CONFIG):
        assert np.all(matrix.variants[validation_rows] == 0)


def test_an_image_and_its_variants_stay_on_one_side_of_every_fold(matrices) -> None:
    matrix = matrices[list(matrices)[0]]
    for train_rows, validation_rows in leakage_safe_folds(matrix, CONFIG):
        assert not set(matrix.groups[train_rows]) & set(matrix.groups[validation_rows])


def test_extraction_times_are_recorded_per_row(matrices) -> None:
    for matrix in matrices.values():
        assert matrix.extraction_times.size == matrix.X.shape[0]
        assert np.all(matrix.extraction_times >= 0.0)


# --------------------------------------------------------------------------- #
# The test split is never extracted
# --------------------------------------------------------------------------- #

def _extraction_call_arguments(module_path: Path) -> list[str]:
    """Return the first argument of every extract_all_variants call, in order."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "extract_all_variants"
        ):
            found.append(ast.unparse(node.args[0]))
    if not found:
        raise AssertionError(f"{module_path.name} never calls extract_all_variants")
    return found


def _enclosing_function(module_path: Path, call_name: str, argument: str) -> str:
    """Return the function containing a given call, for locating the sweep's one exception."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == call_name
                and node.args
                and ast.unparse(node.args[0]) == argument
            ):
                return function.name
    raise AssertionError(f"no call to {call_name}({argument}) found")


def _assignment_source(module_path: Path, target: str) -> str:
    """Return the expression a module-level-in-main name is assigned from."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == target for t in node.targets
        ):
            return ast.unparse(node.value)
    raise AssertionError(f"{module_path.name} never assigns {target}")


@pytest.mark.parametrize(
    "module", [Path(e1.__file__), Path(e2.__file__), PROJECT_ROOT / "scripts" / "run_t3_experiments.py"]
)
def test_only_the_training_partition_is_ever_extracted(module: Path) -> None:
    """Asserted against the source, so it holds for the full run too.

    A test that merely ran the driver on a fixture would prove nothing about
    what the driver does with the real partition. This checks the one call that
    could leak the test split, and checks what it is handed.
    """
    arguments = _extraction_call_arguments(module)

    # The sweep itself must never see the test split. Every arm is scored by
    # cross-validation over the training partition, and an arm scored on
    # held-out data would make that data a selection stage.
    sweep_calls = [a for a in arguments if a != "partition.test"]
    assert sweep_calls, f"{module.name} never extracts the training partition"
    for argument in sweep_calls:
        assert argument == "training", (
            f"{module.name} extracts from {argument!r}; the sweep must extract "
            f"from the training partition only"
        )
    assert _assignment_source(module, "training") == "partition.train"

    # Exactly one exception is permitted: the closing evaluation, which scores
    # the already-selected arm once so the report has a held-out number. It is
    # a confirmation, not a criterion - selection is closed before it runs - so
    # it is pinned to that one function and to a single call rather than
    # merely allowed.
    held_out = [a for a in arguments if a == "partition.test"]
    assert len(held_out) <= 1, (
        f"{module.name} extracts the test partition {len(held_out)} times; at "
        f"most one closing evaluation is permitted"
    )
    if held_out:
        where = _enclosing_function(module, "extract_all_variants", "partition.test")
        assert where == "evaluate_winner", (
            f"{module.name} extracts the test partition inside {where!r}; only "
            f"the closing evaluation may touch it"
        )
