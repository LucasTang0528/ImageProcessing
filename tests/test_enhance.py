"""Tests for Phase 5, the enhancement layer.

The image-level pass (:func:`enhance.build_technique_matrices`) needs the dataset
and is exercised by the pilot driver run. Everything testable on synthetic
feature matrices is pinned here: the soft vote, the aligned concatenation, the
fusion pipeline, and that both cross-validators keep to five leakage-safe folds.

Run from the project root::

    python -m pytest tests/test_enhance.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import inspect

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from enhance import (  # noqa: E402
    E1_PCA_VARIANCE,
    MIN_SUBREGION_PX,
    OutOfFoldPredictions,
    build_fusion_pipeline,
    e2_block_columns,
    random_control_mask,
    cross_validate_decision_fusion,
    cross_validate_decision_fusion_detailed,
    cross_validate_matrix,
    cross_validate_matrix_detailed,
    fit_decision_fusion,
    macro_f1_weights,
    stack_feature_matrices,
    weighted_soft_vote,
    _regional_vector,
)
from evaluate import classification_metrics  # noqa: E402
from scripts.run_enhancements import MetricStore  # noqa: E402
from features.t1_dominant_colour import DominantColourExtractor
from features.t2_glcm import GLCMExtractor
from harness import FeatureMatrix


# --------------------------------------------------------------------------- #
# Synthetic feature matrices
# --------------------------------------------------------------------------- #

def make_matrix(n_groups: int = 30, dim: int = 6, technique: str = "T", seed: int = 0) -> FeatureMatrix:
    """A leakage-safe, originals-only matrix with a learnable class signal."""
    rng = np.random.default_rng(seed)
    y = np.array([g % 3 for g in range(n_groups)], dtype=np.int64)
    centres = rng.normal(scale=3.0, size=(3, dim))
    X = np.vstack([centres[label] + rng.normal(scale=1.0, size=dim) for label in y])
    groups = np.arange(n_groups, dtype=np.int64)
    return FeatureMatrix(
        X=X, y=y, technique=technique,
        extraction_times=np.full(n_groups, 0.01),
        n_augmented=0, failures=[], excluded=0,
        paths=[Path(f"img_{i}.jpg") for i in range(n_groups)],
        groups=groups, variants=np.zeros(n_groups, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# weighted_soft_vote
# --------------------------------------------------------------------------- #

def test_soft_vote_blends_probabilities_by_weight():
    probs = {
        "T1": np.array([[0.6, 0.3, 0.1], [0.1, 0.1, 0.8]]),
        "T2": np.array([[0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]),
    }
    # Equal weights: row 0 -> class 0 vs 1 tie broken by argmax on the mean
    # (0.4, 0.5, 0.1) -> class 1; row 1 -> class 2.
    np.testing.assert_array_equal(
        weighted_soft_vote(probs, {"T1": 1.0, "T2": 1.0}), [1, 2]
    )
    # Heavily favouring T1 pulls row 0 back to class 0.
    np.testing.assert_array_equal(
        weighted_soft_vote(probs, {"T1": 9.0, "T2": 1.0}), [0, 2]
    )


def test_soft_vote_drops_a_zero_weight_technique():
    probs = {
        "good": np.array([[0.9, 0.05, 0.05]]),
        "noise": np.array([[0.0, 0.0, 1.0]]),
    }
    np.testing.assert_array_equal(
        weighted_soft_vote(probs, {"good": 1.0, "noise": 0.0}), [0]
    )


def test_soft_vote_needs_a_positive_weight():
    with pytest.raises(ValueError, match="positive weight"):
        weighted_soft_vote({"a": np.zeros((2, 3))}, {"a": 0.0})


def test_soft_vote_rejects_mismatched_shapes():
    probs = {"a": np.zeros((2, 3)), "b": np.zeros((2, 4))}
    with pytest.raises(ValueError, match="shape"):
        weighted_soft_vote(probs, {"a": 1.0, "b": 1.0})


# --------------------------------------------------------------------------- #
# stack_feature_matrices
# --------------------------------------------------------------------------- #

def test_stack_concatenates_features_and_sums_times():
    a = make_matrix(dim=4, technique="T1", seed=1)
    b = make_matrix(dim=5, technique="T2", seed=1)  # same seed -> same y/groups
    stacked = stack_feature_matrices([a, b], technique="E1")
    assert stacked.X.shape == (30, 9)
    assert stacked.technique == "E1"
    np.testing.assert_allclose(stacked.X[:, :4], a.X)
    np.testing.assert_allclose(stacked.X[:, 4:], b.X)
    np.testing.assert_allclose(stacked.extraction_times, a.extraction_times + b.extraction_times)


def test_stack_refuses_misaligned_labels():
    from dataclasses import replace

    a = make_matrix(technique="T1", seed=1)
    b = replace(make_matrix(technique="T2", seed=1), y=a.y[::-1].copy())
    with pytest.raises(ValueError, match="not aligned"):
        stack_feature_matrices([a, b])


def test_stack_refuses_an_empty_list():
    with pytest.raises(ValueError, match="nothing to stack"):
        stack_feature_matrices([])


# --------------------------------------------------------------------------- #
# build_fusion_pipeline
# --------------------------------------------------------------------------- #

def test_fusion_pipeline_is_scaler_then_pca_then_svm():
    pipeline = build_fusion_pipeline()
    assert list(pipeline.named_steps) == ["scaler", "pca", "svc"]
    assert pipeline.named_steps["pca"].n_components == E1_PCA_VARIANCE


def test_fusion_pipeline_reduces_dimensionality_on_redundant_features():
    matrix = make_matrix(dim=6, seed=3)
    # Duplicate every column: PCA at 95% variance must not keep all 12.
    padded = np.hstack([matrix.X, matrix.X])
    pipeline = build_fusion_pipeline()
    pipeline.fit(padded, matrix.y)
    assert pipeline.named_steps["pca"].n_components_ < padded.shape[1]


# --------------------------------------------------------------------------- #
# cross-validation
# --------------------------------------------------------------------------- #

def test_cross_validate_matrix_returns_five_folds():
    report = cross_validate_matrix(make_matrix(seed=4), technique="T1")
    assert report.accuracy_folds.shape == (5,)
    assert report.macro_f1_folds.shape == (5,)
    assert 0.0 <= report.mean_accuracy <= 1.0


def test_cross_validate_matrix_accepts_a_pipeline_override():
    matrix = make_matrix(dim=6, seed=5)
    report = cross_validate_matrix(
        matrix, technique="E1", pipeline=build_fusion_pipeline()
    )
    assert report.accuracy_folds.shape == (5,)


def test_macro_f1_weights_are_one_per_technique():
    matrices = {
        "T1": make_matrix(seed=6, technique="T1"),
        "T2": make_matrix(seed=7, technique="T2"),
        "T3": make_matrix(seed=8, technique="T3"),
    }
    weights = macro_f1_weights(matrices)
    assert set(weights) == {"T1", "T2", "T3"}
    assert all(0.0 <= value <= 1.0 for value in weights.values())


def test_decision_fusion_fits_and_predicts_aligned_matrices():
    matrices = {name: make_matrix(seed=9, technique=name) for name in ("T1", "T2", "T3")}
    fusion = fit_decision_fusion(matrices, weights={"T1": 1.0, "T2": 1.0, "T3": 1.0})
    predictions = fusion.predict(matrices)
    assert predictions.shape == (30,)
    assert set(np.unique(predictions)).issubset({0, 1, 2})


def test_cross_validate_decision_fusion_returns_five_folds():
    matrices = {name: make_matrix(seed=10, technique=name) for name in ("T1", "T2", "T3")}
    report = cross_validate_decision_fusion(
        matrices, weights={"T1": 1.0, "T2": 1.0, "T3": 1.0}
    )
    assert report.accuracy_folds.shape == (5,)


# --------------------------------------------------------------------------- #
# E2 sub-region descriptor
# --------------------------------------------------------------------------- #

def _apple_and_masks():
    """A 120x120 textured disc, a fruit mask and a small blemish patch."""
    size = 120
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - 60) ** 2 + (xx - 60) ** 2 <= 45 ** 2
    image = np.full((size, size, 3), 200, dtype=np.uint8)
    image[disc] = (60, 90, 150)
    image[::3, :] = np.clip(image[::3, :].astype(int) - 25, 0, 255).astype(np.uint8)
    fruit = (disc.astype(np.uint8)) * 255
    blemish = np.zeros((size, size), dtype=np.uint8)
    blemish[50:70, 50:70] = 255
    return image, fruit, blemish


def test_regional_vector_has_two_blocks_each_for_t1_and_t2():
    image, fruit, blemish = _apple_and_masks()
    t1, t2 = DominantColourExtractor(), GLCMExtractor()
    vector = _regional_vector(t1, t2, image, fruit, blemish)
    assert vector.shape == (2 * t1.dim + 2 * t2.dim,)
    assert np.all(np.isfinite(vector))


def test_regional_vector_zero_fills_a_missing_blemish_region():
    image, fruit, _ = _apple_and_masks()
    t1, t2 = DominantColourExtractor(), GLCMExtractor()
    empty_blemish = np.zeros(fruit.shape, dtype=np.uint8)
    vector = _regional_vector(t1, t2, image, fruit, empty_blemish)
    # Layout: [T1 healthy | T1 blemished | T2 healthy | T2 blemished].
    t1_blemished = vector[t1.dim : 2 * t1.dim]
    t2_blemished = vector[2 * t1.dim + t2.dim :]
    assert np.count_nonzero(t1_blemished) == 0
    assert np.count_nonzero(t2_blemished) == 0
    # The healthy blocks still describe the whole fruit.
    assert np.count_nonzero(vector[: t1.dim]) > 0


def test_min_subregion_px_is_a_positive_pixel_count():
    assert MIN_SUBREGION_PX > 0


# --------------------------------------------------------------------------- #
# Pooled out-of-fold predictions
#
# The per-class tables and the ablation confusion matrices are built from these,
# so they carry the leakage guarantees the accuracy numbers carry. An augmented
# variant reaching the pool would mean a near-duplicate of a training image was
# scored as though it were held out, and the per-class recall would be measuring
# memorisation.
# --------------------------------------------------------------------------- #

def make_augmented_matrix(
    n_groups: int = 30,
    variants_per_image: int = 3,
    dim: int = 6,
    technique: str = "T",
    seed: int = 0,
) -> FeatureMatrix:
    """A matrix where every source image also carries near-duplicate variants."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=3.0, size=(3, dim))

    rows, labels, groups, variants, paths = [], [], [], [], []
    for group in range(n_groups):
        label = group % 3
        base = centres[label] + rng.normal(scale=1.0, size=dim)
        for variant in range(variants_per_image + 1):
            # Variants sit almost on top of their original, which is exactly
            # what makes a row-wise split unsafe.
            rows.append(base if variant == 0 else base + rng.normal(scale=0.01, size=dim))
            labels.append(label)
            groups.append(group)
            variants.append(variant)
            paths.append(Path(f"img_{group}_v{variant}.jpg"))

    n = len(rows)
    return FeatureMatrix(
        X=np.vstack(rows),
        y=np.asarray(labels, dtype=np.int64),
        technique=technique,
        extraction_times=np.full(n, 0.01),
        n_augmented=n - n_groups,
        failures=[], excluded=0, paths=paths,
        groups=np.asarray(groups, dtype=np.int64),
        variants=np.asarray(variants, dtype=np.int64),
    )


def test_detailed_matrix_cv_reports_exactly_what_the_plain_one_does():
    """The wrapper must not be able to drift from the loop it wraps.

    This is the guarantee that adding the per-class outputs cannot move a
    number already in enhancement_matrix.csv or ablation.csv.
    """
    matrix = make_matrix(seed=3)
    plain = cross_validate_matrix(matrix, technique="X")
    detailed, _ = cross_validate_matrix_detailed(matrix, technique="X")
    np.testing.assert_array_equal(plain.accuracy_folds, detailed.accuracy_folds)
    np.testing.assert_array_equal(plain.macro_f1_folds, detailed.macro_f1_folds)
    assert plain.technique == detailed.technique


def test_detailed_decision_fusion_reports_exactly_what_the_plain_one_does():
    matrices = {name: make_matrix(technique=name, seed=s)
                for s, name in enumerate(("T1", "T2", "T3"))}
    plain = cross_validate_decision_fusion(matrices)
    detailed, _ = cross_validate_decision_fusion_detailed(matrices)
    np.testing.assert_array_equal(plain.accuracy_folds, detailed.accuracy_folds)
    np.testing.assert_array_equal(plain.macro_f1_folds, detailed.macro_f1_folds)


def test_pooled_predictions_hold_one_row_per_source_image():
    matrix = make_matrix(n_groups=30, seed=4)
    _, pooled = cross_validate_matrix_detailed(matrix, technique="X")
    assert pooled.y_true.size == pooled.y_pred.size == pooled.rows.size == 30
    assert sorted(pooled.rows.tolist()) == list(range(30))


def test_pooled_predictions_never_include_an_augmented_variant():
    """The leakage invariant, stated over the pool rather than over the folds."""
    matrix = make_augmented_matrix(n_groups=25, variants_per_image=3, seed=5)
    _, pooled = cross_validate_matrix_detailed(matrix, technique="X")

    assert np.all(matrix.variants[pooled.rows] == 0)
    assert pooled.rows.size == 25
    # Every source image is represented, and none of them twice.
    assert sorted(matrix.groups[pooled.rows].tolist()) == list(range(25))


def test_pooled_predictions_carry_the_labels_of_the_rows_they_name():
    matrix = make_augmented_matrix(n_groups=18, seed=6)
    _, pooled = cross_validate_matrix_detailed(matrix, technique="X")
    np.testing.assert_array_equal(pooled.y_true, matrix.y[pooled.rows])


def test_pooled_accuracy_sits_inside_the_fold_accuracies():
    """Pooling is a re-average, not a different measurement."""
    matrix = make_matrix(n_groups=45, seed=7)
    report, pooled = cross_validate_matrix_detailed(matrix, technique="X")
    pooled_accuracy = float(np.mean(pooled.y_true == pooled.y_pred))
    assert report.accuracy_folds.min() - 1e-9 <= pooled_accuracy
    assert pooled_accuracy <= report.accuracy_folds.max() + 1e-9


def test_pooled_decision_fusion_predictions_are_originals_only():
    matrices = {
        name: make_augmented_matrix(n_groups=21, technique=name, seed=s)
        for s, name in enumerate(("T1", "T2", "T3"))
    }
    _, pooled = cross_validate_decision_fusion_detailed(matrices)
    reference = matrices["T1"]
    assert np.all(reference.variants[pooled.rows] == 0)
    assert pooled.rows.size == 21


# --------------------------------------------------------------------------- #
# The metric store
# --------------------------------------------------------------------------- #

def test_metric_store_serialises_without_recomputing_anything():
    """It records what it is handed; it must not derive a metric of its own."""
    truth = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 1, 1, 1, 2, 0])
    names = ["Unripe", "Ripe", "Rotten"]
    report = classification_metrics(truth, predicted, names, technique="X")

    store = MetricStore(names)
    store.add("X", "individual", "test", report)

    assert len(store.rows) == 3
    for row, label in zip(store.rows, names):
        entry = report.per_class.loc[label]
        assert row["class"] == label
        assert row["precision"] == pytest.approx(float(entry["precision"]))
        assert row["recall"] == pytest.approx(float(entry["recall"]))
        assert row["f1"] == pytest.approx(float(entry["f1"]))
        assert row["support"] == int(entry["support"])
        assert row["accuracy"] == pytest.approx(report.accuracy)
        assert row["source"] == "test"


def test_metric_store_keeps_the_two_provenances_apart(tmp_path):
    names = ["Unripe", "Ripe", "Rotten"]
    truth = np.array([0, 1, 2, 0, 1, 2])
    store = MetricStore(names)
    store.add("T1", "individual", "test",
              classification_metrics(truth, truth, names, technique="T1"))
    store.add("T1", "individual", "cv_out_of_fold",
              classification_metrics(truth, truth[::-1], names, technique="T1"))
    store.write(tmp_path)

    frame = pd.read_csv(tmp_path / "per_class.csv")
    assert set(frame["source"]) == {"test", "cv_out_of_fold"}
    # One configuration, two provenances, three classes each.
    assert len(frame) == 6
    assert (tmp_path / "confusion" / "T1.csv").exists()
    assert (tmp_path / "confusion" / "cv" / "T1.csv").exists()


def test_metric_store_writes_row_normalised_confusion_matrices(tmp_path):
    names = ["Unripe", "Ripe", "Rotten"]
    truth = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 1, 1, 1, 2, 0])
    store = MetricStore(names)
    store.add_out_of_fold(
        "E1_feature_fusion__T1+T2", "ablation",
        _pooled(truth, predicted),
    )
    store.write(tmp_path)

    written = tmp_path / "confusion" / "cv" / "E1_feature_fusion__T1+T2.csv"
    matrix = pd.read_csv(written, index_col=0)
    assert list(matrix.index) == names
    assert list(matrix.columns) == names
    np.testing.assert_allclose(matrix.to_numpy().sum(axis=1), np.ones(3))


def _pooled(truth: np.ndarray, predicted: np.ndarray) -> OutOfFoldPredictions:
    return OutOfFoldPredictions(
        technique="arm",
        y_true=truth,
        y_pred=predicted,
        rows=np.arange(truth.size),
    )


# --------------------------------------------------------------------------- #
# E2's block layout
#
# The ablation arms are column slices of the E2 matrix, so if the layout these
# indices describe ever stops matching what _regional_vector emits, an arm
# would be quietly built from the wrong half of the vector and still produce a
# plausible accuracy. These tests are what make that impossible.
# --------------------------------------------------------------------------- #

def test_block_columns_tile_the_whole_vector_without_overlap():
    blocks = e2_block_columns(37, 40)
    covered = np.concatenate([blocks[name] for name in
                              ("t1_healthy", "t1_blemished", "t2_healthy", "t2_blemished")])
    assert covered.tolist() == list(range(154))


def test_block_columns_match_the_order_regional_vector_emits():
    """T1 healthy, T1 blemished, T2 healthy, T2 blemished - in that order."""
    image, fruit, blemish = _apple_and_masks()
    t1 = DominantColourExtractor(n_colours=3, seed=0)
    t2 = GLCMExtractor(distances=(1,), angles_deg=(0.0,), levels=8)
    vector = _regional_vector(t1, t2, image, fruit, blemish)

    blocks = e2_block_columns(t1.dim, t2.dim)
    assert vector.size == 2 * t1.dim + 2 * t2.dim
    # The healthy region is non-empty and the blemished one is too, so no block
    # is the zero-fill; each must equal the extractor run on its own region.
    healthy = (fruit.astype(bool) & ~blemish.astype(bool)).astype(np.uint8) * 255
    np.testing.assert_allclose(
        vector[blocks["t1_healthy"]], t1(image.copy(), healthy)
    )
    np.testing.assert_allclose(
        vector[blocks["t2_healthy"]], t2(image.copy(), healthy)
    )


# --------------------------------------------------------------------------- #
# The random-mask control
#
# This is the arm that decides whether E2's gain is about locating damage or
# merely about having twice as many columns, so the control has to hold
# everything except the location fixed. Each property below is one of the
# things it must not accidentally vary.
# --------------------------------------------------------------------------- #

def _disc_masks():
    """A circular fruit with two blemishes on it."""
    fruit = np.zeros((224, 224), dtype=np.uint8)
    cv2.circle(fruit, (112, 112), 80, 255, thickness=-1)
    blemish = np.zeros((224, 224), dtype=np.uint8)
    cv2.circle(blemish, (92, 92), 18, 255, thickness=-1)
    cv2.circle(blemish, (138, 128), 11, 255, thickness=-1)
    return fruit, (blemish & fruit)


def test_control_mask_preserves_the_blemished_area():
    """Equivalent area is the whole point: otherwise the split differs too."""
    fruit, blemish = _disc_masks()
    control = random_control_mask(fruit, blemish, np.random.default_rng(0))
    target = int(np.count_nonzero(blemish))
    assert int(np.count_nonzero(control)) >= 0.95 * target


def test_control_mask_stays_inside_the_fruit():
    fruit, blemish = _disc_masks()
    control = random_control_mask(fruit, blemish, np.random.default_rng(1))
    assert not np.any((control > 0) & (fruit == 0))


def test_control_mask_preserves_the_component_structure():
    """Rotating keeps the coherency T1's block B measures; scattering would not."""
    fruit, blemish = _disc_masks()
    control = random_control_mask(fruit, blemish, np.random.default_rng(2))
    real, _ = cv2.connectedComponents((blemish > 0).astype(np.uint8))
    spun, _ = cv2.connectedComponents((control > 0).astype(np.uint8))
    assert spun == real


def test_control_mask_actually_moves_the_blemish():
    """A control that landed back on the real blemish would test nothing."""
    fruit, blemish = _disc_masks()
    overlaps = []
    for seed in range(6):
        control = random_control_mask(fruit, blemish, np.random.default_rng(seed))
        overlaps.append(
            np.count_nonzero((control > 0) & (blemish > 0)) / np.count_nonzero(blemish)
        )
    assert min(overlaps) < 0.5


def test_control_mask_is_deterministic_for_one_image():
    fruit, blemish = _disc_masks()
    first = random_control_mask(fruit, blemish, np.random.default_rng([42, 7, 0]))
    second = random_control_mask(fruit, blemish, np.random.default_rng([42, 7, 0]))
    assert np.array_equal(first, second)


def test_control_mask_of_a_clean_apple_is_empty():
    """The Unripe class has genuinely clean fruit; there is nothing to relocate."""
    fruit, _ = _disc_masks()
    control = random_control_mask(fruit, np.zeros_like(fruit), np.random.default_rng(0))
    assert control.shape == fruit.shape
    assert not np.any(control)


def test_control_mask_reads_no_label_and_no_other_image():
    """The leakage invariant for the control: it is a function of one image.

    The mask is derived from that image's own fruit and blemish masks plus a
    seed, so nothing about the class it belongs to, the fold it lands in, or
    any other image can reach it.
    """
    fruit, blemish = _disc_masks()
    signature = inspect.signature(random_control_mask)
    assert list(signature.parameters) == ["fruit_mask", "blemish_mask", "rng", "attempts"]
    # Same inputs, same seed, same answer - regardless of what ran before it.
    np.random.default_rng(999).uniform(size=1000)
    control = random_control_mask(fruit, blemish, np.random.default_rng([1, 2, 3]))
    np.random.default_rng(1).uniform(size=1000)
    again = random_control_mask(fruit, blemish, np.random.default_rng([1, 2, 3]))
    assert np.array_equal(control, again)

