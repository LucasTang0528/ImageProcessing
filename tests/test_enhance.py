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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from enhance import (  # noqa: E402
    E1_PCA_VARIANCE,
    MIN_SUBREGION_PX,
    build_fusion_pipeline,
    cross_validate_decision_fusion,
    cross_validate_matrix,
    fit_decision_fusion,
    macro_f1_weights,
    stack_feature_matrices,
    weighted_soft_vote,
    _regional_vector,
)
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
