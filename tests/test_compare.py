"""Tests for Phase 4, the ranked significance-tested comparison.

Everything here runs on synthetic per-fold scores. The comparison consumes
numbers that Phase 3 already produced, so it needs neither the dataset nor a
fitted model, and these tests pin the statistics rather than the pipeline.

Run from the project root::

    python -m pytest tests/test_compare.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from compare import (  # noqa: E402
    background_only_accuracy,
    compare,
    holm_adjust,
    load_fold_scores,
    paired_fold_t_test,
)


# --------------------------------------------------------------------------- #
# paired_fold_t_test
# --------------------------------------------------------------------------- #

def test_identical_fold_scores_are_not_significant():
    """Two techniques that scored the same on every fold give p = 1.0."""
    scores = [0.71, 0.73, 0.70, 0.72, 0.69]
    result = paired_fold_t_test(scores, scores, "T1", "T2")
    assert result.mean_difference == 0.0
    assert result.p_value == 1.0
    assert result.better is None


def test_a_clear_separation_is_flagged_significant():
    """A consistent per-fold gap produces a small p-value and names the winner."""
    strong = [0.82, 0.84, 0.81, 0.83, 0.80]
    weak = [0.71, 0.73, 0.70, 0.72, 0.69]
    result = paired_fold_t_test(strong, weak, "T1", "T3")
    assert result.mean_difference == pytest.approx(0.11, abs=1e-9)
    assert result.better == "T1"
    assert result.p_value < 0.001
    assert result.n_folds == 5


def test_the_test_is_symmetric_in_its_arguments():
    """Swapping the arguments flips the sign but not the p-value."""
    a = [0.80, 0.78, 0.83, 0.79, 0.81]
    b = [0.74, 0.75, 0.72, 0.77, 0.73]
    forward = paired_fold_t_test(a, b, "A", "B")
    backward = paired_fold_t_test(b, a, "B", "A")
    assert forward.p_value == pytest.approx(backward.p_value)
    assert forward.mean_difference == pytest.approx(-backward.mean_difference)


def test_mismatched_fold_counts_are_refused():
    with pytest.raises(ValueError, match="aligned"):
        paired_fold_t_test([0.1, 0.2, 0.3], [0.1, 0.2], "A", "B")


def test_a_single_fold_is_refused():
    with pytest.raises(ValueError, match="at least two folds"):
        paired_fold_t_test([0.8], [0.7], "A", "B")


# --------------------------------------------------------------------------- #
# holm_adjust
# --------------------------------------------------------------------------- #

def test_holm_on_an_empty_family_is_empty():
    assert holm_adjust([]) == []


def test_holm_leaves_a_single_p_value_untouched():
    assert holm_adjust([0.04]) == [pytest.approx(0.04)]


def test_holm_is_step_down_and_monotone():
    """The smallest p is scaled by m, the next by m-1, and the sequence
    never decreases once sorted."""
    raw = [0.01, 0.02, 0.04]
    adjusted = holm_adjust(raw)
    assert adjusted[0] == pytest.approx(0.03)   # 3 * 0.01
    assert adjusted[1] == pytest.approx(0.04)   # max(0.03, 2 * 0.02)
    assert adjusted[2] == pytest.approx(0.04)   # max(0.04, 1 * 0.04)
    assert adjusted == sorted(adjusted)


def test_holm_preserves_input_order_and_clips_at_one():
    raw = [0.9, 0.01, 0.5]
    adjusted = holm_adjust(raw)
    assert adjusted[1] == pytest.approx(0.03)
    assert all(value <= 1.0 for value in adjusted)


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #

def _folds():
    return {
        "T1": [0.80, 0.82, 0.79, 0.81, 0.80],
        "T2": [0.70, 0.72, 0.69, 0.71, 0.70],
        "T3": [0.71, 0.70, 0.72, 0.71, 0.71],
    }


def test_ranking_is_by_mean_fold_score():
    report = compare(_folds())
    assert report.ranking == ["T1", "T3", "T2"]
    assert report.best == "T1"
    assert report.runner_up == "T3"


def test_every_unordered_pair_is_tested_once():
    report = compare(_folds())
    pairs = {frozenset((t.technique_a, t.technique_b)) for t in report.pairwise}
    assert pairs == {
        frozenset(("T1", "T2")),
        frozenset(("T1", "T3")),
        frozenset(("T2", "T3")),
    }
    assert len(report.holm_p) == 3


def test_best_beats_runner_up_uses_the_holm_adjusted_p():
    report = compare(_folds(), alpha=0.05)
    test, adjusted = report.pair(report.best, report.runner_up)
    assert report.best_beats_runner_up == (adjusted < 0.05)
    # T1 is ~10 points clear of T3 on every fold: this must land significant.
    assert report.best_beats_runner_up is True


def test_a_dead_heat_at_the_top_is_reported_as_not_significant():
    folds = {
        "T1": [0.750, 0.752, 0.749, 0.751, 0.750],
        "T2": [0.748, 0.751, 0.750, 0.749, 0.752],
    }
    report = compare(folds)
    assert report.best_beats_runner_up is False
    assert "does not support calling it the winner" in report.summary_text()


def test_disagreeing_fold_counts_are_refused():
    with pytest.raises(ValueError, match="fold count"):
        compare({"T1": [0.8, 0.8, 0.8], "T2": [0.7, 0.7]})


def test_compare_needs_at_least_one_technique():
    with pytest.raises(ValueError, match="at least one technique"):
        compare({})


def test_single_technique_has_no_comparison():
    report = compare({"T1": [0.8, 0.81, 0.79, 0.8, 0.82]})
    assert report.best_beats_runner_up is None
    assert "no comparison" in report.summary_text().lower()


def test_matrix_frame_carries_rank_cv_and_target_columns():
    report = compare(
        _folds(),
        test_accuracy={"T1": 0.79, "T2": 0.68, "T3": 0.70},
        dimensionality={"T1": 37, "T2": 40, "T3": 36},
    )
    frame = report.matrix_frame()
    assert list(frame.index) == ["T1", "T3", "T2"]
    assert frame.loc["T1", "rank"] == 1
    assert frame.loc["T1", "dimensionality"] == 37
    assert frame.loc["T1", "test_accuracy"] == pytest.approx(0.79)
    # Mean of T1's folds is 0.804, over the 0.80 bar; T3 is under it.
    assert bool(frame.loc["T1", "clears_target"]) is True
    assert bool(frame.loc["T3", "clears_target"]) is False


def test_pairwise_frame_has_one_row_per_pair_with_holm_column():
    report = compare(_folds())
    frame = report.pairwise_frame()
    assert len(frame) == 3
    assert {"p_value", "p_value_holm", "significant_holm", "better"} <= set(frame.columns)


def test_control_margin_appears_in_the_summary():
    report = compare(_folds(), control_accuracy=0.74)
    text = report.summary_text()
    assert "Background-only control: 0.7400" in text
    assert "+0.06" in text or "+0.0640" in text


# --------------------------------------------------------------------------- #
# loading from disk
# --------------------------------------------------------------------------- #

def test_load_fold_scores_round_trips_the_benchmark_matrix(tmp_path):
    frame = pd.DataFrame(
        {
            "technique": ["T1", "T2"],
            "accuracy": [0.79, 0.68],
            "cv_fold1_accuracy": [0.80, 0.70],
            "cv_fold2_accuracy": [0.82, 0.72],
            "cv_fold3_accuracy": [0.79, 0.69],
            "cv_fold1_macro_f1": [0.79, 0.69],
            "cv_fold2_macro_f1": [0.81, 0.71],
            "cv_fold3_macro_f1": [0.78, 0.68],
        }
    ).set_index("technique")
    path = tmp_path / "benchmark_matrix.csv"
    frame.to_csv(path)

    accuracy = load_fold_scores(path, metric="accuracy")
    assert set(accuracy) == {"T1", "T2"}
    np.testing.assert_allclose(accuracy["T1"], [0.80, 0.82, 0.79])

    macro = load_fold_scores(path, metric="macro_f1")
    np.testing.assert_allclose(macro["T2"], [0.69, 0.71, 0.68])


def test_load_fold_scores_orders_folds_numerically(tmp_path):
    """fold10 must not sort before fold2."""
    columns = {f"cv_fold{k}_accuracy": [0.5 + 0.01 * k] for k in range(1, 12)}
    frame = pd.DataFrame({"technique": ["T1"], **columns}).set_index("technique")
    path = tmp_path / "benchmark_matrix.csv"
    frame.to_csv(path)
    scores = load_fold_scores(path, metric="accuracy")
    np.testing.assert_allclose(scores["T1"], [0.5 + 0.01 * k for k in range(1, 12)])


def test_load_fold_scores_without_the_metric_columns_is_an_error(tmp_path):
    frame = pd.DataFrame({"technique": ["T1"], "accuracy": [0.8]}).set_index("technique")
    path = tmp_path / "benchmark_matrix.csv"
    frame.to_csv(path)
    with pytest.raises(ValueError, match="per-fold"):
        load_fold_scores(path, metric="accuracy")


def test_load_fold_scores_missing_file():
    with pytest.raises(FileNotFoundError):
        load_fold_scores(Path("does/not/exist.csv"))


def test_background_only_accuracy_averages_the_per_class_recall(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    pd.DataFrame(
        {"class": ["Unripe", "Ripe", "Rotten"],
         "background_only_recall": [0.925, 0.45625, 0.8375]}
    ).to_csv(audit / "primary_background_only.csv", index=False)
    value = background_only_accuracy(tmp_path, dataset="primary")
    assert value == pytest.approx(0.7396, abs=1e-3)


def test_background_only_accuracy_is_none_when_the_audit_is_absent(tmp_path):
    assert background_only_accuracy(tmp_path, dataset="primary") is None
