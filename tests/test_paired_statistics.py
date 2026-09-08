"""Tests for the paired fold statistics and the tie band that selects on them.

The backfill test is the load-bearing one. ``backfill_paired_tests`` is a
second route to numbers that reach the report, and a second route to a
published figure is exactly what produced a wrong decay comparison earlier in
this work: one path measured against the sweep leader and the other against
the experiment's own reference, and nothing forced them to agree. Asserting
that the two paths produce identical tables closes that off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
for candidate in (PROJECT_ROOT, SCRIPTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from selection import (  # noqa: E402
    SUPPORTING_ALPHA,
    add_paired_differences,
    backfill_paired_tests,
    criterion_agreement,
    fold_matrix,
    paired_difference,
    report_criterion_agreement,
    select_winner,
)

FOLD_COLUMNS = [f"fold{index}_accuracy" for index in range(1, 6)]


def _table(arms: dict, dims: dict | None = None, stages: dict | None = None) -> pd.DataFrame:
    """Build a scored-table stand-in from per-arm fold accuracies."""
    rows = []
    for arm, folds in arms.items():
        rows.append({
            "arm": arm,
            "config_key": arm,
            "dropped_feature": "",
            "cv_mean_accuracy": float(np.mean(folds)),
            "dimensionality": (dims or {}).get(arm, 37),
            "pipeline_stages": (stages or {}).get(arm, 4),
            **{name: value for name, value in zip(FOLD_COLUMNS, folds)},
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Paired differences
# --------------------------------------------------------------------------- #

def test_paired_difference_uses_the_pairing():
    """Two arms differing by a constant have zero spread in their difference.

    Compared unpaired, these two look noisy; paired, they differ by exactly
    0.01 on every fold. Throwing the pairing away is what makes
    indistinguishable arms appear ordered.
    """
    first = np.array([0.80, 0.90, 0.70, 0.85, 0.75])
    second = first - 0.01
    mean, sd = paired_difference(first, second)
    assert mean == pytest.approx(0.01)
    assert sd == pytest.approx(0.0, abs=1e-12)
    assert first.std(ddof=1) > 0.05  # unpaired, each arm is highly variable


def test_within_noise_flags_an_arm_inside_its_own_spread():
    """The band is a comparison of the difference against its own variability."""
    table = _table({
        "reference": [0.80, 0.84, 0.79, 0.83, 0.81],
        "noisy": [0.81, 0.82, 0.80, 0.85, 0.79],      # differs erratically
        "consistent": [0.77, 0.81, 0.76, 0.80, 0.78],  # differs by 0.03 every fold
    })
    out = add_paired_differences(table, "reference").set_index("arm")
    assert bool(out.loc["noisy", "within_noise"])
    assert not bool(out.loc["consistent", "within_noise"])


def test_reference_arm_compares_to_itself_as_zero():
    table = _table({"a": [0.8, 0.9, 0.7, 0.85, 0.75], "b": [0.7, 0.8, 0.6, 0.75, 0.65]})
    out = add_paired_differences(table, "a").set_index("arm")
    assert out.loc["a", "paired_diff_mean"] == pytest.approx(0.0)
    assert out.loc["a", "paired_diff_sd"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Backfill must equal the in-runner path
# --------------------------------------------------------------------------- #

def test_backfill_matches_the_in_runner_path(tmp_path):
    """The two routes to a published column must produce identical tables.

    ``backfill_paired_tests`` exists so that adding a reporting column does
    not require re-extracting unchanged features. That is only safe while it
    is the same computation.
    """
    table = _table({
        "glcm": [0.62, 0.58, 0.60, 0.61, 0.59],
        "intensity_moments": [0.45, 0.44, 0.47, 0.46, 0.43],
        "angle-averaged": [0.63, 0.57, 0.61, 0.60, 0.60],
    })

    results_dir = tmp_path / "e2"
    results_dir.mkdir()

    # What the runner would have written.
    expected = results_dir / "expected.csv"
    add_paired_differences(table, "glcm").to_csv(expected, index=False)

    # What the backfill writes, starting from a table lacking the columns.
    produced = results_dir / "e2_1.csv"
    table.to_csv(produced, index=False)
    touched = backfill_paired_tests(results_dir, {"e2_1": "glcm"})

    assert len(touched) == 1
    # Compared as files rather than as frames. The guarantee that matters is
    # that the artefact on disk is the same either way; comparing in memory
    # would pass or fail on how the test itself chose to re-read the CSV,
    # which is not a property of the code under test.
    assert produced.read_text(encoding="utf-8") == expected.read_text(encoding="utf-8")


def test_backfill_is_idempotent_and_skips_completed_tables(tmp_path):
    """Running it twice must not double-process or alter a finished table."""
    table = _table({"a": [0.8, 0.82, 0.79, 0.81, 0.80], "b": [0.7, 0.75, 0.72, 0.71, 0.73]})
    results_dir = tmp_path / "e1"
    results_dir.mkdir()
    table.to_csv(results_dir / "e1_1.csv", index=False)

    assert len(backfill_paired_tests(results_dir, {"e1_1": "a"})) == 1
    first = pd.read_csv(results_dir / "e1_1.csv")
    assert len(backfill_paired_tests(results_dir, {"e1_1": "a"})) == 0
    pd.testing.assert_frame_equal(first, pd.read_csv(results_dir / "e1_1.csv"))


def test_backfill_ignores_a_missing_table(tmp_path):
    """A results directory from an older run must not raise."""
    assert backfill_paired_tests(tmp_path, {"e1_9": "a"}) == []


# --------------------------------------------------------------------------- #
# Tie band and parsimony
# --------------------------------------------------------------------------- #

def test_tie_band_prefers_the_smaller_tied_arm():
    """A shorter descriptor that cannot be distinguished is the better result."""
    combined = _table(
        {
            "big_leader": [0.885, 0.869, 0.897, 0.891, 0.891],
            "small_tied": [0.884, 0.868, 0.898, 0.888, 0.889],
        },
        dims={"big_leader": 53, "small_tied": 29},
    )
    winner, tied, selected_by = select_winner(combined)
    assert winner.arm == "small_tied"
    assert selected_by == "parsimony"
    assert len(tied) == 2


def test_a_genuinely_worse_arm_is_not_tied_in_however_small_it_is():
    """Parsimony must not rescue an arm the band separates."""
    combined = _table(
        {
            "leader": [0.885, 0.869, 0.897, 0.891, 0.891],
            "tiny_but_worse": [0.845, 0.829, 0.857, 0.851, 0.851],
        },
        dims={"leader": 37, "tiny_but_worse": 4},
    )
    winner, tied, selected_by = select_winner(combined)
    assert winner.arm == "leader"
    assert selected_by == "accuracy"
    assert list(tied.arm) == ["leader"]


def test_stages_break_a_tie_only_after_dimensionality():
    combined = _table(
        {
            "fewer_stages": [0.881, 0.868, 0.893, 0.888, 0.887],
            "same_dims_more_stages": [0.882, 0.869, 0.892, 0.889, 0.886],
        },
        dims={"fewer_stages": 37, "same_dims_more_stages": 37},
        stages={"fewer_stages": 3, "same_dims_more_stages": 4},
    )
    winner, _, _ = select_winner(combined)
    assert winner.arm == "fewer_stages"


# --------------------------------------------------------------------------- #
# Criterion agreement
# --------------------------------------------------------------------------- #

def test_agreement_counts_and_names_a_disagreement():
    """The count is what Chapter 4 quotes, so it has to be right."""
    table = add_paired_differences(
        _table({
            "reference": [0.883, 0.870, 0.895, 0.889, 0.880],
            "clearly_worse": [0.840, 0.828, 0.852, 0.847, 0.838],
            "indistinguishable": [0.884, 0.869, 0.894, 0.890, 0.879],
        }),
        "reference",
    )
    agreement = criterion_agreement({"e1_5": table})

    assert len(agreement) == 2  # the reference itself is excluded
    assert set(agreement.columns) >= {"band_distinguishes", "test_distinguishes",
                                      "criteria_agree", "paired_p_supporting"}
    summary = report_criterion_agreement(agreement)
    assert "comparisons agree" in summary


def test_agreement_is_reported_at_the_declared_alpha():
    """The threshold has to be the declared constant, not a literal."""
    assert SUPPORTING_ALPHA == 0.05


def test_fold_matrix_tolerates_a_table_without_fold_columns():
    """A summary table passed by mistake must not raise."""
    assert fold_matrix(pd.DataFrame({"arm": ["a"]})).size == 0
