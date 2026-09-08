"""Tests for per-class metric collection and serialisation.

Per-class precision and recall cannot be recovered from a macro F1 after the
fact, so the per-fold predictions have to be retained while the fold is being
scored. These pin that the new field is additive, that pooling is done on
counts rather than on ratios, and that a configuration failing one class is
reported as failing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluate import (
    CrossValidationReport,
    collect_per_class,
    per_class_pass_table,
    per_class_rows,
)

# --------------------------------------------------------------------------- #
# Per-class metrics
# --------------------------------------------------------------------------- #

def _confusions() -> np.ndarray:
    """Two folds whose pooled counts are easy to verify by hand."""
    fold = np.array([[8, 2, 0], [1, 7, 2], [0, 3, 7]], dtype=np.float64)
    return np.stack([fold, fold])


def test_cross_validation_report_defaults_to_no_per_class():
    """The new field is additive: an old-style report still constructs."""
    report = CrossValidationReport(
        technique="T1",
        accuracy_folds=np.array([0.8, 0.9]),
        macro_f1_folds=np.array([0.8, 0.9]),
    )
    assert not report.has_per_class
    assert report.mean_accuracy == pytest.approx(0.85)
    assert report.per_class_frame(["Unripe", "Ripe", "Rotten"]).empty


def test_per_class_frame_pools_counts_across_folds():
    """Precision and recall come from pooled counts, not averaged ratios."""
    report = CrossValidationReport(
        technique="T1",
        accuracy_folds=np.array([0.73, 0.73]),
        macro_f1_folds=np.array([0.73, 0.73]),
        fold_confusions=_confusions(),
    )
    frame = report.per_class_frame(["Unripe", "Ripe", "Rotten"]).set_index("class")

    # Unripe: 16 correct, 2 predicted-Unripe from Ripe -> precision 16/18.
    assert frame.loc["Unripe", "precision"] == pytest.approx(16 / 18)
    assert frame.loc["Unripe", "recall"] == pytest.approx(16 / 20)
    assert frame.loc["Unripe", "support"] == 20
    assert frame.support.sum() == 60


def test_per_class_rows_accepts_a_class_index_or_a_class_column():
    """``class`` is a Python keyword, which breaks the obvious implementation."""
    indexed = pd.DataFrame(
        {"precision": [0.9], "recall": [0.8], "f1": [0.85], "support": [10]},
        index=pd.Index(["Unripe"], name="class"),
    )
    columned = pd.DataFrame(
        {"class": ["Unripe"], "precision": [0.9], "recall": [0.8],
         "f1": [0.85], "support": [10]}
    )
    from_index = per_class_rows("T1", "test", indexed)
    from_column = per_class_rows("T1", "cv", columned)

    assert from_index[0]["class"] == "Unripe"
    assert from_column[0]["class"] == "Unripe"
    assert from_index[0]["precision"] == pytest.approx(0.9)


def test_pass_table_fails_a_config_whose_weakest_class_misses_the_target():
    """A configuration can average well and still fail a class."""
    per_class = pd.DataFrame(
        [
            {"config": "T1", "partition": "test", "class": "Unripe",
             "precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 60},
            {"config": "T1", "partition": "test", "class": "Ripe",
             "precision": 0.92, "recall": 0.90, "f1": 0.91, "support": 60},
            {"config": "T1", "partition": "test", "class": "Rotten",
             "precision": 0.88, "recall": 0.55, "f1": 0.68, "support": 60},
        ]
    )
    table = per_class_pass_table(per_class)
    assert not bool(table.per_class_pass.iloc[0])
    assert table.weakest_class.iloc[0] == "Rotten"
    assert table.min_recall.iloc[0] == pytest.approx(0.55)


def test_collect_per_class_labels_both_partitions():
    """Test and cross-validated rows must be distinguishable, not merged."""
    class _Report:
        per_class = pd.DataFrame(
            {"precision": [0.9, 0.8, 0.7], "recall": [0.9, 0.8, 0.7],
             "f1": [0.9, 0.8, 0.7], "support": [10, 10, 10]},
            index=pd.Index(["Unripe", "Ripe", "Rotten"], name="class"),
        )

    cv = CrossValidationReport(
        technique="T1",
        accuracy_folds=np.array([0.73, 0.73]),
        macro_f1_folds=np.array([0.73, 0.73]),
        fold_confusions=_confusions(),
    )
    frame = collect_per_class({"T1": _Report()}, {"T1": cv},
                              ["Unripe", "Ripe", "Rotten"])
    assert set(frame.partition) == {"test", "cv"}
    assert len(frame) == 6


