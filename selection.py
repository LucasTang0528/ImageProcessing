"""Comparing sweep arms, and choosing between them without selecting on noise.

The sub-experiment sweeps produce many arms scored on the same five folds. Two
things follow from that, and this module exists because neither is handled by
comparing mean accuracies.

**The arms are paired.** Every arm sees the same folds, drawn from the same
seed over the same images, so the fold-wise *difference* between two arms
carries far less variance than either arm's accuracy does. Comparing means
discards that pairing and makes indistinguishable arms look ordered.

**Most arms are indistinguishable.** On these sweeps the great majority of
comparisons sit inside their own spread. Ranking on mean accuracy therefore
picks a winner on noise, and if a single held-out evaluation is then spent on
that winner it reports a distinction the data does not contain.

The rule here is a band, not a hypothesis test: two arms are tied when the
mean of their paired fold differences is smaller than the standard deviation
of those same differences. A t statistic is recorded alongside as a supporting
note, and deliberately is not the claim - many arms are compared against one
reference with no correction for multiplicity, and cross-validation folds share
training rows, so the differences are not independent draws. A band assumes
neither.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

#: Fold-accuracy columns the experiment runners emit.
FOLD_COLUMNS = tuple(f"fold{index}_accuracy" for index in range(1, 6))

#: Columns that are text rather than measurements. Listed so a value of ""
#: survives a CSV round-trip instead of returning as a missing number, which
#: would make a backfilled table stop matching the one a run produced.
STRING_COLUMNS = (
    "arm", "config_key", "dropped_feature", "reference_arm", "representation",
    "prediction_sha256", "distances", "blocks", "space", "selected_by",
    "highest_accuracy_arm", "selected_on", "blemish_method", "se_shape",
)

#: The threshold the supporting t-test would be read at, if it were the claim.
#: It is not; it is recorded so a report can state exactly how often the two
#: criteria would disagree, and where.
SUPPORTING_ALPHA = 0.05


def fold_matrix(table: pd.DataFrame) -> np.ndarray:
    """Return the per-fold accuracies of a scored table as ``(arms, folds)``."""
    present = [column for column in FOLD_COLUMNS if column in table.columns]
    return table[present].to_numpy(dtype=float) if present else np.empty((0, 0))


def paired_difference(first: Sequence[float], second: Sequence[float]) -> Tuple[float, float]:
    """Mean and standard deviation of the fold-wise difference between two arms."""
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    if difference.size < 2:
        return (float(difference.mean()) if difference.size else 0.0), 0.0
    return float(difference.mean()), float(difference.std(ddof=1))


def add_paired_differences(table: pd.DataFrame, reference_arm: str) -> pd.DataFrame:
    """Attach paired fold-difference statistics against one reference arm.

    ``within_noise`` is the reportable quantity: an arm whose mean fold-wise
    difference from the reference is smaller than the standard deviation of
    those differences lies inside the stated noise band, and the sweep does
    not distinguish it from the reference.

    ``paired_t_supporting`` and ``paired_p_supporting`` are a supporting note
    only, for the reasons in the module docstring.
    """
    folds = fold_matrix(table)
    if folds.size == 0 or reference_arm not in set(table.arm):
        return table
    reference = folds[list(table.arm).index(reference_arm)]

    means, sds, within, t_stats, p_values = [], [], [], [], []
    for row in range(len(table)):
        mean, sd = paired_difference(folds[row], reference)
        means.append(mean)
        sds.append(sd)
        within.append(bool(abs(mean) < sd) if sd > 0 else bool(mean == 0.0))
        if sd > 0 and folds.shape[1] > 1:
            statistic, p_value = stats.ttest_rel(folds[row], reference)
            t_stats.append(float(statistic))
            p_values.append(float(p_value))
        else:
            t_stats.append(float("nan"))
            p_values.append(float("nan"))

    out = table.copy()
    out["reference_arm"] = reference_arm
    out["paired_diff_mean"] = means
    out["paired_diff_sd"] = sds
    out["within_noise"] = within
    out["paired_t_supporting"] = t_stats
    out["paired_p_supporting"] = p_values
    return out


def backfill_paired_tests(results_dir: Path, references: Mapping[str, str]) -> List[Path]:
    """Add the paired columns to tables written before they existed.

    Computed from the per-fold accuracies already stored in each CSV, so a
    backfilled table is identical to one produced by a fresh run. A test
    asserts that, because a second route to a published number is how a wrong
    comparison reached a table earlier in this work.
    """
    touched: List[Path] = []
    for name, reference_arm in references.items():
        path = Path(results_dir) / f"{name}.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        for column in STRING_COLUMNS:
            if column in table.columns:
                table[column] = table[column].fillna("").astype(str)
        if "paired_t_supporting" in table.columns or "arm" not in table.columns:
            continue
        table = table.drop(columns=[c for c in ("reference_arm", "paired_diff_mean",
                                                "paired_diff_sd", "within_noise")
                                    if c in table.columns])
        add_paired_differences(table, reference_arm).to_csv(path, index=False)
        touched.append(path)
    return touched


def select_winner(combined: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, str]:
    """Choose the arm to spend the single held-out evaluation on.

    Two arms are tied when their paired fold difference lies inside the stated
    noise band. Among the tied set the choice is parsimony - fewest
    dimensions, then fewest declared processing stages where that column
    exists, then arm name for determinism. A shorter descriptor that cannot be
    distinguished from a longer one is the better result, and choosing it
    makes the report defend the weaker claim rather than the stronger one.

    Returns the winning row, the tied set, and how the choice was made.
    """
    folds = fold_matrix(combined)
    leader = int(combined.cv_mean_accuracy.idxmax())
    leader_folds = folds[leader]

    tied_rows = []
    for row in range(len(combined)):
        mean, sd = paired_difference(leader_folds, folds[row])
        if row == leader or (sd > 0 and abs(mean) < sd) or mean == 0.0:
            tied_rows.append(row)

    tied = combined.iloc[tied_rows].copy()
    keys = ["dimensionality"]
    if "pipeline_stages" in tied.columns:
        keys.append("pipeline_stages")
    keys.append("arm")
    ordered = tied.sort_values(keys, ascending=[True] * len(keys))
    winner = ordered.iloc[0]
    selected_by = "accuracy" if winner.arm == combined.iloc[leader].arm else "parsimony"
    return winner, tied, selected_by


def criterion_agreement(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Count where the noise band and the supporting t-test would disagree.

    Stated rather than left in a column. If the two criteria agreed
    everywhere, the choice between them would not matter and fixing one in
    advance would be an empty precaution.
    """
    rows: List[Dict[str, object]] = []
    for name, table in tables.items():
        if "within_noise" not in table.columns:
            continue
        for row in table.itertuples():
            if row.arm == row.reference_arm:
                continue
            p_value = float(getattr(row, "paired_p_supporting", float("nan")))
            band_says = not bool(row.within_noise)
            test_says = bool(p_value < SUPPORTING_ALPHA) if p_value == p_value else band_says
            rows.append({
                "table": name,
                "arm": row.arm,
                "reference_arm": row.reference_arm,
                "paired_diff_mean": float(row.paired_diff_mean),
                "paired_diff_sd": float(row.paired_diff_sd),
                "band_distinguishes": band_says,
                "test_distinguishes": test_says,
                "paired_p_supporting": p_value,
                "criteria_agree": bool(band_says == test_says),
            })
    return pd.DataFrame(rows)


def report_criterion_agreement(agreement: pd.DataFrame) -> str:
    """One-paragraph summary of the agreement table, for the console and file."""
    if agreement.empty:
        return "No comparisons to summarise."
    total = len(agreement)
    agree = int(agreement.criteria_agree.sum())
    lines = [
        f"Noise band against the supporting t-test at alpha {SUPPORTING_ALPHA}:",
        f"  {agree} of {total} comparisons agree.",
    ]
    for row in agreement[~agreement.criteria_agree].itertuples():
        lines.append(
            f"  disagreement: {row.arm} vs {row.reference_arm} - "
            f"band {'distinguishes' if row.band_distinguishes else 'does not'}, "
            f"test {'distinguishes' if row.test_distinguishes else 'does not'} "
            f"(p = {row.paired_p_supporting:.4f}, "
            f"diff {row.paired_diff_mean * 100:.2f} pp, "
            f"sd {row.paired_diff_sd * 100:.2f} pp)"
        )
    if agree < total:
        lines.append(
            "  The criterion was fixed before these were computed. The band is "
            "the claim; the t-test is a supporting note."
        )
    return "\n".join(lines)


__all__ = [
    "FOLD_COLUMNS",
    "STRING_COLUMNS",
    "SUPPORTING_ALPHA",
    "add_paired_differences",
    "backfill_paired_tests",
    "criterion_agreement",
    "fold_matrix",
    "paired_difference",
    "report_criterion_agreement",
    "select_winner",
]
