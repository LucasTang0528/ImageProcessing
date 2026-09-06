"""Phase 4 - turn a benchmark run into a ranked, significance-tested comparison.

Phase 3 scores each technique in isolation. This module answers the question the
study exists to ask: which descriptor family is best, and is the gap between the
top two real or an artefact of one particular train/test split?

A difference of two or three accuracy points on a single held-out split can
reflect nothing more than which images happened to land in the test set. So the
ranking here is built on the five cross-validation folds drawn on the *training*
partition, and every pairwise gap is checked with a paired t-test over those
folds, comparing the two techniques fold by fold (Demsar, 2006). The held-out
test accuracy is reported alongside, but the ranking and every significance
claim rest on the folds, not on the single test split.

Nothing here reads the test partition or fits a model. It consumes the per-fold
scores that :mod:`scripts.run_benchmarks` already wrote and produces:

* ``comparison_matrix.csv`` - one row per technique, CV and test metrics together
* ``pairwise_ttests.csv``   - every technique pair: mean gap, t, p, Holm-adjusted p
* ``ranking.txt``           - the ranked list, the significance verdict, the controls

Two controls are printed next to the ranking rather than left implicit. The
**target** is the assignment's 80% bar for the best individual technique. The
**background-only control** is the accuracy the shared classifier reaches on the
border ring alone, with no fruit pixels in view (``scripts/audit_dataset.py``);
on this dataset it is far above chance, so an absolute accuracy only means
something read against it. Neither control affects the ranking - the comparison
between techniques stays valid because all three see identical masks - but a
headline number quoted without them is misleading.

Reference
---------
Demsar, J. (2006). Statistical comparisons of classifiers over multiple data
    sets. Journal of Machine Learning Research, 7, 1-30.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_ALPHA = 0.05
#: The assignment's accuracy bar for the strongest individual technique.
TARGET_ACCURACY = 0.80


# --------------------------------------------------------------------------- #
# Paired t-test over cross-validation folds
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FoldTTest:
    """A paired t-test between two techniques, fold by fold.

    Attributes:
        technique_a: Short name of the first technique.
        technique_b: Short name of the second technique.
        mean_a: Mean fold score for ``technique_a``.
        mean_b: Mean fold score for ``technique_b``.
        mean_difference: ``mean_a - mean_b``. Positive means ``a`` scored higher.
        t_statistic: Paired t statistic for the per-fold differences.
        p_value: Two-sided p-value, before any multiple-comparison correction.
        n_folds: Number of paired folds the test ran over.
    """

    technique_a: str
    technique_b: str
    mean_a: float
    mean_b: float
    mean_difference: float
    t_statistic: float
    p_value: float
    n_folds: int

    @property
    def better(self) -> Optional[str]:
        """The higher-scoring technique, or ``None`` when the means are equal."""
        if self.mean_a > self.mean_b:
            return self.technique_a
        if self.mean_b > self.mean_a:
            return self.technique_b
        return None


def paired_fold_t_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    technique_a: str = "A",
    technique_b: str = "B",
) -> FoldTTest:
    """Run a paired t-test on two techniques' per-fold scores.

    The two sequences must be aligned: entry ``i`` of each is the score on the
    same cross-validation fold, so the difference ``scores_a[i] - scores_b[i]``
    removes the fold-to-fold variation that both techniques share and isolates
    the variation between the techniques.

    Args:
        scores_a: One score per fold for the first technique.
        scores_b: One score per fold for the second technique, same fold order.
        technique_a: Short name for the first technique.
        technique_b: Short name for the second technique.

    Returns:
        A :class:`FoldTTest`.

    Raises:
        ValueError: If the sequences differ in length or hold fewer than two
            folds, in which case a paired t-test is not defined.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"fold scores must be aligned: {a.size} vs {b.size} folds"
        )
    if a.size < 2:
        raise ValueError("a paired t-test needs at least two folds")

    difference = a - b
    # ttest_rel warns and returns NaN when every paired difference is identical
    # (variance zero). Two techniques that scored the same on every fold are not
    # distinguishable, which is p = 1.0, not a missing value.
    if np.allclose(difference, difference[0]):
        t_statistic = 0.0 if np.isclose(difference[0], 0.0) else float("inf")
        p_value = 1.0 if np.isclose(difference[0], 0.0) else 0.0
    else:
        result = stats.ttest_rel(a, b)
        t_statistic = float(result.statistic)
        p_value = float(result.pvalue)

    return FoldTTest(
        technique_a=technique_a,
        technique_b=technique_b,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_difference=float(difference.mean()),
        t_statistic=t_statistic,
        p_value=p_value,
        n_folds=int(a.size),
    )


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjustment for a family of p-values.

    Running one t-test per technique pair inflates the chance that at least one
    comes out significant by luck. Holm's correction controls that family-wise
    error rate while being less conservative than a plain Bonferroni divide
    (Demsar, 2006, section 5).

    Args:
        p_values: The raw p-values, in any order.

    Returns:
        The adjusted p-values, in the original order, each clipped to ``1.0``
        and kept monotone with the raw ranking.
    """
    raw = np.asarray(p_values, dtype=np.float64)
    count = raw.size
    if count == 0:
        return []

    order = np.argsort(raw)
    adjusted = np.empty(count, dtype=np.float64)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * raw[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)
    return [float(value) for value in adjusted]


# --------------------------------------------------------------------------- #
# The comparison report
# --------------------------------------------------------------------------- #

@dataclass
class ComparisonReport:
    """A ranked comparison of every benchmarked technique.

    Attributes:
        metric: Name of the per-fold metric the ranking is built on.
        fold_scores: Per-fold scores per technique, in fold order.
        ranking: Technique short names, best first, by mean fold score.
        pairwise: One :class:`FoldTTest` per unordered technique pair.
        holm_p: Holm-adjusted p-value per pair, aligned with ``pairwise``.
        alpha: Significance threshold applied to the adjusted p-values.
        test_accuracy: Held-out test accuracy per technique, when supplied.
        dimensionality: Feature vector length per technique, when supplied.
        target_accuracy: The assignment's bar for the best individual technique.
        control_accuracy: Background-only accuracy, when known.
    """

    metric: str
    fold_scores: Dict[str, np.ndarray]
    ranking: List[str]
    pairwise: List[FoldTTest]
    holm_p: List[float]
    alpha: float
    test_accuracy: Dict[str, float]
    dimensionality: Dict[str, int]
    target_accuracy: float = TARGET_ACCURACY
    control_accuracy: Optional[float] = None

    @property
    def best(self) -> str:
        """Short name of the top-ranked technique."""
        return self.ranking[0]

    @property
    def runner_up(self) -> Optional[str]:
        """Short name of the second-ranked technique, if there is one."""
        return self.ranking[1] if len(self.ranking) > 1 else None

    def mean_score(self, technique: str) -> float:
        """Mean fold score for one technique."""
        return float(np.mean(self.fold_scores[technique]))

    def std_score(self, technique: str) -> float:
        """Standard deviation of the fold scores for one technique."""
        folds = self.fold_scores[technique]
        return float(np.std(folds, ddof=1)) if folds.size > 1 else 0.0

    def pair(self, technique_a: str, technique_b: str) -> tuple[FoldTTest, float]:
        """Return the t-test and its Holm-adjusted p for one technique pair."""
        wanted = {technique_a, technique_b}
        for test, adjusted in zip(self.pairwise, self.holm_p):
            if {test.technique_a, test.technique_b} == wanted:
                return test, adjusted
        raise KeyError(f"no comparison between {technique_a} and {technique_b}")

    @property
    def best_beats_runner_up(self) -> Optional[bool]:
        """Whether the best technique is significantly ahead of the second.

        ``None`` when there is only one technique to compare.
        """
        if self.runner_up is None:
            return None
        _, adjusted = self.pair(self.best, self.runner_up)
        return adjusted < self.alpha

    # ---- serialisation --------------------------------------------------- #

    def matrix_frame(self) -> pd.DataFrame:
        """One row per technique: CV mean/std, test accuracy, dimensionality."""
        rows = []
        for rank, technique in enumerate(self.ranking, start=1):
            rows.append(
                {
                    "technique": technique,
                    "rank": rank,
                    f"cv_mean_{self.metric}": self.mean_score(technique),
                    f"cv_std_{self.metric}": self.std_score(technique),
                    "test_accuracy": self.test_accuracy.get(technique, float("nan")),
                    "dimensionality": self.dimensionality.get(technique, -1),
                    "clears_target": (
                        self.mean_score(technique) >= self.target_accuracy
                        if self.metric == "accuracy"
                        else None
                    ),
                }
            )
        return pd.DataFrame(rows).set_index("technique")

    def pairwise_frame(self) -> pd.DataFrame:
        """One row per technique pair, with raw and Holm-adjusted p-values."""
        rows = []
        for test, adjusted in zip(self.pairwise, self.holm_p):
            rows.append(
                {
                    "technique_a": test.technique_a,
                    "technique_b": test.technique_b,
                    "mean_a": test.mean_a,
                    "mean_b": test.mean_b,
                    "mean_difference": test.mean_difference,
                    "t_statistic": test.t_statistic,
                    "p_value": test.p_value,
                    "p_value_holm": adjusted,
                    "significant_holm": adjusted < self.alpha,
                    "better": test.better or "tie",
                    "n_folds": test.n_folds,
                }
            )
        return pd.DataFrame(rows)

    def summary_text(self) -> str:
        """A human-readable ranking, verdict and control block."""
        lines: List[str] = []
        lines.append(f"Ranking by cross-validated {self.metric} "
                     f"(5-fold, training partition):")
        for rank, technique in enumerate(self.ranking, start=1):
            test_acc = self.test_accuracy.get(technique)
            tail = f"  test acc {test_acc:.4f}" if test_acc is not None else ""
            lines.append(
                f"  {rank}. {technique:<6} "
                f"{self.mean_score(technique):.4f} +/- {self.std_score(technique):.4f}"
                f"{tail}"
            )

        lines.append("")
        if self.runner_up is None:
            lines.append("Only one technique was benchmarked; no comparison to make.")
            return "\n".join(lines)

        test, adjusted = self.pair(self.best, self.runner_up)
        verdict = "significant" if adjusted < self.alpha else "NOT significant"
        lines.append(
            f"Best technique: {self.best} "
            f"(+{test.mean_difference:.4f} over {self.runner_up}, "
            f"paired t = {test.t_statistic:.3f}, "
            f"p = {test.p_value:.4f}, Holm p = {adjusted:.4f}) -> {verdict} at "
            f"alpha = {self.alpha}"
        )
        if adjusted >= self.alpha:
            lines.append(
                f"  The top two sit inside each other's fold-to-fold spread. "
                f"{self.best} leads, but the data does not support calling it "
                f"the winner."
            )

        if self.metric == "accuracy":
            lines.append("")
            best_mean = self.mean_score(self.best)
            clears = best_mean >= self.target_accuracy
            lines.append(
                f"Target (>= {self.target_accuracy:.0%} for the best individual "
                f"technique): {self.best} at {best_mean:.4f} "
                f"{'clears it' if clears else 'does NOT clear it'}."
            )
            if self.control_accuracy is not None:
                margin = best_mean - self.control_accuracy
                lines.append(
                    f"Background-only control: {self.control_accuracy:.4f}. "
                    f"The best technique is {margin:+.4f} relative to a classifier "
                    f"that never sees the fruit."
                )

        return "\n".join(lines)


def compare(
    fold_scores: Mapping[str, Sequence[float]],
    *,
    metric: str = "accuracy",
    alpha: float = DEFAULT_ALPHA,
    test_accuracy: Optional[Mapping[str, float]] = None,
    dimensionality: Optional[Mapping[str, int]] = None,
    target_accuracy: float = TARGET_ACCURACY,
    control_accuracy: Optional[float] = None,
) -> ComparisonReport:
    """Rank techniques by their per-fold scores and test every pairwise gap.

    Args:
        fold_scores: ``{technique: [score per fold]}``. Every technique must
            have the same number of folds, in the same fold order, so the
            paired test compares like with like.
        metric: Name of the metric in ``fold_scores``, used only for labels.
        alpha: Significance threshold for the Holm-adjusted p-values.
        test_accuracy: Optional held-out test accuracy per technique, reported
            beside the ranking but never used to build it.
        dimensionality: Optional feature vector length per technique.
        target_accuracy: The assignment's bar for the best individual technique.
        control_accuracy: Optional background-only accuracy, for context.

    Returns:
        A :class:`ComparisonReport`.

    Raises:
        ValueError: If fewer than one technique is given, or the fold counts
            disagree between techniques.
    """
    if not fold_scores:
        raise ValueError("compare() needs at least one technique")

    folds = {name: np.asarray(scores, dtype=np.float64)
             for name, scores in fold_scores.items()}
    fold_counts = {name: array.size for name, array in folds.items()}
    if len(set(fold_counts.values())) > 1:
        raise ValueError(
            f"techniques disagree on fold count: {fold_counts}"
        )

    ranking = sorted(
        folds,
        key=lambda name: (-float(folds[name].mean()),
                          float(np.std(folds[name], ddof=1)) if folds[name].size > 1 else 0.0,
                          name),
    )

    names = list(folds)
    pairwise: List[FoldTTest] = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            pairwise.append(
                paired_fold_t_test(folds[first], folds[second], first, second)
            )
    holm_p = holm_adjust([test.p_value for test in pairwise])

    return ComparisonReport(
        metric=metric,
        fold_scores=folds,
        ranking=ranking,
        pairwise=pairwise,
        holm_p=holm_p,
        alpha=alpha,
        test_accuracy=dict(test_accuracy or {}),
        dimensionality=dict(dimensionality or {}),
        target_accuracy=target_accuracy,
        control_accuracy=control_accuracy,
    )


# --------------------------------------------------------------------------- #
# Loading a benchmark run from disk
# --------------------------------------------------------------------------- #

def load_fold_scores(
    benchmark_matrix_csv: Path,
    metric: str = "accuracy",
) -> Dict[str, np.ndarray]:
    """Read the per-fold scores from a ``benchmark_matrix.csv``.

    :mod:`scripts.run_benchmarks` writes one ``cv_fold<k>_<metric>`` column per
    fold. This pulls them back out, in fold order, as one array per technique.

    Args:
        benchmark_matrix_csv: Path to the CSV written by the Phase 3 driver.
        metric: ``"accuracy"`` or ``"macro_f1"``.

    Returns:
        ``{technique: array of per-fold scores}``.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If no per-fold columns for ``metric`` are present.
    """
    path = Path(benchmark_matrix_csv)
    if not path.exists():
        raise FileNotFoundError(f"no benchmark matrix at {path}")

    frame = pd.read_csv(path, index_col=0)
    prefix, suffix = "cv_fold", f"_{metric}"
    fold_columns = sorted(
        (column for column in frame.columns
         if column.startswith(prefix) and column.endswith(suffix)),
        key=lambda column: int(column[len(prefix):-len(suffix)]),
    )
    if not fold_columns:
        raise ValueError(
            f"{path.name} carries no '{prefix}<k>{suffix}' columns; "
            f"re-run scripts/run_benchmarks.py to record per-fold {metric}"
        )
    return {
        str(technique): frame.loc[technique, fold_columns].to_numpy(dtype=np.float64)
        for technique in frame.index
    }


def load_test_metrics(benchmark_matrix_csv: Path) -> pd.DataFrame:
    """Read the single-split test metrics from a ``benchmark_matrix.csv``."""
    return pd.read_csv(Path(benchmark_matrix_csv), index_col=0)


def background_only_accuracy(
    results_root: Path,
    dataset: str = "primary",
) -> Optional[float]:
    """Overall background-only accuracy from the audit's per-class recall CSV.

    ``scripts/audit_dataset.py`` writes ``<dataset>_background_only.csv`` with
    one recall per class. The apple classes are balanced 800/800/800, so the
    unweighted mean of the per-class recalls is the overall accuracy. Returns
    ``None`` when the audit has not been run.
    """
    path = Path(results_root) / "audit" / f"{dataset}_background_only.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "background_only_recall" not in frame.columns or frame.empty:
        return None
    return float(frame["background_only_recall"].mean())


__all__ = [
    "ComparisonReport",
    "DEFAULT_ALPHA",
    "FoldTTest",
    "TARGET_ACCURACY",
    "background_only_accuracy",
    "compare",
    "holm_adjust",
    "load_fold_scores",
    "load_test_metrics",
    "paired_fold_t_test",
]
