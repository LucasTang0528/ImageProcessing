"""Shared evaluation: metrics, cross-validation, timing and reporting.

Every technique is scored by exactly this code. Accuracy is always reported
alongside macro F1, because the class balance of the apple dataset is not
guaranteed and accuracy alone would flatter a technique that simply favours
the majority class.

Nothing here fits anything on test data. The cross-validation helpers take an
unfitted pipeline and a training partition; the scaler inside that pipeline is
therefore fitted on each training fold alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from config import Config, get_config
from harness import FeatureMatrix, leakage_safe_folds


# --------------------------------------------------------------------------- #
# Single-split metrics
# --------------------------------------------------------------------------- #

@dataclass
class ClassificationReport:
    """The common metric set, computed identically for every technique.

    Attributes:
        technique: Short name of the technique scored.
        accuracy: Overall accuracy on the evaluated partition.
        macro_f1: Unweighted mean of the per-class F1 scores.
        weighted_f1: Support-weighted mean of the per-class F1 scores.
        per_class: One row per class, with precision, recall, F1 and support.
        confusion: Raw confusion matrix, true classes as rows.
        confusion_normalised: Confusion matrix normalised over the true class,
            so each row sums to one.
        class_names: Display names, in label order.
    """

    technique: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: pd.DataFrame
    confusion: np.ndarray
    confusion_normalised: np.ndarray
    class_names: Sequence[str]

    def to_row(self) -> Dict[str, float]:
        """Flatten the headline metrics into one row of a benchmark table."""
        row: Dict[str, float] = {
            "technique": self.technique,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
        }
        for name in self.class_names:
            entry = self.per_class.loc[name]
            row[f"precision_{name}"] = float(entry["precision"])
            row[f"recall_{name}"] = float(entry["recall"])
            row[f"f1_{name}"] = float(entry["f1"])
        return row


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    technique: str = "technique",
) -> ClassificationReport:
    """Compute the shared metric set for one set of predictions.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Display names in label order.
        technique: Short name, recorded in the report.

    Returns:
        A :class:`ClassificationReport`.
    """
    labels = list(range(len(class_names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
        index=list(class_names),
    )

    raw = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_totals = raw.sum(axis=1, keepdims=True)
        normalised = np.divide(
            raw,
            row_totals,
            out=np.zeros(raw.shape, dtype=np.float64),
            where=row_totals != 0,
        )

    return ClassificationReport(
        technique=technique,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        per_class=per_class,
        confusion=raw,
        confusion_normalised=normalised,
        class_names=list(class_names),
    )


# --------------------------------------------------------------------------- #
# Cross-validation on the training partition only
# --------------------------------------------------------------------------- #

@dataclass
class CrossValidationReport:
    """Per-fold scores from 5-fold stratified cross-validation.

    Attributes:
        technique: Short name of the technique scored.
        accuracy_folds: One accuracy per fold, in fold order.
        macro_f1_folds: One macro F1 per fold, in fold order.
        fit_seconds: Wall-clock fit time per fold.
        fold_confusions: ``(n_folds, n_classes, n_classes)`` raw confusion
            matrices, true classes as rows, in fold order. Empty when the
            caller did not collect them.

    ``fold_confusions`` is additive and defaulted: every existing field keeps
    its meaning and its value, so a report built without it scores exactly as
    it did before. It exists because per-class precision and recall cannot be
    recovered from a macro F1 after the fact - the per-fold predictions have
    to be retained while the fold is being scored, or the information is gone.
    """

    technique: str
    accuracy_folds: np.ndarray
    macro_f1_folds: np.ndarray
    fit_seconds: np.ndarray = field(default_factory=lambda: np.array([]))
    fold_confusions: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0)))

    @property
    def has_per_class(self) -> bool:
        """True when per-fold confusions were collected."""
        return bool(self.fold_confusions.size)

    def per_class_frame(self, class_names: Sequence[str]) -> pd.DataFrame:
        """Aggregate per-fold confusions into per-class precision/recall/F1.

        Counts are pooled across folds before the ratios are taken, which is
        the same micro-aggregation the fold scores already use: every
        validation row counts once, so a small fold cannot swing the result
        the way averaging five per-fold ratios would.
        """
        if not self.has_per_class:
            return pd.DataFrame(
                columns=["class", "precision", "recall", "f1", "support"]
            )
        pooled = self.fold_confusions.sum(axis=0)
        rows = []
        for index, name in enumerate(class_names):
            true_positive = float(pooled[index, index])
            predicted = float(pooled[:, index].sum())
            actual = float(pooled[index, :].sum())
            precision = true_positive / predicted if predicted else 0.0
            recall = true_positive / actual if actual else 0.0
            denominator = precision + recall
            rows.append(
                {
                    "class": name,
                    "precision": precision,
                    "recall": recall,
                    "f1": (2 * precision * recall / denominator) if denominator else 0.0,
                    "support": int(actual),
                }
            )
        return pd.DataFrame(rows)

    @property
    def mean_accuracy(self) -> float:
        """Mean cross-validated accuracy."""
        return float(np.mean(self.accuracy_folds))

    @property
    def std_accuracy(self) -> float:
        """Standard deviation of cross-validated accuracy across folds."""
        return float(np.std(self.accuracy_folds, ddof=1))

    @property
    def mean_macro_f1(self) -> float:
        """Mean cross-validated macro F1."""
        return float(np.mean(self.macro_f1_folds))

    @property
    def std_macro_f1(self) -> float:
        """Standard deviation of cross-validated macro F1 across folds."""
        return float(np.std(self.macro_f1_folds, ddof=1))


def cross_validate_technique(
    pipeline: Pipeline,
    matrix: FeatureMatrix,
    technique: Optional[str] = None,
    config: Optional[Config] = None,
) -> CrossValidationReport:
    """Run 5-fold stratified cross-validation on the training partition.

    This deliberately takes a whole :class:`~harness.FeatureMatrix` rather than
    a bare ``X``/``y`` pair. Splitting a raw augmented matrix row-wise would
    scatter near-duplicate variants of the same image across both sides of
    every fold, so the folds must be drawn from the matrix's group and variant
    provenance. Requiring the full object makes that misuse impossible to
    express: there is no argument through which a caller can hand over an
    augmented matrix stripped of the information needed to split it safely.

    Guarantees, all enforced by :func:`~harness.leakage_safe_folds`:

    * folds are drawn over **source images**, so an image and every one of its
      augmented variants stay on the same side of the split;
    * training folds carry originals and their variants;
    * validation folds carry **originals only**, so the reported score
      describes performance on real images.

    The pipeline is cloned per fold and the scaler inside it is fitted on the
    training fold alone, so no validation row influences standardisation.

    Args:
        pipeline: An unfitted scaler-then-SVM pipeline.
        matrix: The **training** feature matrix. Must not contain test rows.
        technique: Short name. Defaults to the matrix's own technique name.
        config: Optional configuration override.

    Returns:
        A :class:`CrossValidationReport` with one score per fold.
    """
    cfg = config or get_config()
    name = technique or matrix.technique

    accuracies: List[float] = []
    macro_f1s: List[float] = []
    fit_times: List[float] = []
    confusions: List[np.ndarray] = []
    labels = list(range(len(cfg.primary.display_names)))

    for train_rows, validation_rows in leakage_safe_folds(matrix, cfg):
        estimator = clone(pipeline)

        start = time.perf_counter()
        estimator.fit(matrix.X[train_rows], matrix.y[train_rows])
        fit_times.append(time.perf_counter() - start)

        predicted = estimator.predict(matrix.X[validation_rows])
        truth = matrix.y[validation_rows]
        accuracies.append(float(accuracy_score(truth, predicted)))
        macro_f1s.append(float(f1_score(truth, predicted, average="macro", zero_division=0)))
        confusions.append(confusion_matrix(truth, predicted, labels=labels))

    return CrossValidationReport(
        technique=name,
        accuracy_folds=np.asarray(accuracies, dtype=np.float64),
        macro_f1_folds=np.asarray(macro_f1s, dtype=np.float64),
        fit_seconds=np.asarray(fit_times, dtype=np.float64),
        fold_confusions=np.asarray(confusions, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

def time_inference(
    fitted_pipeline: Pipeline,
    X: np.ndarray,
    warmup: int = 5,
    repeats: int = 3,
) -> float:
    """Measure mean per-image inference time in seconds.

    A warm-up pass is run and discarded before timing begins, so BLAS thread
    start-up and first-call allocation are not charged to the classifier. The
    remaining passes predict one sample at a time, which is what a deployed
    grading system would do.

    Args:
        fitted_pipeline: A pipeline already fitted on the training partition.
        X: Feature matrix to predict over.
        warmup: Number of single-sample predictions to discard.
        repeats: Number of timed passes over ``X``.

    Returns:
        Mean seconds per image.
    """
    if X.size == 0:
        return float("nan")

    for _ in range(warmup):
        fitted_pipeline.predict(X[:1])

    durations: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for index in range(X.shape[0]):
            fitted_pipeline.predict(X[index : index + 1])
        durations.append((time.perf_counter() - start) / X.shape[0])

    return float(np.mean(durations))


def summarise_extraction_time(times: np.ndarray) -> Dict[str, float]:
    """Return mean, median and standard deviation of extraction times.

    The caller is responsible for having excluded the warm-up call;
    :func:`harness.build_feature_matrix` already does so.
    """
    if times.size == 0:
        return {"mean_s": float("nan"), "median_s": float("nan"), "std_s": float("nan")}
    return {
        "mean_s": float(np.mean(times)),
        "median_s": float(np.median(times)),
        "std_s": float(np.std(times, ddof=1)) if times.size > 1 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_report(
    report: ClassificationReport,
    cv: Optional[CrossValidationReport] = None,
) -> None:
    """Print a classification report, and its CV folds when supplied."""
    print(f"\n{report.technique} - held-out test partition")
    print(f"  accuracy    : {report.accuracy:.4f}")
    print(f"  macro F1    : {report.macro_f1:.4f}")
    print(f"  weighted F1 : {report.weighted_f1:.4f}")
    print("\n  per class:")
    print(report.per_class.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n  normalised confusion matrix (rows = true class):")
    frame = pd.DataFrame(
        report.confusion_normalised,
        index=list(report.class_names),
        columns=list(report.class_names),
    )
    print(frame.to_string(float_format=lambda v: f"{v:.3f}"))

    if cv is not None:
        folds = ", ".join(f"{score:.4f}" for score in cv.accuracy_folds)
        print(f"\n  cross-validated accuracy per fold: {folds}")
        print(f"  mean {cv.mean_accuracy:.4f} (sd {cv.std_accuracy:.4f})")


def save_confusion_matrix(
    report: ClassificationReport,
    output_path: Path,
    title: Optional[str] = None,
) -> Path:
    """Render the normalised confusion matrix to a PNG.

    Args:
        report: The report to plot.
        output_path: Destination PNG path. Parent directories are created.
        title: Optional plot title; defaults to the technique name.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(report.class_names)

    figure, axes = plt.subplots(figsize=(1.6 * len(names) + 2.4, 1.6 * len(names) + 2.0))
    image = axes.imshow(report.confusion_normalised, cmap="Blues", vmin=0.0, vmax=1.0)

    axes.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axes.set_yticks(range(len(names)), names)
    axes.set_xlabel("Predicted class")
    axes.set_ylabel("True class")
    axes.set_title(title or f"{report.technique}: normalised confusion matrix")

    for row in range(len(names)):
        for column in range(len(names)):
            value = report.confusion_normalised[row, column]
            axes.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=10,
            )

    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label="Proportion of true class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


#: Per-class precision and recall each have to clear this for a configuration
#: to be judged usable class-by-class rather than only on average. Fixed
#: before the benchmark ran; nothing is tuned towards it.
PER_CLASS_TARGET = 0.80


def per_class_rows(
    config_name: str,
    partition: str,
    frame: pd.DataFrame,
) -> List[Dict[str, object]]:
    """Flatten a per-class table into long-form rows tagged by partition.

    Accepts both shapes the codebase produces: ``ClassificationReport``
    carries the class as the index, while the cross-validated table carries it
    as a column. ``itertuples`` is deliberately avoided - ``class`` is a Python
    keyword, so pandas silently renames the field and the lookup fails.
    """
    working = frame.copy()
    if "class" not in working.columns:
        working = working.reset_index()
        working = working.rename(columns={working.columns[0]: "class"})
    return [
        {
            "config": config_name,
            "partition": partition,
            "class": str(record["class"]),
            "precision": float(record["precision"]),
            "recall": float(record["recall"]),
            "f1": float(record["f1"]),
            "support": int(record["support"]),
        }
        for record in working.to_dict("records")
    ]


def collect_per_class(
    reports: Mapping[str, ClassificationReport],
    cv_reports: Mapping[str, CrossValidationReport],
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Build the long-form per-class table for every configuration.

    Two partitions per configuration: ``test`` from the held-out evaluation,
    and ``cv`` pooled over the cross-validation folds. Both are reported
    because they answer different questions - the test row is the headline
    figure, the cv row is the one with folds behind it - and a configuration
    that clears the target on one and not the other is worth seeing.
    """
    rows: List[Dict[str, object]] = []
    for name in sorted(set(reports) | set(cv_reports)):
        if name in reports:
            rows.extend(per_class_rows(name, "test", reports[name].per_class))
        cv = cv_reports.get(name)
        if cv is not None and cv.has_per_class:
            rows.extend(per_class_rows(name, "cv", cv.per_class_frame(class_names)))
    return pd.DataFrame(rows)


def per_class_pass_table(per_class: pd.DataFrame) -> pd.DataFrame:
    """Report, per configuration and partition, whether every class clears the target.

    The weakest class is carried alongside the verdict: a configuration can
    miss the bar on one class while averaging well above it, and the average
    is what the headline metrics already show.
    """
    rows: List[Dict[str, object]] = []
    for (config_name, partition), group in per_class.groupby(["config", "partition"]):
        worst = group.loc[group[["precision", "recall"]].min(axis=1).idxmin()]
        rows.append(
            {
                "config": config_name,
                "partition": partition,
                "min_precision": float(group.precision.min()),
                "min_recall": float(group.recall.min()),
                "weakest_class": str(worst["class"]),
                "per_class_pass": bool(
                    group.precision.min() >= PER_CLASS_TARGET
                    and group.recall.min() >= PER_CLASS_TARGET
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["partition", "config"]).reset_index(drop=True)


def save_per_class_outputs(
    reports: Mapping[str, ClassificationReport],
    cv_reports: Mapping[str, CrossValidationReport],
    class_names: Sequence[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Write per_class.csv, the confusion CSVs, and return the pass table.

    Pure serialisation of values the metric code already produced. Nothing
    here fits, predicts or scores, so it cannot move a published figure.
    """
    per_class = collect_per_class(reports, cv_reports, class_names)
    save_dataframe(per_class, output_dir / "per_class.csv", index=False)

    confusion_dir = output_dir / "confusion"
    for name, report in reports.items():
        save_dataframe(
            pd.DataFrame(
                report.confusion_normalised,
                index=list(class_names),
                columns=list(class_names),
            ),
            confusion_dir / f"{name}.csv",
        )

    passes = per_class_pass_table(per_class)
    save_dataframe(passes, output_dir / "per_class_pass.csv", index=False)
    return passes


def save_dataframe(frame: pd.DataFrame, output_path: Path, index: bool = True) -> Path:
    """Write a data frame to CSV, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=index)
    return output_path


def save_segmentation_failures(failures: Sequence, output_path: Path) -> Path:
    """Write the logged segmentation failures to CSV.

    Args:
        failures: A sequence of :class:`harness.SegmentationFailure`.
        output_path: Destination CSV path.

    Returns:
        The path written. An empty but correctly headed file is written when
        there were no failures, so a missing file always means the check was
        never run rather than that nothing failed.
    """
    columns = ["path", "class_folder", "source", "variant", "coverage", "reason", "substituted"]
    rows = [
        {
            "path": str(failure.path),
            "class_folder": failure.class_folder,
            "source": failure.source,
            "variant": failure.variant,
            "coverage": failure.coverage,
            "reason": failure.reason,
            "substituted": failure.substituted,
        }
        for failure in failures
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return save_dataframe(frame, output_path, index=False)
