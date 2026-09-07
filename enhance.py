"""Phase 5 - the enhancement layer: three ways to combine the three techniques.

Report section 3.8. Once T1, T2 and T3 are benchmarked in isolation, the study
combines them. Three strategies are implemented, and each is scored on its own
against the strongest individual technique, so its contribution is attributable
rather than reported as one aggregate improvement - the shortcoming the review
called Gap 1.

====  ======================================================================
E1    Feature-level fusion. Standardise the three vectors, concatenate them
      (37 + 40 + 36 = 113 dimensions), and reduce with PCA retaining 95% of
      the variance before the shared SVM. The report text quotes 111/34-D from
      an earlier T3 design; the code follows the vectors the extractors
      actually emit.
E2    Blemish-aware regional weighting. The T3 blemish mask splits the fruit
      into a healthy and a blemished sub-region. T1 and T2 are recomputed over
      each sub-region separately and concatenated, so a localised defect
      signal is no longer averaged away across the whole peel. This is the
      study's principal novel contribution (Problem 3, Gap 4).
E3    Weighted decision-level fusion. One SVM per technique, wrapped in
      ``CalibratedClassifierCV``, produces calibrated class probabilities;
      these are combined by weighted soft voting, the weight of each technique
      being its cross-validated macro F1 (Kittler et al., 1998). Under
      cross-validation the weights are recomputed inside each fold, so no fold
      is scored with a weight its own validation rows helped set.
====  ======================================================================

Nothing here changes the shared harness, the partition, the folds or the
classifier hyperparameters. E1 and E2 are ordinary :class:`~harness.FeatureMatrix`
objects and go through exactly the same cross-validator and metrics as an
individual technique. E3 needs its own fold loop only because it fuses three
fitted pipelines rather than one feature vector; it still draws its folds from
:func:`~harness.leakage_safe_folds` so an augmented variant can never cross into
a validation fold.

Reference
---------
Kittler, J., Hatef, M., Duin, R. P. W., & Matas, J. (1998). On combining
    classifiers. IEEE TPAMI, 20(3), 226-239.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import Config, get_config
from data import ImageRecord
from evaluate import CrossValidationReport
from features.base import FeatureExtractionError, FeatureExtractor
from features.t1_dominant_colour import DominantColourExtractor
from features.t2_glcm import GLCMExtractor
from features.t3_morphological import T3MorphologicalExtractor
from harness import (
    FeatureMatrix,
    SegmentationFailure,
    build_pipeline,
    build_probability_pipeline,
    collect_failures,
    iter_prepared,
    leakage_safe_folds,
)

#: Smallest sub-region, in pixels, that E2 will try to describe. A blemished
#: region below this is treated as "no blemish": the healthy blocks carry the
#: whole fruit and the blemished blocks are zero-filled. The Unripe class has
#: many genuinely clean apples, so this is a normal path, not an error.
MIN_SUBREGION_PX = 40

#: Variance PCA retains in E1, per the report.
E1_PCA_VARIANCE = 0.95


# --------------------------------------------------------------------------- #
# One shared extraction pass
# --------------------------------------------------------------------------- #

@dataclass
class TechniqueMatrices:
    """Row-aligned feature matrices for every technique and enhancement.

    Every matrix here describes the *same* images in the *same* row order, drawn
    from one segmentation pass, so they can be concatenated (E1), split by
    region (E2) or fused at the decision stage (E3) without any risk of a
    misalignment silently pairing one image's colour with another's texture.

    Attributes:
        matrices: ``{name: FeatureMatrix}`` for ``T1``, ``T2``, ``T3``, ``E1``
            and ``E2``. ``E3`` is not a matrix - it fuses the T1/T2/T3 pipelines.
        excluded: Samples dropped because segmentation failed.
        failures: The logged segmentation failures.
    """

    matrices: Dict[str, FeatureMatrix]
    excluded: int
    failures: List[SegmentationFailure]

    def __getitem__(self, key: str) -> FeatureMatrix:
        return self.matrices[key]

    @property
    def individual(self) -> Dict[str, FeatureMatrix]:
        """Just the three individual techniques, in T1/T2/T3 order."""
        return {name: self.matrices[name] for name in ("T1", "T2", "T3")}


def default_extractors(config: Optional[Config] = None) -> Dict[str, FeatureExtractor]:
    """Build the three technique extractors from ``config.json``."""
    cfg = config or get_config()
    return {
        "T1": DominantColourExtractor.from_config(cfg),
        "T2": GLCMExtractor.from_config(cfg),
        "T3": T3MorphologicalExtractor.from_config(cfg),
    }


#: How many random placements the control mask tries before keeping the best.
#: A spun mask can overhang the fruit and lose area; a handful of attempts is
#: enough to find one that does not, and the loop stops as soon as it does.
CONTROL_MASK_ATTEMPTS = 8


def e2_block_columns(t1_dim: int, t2_dim: int) -> Dict[str, np.ndarray]:
    """Column indices of E2's four blocks, in the order :func:`_regional_vector` emits them.

    The layout is defined by that function and nowhere else, so the ablation
    reads it from here rather than recomputing offsets at the call site, where
    a change to the layout would silently mislabel an arm instead of failing.

    Args:
        t1_dim: Length of one T1 block.
        t2_dim: Length of one T2 block.

    Returns:
        ``{"t1_healthy", "t1_blemished", "t2_healthy", "t2_blemished"}`` mapped
        to index arrays.
    """
    bounds = np.cumsum([0, t1_dim, t1_dim, t2_dim, t2_dim])
    names = ("t1_healthy", "t1_blemished", "t2_healthy", "t2_blemished")
    return {
        name: np.arange(bounds[i], bounds[i + 1])
        for i, name in enumerate(names)
    }


def random_control_mask(
    fruit_mask: np.ndarray,
    blemish_mask: np.ndarray,
    rng: np.random.Generator,
    attempts: int = CONTROL_MASK_ATTEMPTS,
) -> np.ndarray:
    """A blemish mask of the same size and shape, moved somewhere else on the fruit.

    This is E2's null hypothesis made concrete. E2 splits the peel in two and
    describes each half, which doubles the dimensionality; a gain over the
    unsplit descriptors could therefore be nothing but the extra dimensions and
    the extra parameters they buy. The control keeps every one of those - same
    154 columns, same two sub-regions, same areas - and changes only **where**
    the boundary between them falls. If the gain survives, it is about the
    blemish mask locating real damage. If it does not, it never was.

    The mask is **rotated about the fruit's centroid** rather than replaced by
    random pixels. That matters: scattering the same number of pixels across
    the fruit would also destroy the spatial coherency that T1's block B
    measures, so a beaten control would show only that coherent regions beat
    speckle - a different and much weaker claim. Rotating preserves the area,
    the shape, the number of connected components and their coherency exactly,
    and varies the one thing under test.

    Args:
        fruit_mask: The fruit mask; the control never leaves it.
        blemish_mask: The real blemish mask for this image.
        rng: Seeded generator. Seed it per image, not per pass, so the control
            for one image does not depend on how many images preceded it.
        attempts: Rotations to try before keeping the largest.

    Returns:
        A ``uint8`` mask, 255 inside the control region, the shape of
        ``fruit_mask``. An empty blemish mask returns an empty control, which
        is correct: a clean apple has nothing to relocate, and the arm then
        describes it exactly as E2 does.
    """
    fruit = np.asarray(fruit_mask).astype(bool)
    blemish = np.asarray(blemish_mask).astype(bool) & fruit
    target = int(np.count_nonzero(blemish))
    if target == 0:
        return np.zeros(fruit.shape, dtype=np.uint8)

    rows, columns = np.nonzero(fruit)
    centre = (float(columns.mean()), float(rows.mean()))
    source = (blemish.astype(np.uint8) * 255)
    height, width = fruit.shape

    best = np.zeros(fruit.shape, dtype=bool)
    best_area = -1
    for _ in range(max(1, attempts)):
        angle = float(rng.uniform(0.0, 360.0))
        rotation = cv2.getRotationMatrix2D(centre, angle, 1.0)
        spun = cv2.warpAffine(
            source, rotation, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        candidate = (spun > 127) & fruit
        area = int(np.count_nonzero(candidate))
        if area > best_area:
            best, best_area = candidate, area
        if area >= 0.95 * target:
            break

    return best.astype(np.uint8) * 255


def _regional_vector(
    t1: FeatureExtractor,
    t2: FeatureExtractor,
    image: np.ndarray,
    fruit_mask: np.ndarray,
    blemish_mask: np.ndarray,
) -> np.ndarray:
    """E2's descriptor for one fruit: T1 and T2 over healthy then blemished peel.

    Layout: ``[T1(healthy) | T1(blemished) | T2(healthy) | T2(blemished)]``.
    A sub-region smaller than :data:`MIN_SUBREGION_PX` is described as a zero
    block rather than by forcing an extractor onto a handful of pixels.
    """
    fruit = np.asarray(fruit_mask).astype(bool)
    blem = np.asarray(blemish_mask).astype(bool) & fruit
    healthy = fruit & ~blem

    def describe(extractor: FeatureExtractor, region: np.ndarray) -> np.ndarray:
        if int(np.count_nonzero(region)) < MIN_SUBREGION_PX:
            return np.zeros(extractor.dim, dtype=np.float64)
        try:
            return np.asarray(
                extractor(image.copy(), (region.astype(np.uint8) * 255)),
                dtype=np.float64,
            )
        except FeatureExtractionError:
            return np.zeros(extractor.dim, dtype=np.float64)

    return np.concatenate(
        [
            describe(t1, healthy),
            describe(t1, blem),
            describe(t2, healthy),
            describe(t2, blem),
        ]
    )


def build_technique_matrices(
    records: Sequence[ImageRecord],
    config: Optional[Config] = None,
    augment: bool = False,
    extractors: Optional[Mapping[str, FeatureExtractor]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    include_random_control: bool = False,
) -> TechniqueMatrices:
    """Run all three techniques, and derive E1 and E2, in a single pass.

    Preprocessing and seeded GrabCut dominate the per-image cost and are
    identical for every technique by construction, so they are run once and
    every descriptor is taken from the same mask. Row inclusion follows the
    harness exactly: a sample whose segmentation failed is dropped unless a
    substitute mask was supplied.

    Args:
        records: Records to describe. Pass the training records for a training
            matrix and the test records for a test matrix - never mixed.
        config: Optional configuration override.
        augment: Apply the training-only augmentation plan. Must be ``False``
            for the test partition.
        extractors: Optional ``{"T1":..., "T2":..., "T3":...}`` override.
        progress: Optional ``(done, total)`` console callback.
        include_random_control: Also build ``E2_random``, the ablation control
            in which each image's blemish mask is relocated on the fruit by
            :func:`random_control_mask`. Off by default, so nothing that does
            not ask for it pays the second regional pass or sees a new key.

    Returns:
        A :class:`TechniqueMatrices`. Its ``matrices`` holds ``T1``, ``T2``,
        ``T3``, ``E1`` and ``E2``, plus ``E2_random`` when it was asked for.
    """
    cfg = config or get_config()
    ext = dict(extractors or default_extractors(cfg))
    t1, t2, t3 = ext["T1"], ext["T2"], ext["T3"]
    if not isinstance(t3, T3MorphologicalExtractor):
        raise TypeError("E2 needs the T3 extractor to expose extract_with_aux")

    names = ["T1", "T2", "T3", "E1", "E2"]
    if include_random_control:
        names.append("E2_random")
    rows: Dict[str, List[np.ndarray]] = {k: [] for k in names}
    times: Dict[str, List[float]] = {k: [] for k in names}
    labels: List[int] = []
    paths: List = []
    groups: List[int] = []
    variants: List[int] = []
    failures: List[SegmentationFailure] = []
    excluded = 0
    n_augmented = 0
    warmed = False

    total = len(records) * (
        1 + (cfg.augmentation.variants_per_image
             if augment and cfg.augmentation.enabled else 0)
    )
    done = 0

    for sample in iter_prepared(records, augment=augment, config=cfg):
        done += 1
        if progress is not None:
            progress(done, total)

        if sample.segmentation.failed:
            failures.append(collect_failures([sample])[0])
            if not sample.segmentation.substituted:
                excluded += 1
                continue

        image = sample.image
        mask = sample.mask

        if not warmed:
            t1(image.copy(), mask.copy())
            t2(image.copy(), mask.copy())
            t3.extract_with_aux(image.copy(), mask.copy())
            _regional_vector(t1, t2, image, mask, np.zeros_like(mask))
            warmed = True

        start = time.perf_counter()
        v1 = np.asarray(t1(image.copy(), mask.copy()), dtype=np.float64)
        times["T1"].append(time.perf_counter() - start)

        start = time.perf_counter()
        v2 = np.asarray(t2(image.copy(), mask.copy()), dtype=np.float64)
        times["T2"].append(time.perf_counter() - start)

        start = time.perf_counter()
        v3, aux = t3.extract_with_aux(image.copy(), mask.copy())
        v3 = np.asarray(v3, dtype=np.float64)
        times["T3"].append(time.perf_counter() - start)

        start = time.perf_counter()
        v_e2 = _regional_vector(t1, t2, image, mask, aux["blemish_mask"])
        times["E2"].append(time.perf_counter() - start + times["T3"][-1])

        if include_random_control:
            # Seeded from the image's own identity rather than from a counter
            # advanced through the pass, so the control mask for one image is
            # the same whether it is described alone or in the middle of the
            # whole dataset, and does not shift when an earlier image is
            # excluded by a segmentation failure.
            control_rng = np.random.default_rng(
                [cfg.seed, sample.record_index, sample.variant]
            )
            start = time.perf_counter()
            control = random_control_mask(mask, aux["blemish_mask"], control_rng)
            v_control = _regional_vector(t1, t2, image, mask, control)
            times["E2_random"].append(time.perf_counter() - start + times["T3"][-1])
            rows["E2_random"].append(v_control)

        rows["T1"].append(v1)
        rows["T2"].append(v2)
        rows["T3"].append(v3)
        rows["E1"].append(np.concatenate([v1, v2, v3]))
        rows["E2"].append(v_e2)
        times["E1"].append(times["T1"][-1] + times["T2"][-1] + times["T3"][-1])

        labels.append(sample.label)
        paths.append(sample.record.path)
        groups.append(sample.record_index)
        variants.append(sample.variant)
        if sample.is_augmented:
            n_augmented += 1

    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups, dtype=np.int64)
    variant_array = np.asarray(variants, dtype=np.int64)

    def matrix(name: str) -> FeatureMatrix:
        stacked = (
            np.vstack(rows[name]) if rows[name]
            else np.empty((0, 0), dtype=np.float64)
        )
        return FeatureMatrix(
            X=stacked,
            y=y,
            technique=name,
            extraction_times=np.asarray(times[name], dtype=np.float64),
            n_augmented=n_augmented,
            failures=failures,
            excluded=excluded,
            paths=list(paths),
            groups=group_array,
            variants=variant_array,
        )

    return TechniqueMatrices(
        matrices={name: matrix(name) for name in names},
        excluded=excluded,
        failures=failures,
    )


# --------------------------------------------------------------------------- #
# E1 - feature-level fusion
# --------------------------------------------------------------------------- #

def stack_feature_matrices(
    matrices: Sequence[FeatureMatrix],
    technique: str = "E1",
) -> FeatureMatrix:
    """Concatenate row-aligned feature matrices along the feature axis.

    The matrices must describe the same images in the same order. That is
    checked on the labels and the group/variant provenance, not assumed:
    concatenating misaligned matrices would pair one fruit's colour with
    another's texture and the error would never surface as an exception.
    """
    if not matrices:
        raise ValueError("nothing to stack")
    first = matrices[0]
    for other in matrices[1:]:
        if other.X.shape[0] != first.X.shape[0]:
            raise ValueError("row counts differ between matrices")
        if not np.array_equal(other.y, first.y):
            raise ValueError("labels differ between matrices; rows are not aligned")
        if not np.array_equal(other.groups, first.groups):
            raise ValueError("group provenance differs; rows are not aligned")
        if not np.array_equal(other.variants, first.variants):
            raise ValueError("variant provenance differs; rows are not aligned")

    return replace(
        first,
        X=np.hstack([m.X for m in matrices]),
        technique=technique,
        extraction_times=np.sum(
            [m.extraction_times for m in matrices], axis=0
        ),
    )


def build_fusion_pipeline(
    config: Optional[Config] = None,
    pca_variance: float = E1_PCA_VARIANCE,
) -> Pipeline:
    """Standardise, then PCA to ``pca_variance``, then the shared SVM.

    Standardisation comes before PCA because PCA is scale-sensitive and the
    three blocks are on different numeric scales (colour percentages in [0, 1],
    unbounded GLCM contrast, normalised morphological responses). Every step is
    inside one :class:`~sklearn.pipeline.Pipeline`, so under cross-validation
    the scaler and the PCA basis are both fitted on the training fold alone.
    """
    cfg = config or get_config()
    clf = cfg.classifier
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, random_state=cfg.seed)),
            (
                "svc",
                SVC(
                    kernel=clf.kernel,
                    C=clf.C,
                    gamma=clf.gamma,
                    random_state=cfg.seed,
                ),
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# E3 - weighted decision-level fusion
# --------------------------------------------------------------------------- #

def weighted_soft_vote(
    probabilities: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Combine per-technique class probabilities into one label per row.

    Args:
        probabilities: ``{technique: (n_samples, n_classes) probability array}``.
            Every array must share the shape and the class column order.
        weights: ``{technique: weight}``. Rescaled to sum to one; a technique
            absent from this mapping is dropped from the vote.

    Returns:
        ``(n_samples,)`` predicted class indices.
    """
    names = [name for name in probabilities if weights.get(name, 0.0) > 0.0]
    if not names:
        raise ValueError("no technique carries a positive weight")

    total = float(sum(weights[name] for name in names))
    shape = probabilities[names[0]].shape
    blended = np.zeros(shape, dtype=np.float64)
    for name in names:
        proba = np.asarray(probabilities[name], dtype=np.float64)
        if proba.shape != shape:
            raise ValueError(f"probability arrays disagree on shape for {name}")
        blended += (weights[name] / total) * proba
    return np.argmax(blended, axis=1)


@dataclass
class DecisionFusion:
    """Three fitted pipelines and the weights their votes are combined with."""

    pipelines: Dict[str, Pipeline]
    weights: Dict[str, float]

    def predict(self, matrices: Mapping[str, FeatureMatrix]) -> np.ndarray:
        """Predict labels for a set of row-aligned technique matrices."""
        probabilities = {
            name: pipeline.predict_proba(matrices[name].X)
            for name, pipeline in self.pipelines.items()
        }
        return weighted_soft_vote(probabilities, self.weights)


def macro_f1_weights(
    matrices: Mapping[str, FeatureMatrix],
    config: Optional[Config] = None,
) -> Dict[str, float]:
    """Cross-validated macro F1 per technique, on the training partition.

    These are E3's voting weights: a technique that separates the classes
    better on held-out folds is trusted more in the vote (Kittler et al., 1998).
    """
    cfg = config or get_config()
    weights: Dict[str, float] = {}
    for name, matrix in matrices.items():
        report = cross_validate_matrix(matrix, cfg, technique=name)
        weights[name] = report.mean_macro_f1
    return weights


def fit_decision_fusion(
    train_matrices: Mapping[str, FeatureMatrix],
    config: Optional[Config] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> DecisionFusion:
    """Fit one SVM per technique and set the soft-vote weights.

    Args:
        train_matrices: ``{technique: training FeatureMatrix}``, row-aligned.
        config: Optional configuration override.
        weights: Optional explicit weights. When omitted they are the
            cross-validated macro F1 of each technique.

    Each sub-model is the calibrated form of the shared pipeline. Soft voting
    only means anything if the three probability vectors are on a common
    scale, and an uncalibrated SVM decision function is not a probability, so
    the brief specifies ``CalibratedClassifierCV`` here and nowhere else.

    Returns:
        A fitted :class:`DecisionFusion`.
    """
    cfg = config or get_config()
    resolved = dict(weights) if weights is not None else macro_f1_weights(train_matrices, cfg)

    pipelines: Dict[str, Pipeline] = {}
    for name, matrix in train_matrices.items():
        pipeline = build_probability_pipeline(cfg)
        pipeline.fit(matrix.X, matrix.y)
        pipelines[name] = pipeline

    return DecisionFusion(pipelines=pipelines, weights=resolved)


# --------------------------------------------------------------------------- #
# Cross-validation for the enhancements
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OutOfFoldPredictions:
    """Every validation prediction the cross-validation made, pooled.

    A fold's predictions are thrown away today: only its accuracy and macro F1
    survive, which is enough to rank a configuration and not enough to say
    *which* class it fails on. Keeping them costs nothing - they are already
    computed - and it is the only way to give an ablation arm a per-class
    breakdown, because an ablation arm is never fitted on the training
    partition and never predicts the test split.

    Every row here is a validation row, so no prediction was made by a model
    that had seen that image. :func:`~harness.leakage_safe_folds` puts only
    originals on the validation side, so the pooled set holds one prediction
    per original image and none for an augmented variant.

    Attributes:
        technique: Short name of the configuration scored.
        y_true: Ground-truth labels, in pooling order.
        y_pred: Predicted labels, in the same order.
        rows: Index into the source matrix each prediction came from, so a
            wrong answer can be traced back to the image that produced it.
    """

    technique: str
    y_true: np.ndarray
    y_pred: np.ndarray
    rows: np.ndarray


def cross_validate_matrix_detailed(
    matrix: FeatureMatrix,
    config: Optional[Config] = None,
    technique: Optional[str] = None,
    pipeline: Optional[Pipeline] = None,
) -> Tuple[CrossValidationReport, OutOfFoldPredictions]:
    """:func:`cross_validate_matrix`, also returning the pooled predictions.

    This holds the fold loop; :func:`cross_validate_matrix` is a wrapper that
    discards the second return value. Structuring it this way rather than
    writing a second loop is deliberate - two loops could drift, and then a
    per-class table would describe a slightly different cross-validation from
    the accuracy printed beside it.
    """
    cfg = config or get_config()
    name = technique or matrix.technique
    estimator = pipeline or build_pipeline(cfg)

    accuracies: List[float] = []
    macro_f1s: List[float] = []
    fit_times: List[float] = []
    pooled_true: List[np.ndarray] = []
    pooled_pred: List[np.ndarray] = []
    pooled_rows: List[np.ndarray] = []

    for train_rows, validation_rows in leakage_safe_folds(matrix, cfg):
        fold = clone(estimator)
        start = time.perf_counter()
        fold.fit(matrix.X[train_rows], matrix.y[train_rows])
        fit_times.append(time.perf_counter() - start)
        predicted = fold.predict(matrix.X[validation_rows])
        truth = matrix.y[validation_rows]
        accuracies.append(float(accuracy_score(truth, predicted)))
        macro_f1s.append(float(f1_score(truth, predicted, average="macro", zero_division=0)))
        pooled_true.append(np.asarray(truth))
        pooled_pred.append(np.asarray(predicted))
        pooled_rows.append(np.asarray(validation_rows))

    report = CrossValidationReport(
        technique=name,
        accuracy_folds=np.asarray(accuracies, dtype=np.float64),
        macro_f1_folds=np.asarray(macro_f1s, dtype=np.float64),
        fit_seconds=np.asarray(fit_times, dtype=np.float64),
    )
    return report, _pool(name, pooled_true, pooled_pred, pooled_rows)


def cross_validate_matrix(
    matrix: FeatureMatrix,
    config: Optional[Config] = None,
    technique: Optional[str] = None,
    pipeline: Optional[Pipeline] = None,
) -> CrossValidationReport:
    """5-fold leakage-safe CV for one feature matrix (E1, E2, or a technique).

    A thin wrapper over :func:`~harness.leakage_safe_folds` that lets the
    pipeline be overridden, which :func:`~evaluate.cross_validate_technique`
    does not: E1 needs the scaler-PCA-SVM pipeline, not the plain one.
    """
    return cross_validate_matrix_detailed(matrix, config, technique, pipeline)[0]


def _pool(
    technique: str,
    truths: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    rows: Sequence[np.ndarray],
) -> OutOfFoldPredictions:
    """Concatenate per-fold validation results into one set."""
    empty = np.empty(0, dtype=np.int64)
    return OutOfFoldPredictions(
        technique=technique,
        y_true=np.concatenate(truths) if truths else empty,
        y_pred=np.concatenate(predictions) if predictions else empty,
        rows=np.concatenate(rows) if rows else empty,
    )


def cross_validate_decision_fusion(
    train_matrices: Mapping[str, FeatureMatrix],
    config: Optional[Config] = None,
    weights: Optional[Mapping[str, float]] = None,
    technique: str = "E3",
) -> CrossValidationReport:
    """5-fold CV for the decision-level fusion.

    The folds are drawn once, from any one of the row-aligned matrices, and
    reused for all three sub-pipelines so every technique is trained and scored
    on exactly the same images each fold. When ``weights`` is omitted it is
    recomputed inside each fold from that fold's training rows, so the weight a
    technique gets is never informed by the rows it is about to be scored on.
    """
    return cross_validate_decision_fusion_detailed(
        train_matrices, config, weights, technique
    )[0]


def cross_validate_decision_fusion_detailed(
    train_matrices: Mapping[str, FeatureMatrix],
    config: Optional[Config] = None,
    weights: Optional[Mapping[str, float]] = None,
    technique: str = "E3",
) -> Tuple[CrossValidationReport, OutOfFoldPredictions]:
    """:func:`cross_validate_decision_fusion`, also returning the predictions.

    Same relationship as :func:`cross_validate_matrix_detailed` has to
    :func:`cross_validate_matrix`: this holds the fold loop, the public
    function discards the second value.
    """
    cfg = config or get_config()
    names = list(train_matrices)
    reference = train_matrices[names[0]]

    accuracies: List[float] = []
    macro_f1s: List[float] = []
    pooled_true: List[np.ndarray] = []
    pooled_pred: List[np.ndarray] = []
    pooled_rows: List[np.ndarray] = []
    for train_rows, validation_rows in leakage_safe_folds(reference, cfg):
        fold_train = {
            name: replace(
                m, X=m.X[train_rows], y=m.y[train_rows],
                groups=m.groups[train_rows], variants=m.variants[train_rows],
                extraction_times=m.extraction_times[train_rows]
                if m.extraction_times.size == m.y.size else m.extraction_times,
                paths=[m.paths[i] for i in train_rows] if len(m.paths) == m.y.size else m.paths,
            )
            for name, m in train_matrices.items()
        }
        fusion = fit_decision_fusion(fold_train, cfg, weights=weights)
        probabilities = {
            name: fusion.pipelines[name].predict_proba(train_matrices[name].X[validation_rows])
            for name in names
        }
        predicted = weighted_soft_vote(probabilities, fusion.weights)
        truth = reference.y[validation_rows]
        accuracies.append(float(accuracy_score(truth, predicted)))
        macro_f1s.append(float(f1_score(truth, predicted, average="macro", zero_division=0)))
        pooled_true.append(np.asarray(truth))
        pooled_pred.append(np.asarray(predicted))
        pooled_rows.append(np.asarray(validation_rows))

    report = CrossValidationReport(
        technique=technique,
        accuracy_folds=np.asarray(accuracies, dtype=np.float64),
        macro_f1_folds=np.asarray(macro_f1s, dtype=np.float64),
    )
    return report, _pool(technique, pooled_true, pooled_pred, pooled_rows)


__all__ = [
    "CONTROL_MASK_ATTEMPTS",
    "DecisionFusion",
    "E1_PCA_VARIANCE",
    "MIN_SUBREGION_PX",
    "OutOfFoldPredictions",
    "TechniqueMatrices",
    "build_fusion_pipeline",
    "build_technique_matrices",
    "cross_validate_decision_fusion",
    "cross_validate_decision_fusion_detailed",
    "cross_validate_matrix",
    "cross_validate_matrix_detailed",
    "default_extractors",
    "e2_block_columns",
    "fit_decision_fusion",
    "macro_f1_weights",
    "random_control_mask",
    "stack_feature_matrices",
    "weighted_soft_vote",
]
