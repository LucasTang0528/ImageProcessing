"""E1 sub-experiments: what the dominant colour descriptor's choices are worth.

Six questions, each isolating one decision taken when T1 was specified:

======  =========================================================  =========
E1.1    dominant colours against a colour-histogram baseline       e1_1.csv
E1.2    how many dominant colours, N = 3, 4, 5, 6                  e1_2.csv
E1.3    which space the clustering runs in, Lab, HSV, RGB          e1_3.csv
E1.4    whether specular highlights are excluded                   e1_4.csv
E1.5    which blocks of the descriptor carry the signal            e1_5.csv
E1.6    the decay index with shadowed green peel excluded         e1_6.csv
======  =========================================================  =========

Every arm is scored on **one shared segmentation pass**. Segmentation is the
expensive stage and is identical for all of them - no arm here changes the
mask - so re-segmenting per arm would cost hours and buy nothing. Building the
matrices together also guarantees the arms are compared on exactly the same
images, including the same segmentation failures.

All scoring is 5-fold cross-validation on the **training partition only**,
through :func:`evaluate.cross_validate_technique`, which draws leakage-safe
folds over source images and validates on originals only. The test partition
is never touched: these experiments choose nothing, they explain a choice
already made, and a sweep scored on held-out data would turn that data into a
tuning set.

Run from the project root::

    python scripts/run_e1_experiments.py --per-class 100    # quick look
    python scripts/run_e1_experiments.py                    # the reported run

Results land in ``results/e1/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config import Config, get_config  # noqa: E402
from data import ImageRecord, load_primary  # noqa: E402
from evaluate import (  # noqa: E402
    classification_metrics,
    cross_validate_technique,
    print_report,
    save_dataframe,
)
from features.t1_dominant_colour import DominantColourExtractor  # noqa: E402
from features.t1_histogram_baseline import ColourHistogramExtractor  # noqa: E402
from harness import (  # noqa: E402
    FeatureMatrix,
    build_pipeline,
    collect_failures,
    iter_prepared,
    set_global_seed,
    stratified_split,
)
from run_benchmarks import balanced_subset  # noqa: E402

#: The dimension E1.5 removes. Named rather than indexed, because the position
#: of a feature moves whenever N changes and an index would silently start
#: deleting a different column.
DECAY_FEATURE = "T1_decay_share"

#: The prediction for E1.6, written before the arm was run. Its hash goes into
#: the result row so that "the prediction preceded the result" is a checkable
#: claim rather than an asserted one: the file cannot be edited after the fact
#: without the recorded digest ceasing to match.
PREDICTION_FILE = PROJECT_ROOT / "results" / "predictions" / "e1_6_decay_green_exclusion.txt"


def prediction_digest(path: Path = PREDICTION_FILE) -> str:
    """SHA-256 of the recorded prediction, or a marker if it is absent."""
    if not path.exists():
        return "ABSENT"
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: Image-processing operations each configuration performs before it emits a
#: vector, used only as the second parsimony tie-break. Counted explicitly
#: rather than inferred, so the number is auditable rather than a guess:
#: colour conversion, specular exclusion, clustering, and summarising are one
#: each. Selecting blocks or changing N adds no operation, so those arms all
#: carry the same count as the default.
PIPELINE_STAGES = {
    "t1_default": 4,            # convert, exclude specular, cluster, summarise
    "specular_off": 3,          # convert, cluster, summarise
    "histogram": 2,             # convert, histogram and moments
    "n3": 4, "n5": 4, "n6": 4,
    "space_HSV": 4, "space_RGB": 4,
    "blocks_A": 4, "blocks_AB": 4,
    "decay_green_excluded": 4,
}

#: Columns that are text rather than measurements. Listed so that a value of
#: "" survives a CSV round-trip instead of returning as a missing number.
STRING_COLUMNS = (
    "arm", "config_key", "dropped_feature", "reference_arm", "representation",
    "prediction_sha256", "distances", "blocks", "space", "selected_by",
    "highest_accuracy_arm", "selected_on",
)

#: Fold-accuracy column names produced by :func:`score`.
FOLD_COLUMNS = tuple(f"fold{index}_accuracy" for index in range(1, 6))


def fold_matrix(table: pd.DataFrame) -> np.ndarray:
    """Return the per-fold accuracies of a scored table as ``(arms, folds)``."""
    present = [column for column in FOLD_COLUMNS if column in table.columns]
    return table[present].to_numpy(dtype=float)


def paired_difference(first: np.ndarray, second: np.ndarray) -> Tuple[float, float]:
    """Mean and standard deviation of the fold-wise difference between two arms.

    Every arm is scored on the same folds, drawn from the same seed over the
    same images, so the fold scores are paired and their differences carry far
    less variance than the two accuracy figures do separately. Comparing the
    means alone throws that pairing away and makes indistinguishable arms look
    ordered.
    """
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    if difference.size < 2:
        return float(difference.mean()) if difference.size else 0.0, 0.0
    return float(difference.mean()), float(difference.std(ddof=1))


def add_paired_differences(table: pd.DataFrame, reference_arm: str) -> pd.DataFrame:
    """Attach paired fold-difference statistics against one reference arm.

    ``within_noise`` is the reportable quantity: an arm whose mean fold-wise
    difference from the reference is smaller than the standard deviation of
    those same differences lies **inside the stated noise band**, and this
    sweep does not distinguish it from the reference. On these data almost
    every arm qualifies, which is the finding rather than an inconvenience.

    ``paired_t`` and ``paired_p`` are recorded as a supporting note and are
    deliberately not the claim. Two assumptions behind reading them as
    significance both fail here. Each sweep compares many arms against one
    reference without any correction for multiplicity, so the smallest p-value
    among fifteen comparisons is not the p-value it appears to be. And
    cross-validation folds share training data by construction, so the five
    differences are not independent draws and the t statistic's reference
    distribution does not apply. The noise band makes no distributional
    assumption and is what the text should quote.
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
    # Supporting note only. See the docstring: uncorrected multiplicity and
    # dependent folds both bar these from carrying a significance claim.
    out["paired_t_supporting"] = t_stats
    out["paired_p_supporting"] = p_values
    return out


def backfill_paired_tests(results_dir: Path, references: Dict[str, str]) -> List[Path]:
    """Add the supporting t-test columns to tables written before they existed.

    The statistics are computed from the per-fold accuracies already stored in
    each CSV, so a backfilled table is identical to one produced by a fresh
    run. This exists so that adding a reporting column does not require
    re-extracting features that have not changed.
    """
    touched: List[Path] = []
    for name, reference_arm in references.items():
        path = results_dir / f"{name}.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        # An empty string round-trips through CSV as a missing value, so a
        # column such as dropped_feature comes back as NaN and the backfilled
        # table stops matching the one the runner produced. These columns are
        # strings whose empty value is "", not absence; restoring that is what
        # keeps the two paths identical rather than merely equivalent.
        for column in STRING_COLUMNS:
            if column in table.columns:
                table[column] = table[column].fillna("").astype(str)
        if "paired_t_supporting" in table.columns or "arm" not in table.columns:
            continue
        table = table.drop(columns=[c for c in
                                    ("reference_arm", "paired_diff_mean",
                                     "paired_diff_sd", "within_noise")
                                    if c in table.columns])
        add_paired_differences(table, reference_arm).to_csv(path, index=False)
        touched.append(path)
    return touched


def build_extractors(config: Config) -> Dict[str, object]:
    """Every distinct extractor configuration the five experiments need.

    Deliberately a flat mapping built once. Several arms share a configuration
    - the default descriptor is the N=4 arm of E1.2, the Lab arm of E1.3, the
    specular-on arm of E1.4 and the A+B+C arm of E1.5 - and extracting it once
    keeps those arms numerically identical rather than merely equivalent.
    """
    block = dict(getattr(config, "t1_colour", {}) or {})
    block.pop("_comment", None)

    def dominant(**overrides) -> DominantColourExtractor:
        settings = {**block, **overrides}
        settings.pop("histogram_bins", None)
        return DominantColourExtractor(seed=config.seed, **settings)

    extractors: Dict[str, object] = {
        "t1_default": dominant(),
        "histogram": ColourHistogramExtractor.from_config(config),
    }
    for n_colours in (3, 5, 6):
        extractors[f"n{n_colours}"] = dominant(n_colours=n_colours)
    for space in ("HSV", "RGB"):
        extractors[f"space_{space}"] = dominant(space=space)
    extractors["specular_off"] = dominant(exclude_specular=False)
    extractors["blocks_A"] = dominant(blocks="A")
    extractors["blocks_AB"] = dominant(blocks="AB")
    # E1.6. An arm, not a redefinition: the flag defaults to false in
    # config.json, so the published descriptor is untouched and both readings
    # of the decay index are reported side by side.
    extractors["decay_green_excluded"] = dominant(decay_exclude_green=True)
    return extractors


def extract_all(
    records: Sequence[ImageRecord],
    extractors: Dict[str, object],
    augment: bool,
    config: Config,
) -> Dict[str, FeatureMatrix]:
    """Build one feature matrix per configuration in a single pass.

    Row inclusion follows the harness exactly: a sample whose segmentation
    failed is dropped unless a substitute mask was supplied, and group and
    variant provenance is carried through so cross-validation can still draw
    leakage-safe folds.
    """
    vectors: Dict[str, List[np.ndarray]] = {name: [] for name in extractors}
    times: Dict[str, List[float]] = {name: [] for name in extractors}
    warmed = {name: False for name in extractors}

    labels: List[int] = []
    paths: List[Path] = []
    groups: List[int] = []
    variant_ids: List[int] = []
    failures: List = []
    excluded = 0
    n_augmented = 0

    total = len(records) * (
        1 + (config.augmentation.variants_per_image
             if augment and config.augmentation.enabled else 0)
    )
    done = 0

    for sample in iter_prepared(records, augment=augment, config=config):
        done += 1
        if done % 25 == 0 or done == total:
            print(f"    prepared {done}/{total}", end="\r", flush=True)

        if sample.segmentation.failed:
            failures.append(collect_failures([sample])[0])
            if not sample.segmentation.substituted:
                excluded += 1
                continue

        labels.append(sample.label)
        paths.append(sample.record.path)
        groups.append(sample.record_index)
        variant_ids.append(sample.variant)
        if sample.is_augmented:
            n_augmented += 1

        for name, extractor in extractors.items():
            image, mask = sample.image.copy(), sample.mask.copy()
            if not warmed[name]:
                extractor(image, mask)  # Warm-up, not timed.
                warmed[name] = True
            start = time.perf_counter()
            vectors[name].append(extractor(image, mask))
            times[name].append(time.perf_counter() - start)

    print()
    return {
        name: FeatureMatrix(
            X=np.vstack(rows) if rows else np.empty((0, extractors[name].dim)),
            y=np.asarray(labels, dtype=np.int64),
            technique=name,
            extraction_times=np.asarray(times[name], dtype=np.float64),
            n_augmented=n_augmented,
            failures=failures,
            excluded=excluded,
            paths=paths,
            groups=np.asarray(groups, dtype=np.int64),
            variants=np.asarray(variant_ids, dtype=np.int64),
        )
        for name, rows in vectors.items()
    }


def drop_feature(matrix: FeatureMatrix, names: Sequence[str], drop: str) -> FeatureMatrix:
    """Return the matrix with one named column removed.

    Removing a column here rather than adding a parameter to the extractor
    keeps the descriptor untouched and keeps the arm on the same extraction
    pass as its parent, so the only difference between them really is the one
    column.
    """
    keep = [index for index, name in enumerate(names) if name != drop]
    if len(keep) == len(names):
        raise KeyError(f"{drop!r} is not among the feature names")
    return FeatureMatrix(
        X=matrix.X[:, keep],
        y=matrix.y,
        technique=f"{matrix.technique}_minus_{drop}",
        extraction_times=matrix.extraction_times,
        n_augmented=matrix.n_augmented,
        failures=matrix.failures,
        excluded=matrix.excluded,
        paths=matrix.paths,
        groups=matrix.groups,
        variants=matrix.variants,
    )


def score(matrix: FeatureMatrix, arm: str, config: Config, **extra: object) -> dict:
    """Cross-validate one arm on the training partition and return its row."""
    report = cross_validate_technique(
        build_pipeline(config), matrix, technique=arm, config=config
    )
    # Per-class recall is emitted for every arm, not derived afterwards. A
    # claim about which classes a change helps is only auditable if the
    # per-class numbers sit in the same row as the accuracy they explain.
    display = list(config.primary.display_names)
    per_class = report.per_class_frame(display)
    if not per_class.empty:
        indexed = per_class.set_index("class")
        recalls = {f"recall_{name}": float(indexed.loc[name, "recall"]) for name in display}
        precisions = {f"precision_{name}": float(indexed.loc[name, "precision"]) for name in display}
    else:
        recalls = {f"recall_{name}": float("nan") for name in display}
        precisions = {f"precision_{name}": float("nan") for name in display}
    print(
        f"    {arm:<26} dim {matrix.dim:>3}  "
        f"accuracy {report.mean_accuracy:.4f} +/- {report.std_accuracy:.4f}  "
        f"macro F1 {report.mean_macro_f1:.4f}"
    )
    # Read before the dict literal pops them: a dict literal evaluates its
    # values in order, so looking the key up after popping it silently yields
    # the not-found sentinel.
    config_key = str(extra.pop("config_key", ""))
    dropped_feature = str(extra.pop("dropped_feature", ""))
    stages = PIPELINE_STAGES.get(config_key, -1)
    if stages < 0:
        raise KeyError(
            f"PIPELINE_STAGES has no entry for {config_key!r}. The parsimony "
            f"tie-break would silently rank this arm first; add its stage count."
        )

    return {
        "arm": arm,
        "config_key": config_key,
        "dropped_feature": dropped_feature,
        "dimensionality": matrix.dim,
        "cv_mean_accuracy": report.mean_accuracy,
        "cv_std_accuracy": report.std_accuracy,
        "cv_mean_macro_f1": report.mean_macro_f1,
        "cv_std_macro_f1": report.std_macro_f1,
        **{f"fold{i + 1}_accuracy": float(v) for i, v in enumerate(report.accuracy_folds)},
        **recalls,
        **precisions,
        "extraction_mean_s": float(np.mean(matrix.extraction_times)),
        "fit_seconds_mean": float(np.mean(report.fit_seconds)) if report.fit_seconds.size else float("nan"),
        "pipeline_stages": stages,
        "rows": int(matrix.X.shape[0]),
        **extra,
    }


# --------------------------------------------------------------------------- #
# The five experiments
# --------------------------------------------------------------------------- #

def e1_1_baseline(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E1.1 - dominant colours against a colour histogram."""
    print("\n  E1.1  representation: dominant colours vs histogram")
    return add_paired_differences(pd.DataFrame([
        score(matrices["t1_default"], "dominant_colour", config, config_key="t1_default",
              representation="MPEG-7 dominant colour"),
        score(matrices["histogram"], "colour_histogram", config, config_key="histogram",
              representation="32-bin histogram plus moments"),
    ]), "dominant_colour")


def e1_2_n_colours(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E1.2 - how many dominant colours.

    Dimensionality is reported rather than equalised. Padding a shorter vector
    would invent features the descriptor never produced, and truncating a
    longer one would score a different descriptor from the one named.
    """
    print("\n  E1.2  number of dominant colours")
    rows = []
    for n_colours in (3, 4, 5, 6):
        key = "t1_default" if n_colours == 4 else f"n{n_colours}"
        rows.append(score(matrices[key], f"N={n_colours}", config, config_key=key,
                          n_colours=n_colours))
    return add_paired_differences(pd.DataFrame(rows), "N=4")


def e1_3_space(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E1.3 - which colour space the clustering runs in.

    The five derived indices are computed in CIE units whatever space the
    clustering used, so this arm changes where the clusters land and not what
    the indices mean.
    """
    print("\n  E1.3  clustering space")
    rows = [score(matrices["t1_default"], "space=LAB", config,
                  config_key="t1_default", space="LAB")]
    for space in ("HSV", "RGB"):
        rows.append(score(matrices[f"space_{space}"], f"space={space}", config,
                          config_key=f"space_{space}", space=space))
    return add_paired_differences(pd.DataFrame(rows), "space=LAB")


def e1_4_specular(
    matrices: Dict[str, FeatureMatrix],
    extractors: Dict[str, object],
    config: Config,
) -> pd.DataFrame:
    """E1.4 - whether specular highlights are excluded before clustering.

    The mean fraction of masked pixels removed is reported alongside the
    accuracy. Without it a null result is unreadable: exclusion that changes
    nothing because it fired on almost no pixels is a different finding from
    exclusion that removed a tenth of the fruit and still changed nothing.
    """
    print("\n  E1.4  specular highlight exclusion")
    rows = []
    for arm, key in (("specular=on", "t1_default"), ("specular=off", "specular_off")):
        diagnostics = getattr(extractors[key], "diagnostics", [])
        excluded = [d.specular_fraction for d in diagnostics]
        fallbacks = [d.specular_fallback for d in diagnostics]
        rows.append(score(
            matrices[key], arm, config, config_key=key,
            mean_pixels_excluded_pct=100.0 * float(np.mean(excluded)) if excluded else 0.0,
            max_pixels_excluded_pct=100.0 * float(np.max(excluded)) if excluded else 0.0,
            fallback_images=int(np.sum(fallbacks)) if fallbacks else 0,
        ))
    return add_paired_differences(pd.DataFrame(rows), "specular=on")


def e1_5_blocks(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E1.5 - which blocks of the descriptor carry the signal.

    The final arm removes the decay share from the complete descriptor. On
    held-out exemplars that index reads 26.5% on an unripe apple and 0.0% on a
    rotten one, inverted from its design: a dark, low-chroma green cluster
    satisfies both its chroma and lightness bounds, so it is detecting shadow
    on green peel rather than decay. This arm measures what carrying it costs.
    """
    print("\n  E1.5  block ablation")
    full = matrices["t1_default"]
    names = list(build_extractors(config)["t1_default"].feature_names)
    rows = [
        score(matrices["blocks_A"], "A (clusters)", config, config_key="blocks_A", blocks="A"),
        score(matrices["blocks_AB"], "A+B (with coherency)", config, config_key="blocks_AB",
              blocks="AB"),
        score(full, "A+B+C (complete)", config, config_key="t1_default", blocks="ABC"),
        score(drop_feature(full, names, DECAY_FEATURE), "A+B+C minus decay share",
              config, config_key="t1_default", dropped_feature=DECAY_FEATURE,
              blocks="ABC-decay"),
    ]
    return add_paired_differences(pd.DataFrame(rows), "A+B+C (complete)")


def e1_6_decay_green_exclusion(
    matrices: Dict[str, FeatureMatrix],
    config: Config,
) -> pd.DataFrame:
    """E1.6 - removing shadowed green peel from the decay index.

    The decay index counts clusters that are dark and weakly chromatic.
    Shadowed green peel satisfies both, so on unripe fruit the index counts
    shade as decay: measured over 590 training images it reads 10.97 on
    Unripe against 12.97 on Rotten, and 93.7% of the Unripe reading is
    removed by additionally requiring the cluster to be non-green, while
    Rotten loses exactly none of its own.

    This arm is not evidence that the index is inverted. Across the
    population it already orders the classes correctly, Rotten highest; the
    inversion reported on two held-out exemplars does not generalise. What is
    real is the contamination, and this measures what removing it is worth.

    The prediction for this arm was recorded before it was run, in
    results/predictions/e1_6_decay_green_exclusion.txt: a gain of about +0.3
    points, plausibly nothing at all, because T1_green_share already
    identifies unripe fruit and the classifier can discount an inflated decay
    reading from it. A flat result is a redundancy finding, not a failure.

    The prediction is left exactly as recorded, and its digest travels in the
    result row. It is judged in the paired frame, on the same tie band as
    every other arm: an unpaired reading of this arm against an ostensibly
    paired reading of the others would not be comparing like with like, and
    the falsifiable check on where any gain lands would be meaningless.
    """
    print("\n  E1.6  decay index with shadowed green peel excluded")
    digest = prediction_digest()
    print(f"    prediction on file, sha256 {digest[:16]}...")
    return add_paired_differences(pd.DataFrame([
        score(matrices["t1_default"], "decay as published", config,
              config_key="t1_default", decay_exclude_green=False,
              prediction_sha256=digest),
        score(matrices["decay_green_excluded"], "decay excluding green", config,
              config_key="decay_green_excluded", decay_exclude_green=True,
              prediction_sha256=digest),
    ]), "decay as published")


def e1_6_sample_sensitivity(
    matrices: Dict[str, FeatureMatrix],
    config: Config,
    draws: int = 400,
) -> pd.DataFrame:
    """How the decay index's apparent strength depends on how many images you look at.

    This is a finding, not a caveat. Measured at 60 images per class the decay
    index separates Rotten from Unripe by 1.86x; at 200 per class the same
    quantity reads 1.18x; and the full training partition is larger still. The
    dimension does not weaken - the small-sample estimate was optimistic, and
    an experiment that stopped at sixty images would have reported an effect
    roughly half again as large as the one that is there.

    The resampling is free: every training image's decay value is already in
    the extracted matrix, so subsets are drawn from those values rather than
    re-extracted. Draws are without replacement within each class, which makes
    this the sampling distribution of the ratio under the study's own design
    rather than a bootstrap approximation to it.

    A ratio is undefined when a draw's Unripe mean is zero, which cannot
    happen here but is guarded anyway; such draws are excluded and counted.
    """
    names = list(build_extractors(config)["t1_default"].feature_names)
    decay = matrices["t1_default"].X[:, names.index(DECAY_FEATURE)]
    labels = matrices["t1_default"].y
    display = list(config.primary.display_names)
    unripe = decay[labels == display.index("Unripe")]
    rotten = decay[labels == display.index("Rotten")]

    print("\n  E1.6b sample-size sensitivity of the decay separation")
    rng = np.random.default_rng(config.seed)
    rows = []
    smallest = min(unripe.size, rotten.size)
    for per_class in (30, 45, 60, 100, 200, 400, smallest):
        if per_class > smallest:
            continue
        ratios = []
        for _ in range(draws if per_class < smallest else 1):
            u = rng.choice(unripe, size=per_class, replace=False).mean()
            r = rng.choice(rotten, size=per_class, replace=False).mean()
            if u > 0:
                ratios.append(r / u)
        ratios = np.asarray(ratios, dtype=np.float64)
        rows.append({
            "images_per_class": int(per_class),
            "draws": int(ratios.size),
            "ratio_rotten_over_unripe_mean": float(ratios.mean()),
            "ratio_sd": float(ratios.std(ddof=1)) if ratios.size > 1 else 0.0,
            "ratio_p05": float(np.percentile(ratios, 5)),
            "ratio_p95": float(np.percentile(ratios, 95)),
            "unripe_decay_mean": float(unripe.mean()),
            "rotten_decay_mean": float(rotten.mean()),
            "is_full_partition": bool(per_class == smallest),
        })
        print(f"    {per_class:>4} per class   ratio {ratios.mean():.2f}"
              f"  (5th-95th {np.percentile(ratios, 5):.2f}-{np.percentile(ratios, 95):.2f})")
    return pd.DataFrame(rows)


#: The threshold the supporting t-test would be read at, if it were the
#: claim. It is not; it is recorded so the report can state exactly how often
#: the two criteria would disagree, and where.
SUPPORTING_ALPHA = 0.05


def criterion_agreement(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count where the noise band and the supporting t-test would disagree.

    Chapter 4 needs this stated rather than left in a column. If the two
    criteria agreed everywhere, the choice between them would not matter and
    fixing one in advance would be an empty precaution. They do not agree
    everywhere, and the single arm on which they part company sits at
    p = 0.0517 - close enough to the conventional cliff that the verdict is
    decided by the threshold rather than by the data. That is the concrete
    case for having chosen the criterion before seeing it.

    The reference arm of each table is excluded: its difference from itself is
    zero by construction and has no t statistic.
    """
    rows = []
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


def select_winner(combined: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, str]:
    """Choose the arm to spend the single test evaluation on.

    Selecting on mean accuracy alone selects on noise. Every comparison in
    this sweep sits inside its own fold spread, so the top-ranked arm is
    frequently ahead by less than the run-to-run variation that produced the
    ranking, and spending the one held-out evaluation on a margin of a few
    thousandths would report a distinction the data does not contain.

    Two arms are treated as tied when the mean of their paired fold-wise
    differences lies inside the stated noise band - smaller than the standard
    deviation of those same differences. The pairing matters: the arms share
    folds, seed and images, so the difference is far less variable than either
    accuracy alone.

    This is deliberately a band and not a hypothesis test. Each sweep compares
    many arms against one leader with no correction for multiplicity, and
    cross-validation folds share training rows so the per-fold differences are
    not independent. A band makes neither assumption.

    Among the tied set the choice is parsimony - fewest dimensions, then
    fewest image-processing stages, then arm name for determinism. A shorter
    descriptor that cannot be distinguished from a longer one is the better
    result, and choosing it makes the report defend the weaker claim rather
    than the stronger one.

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
    ordered = tied.sort_values(
        ["dimensionality", "pipeline_stages", "arm"], ascending=[True, True, True]
    )
    winner = ordered.iloc[0]
    selected_by = "accuracy" if winner.arm == combined.iloc[leader].arm else "parsimony"
    return winner, tied, selected_by


def evaluate_winner(
    tables: Dict[str, pd.DataFrame],
    train_matrices: Dict[str, FeatureMatrix],
    extractors: Dict[str, object],
    partition,
    config: Config,
) -> pd.DataFrame:
    """Score the single winning arm on the test partition, once.

    The only place in E1 that touches held-out data, and it runs after every
    table above is closed. The sweep selects; this reports. Scoring more than
    one arm here would turn the test partition into a second selection stage.
    """
    combined = pd.concat(
        [table for name, table in tables.items() if "arm" in table.columns],
        ignore_index=True,
    ).drop_duplicates(subset=["arm"]).reset_index(drop=True)

    winner, tied, selected_by = select_winner(combined)
    leader = combined.loc[combined.cv_mean_accuracy.idxmax()]
    key = str(winner.config_key)
    dropped = str(winner.dropped_feature) if winner.dropped_feature else ""

    print("\n  Closing evaluation - one arm, one pass over the test partition")
    print(f"    highest accuracy  : {leader.arm} (cv {leader.cv_mean_accuracy:.4f})")
    print(f"    tied with it      : {len(tied)} of {len(combined)} arms, "
          f"by paired fold differences")
    print(f"    selected          : {winner.arm} "
          f"(dim {int(winner.dimensionality)}, {int(winner.pipeline_stages)} stages) "
          f"by {selected_by}")
    print(f"    extracting test features for {key} only ...")

    test = extract_all(partition.test, {key: extractors[key]}, False, config)[key]
    train = train_matrices[key]
    if dropped:
        names = list(extractors[key].feature_names)
        train = drop_feature(train, names, dropped)
        test = drop_feature(test, names, dropped)

    fitted = build_pipeline(config).fit(train.X, train.y)
    report = classification_metrics(
        test.y, fitted.predict(test.X), list(config.primary.display_names),
        technique=str(winner.arm),
    )
    print_report(report)

    per_class = report.per_class.set_index(report.per_class.index)
    return pd.DataFrame([{
        "arm": winner.arm,
        "config_key": key,
        "dropped_feature": dropped,
        "selected_by": selected_by,
        "dimensionality": int(winner.dimensionality),
        "pipeline_stages": int(winner.pipeline_stages),
        "highest_accuracy_arm": leader.arm,
        "highest_accuracy_cv": float(leader.cv_mean_accuracy),
        "winner_cv_accuracy": float(winner.cv_mean_accuracy),
        "margin_over_selected": float(leader.cv_mean_accuracy - winner.cv_mean_accuracy),
        "arms_tied_with_leader": int(len(tied)),
        "arms_compared": int(len(combined)),
        "arms_scored_on_test": 1,
        "test_accuracy": report.accuracy,
        "test_macro_f1": report.macro_f1,
        "test_weighted_f1": report.weighted_f1,
        "test_rows": int(test.X.shape[0]),
        "selected_on": "5-fold CV over the training partition, tie band then parsimony",
    }])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the five E1 experiments on one segmentation pass."""
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=0,
                        help="Images per class; 0 uses the whole dataset.")
    parser.add_argument("--augment", action="store_true",
                        help="Augment the training partition, as the benchmark does.")
    parser.add_argument("--tag", default="e1",
                        help="Sub-directory of results/ to write into.")
    args = parser.parse_args(argv)

    set_global_seed(config)
    records = load_primary(config)
    if args.per_class:
        records = balanced_subset(records, args.per_class, config)
    partition = stratified_split(records, config)
    extractors = build_extractors(config)

    print("=" * 74)
    print("E1 sub-experiments - MPEG-7 dominant colour descriptor")
    print("=" * 74)
    print(f"  images          : {len(records)} ({args.per_class or 'all'} per class)")
    print(f"  scored on       : {len(partition.train)} training images, "
          f"5-fold CV; test partition untouched")
    print(f"  augmentation    : {'on' if args.augment else 'OFF'}")
    print(f"  configurations  : {len(extractors)}, one segmentation pass")
    print()

    matrices = extract_all(partition.train, extractors, args.augment, config)

    tables = {
        "e1_1": e1_1_baseline(matrices, config),
        "e1_2": e1_2_n_colours(matrices, config),
        "e1_3": e1_3_space(matrices, config),
        "e1_4": e1_4_specular(matrices, extractors, config),
        "e1_5": e1_5_blocks(matrices, config),
        "e1_6": e1_6_decay_green_exclusion(matrices, config),
        "e1_6b_sample_sensitivity": e1_6_sample_sensitivity(matrices, config),
    }

    final = evaluate_winner(tables, matrices, extractors, partition, config)
    agreement = criterion_agreement(tables)
    print()
    print(report_criterion_agreement(agreement))

    output = config.paths.results_subdir(args.tag)
    for name, table in tables.items():
        save_dataframe(table, output / f"{name}.csv", index=False)
    save_dataframe(final, output / "e1_final_test.csv", index=False)
    save_dataframe(agreement, output / "criterion_agreement.csv", index=False)
    (output / "criterion_agreement.txt").write_text(
        report_criterion_agreement(agreement) + "\n", encoding="utf-8"
    )

    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "per_class": args.per_class or "all",
                "augmentation": args.augment,
                "n_images": len(records),
                "n_train_images": len(partition.train),
                "scored_partition": "train (5-fold CV)",
                "test_partition_used_for": "one closing evaluation of the winning arm",
                "seed": config.seed,
                "segmentation_method": config.segmentation.method,
                "configurations": sorted(extractors),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n  written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
