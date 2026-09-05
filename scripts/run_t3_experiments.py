"""T3 internal comparative experiments (E3.1 to E3.4).

Four questions about the multiscale morphological descriptor, each answered by
5-fold stratified cross-validation on the **training partition only**:

===== =========================================================================
E3.1  What does the multiscale spectrum contribute? Full 36 against Blocks C+D
      alone (10 dimensions) and Blocks A+B alone (18 dimensions).
E3.2  Does the structuring element shape matter? Ellipse, rectangle, cross.
E3.3  How far should the granulometry run? ``r_max`` in {5, 7, 9, 11}.
E3.4  How should blemishes be segmented? Black top hat with Otsu, a fixed
      intensity threshold, and the dropped hue-deviation baseline.
===== =========================================================================

The test split is never read. It is drawn by the shared harness and then left
alone, so nothing tuned here can be justified by the numbers it will later be
judged against.

Every variant is scored on one shared segmentation pass. Preprocessing and
seeded GrabCut cost far more than the descriptor does, and they are identical
across variants by construction - the descriptor family is the only thing being
changed - so running them once and fanning out to all eight configurations is
both faster and stricter than eight independent passes, which could otherwise
drift apart.

Run from the project root::

    # Pilot: fast enough to check the shape of the answer.
    python scripts/run_t3_experiments.py --per-class 150

    # The full run the report quotes.
    python scripts/run_t3_experiments.py

Results land in ``results/t3/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from config import Config, get_config  # noqa: E402
from data import ImageRecord, load_primary  # noqa: E402
from evaluate import cross_validate_technique, save_dataframe  # noqa: E402
from features.t3_morphological import (  # noqa: E402
    DEFAULT_CONFIG,
    T3MorphologicalExtractor,
    block_indices,
    dimension_for,
    feature_names,
)
from harness import (  # noqa: E402
    FeatureMatrix,
    build_pipeline,
    collect_failures,
    iter_prepared,
    preprocess,
    segment_fruit,
    set_global_seed,
    stratified_split,
)
from scripts.annotate_blemishes import ANNOTATION_ROOT, INDEX_NAME  # noqa: E402

#: Ground-truth blemish masks for E3.4, written by scripts/annotate_blemishes.py.
ANNOTATION_INDEX = ANNOTATION_ROOT / INDEX_NAME

#: Every distinct extractor configuration the four experiments need, named.
#: ``baseline`` is shared by all four, so it is extracted once.
VARIANTS: Dict[str, dict] = {
    "baseline": {},
    "se_rect": {"se_shape": "rect"},
    "se_cross": {"se_shape": "cross"},
    "rmax_5": {"r_max": 5},
    "rmax_9": {"r_max": 9},
    "rmax_11": {"r_max": 11},
    "seg_fixed": {"blemish_method": "fixed"},
    "seg_hue": {"blemish_method": "hue_deviation"},
}


# --------------------------------------------------------------------------- #
# One shared pass
# --------------------------------------------------------------------------- #

def balanced_subset(
    records: Sequence[ImageRecord],
    per_class: int,
    config: Config,
) -> List[ImageRecord]:
    """Draw ``per_class`` records from each class, reproducibly.

    Mirrors the benchmark driver's subset so that a pilot experiment and a
    pilot benchmark describe the same images.
    """
    rng = np.random.default_rng(config.seed)
    chosen: List[int] = []
    for label in range(config.primary.n_classes):
        pool = [i for i, record in enumerate(records) if record.label == label]
        take = min(per_class, len(pool))
        chosen.extend(int(pool[i]) for i in rng.choice(len(pool), size=take, replace=False))
    return [records[i] for i in sorted(chosen)]


def extract_all_variants(
    records: Sequence[ImageRecord],
    extractors: Dict[str, T3MorphologicalExtractor],
    augment: bool,
    config: Config,
) -> Dict[str, FeatureMatrix]:
    """Build one :class:`~harness.FeatureMatrix` per variant, in a single pass.

    Row inclusion follows the harness exactly: a sample whose segmentation
    failed is dropped unless a substitute mask was supplied, and the group and
    variant provenance is carried through so that
    :func:`~evaluate.cross_validate_technique` can still draw leakage-safe folds.
    """
    vectors: Dict[str, List[np.ndarray]] = {name: [] for name in extractors}
    times: Dict[str, List[float]] = {name: [] for name in extractors}
    warmed: Dict[str, bool] = {name: False for name in extractors}

    labels: List[int] = []
    paths: List[Path] = []
    groups: List[int] = []
    variant_ids: List[int] = []
    failures: List = []
    excluded = 0
    n_augmented = 0

    total = len(records) * (
        1 + (config.augmentation.variants_per_image if augment and config.augmentation.enabled else 0)
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


def score(
    matrix: FeatureMatrix,
    arm: str,
    config: Config,
    **extra: object,
) -> dict:
    """Cross-validate one arm on the training partition and return its row."""
    report = cross_validate_technique(build_pipeline(config), matrix, technique=arm, config=config)
    print(
        f"    {arm:<22} dim {matrix.dim:>3}  "
        f"accuracy {report.mean_accuracy:.4f} +/- {report.std_accuracy:.4f}  "
        f"macro F1 {report.mean_macro_f1:.4f}"
    )
    return {
        "arm": arm,
        "dimensionality": matrix.dim,
        "cv_mean_accuracy": report.mean_accuracy,
        "cv_std_accuracy": report.std_accuracy,
        "cv_mean_macro_f1": report.mean_macro_f1,
        "cv_std_macro_f1": report.std_macro_f1,
        **{f"fold{i + 1}_accuracy": float(value) for i, value in enumerate(report.accuracy_folds)},
        "extraction_mean_s": float(np.mean(matrix.extraction_times)),
        "rows": int(matrix.X.shape[0]),
        **extra,
    }


# --------------------------------------------------------------------------- #
# The four experiments
# --------------------------------------------------------------------------- #

def e3_1_ablation(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E3.1 - what the multiscale spectrum contributes.

    The blocks are computed independently, so an arm is obtained by selecting
    columns from the full vector rather than by re-extracting. The values are
    identical either way, and selecting keeps the three arms on exactly the
    same images.
    """
    print("\n  E3.1  multiscale contribution")
    full = matrices["baseline"]
    blocks = block_indices(int(DEFAULT_CONFIG["r_max"]))
    arms = {
        "full_36": np.arange(full.dim),
        "blocks_CD_only": np.concatenate([blocks["C"], blocks["D"]]),
        "blocks_AB_only": np.concatenate([blocks["A"], blocks["B"]]),
    }
    names = feature_names(int(DEFAULT_CONFIG["r_max"]))
    return pd.DataFrame(
        [
            score(
                replace(full, X=full.X[:, columns], technique=arm),
                arm,
                config,
                blocks="ABCDE" if arm == "full_36" else arm.split("_")[1],
                first_feature=names[int(columns[0])],
                last_feature=names[int(columns[-1])],
            )
            for arm, columns in arms.items()
        ]
    )


def e3_2_se_shape(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E3.2 - whether the structuring element shape matters."""
    print("\n  E3.2  structuring element shape")
    arms = {"ellipse": "baseline", "rect": "se_rect", "cross": "se_cross"}
    return pd.DataFrame(
        [
            score(matrices[key], shape, config, se_shape=shape)
            for shape, key in arms.items()
        ]
    )


def e3_3_rmax(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E3.3 - how far the granulometry should run.

    The vector length changes with the radius, so dimensionality is reported
    alongside accuracy. Nothing is padded or truncated to make the arms
    comparable: that the arms are *not* the same length is the finding.
    """
    print("\n  E3.3  maximum granulometric radius")
    arms = {5: "rmax_5", 7: "baseline", 9: "rmax_9", 11: "rmax_11"}
    return pd.DataFrame(
        [
            score(
                matrices[key],
                f"r_max_{r_max}",
                config,
                r_max=r_max,
                expected_dim=dimension_for(r_max),
            )
            for r_max, key in arms.items()
        ]
    )


def annotation_errors(
    extractors: Dict[str, T3MorphologicalExtractor],
    arms: Dict[str, str],
    config: Config,
) -> Dict[str, float]:
    """Mean absolute blemish-ratio error against the painted masks, per method.

    This is the metric E3.4 is specified to be scored on. It needs ground truth
    and therefore only runs when ``scripts/annotate_blemishes.py`` has been
    used; without it the column is left as NaN rather than filled with
    classification accuracy, which answers a different question - a method can
    mislabel which pixels are blemished and still feed the classifier something
    separable.
    """
    index_path = ANNOTATION_INDEX
    if not index_path.exists():
        return {}

    index = pd.read_csv(index_path)
    if index.empty:
        return {}

    print(f"    scoring against {len(index)} annotated mask(s)")
    absolute_errors: Dict[str, List[float]] = {method: [] for method in arms}

    for row in index.itertuples():
        painted = cv2.imread(str(row.annotation), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(row.path), cv2.IMREAD_COLOR)
        if painted is None or image is None:
            continue

        prepared = preprocess(image, config)
        segmentation = segment_fruit(prepared, config)
        fruit_px = max(int(np.count_nonzero(segmentation.mask)), 1)
        truth = 100.0 * float(np.count_nonzero(painted > 127)) / fruit_px

        for method, key in arms.items():
            _, aux = extractors[key].extract_with_aux(prepared, segmentation.mask)
            absolute_errors[method].append(abs(aux["blemish_ratio_pct"] - truth))

    return {
        method: float(np.mean(errors))
        for method, errors in absolute_errors.items()
        if errors
    }


def e3_4_segmentation(
    matrices: Dict[str, FeatureMatrix],
    extractors: Dict[str, T3MorphologicalExtractor],
    config: Config,
) -> pd.DataFrame:
    """E3.4 - how blemishes should be segmented.

    Scored on mean absolute blemish-ratio error against manually annotated
    masks when those exist, which is what the specification asks for. What can
    be measured without ground truth is reported either way: the blemish ratio
    each method produces per class, and how far the two comparators depart from
    the method the technique actually uses. Classification accuracy is carried
    as a secondary column and labelled as such.
    """
    print("\n  E3.4  blemish segmentation")
    arms = {"bth_otsu": "baseline", "fixed": "seg_fixed", "hue_deviation": "seg_hue"}
    names = np.asarray(feature_names(int(DEFAULT_CONFIG["r_max"])))
    ratio_column = int(np.flatnonzero(names == "blemish_ratio")[0])
    count_column = int(np.flatnonzero(names == "blemish_count")[0])

    errors = annotation_errors(extractors, arms, config)
    if not errors:
        print("    no annotated masks found; the ratio-error column stays NaN")

    reference = matrices["baseline"].X[:, ratio_column]
    display = list(config.primary.display_names)
    rows = []

    for method, key in arms.items():
        matrix = matrices[key]
        ratios = matrix.X[:, ratio_column]
        row = score(
            matrix,
            method,
            config,
            blemish_method=method,
            mean_blemish_ratio_pct=float(np.mean(ratios)),
            mean_blemish_count=float(np.mean(matrix.X[:, count_column])),
            pct_images_with_no_blemish=float(100.0 * np.mean(matrix.X[:, count_column] == 0)),
            mean_abs_deviation_from_bth_otsu_pct=float(np.mean(np.abs(ratios - reference))),
            mean_abs_ratio_error_vs_annotation=errors.get(method, float("nan")),
            annotations_available=bool(errors),
        )
        for label, name in enumerate(display):
            row[f"mean_blemish_ratio_{name}"] = float(np.mean(ratios[matrix.y == label]))
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=0,
                        help="Images per class; 0 uses the whole dataset.")
    parser.add_argument("--augment", action="store_true",
                        help="Apply the training augmentation plan. Slower.")
    parser.add_argument("--tag", default="t3",
                        help="Sub-directory of results/ to write into.")
    args = parser.parse_args(argv)

    set_global_seed(config)
    records = load_primary(config)
    if args.per_class:
        records = balanced_subset(records, args.per_class, config)

    # The test split is drawn and then deliberately never touched again.
    partition = stratified_split(records, config)
    training = partition.train

    # Only a reduced image count makes a run a pilot. Augmentation is off by
    # default here on purpose: these experiments tune the descriptor against
    # real images, and labelling that "reduced data" would misdescribe it.
    mode = "PILOT" if args.per_class else "FULL"
    print("=" * 74)
    print(f"T3 internal experiments - {mode}")
    print("=" * 74)
    print(f"  images          : {len(records)} ({args.per_class or 'all'} per class)")
    print(f"  training rows   : {len(training)}  (test split of {len(partition.test)} untouched)")
    print(f"  augmentation    : {'on' if args.augment else 'OFF'}")
    print(f"  variants        : {len(VARIANTS)} extractor configurations, one segmentation pass")
    print(f"  cross-validation: {config.partition.cv_folds}-fold stratified, training partition only")
    if mode == "PILOT":
        print(f"  NOTE            : {args.per_class} images per class, not the full set. "
              f"Indicative only.")

    extractors = {
        name: T3MorphologicalExtractor(config=overrides, seed=config.seed)
        for name, overrides in VARIANTS.items()
    }

    print("\n  Building features ...")
    started = time.perf_counter()
    matrices = extract_all_variants(training, extractors, args.augment, config)
    elapsed = time.perf_counter() - started
    baseline = matrices["baseline"]
    print(f"  {baseline.X.shape[0]} rows x {len(VARIANTS)} variants in {elapsed:.1f}s "
          f"({baseline.excluded} excluded by segmentation failure)")

    output = config.paths.results_subdir(args.tag)
    tables = {
        "e3_1_ablation.csv": e3_1_ablation(matrices, config),
        "e3_2_se_shape.csv": e3_2_se_shape(matrices, config),
        "e3_3_rmax.csv": e3_3_rmax(matrices, config),
        "e3_4_segmentation.csv": e3_4_segmentation(matrices, extractors, config),
    }
    for filename, frame in tables.items():
        save_dataframe(frame.set_index("arm"), output / filename)

    # Section 12: the configuration that produced these numbers travels with them.
    log = {
        "mode": mode,
        "seed": config.seed,
        "per_class": args.per_class or None,
        "augmented": args.augment,
        "n_images": len(records),
        "n_training_records": len(training),
        "n_training_rows": int(baseline.X.shape[0]),
        "n_excluded_by_segmentation": int(baseline.excluded),
        "cv_folds": config.partition.cv_folds,
        "test_split_touched": False,
        "dataset_root": str(config.paths.primary_root),
        "dataset_audit_verdict": "NOT SUITABLE - see README; 74.0% background-only "
                                 "against a 33.3% chance level. Relative comparisons "
                                 "between these arms are unaffected; absolute "
                                 "accuracies carry that ceiling.",
        "segmentation_method": config.segmentation.method,
        "default_config": dict(DEFAULT_CONFIG),
        "variants": VARIANTS,
        "e3_4_annotations_available": ANNOTATION_INDEX.exists(),
        "elapsed_seconds": round(elapsed, 1),
    }
    (output / "run_config.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\n  Written to {output}:")
    for filename in tables:
        print(f"    {filename}")
    print("    run_config.json")
    print("\n  E3.4 note: mean_abs_ratio_error_vs_annotation is NaN because no "
          "manually\n             annotated blemish masks exist yet. Build them with "
          "scripts/annotate_blemishes.py\n             and re-run to fill that column.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
