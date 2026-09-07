"""T2 internal comparative experiments (E2.1 to E2.4).

Four questions about the GLCM texture descriptor, each answered by 5-fold
stratified cross-validation on the **training partition only**:

===== =========================================================================
E2.1  Does the co-occurrence machinery beat first-order intensity? 40 Haralick
      dimensions against four grey-level moments.
E2.2  How far apart should the pixel pairs be? Distance sets [1], [1,2], [1,2,3].
E2.3  How finely should grey be quantised? 16, 32, 64 levels.
E2.4  Per-angle descriptors against the angle-averaged, rotation-invariant form.
===== =========================================================================

The naming collides with Phase 5, and it is worth saying plainly: **E2.1 to
E2.4 here are sweeps of the T2 descriptor**, and have nothing to do with the
enhancement ``scripts/run_enhancements.py`` calls E2, which is blemish-aware
regional weighting. Different experiment, different file, different directory.

This sweep is diagnostic as well as required
--------------------------------------------

T2 cross-validates at 0.6235 on the full run, against a background-only control
of 0.7396. It scores *below* a classifier that never sees the fruit, which is
not a result one reports and moves on from - it means either the descriptor is
misconfigured or it is unsuited to the task. These four arms separate those two
readings, and E2.1 does most of the work: if four first-order moments match or
beat forty Haralick statistics, the co-occurrence machinery is contributing
nothing on a problem whose signal is chromatic, and no choice of distance,
quantisation or angle will rescue it. 32 levels and distances [1, 2] are
defaults inherited from the literature, not measured choices, and until this
runs nobody can say which they are.

The test split is never read. Nothing here is tuned towards the report's
acceptance targets: a worse number is a result, and given where T2 starts, a
worse number is a likely one.

Every variant is scored on one shared segmentation pass. Preprocessing and
seeded GrabCut cost about 400 ms per image against the descriptor's 2 ms, so
the segmentation is more than 99% of the work and running it once for all seven
configurations is both far faster and stricter than seven independent passes,
which could otherwise drift apart.

Run from the project root::

    # Smoke: exercises all four experiments end to end in about a minute.
    python scripts/run_e2_experiments.py --per-class 25 --no-augment --tag smoke

    # The full run the report quotes. Roughly 20 minutes.
    python scripts/run_e2_experiments.py

Results land in ``results/e2/``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config import Config, get_config  # noqa: E402
from data import ImageRecord, load_primary  # noqa: E402
from evaluate import cross_validate_technique, save_dataframe  # noqa: E402
from features.base import FeatureExtractionError  # noqa: E402
from features.intensity_baseline import IntensityMomentExtractor  # noqa: E402
from features.t2_glcm import PROPERTIES, GLCMExtractor  # noqa: E402
from harness import (  # noqa: E402
    FeatureMatrix,
    build_pipeline,
    collect_failures,
    iter_prepared,
    set_global_seed,
    stratified_split,
)

#: Every distinct GLCM configuration the four experiments need, named.
#: ``baseline`` is the descriptor as ``config.json`` configures it and is
#: shared by all four, so it is extracted once.
GLCM_VARIANTS: Dict[str, dict] = {
    "baseline": {},
    "distances_1": {"distances": (1,)},
    "distances_123": {"distances": (1, 2, 3)},
    "levels_16": {"levels": 16},
    "levels_64": {"levels": 64},
    "angle_averaged": {"angle_averaged": True},
}

#: The one arm that is not a co-occurrence descriptor at all.
INTENSITY_VARIANT = "intensity_moments"


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def build_glcm(config: Config, **overrides: object) -> GLCMExtractor:
    """Build a descriptor from ``config.json``, then apply ``overrides``.

    Reading the configuration first matters more than it looks. Constructing
    the extractor bare would silently fall back to the module defaults, so an
    edit to ``config.json`` would move every other driver's T2 and leave this
    one alone - the sweep would then be centred on a configuration nobody else
    uses, and every arm would be wrong together.

    Args:
        config: The loaded configuration.
        **overrides: Constructor arguments to replace for this variant.

    Returns:
        A configured :class:`GLCMExtractor`.

    Raises:
        KeyError: If an override names a parameter the extractor does not take,
            which would otherwise be silently ignored.
    """
    accepted = set(inspect.signature(GLCMExtractor.__init__).parameters) - {"self"}
    unknown = set(overrides) - accepted
    if unknown:
        raise KeyError(
            f"{sorted(unknown)} are not constructor arguments of GLCMExtractor; "
            f"the variant table is out of date"
        )

    block = {
        key: value
        for key, value in dict(config.t2_glcm).items()
        if key in accepted
    }
    block["seed"] = config.seed
    block.update(overrides)
    return GLCMExtractor(**block)


def build_extractors(config: Config) -> Dict[str, object]:
    """Return every extractor configuration the four experiments need."""
    extractors: Dict[str, object] = {
        name: build_glcm(config, **overrides) for name, overrides in GLCM_VARIANTS.items()
    }
    extractors[INTENSITY_VARIANT] = IntensityMomentExtractor.from_config(config)
    return extractors


def expected_dim(extractor: GLCMExtractor) -> int:
    """Dimensionality the configuration implies, independent of the extractor.

    Computed from the sweep's own understanding of the layout rather than read
    off ``extractor.dim``, so that an arm whose length is not what the
    experiment believes it to be fails loudly instead of being reported under
    the wrong description.
    """
    per_property = len(extractor.distances) * (
        1 if extractor.angle_averaged else len(extractor.angles_deg)
    )
    return len(PROPERTIES) * per_property


# --------------------------------------------------------------------------- #
# One shared pass
# --------------------------------------------------------------------------- #

def balanced_subset(
    records: Sequence[ImageRecord],
    per_class: int,
    config: Config,
) -> List[ImageRecord]:
    """Draw ``per_class`` records from each class, reproducibly.

    Mirrors the benchmark and the T1/T3 experiment drivers, so a pilot of any
    of them describes the same images.
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
    extractors: Dict[str, object],
    augment: bool,
    config: Config,
) -> Tuple[Dict[str, FeatureMatrix], int]:
    """Build one :class:`~harness.FeatureMatrix` per variant, in a single pass.

    Row inclusion follows the harness exactly: a sample whose segmentation
    failed is dropped unless a substitute mask was supplied, and the group and
    variant provenance is carried through so that
    :func:`~evaluate.cross_validate_technique` can still draw leakage-safe
    folds.

    A row is committed only once **every** variant has described it. If any
    extractor refuses the sample the row is dropped from all of them, so the
    arms are never scored on subtly different image sets - a difference in
    accuracy has to be a difference in the descriptor, not in which apples each
    arm was shown.

    Returns:
        The matrices, and the number of rows dropped by an extraction refusal.
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
    refused = 0
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

        row: Dict[str, np.ndarray] = {}
        elapsed: Dict[str, float] = {}
        try:
            for name, extractor in extractors.items():
                image, mask = sample.image.copy(), sample.mask.copy()
                if not warmed[name]:
                    extractor(image, mask)  # Warm-up, not timed and not kept.
                    extractor.reset_diagnostics()
                    warmed[name] = True
                start = time.perf_counter()
                row[name] = extractor(image, mask)
                elapsed[name] = time.perf_counter() - start
        except FeatureExtractionError as error:
            # Roll the diagnostics back to the committed length, so the E2.1
            # and E2.3 diagnostic columns stay aligned with the matrix rows.
            for extractor in extractors.values():
                del extractor.diagnostics[len(labels):]
            refused += 1
            print(f"\n    refused {sample.record.path.name}: {error}")
            continue

        labels.append(sample.label)
        paths.append(sample.record.path)
        groups.append(sample.record_index)
        variant_ids.append(sample.variant)
        if sample.is_augmented:
            n_augmented += 1
        for name in extractors:
            vectors[name].append(row[name])
            times[name].append(elapsed[name])

    print()
    matrices = {
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
    return matrices, refused


def score(
    matrix: FeatureMatrix,
    arm: str,
    config: Config,
    **extra: object,
) -> dict:
    """Cross-validate one arm on the training partition and return its row."""
    report = cross_validate_technique(build_pipeline(config), matrix, technique=arm, config=config)
    print(
        f"    {arm:<24} dim {matrix.dim:>4}  "
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


def control_columns(config: Config) -> dict:
    """The background-only control, carried on every row of every table.

    T2's standing is only legible against it. An arm at 0.62 sounds like a
    weak-but-working descriptor until it is put beside a classifier that scores
    0.74 with the fruit masked out, at which point it is something else
    entirely. Reading the number off the audit rather than restating it keeps
    the two from drifting.
    """
    path = config.paths.results_root / "audit" / "primary_background_only.csv"
    if not path.exists():
        return {"background_only_control": float("nan")}
    frame = pd.read_csv(path)
    if "background_only_recall" not in frame.columns or frame.empty:
        return {"background_only_control": float("nan")}
    return {"background_only_control": float(frame["background_only_recall"].mean())}


# --------------------------------------------------------------------------- #
# The four experiments
# --------------------------------------------------------------------------- #

def e2_1_baseline(
    matrices: Dict[str, FeatureMatrix],
    extractors: Dict[str, object],
    config: Config,
) -> pd.DataFrame:
    """E2.1 - co-occurrence texture against first-order intensity.

    Both arms see the same images, the same masks and the same grey levels over
    the same pixel set; the only difference is whether the *arrangement* of
    those grey levels is measured. The baseline is ten times shorter, so if it
    holds its own the forty dimensions are not paying for themselves.
    """
    print("\n  E2.1  co-occurrence against first-order intensity")
    control = control_columns(config)
    arms = {
        "glcm_40": ("baseline", "5 Haralick properties x 2 distances x 4 angles"),
        "intensity_moments_4": (INTENSITY_VARIANT, "mean, variance, skewness, kurtosis"),
    }
    rows = []
    for arm, (key, description) in arms.items():
        row = score(matrices[key], arm, config, descriptor=description, **control)
        row["beats_background_control"] = bool(
            row["cv_mean_accuracy"] > row["background_only_control"]
        ) if np.isfinite(row["background_only_control"]) else None
        rows.append(row)

    constant = [d for d in extractors[INTENSITY_VARIANT].diagnostics if d.constant]
    if constant:
        print(f"    note: {len(constant)} image(s) had a constant grey level")
    return pd.DataFrame(rows)


def e2_2_distances(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E2.2 - how far apart the co-occurring pixels should be.

    The vector length changes with the distance set, so dimensionality is
    reported alongside accuracy and nothing is padded to a common length. A
    larger offset reads coarser structure; whether the peel has any is the
    question.
    """
    print("\n  E2.2  distance set")
    control = control_columns(config)
    arms = {
        "1": ("distances_1", (1,)),
        "1_2": ("baseline", (1, 2)),
        "1_2_3": ("distances_123", (1, 2, 3)),
    }
    return pd.DataFrame(
        [
            score(matrices[key], f"distances_{arm}", config,
                  distances=str(list(distances)), n_distances=len(distances), **control)
            for arm, (key, distances) in arms.items()
        ]
    )


def e2_3_levels(
    matrices: Dict[str, FeatureMatrix],
    extractors: Dict[str, object],
    config: Config,
) -> pd.DataFrame:
    """E2.3 - how finely grey should be quantised.

    Dimensionality is constant across these three arms: quantisation changes
    the size of the co-occurrence matrix, not the number of properties read off
    it. Coarser levels pool more pixels into each bin and so estimate each
    probability from more evidence; finer levels resolve more structure and
    estimate each from less. 32 was inherited, not chosen.
    """
    print("\n  E2.3  grey-level quantisation")
    control = control_columns(config)
    arms = {16: "levels_16", 32: "baseline", 64: "levels_64"}
    rows = []
    for levels, key in arms.items():
        degenerate = [d.degenerate_slices for d in extractors[key].diagnostics]
        rows.append(
            score(
                matrices[key], f"levels_{levels}", config,
                levels=levels,
                glcm_cells=levels * levels,
                mean_degenerate_slices=float(np.mean(degenerate)) if degenerate else float("nan"),
                **control,
            )
        )
    return pd.DataFrame(rows)


def e2_4_angles(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E2.4 - per-angle descriptors against the rotation-invariant form.

    Averaging the four Haralick directions makes the descriptor invariant to
    rotation and costs three quarters of its length. On this dataset the
    trade-off is not obviously worth taking either way: an apple has no
    canonical orientation, which argues for averaging, but the harness augments
    the training set by rotating images, which already teaches the classifier
    to tolerate orientation and so may make the invariance redundant.
    """
    print("\n  E2.4  per-angle against angle-averaged")
    control = control_columns(config)
    arms = {
        "per_angle": ("baseline", False, "4 angles kept separate"),
        "angle_averaged": ("angle_averaged", True, "averaged over the 4 angles"),
    }
    return pd.DataFrame(
        [
            score(matrices[key], arm, config,
                  angle_averaged=averaged, description=description, **control)
            for arm, (key, averaged, description) in arms.items()
        ]
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=0,
                        help="Images per class; 0 uses the whole dataset.")
    parser.add_argument("--augment", dest="augment", action="store_true",
                        help="Apply the training augmentation plan. Roughly four times slower.")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="The default; accepted so the documented smoke command runs.")
    parser.add_argument("--tag", default="e2",
                        help="Sub-directory of results/ to write into.")
    parser.set_defaults(augment=False)
    args = parser.parse_args(argv)

    set_global_seed(config)
    records = load_primary(config)
    if args.per_class:
        records = balanced_subset(records, args.per_class, config)

    # The test split is drawn and then deliberately never touched again.
    partition = stratified_split(records, config)
    training = partition.train

    mode = "PILOT" if args.per_class else "FULL"
    extractors = build_extractors(config)

    print("=" * 74)
    print(f"T2 internal experiments - {mode}")
    print("=" * 74)
    print(f"  images          : {len(records)} ({args.per_class or 'all'} per class)")
    print(f"  training rows   : {len(training)}  (test split of {len(partition.test)} untouched)")
    print(f"  augmentation    : {'on' if args.augment else 'OFF'}")
    print(f"  variants        : {len(extractors)} extractor configurations, one segmentation pass")
    print(f"  cross-validation: {config.partition.cv_folds}-fold stratified, training partition only")
    if mode == "PILOT":
        print(f"  NOTE            : {args.per_class} images per class, not the full set. "
              f"Indicative only.")

    # Every GLCM arm's length must be what the sweep believes it is, or an arm
    # gets reported under the wrong description.
    for name, extractor in extractors.items():
        if isinstance(extractor, GLCMExtractor) and extractor.dim != expected_dim(extractor):
            raise AssertionError(
                f"{name}: T2 declares {extractor.dim} dimensions, the layout implies "
                f"{expected_dim(extractor)}"
            )

    print("\n  Building features ...")
    started = time.perf_counter()
    matrices, refused = extract_all_variants(training, extractors, args.augment, config)
    elapsed = time.perf_counter() - started
    baseline = matrices["baseline"]
    print(f"  {baseline.X.shape[0]} rows x {len(extractors)} variants in {elapsed:.1f}s "
          f"({baseline.excluded} excluded by segmentation failure, "
          f"{refused} by extraction refusal)")

    output = config.paths.results_subdir(args.tag)
    tables = {
        "e2_1.csv": e2_1_baseline(matrices, extractors, config),
        "e2_2.csv": e2_2_distances(matrices, config),
        "e2_3.csv": e2_3_levels(matrices, extractors, config),
        "e2_4.csv": e2_4_angles(matrices, config),
    }
    for filename, frame in tables.items():
        save_dataframe(frame.set_index("arm"), output / filename)

    control = control_columns(config)["background_only_control"]
    best = max(
        (row for frame in tables.values() for row in frame.to_dict("records")),
        key=lambda row: row["cv_mean_accuracy"],
    )

    # The configuration that produced these numbers travels with them.
    log = {
        "mode": mode,
        "seed": config.seed,
        "per_class": args.per_class or None,
        "augmented": args.augment,
        "n_images": len(records),
        "n_training_records": len(training),
        "n_training_rows": int(baseline.X.shape[0]),
        "n_excluded_by_segmentation": int(baseline.excluded),
        "n_refused_by_extraction": int(refused),
        "cv_folds": config.partition.cv_folds,
        "test_split_touched": False,
        "dataset_root": str(config.paths.primary_root),
        "dataset_audit_verdict": "NOT SUITABLE - see README; 74.0% background-only "
                                 "against a 33.3% chance level. Relative comparisons "
                                 "between these arms are unaffected; absolute "
                                 "accuracies carry that ceiling.",
        "segmentation_method": config.segmentation.method,
        "t2_config": {k: v for k, v in dict(config.t2_glcm).items() if k != "_comment"},
        "glcm_variants": {k: {kk: list(vv) if isinstance(vv, tuple) else vv
                              for kk, vv in v.items()}
                          for k, v in GLCM_VARIANTS.items()},
        "intensity_baseline_dim": extractors[INTENSITY_VARIANT].dim,
        "background_only_control": control,
        "best_arm": best["arm"],
        "best_arm_accuracy": best["cv_mean_accuracy"],
        "best_arm_beats_control": bool(best["cv_mean_accuracy"] > control)
                                  if np.isfinite(control) else None,
        "elapsed_seconds": round(elapsed, 1),
    }
    (output / "run_config.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\n  Written to {output}:")
    for filename in tables:
        print(f"    {filename}")
    print("    run_config.json")

    if np.isfinite(control):
        verdict = "clears" if best["cv_mean_accuracy"] > control else "does NOT clear"
        print(f"\n  Best arm: {best['arm']} at {best['cv_mean_accuracy']:.4f}. "
              f"It {verdict} the {control:.4f} background-only control.")
        if best["cv_mean_accuracy"] <= control:
            print("  No configuration of this descriptor beats a classifier that never "
                  "sees the\n  fruit. That is the finding; it is not a reason to keep "
                  "sweeping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
