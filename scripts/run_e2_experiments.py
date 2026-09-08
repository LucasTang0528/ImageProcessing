"""E2 sub-experiments: what the GLCM descriptor's choices are worth.

Four questions, each isolating one decision taken when T2 was specified:

======  =========================================================  =========
E2.1    GLCM against a first-order intensity baseline              e2_1.csv
E2.2    offsets, [1], [1,2], [1,2,3]                               e2_2.csv
E2.3    grey-level quantisation, 16, 32, 64                        e2_3.csv
E2.4    per-angle against angle-averaged descriptors               e2_4.csv
======  =========================================================  =========

This sweep is diagnostic as well as required. T2 scores below the
background-only control on the primary dataset, so it has to establish which
of two things is true: the descriptor is misconfigured, or texture is the
wrong evidence for a task whose classes differ mainly in colour. E2.1 answers
that most directly - if a spatial descriptor cannot beat four moments of the
intensity histogram, no amount of retuning the offsets will save it.

Every arm is scored on **one shared segmentation pass**, and all scoring is
5-fold cross-validation on the **training partition only**. The test partition
is touched exactly once, at the end, for the winning arm alone: the sweep
selects, the closing evaluation reports, and scoring more arms there would
turn held-out data into a second selection stage.

Run from the project root::

    python scripts/run_e2_experiments.py --per-class 100    # quick look
    python scripts/run_e2_experiments.py                    # the reported run

Results land in ``results/e2/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config import Config, get_config  # noqa: E402
from data import load_primary  # noqa: E402
from evaluate import (  # noqa: E402
    classification_metrics,
    print_report,
    save_dataframe,
)
from features.t2_glcm import GLCMExtractor  # noqa: E402
from features.t2_intensity_baseline import IntensityMomentExtractor  # noqa: E402
from harness import (  # noqa: E402
    FeatureMatrix,
    build_pipeline,
    set_global_seed,
    stratified_split,
)
from run_benchmarks import balanced_subset  # noqa: E402
from run_e1_experiments import (  # noqa: E402
    PIPELINE_STAGES,
    add_paired_differences,
    criterion_agreement,
    evaluate_winner,
    extract_all,
    report_criterion_agreement,
    score,
)

#: Image-processing operations per configuration, on the same explicit basis
#: as E1's: grey conversion, quantisation, co-occurrence accumulation and
#: property reduction are one each. Angle averaging reduces properties already
#: computed rather than adding an operation, so it carries the same count and
#: earns its parsimony advantage through dimensionality instead.
PIPELINE_STAGES.update({
    "t2_default": 4,      # grey, quantise, co-occurrence, properties
    "moments": 2,         # grey, moments
    "d1": 4, "d123": 4,
    "levels16": 4, "levels64": 4,
    "angle_averaged": 4,
})


def build_extractors(config: Config) -> Dict[str, object]:
    """Every distinct extractor configuration the four experiments need.

    The default configuration is shared by one arm of each experiment, and is
    extracted once so those arms are numerically identical rather than merely
    equivalent.
    """
    block = dict(getattr(config, "t2_glcm", {}) or {})
    block.pop("_comment", None)

    def glcm(**overrides) -> GLCMExtractor:
        return GLCMExtractor(seed=config.seed, **{**block, **overrides})

    extractors: Dict[str, object] = {
        "t2_default": glcm(),
        "moments": IntensityMomentExtractor.from_config(config),
        "d1": glcm(distances=(1,)),
        "d123": glcm(distances=(1, 2, 3)),
        "levels16": glcm(levels=16),
        "levels64": glcm(levels=64),
        "angle_averaged": glcm(angle_averaged=True),
    }
    return extractors


# --------------------------------------------------------------------------- #
# The four experiments
# --------------------------------------------------------------------------- #

def e2_1_baseline(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E2.1 - spatial co-occurrence against order-blind intensity moments.

    The baseline is invariant to any permutation of the peel pixels, while
    every GLCM property depends on their arrangement. The gap between the two
    is therefore the value of the spatial information specifically, not of
    texture description in general.
    """
    print("\n  E2.1  spatial co-occurrence vs first-order moments")
    return add_paired_differences(pd.DataFrame([
        score(matrices["t2_default"], "glcm", config, config_key="t2_default",
              representation="GLCM, 5 properties"),
        score(matrices["moments"], "intensity_moments", config, config_key="moments",
              representation="mean, variance, skewness, kurtosis"),
    ]), "glcm")


def e2_2_distances(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E2.2 - how far apart the co-occurring pixels are sampled.

    A larger offset set adds coarser structure and dimensions in equal
    measure, so accuracy is reported against dimensionality rather than alone.
    """
    print("\n  E2.2  co-occurrence offsets")
    arms = (("d=[1]", "d1", "[1]"),
            ("d=[1,2]", "t2_default", "[1, 2]"),
            ("d=[1,2,3]", "d123", "[1, 2, 3]"))
    return add_paired_differences(pd.DataFrame([
        score(matrices[key], arm, config, config_key=key, distances=label)
        for arm, key, label in arms
    ]), "d=[1,2]")


def e2_3_levels(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E2.3 - how finely the grey channel is quantised.

    Quantisation trades noise against detail: coarse bins pool genuinely
    different intensities, fine bins spread a fixed number of pixels over a
    quadratically larger matrix and make the estimate sparse. Dimensionality
    does not change with this setting, so the arms are directly comparable.
    """
    print("\n  E2.3  grey-level quantisation")
    arms = (("levels=16", "levels16", 16),
            ("levels=32", "t2_default", 32),
            ("levels=64", "levels64", 64))
    return add_paired_differences(pd.DataFrame([
        score(matrices[key], arm, config, config_key=key, levels=levels)
        for arm, key, levels in arms
    ]), "levels=32")


def e2_4_angles(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E2.4 - per-angle descriptors against rotation-invariant ones.

    Averaging the four Haralick directions buys invariance to how the fruit
    happened to be oriented in front of the camera and costs any genuinely
    directional structure. Apple peel has no preferred orientation, so the
    invariant form should lose little; if it loses a lot, what the per-angle
    form is reading is the photograph rather than the fruit.
    """
    print("\n  E2.4  per-angle vs angle-averaged")
    return add_paired_differences(pd.DataFrame([
        score(matrices["t2_default"], "per-angle", config, config_key="t2_default",
              angle_averaged=False),
        score(matrices["angle_averaged"], "angle-averaged", config,
              config_key="angle_averaged", angle_averaged=True),
    ]), "per-angle")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the four E2 experiments on one segmentation pass."""
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=0,
                        help="Images per class; 0 uses the whole dataset.")
    parser.add_argument("--augment", action="store_true",
                        help="Augment the training partition, as the benchmark does.")
    parser.add_argument("--tag", default="e2",
                        help="Sub-directory of results/ to write into.")
    args = parser.parse_args(argv)

    set_global_seed(config)
    records = load_primary(config)
    if args.per_class:
        records = balanced_subset(records, args.per_class, config)
    partition = stratified_split(records, config)
    extractors = build_extractors(config)

    print("=" * 74)
    print("E2 sub-experiments - GLCM texture descriptors")
    print("=" * 74)
    print(f"  images          : {len(records)} ({args.per_class or 'all'} per class)")
    print(f"  scored on       : {len(partition.train)} training images, 5-fold CV")
    print(f"  augmentation    : {'on' if args.augment else 'OFF'}")
    print(f"  configurations  : {len(extractors)}, one segmentation pass")
    print()

    matrices = extract_all(partition.train, extractors, args.augment, config)

    tables = {
        "e2_1": e2_1_baseline(matrices, config),
        "e2_2": e2_2_distances(matrices, config),
        "e2_3": e2_3_levels(matrices, config),
        "e2_4": e2_4_angles(matrices, config),
    }
    final = evaluate_winner(tables, matrices, extractors, partition, config)
    agreement = criterion_agreement(tables)
    print()
    print(report_criterion_agreement(agreement))

    output = config.paths.results_subdir(args.tag)
    for name, table in tables.items():
        save_dataframe(table, output / f"{name}.csv", index=False)
    save_dataframe(final, output / "e2_final_test.csv", index=False)
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
