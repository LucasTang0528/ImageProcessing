"""Phase 3 - benchmark each technique through the shared harness.

Every technique is run through exactly the same code path: the same
partition, the same segmentation, the same augmentation plan, the same
pipeline and the same metrics. Nothing in this script is technique-specific
beyond which extractor object it hands to :func:`harness.build_feature_matrix`.

Run from the project root::

    # Pilot: a balanced subset, no augmentation. Minutes, not hours.
    python scripts/run_benchmarks.py --per-class 300 --no-augment --tag pilot

    # The full run the report quotes.
    python scripts/run_benchmarks.py --tag full

A pilot is not a substitute for the full run. It uses fewer images and skips
augmentation, so its confidence intervals are wider and its cross-validation
folds are smaller; it exists to show whether the pipeline produces sane
numbers before an hours-long run is committed to. Every output file records
which mode produced it.

Results land in ``results/<tag>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

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
    save_confusion_matrix,
    save_dataframe,
    save_segmentation_failures,
    summarise_extraction_time,
    time_inference,
)
from features.t1_dominant_colour import DominantColourExtractor  # noqa: E402
from features.t2_glcm import GLCMExtractor  # noqa: E402
from harness import (  # noqa: E402
    build_feature_matrix,
    build_pipeline,
    set_global_seed,
    stratified_split,
)


def available_extractors(config: Config) -> Dict[str, object]:
    """Return every technique that is currently implemented.

    T3 is absent from this mapping while it remains unmerged. Listing it here
    before it exists would produce a comparison with a silent hole in it.
    """
    extractors: Dict[str, object] = {
        "T1": DominantColourExtractor.from_config(config),
        "T2": GLCMExtractor.from_config(config),
    }
    try:
        from features.t3_morphological import T3MorphologicalExtractor  # noqa: WPS433

        # T3 carries its own parameter defaults inside the module rather than
        # reading config.json, so there is nothing to hand it here.
        extractors["T3"] = T3MorphologicalExtractor()
    except ImportError:
        pass
    return extractors


def balanced_subset(
    records: Sequence[ImageRecord],
    per_class: int,
    config: Config,
) -> List[ImageRecord]:
    """Draw ``per_class`` records from each class, reproducibly.

    The draw is seeded and the result re-sorted into the original order, so
    the subset - and therefore the partition drawn from it - is identical on
    every machine and for every technique.
    """
    rng = np.random.default_rng(config.seed)
    chosen: List[int] = []
    for label in range(config.primary.n_classes):
        pool = [i for i, record in enumerate(records) if record.label == label]
        take = min(per_class, len(pool))
        chosen.extend(int(pool[i]) for i in rng.choice(len(pool), size=take, replace=False))
    return [records[i] for i in sorted(chosen)]


def progress_reporter(name: str, total: int):
    """Return a callback that prints progress on one line."""

    def report(done: int, expected: int) -> None:
        if done % 100 == 0 or done == expected:
            print(f"    {name}: {done}/{expected}", end="\r", flush=True)

    return report


def benchmark(
    short_name: str,
    extractor: object,
    partition,
    augment: bool,
    config: Config,
) -> Dict[str, object]:
    """Run one technique end to end and collect every reported metric."""
    print(f"\n  {short_name}: building training features "
          f"({'augmented' if augment else 'originals only'}) ...")
    train = build_feature_matrix(
        partition.train, extractor, augment=augment, config=config,
        progress=progress_reporter(f"{short_name} train", len(partition.train)),
    )
    print(f"\n  {short_name}: building test features (never augmented) ...")
    test = build_feature_matrix(
        partition.test, extractor, augment=False, config=config,
        progress=progress_reporter(f"{short_name} test", len(partition.test)),
    )
    print()

    pipeline = build_pipeline(config)
    cv = cross_validate_technique(pipeline, train, technique=short_name, config=config)

    fitted = build_pipeline(config)
    start = time.perf_counter()
    fitted.fit(train.X, train.y)
    fit_seconds = time.perf_counter() - start

    predicted = fitted.predict(test.X)
    report = classification_metrics(
        test.y, predicted, config.primary.display_names, technique=short_name
    )
    inference = time_inference(fitted, test.X)
    extraction = summarise_extraction_time(train.extraction_times)

    print_report(report, cv)
    return {
        "short_name": short_name,
        "report": report,
        "cv": cv,
        "train": train,
        "test": test,
        "row": {
            "technique": short_name,
            "dimensionality": train.dim,
            **{k: v for k, v in report.to_row().items() if k != "technique"},
            "cv_mean_accuracy": cv.mean_accuracy,
            "cv_std_accuracy": cv.std_accuracy,
            "cv_mean_macro_f1": cv.mean_macro_f1,
            "cv_std_macro_f1": cv.std_macro_f1,
            **{f"cv_fold{i + 1}_accuracy": float(s) for i, s in enumerate(cv.accuracy_folds)},
            **{f"cv_fold{i + 1}_macro_f1": float(s) for i, s in enumerate(cv.macro_f1_folds)},
            "extraction_mean_s": extraction["mean_s"],
            "extraction_median_s": extraction["median_s"],
            "inference_mean_s": inference,
            "fit_seconds": fit_seconds,
            "train_rows": int(train.X.shape[0]),
            "test_rows": int(test.X.shape[0]),
            "train_excluded": train.excluded,
            "test_excluded": test.excluded,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark for every implemented technique."""
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=0,
                        help="Images per class; 0 uses the whole dataset.")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="Skip training augmentation. Faster, and weaker.")
    parser.add_argument("--tag", default="phase3",
                        help="Sub-directory of results/ to write into.")
    parser.set_defaults(augment=True)
    args = parser.parse_args(argv)

    set_global_seed(config)
    records = load_primary(config)
    if args.per_class:
        records = balanced_subset(records, args.per_class, config)

    partition = stratified_split(records, config)
    extractors = available_extractors(config)

    mode = "PILOT" if (args.per_class or not args.augment) else "FULL"
    print("=" * 74)
    print(f"Phase 3 benchmark - {mode}")
    print("=" * 74)
    print(f"  images          : {len(records)} "
          f"({args.per_class or 'all'} per class)")
    print(f"  train / test    : {len(partition.train)} / {len(partition.test)}")
    print(f"  augmentation    : {'on' if args.augment else 'OFF'}")
    print(f"  techniques      : {', '.join(extractors)}")
    if "T3" not in extractors:
        print("  note            : T3 is not implemented; it is absent from this comparison.")
    if mode == "PILOT":
        print("  NOTE            : reduced data. Indicative only - not the reported figures.")

    results = [
        benchmark(name, extractor, partition, args.augment, config)
        for name, extractor in extractors.items()
    ]

    output = config.paths.results_subdir(args.tag)
    matrix = pd.DataFrame([r["row"] for r in results]).set_index("technique")
    save_dataframe(matrix, output / "benchmark_matrix.csv")

    for result in results:
        name = result["short_name"]
        save_dataframe(result["report"].per_class, output / f"{name}_per_class.csv")
        save_dataframe(
            pd.DataFrame(
                result["report"].confusion_normalised,
                index=list(config.primary.display_names),
                columns=list(config.primary.display_names),
            ),
            output / f"{name}_confusion_normalised.csv",
        )
        save_confusion_matrix(result["report"], output / f"{name}_confusion.png")
        save_segmentation_failures(
            result["train"].failures + result["test"].failures,
            output / f"{name}_segmentation_failures.csv",
        )

    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "per_class": args.per_class or "all",
                "augmentation": args.augment,
                "n_images": len(records),
                "n_train_images": len(partition.train),
                "n_test_images": len(partition.test),
                "techniques": list(extractors),
                "seed": config.seed,
                "segmentation_method": config.segmentation.method,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 74)
    print("BENCHMARK MATRIX")
    print("=" * 74)
    columns = [
        "dimensionality", "accuracy", "macro_f1", "weighted_f1",
        "cv_mean_accuracy", "cv_std_accuracy", "extraction_mean_s", "inference_mean_s",
    ]
    print(matrix[columns].to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nWritten to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
