"""Visual sanity check for preprocessing and segmentation.

Renders original, preprocessed, mask and contour panels for a random sample of
images from each class, so that the fruit masks can be inspected by eye before
any feature is computed. Also writes a segmentation failure log and a per-class
coverage summary.

Run from the project root::

    python scripts/sanity_check_segmentation.py
    python scripts/sanity_check_segmentation.py --per-class 12 --source primary

Outputs land in ``results/phase1/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# Allow the script to be run directly from anywhere inside the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import Config, DatasetSpec, get_config  # noqa: E402
from data import (  # noqa: E402
    DatasetError,
    ImageRecord,
    load_generalisation,
    load_primary,
    load_robustness,
    read_image,
    summarise,
)
from evaluate import save_dataframe, save_segmentation_failures  # noqa: E402
from harness import (  # noqa: E402
    PreparedSample,
    collect_failures,
    prepare_sample,
    preprocess,
    segment_fruit,
    set_global_seed,
)

SOURCES = {
    "primary": (load_primary, "primary"),
    "generalisation": (load_generalisation, "generalisation"),
    "robustness": (load_robustness, "robustness"),
}


def dataset_spec_for(source: str, config: Config) -> DatasetSpec:
    """Return the class specification matching a dataset source name."""
    return {
        "primary": config.primary,
        "generalisation": config.generalisation,
        "robustness": config.robustness,
    }[source]


def sample_per_class(
    records: Sequence[ImageRecord],
    spec: DatasetSpec,
    per_class: int,
    seed: int,
) -> Dict[str, List[ImageRecord]]:
    """Draw ``per_class`` records at random from each class, reproducibly.

    Args:
        records: All records for the dataset.
        spec: Class specification, used for the class ordering.
        per_class: How many images to draw per class.
        seed: Random seed, so the same images are shown on every run.

    Returns:
        A mapping from display name to the drawn records.
    """
    rng = np.random.default_rng(seed)
    drawn: Dict[str, List[ImageRecord]] = {}
    for label, display in enumerate(spec.display_names):
        pool = [record for record in records if record.label == label]
        take = min(per_class, len(pool))
        indices = rng.choice(len(pool), size=take, replace=False)
        drawn[display] = [pool[int(index)] for index in sorted(indices)]
    return drawn


def contour_overlay(sample: PreparedSample) -> np.ndarray:
    """Draw the fruit contour and bounding box over the preprocessed image."""
    overlay = sample.image.copy()
    contour = sample.segmentation.contour
    if contour.size:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    x, y, width, height = sample.segmentation.bbox
    if width > 0 and height > 0:
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 128, 255), 2)
    return overlay


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB for Matplotlib display."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def render_class_grid(
    display_name: str,
    samples: Sequence[PreparedSample],
    originals: Sequence[np.ndarray],
    output_path: Path,
) -> Path:
    """Render one figure of original / preprocessed / mask / contour rows.

    Each image occupies one row of four panels, so that a failed segmentation
    is obvious at a glance rather than hidden inside an average.
    """
    n_rows = len(samples)
    figure, axes = plt.subplots(
        n_rows, 4, figsize=(10.0, 2.6 * n_rows), squeeze=False
    )

    column_titles = ("Original", "Preprocessed", "Fruit mask", "Contour and box")
    for row, (sample, original) in enumerate(zip(samples, originals)):
        panels = (
            to_rgb(original),
            to_rgb(sample.image),
            sample.mask,
            to_rgb(contour_overlay(sample)),
        )
        for column, panel in enumerate(panels):
            axis = axes[row][column]
            if panel.ndim == 2:
                axis.imshow(panel, cmap="grey", vmin=0, vmax=255)
            else:
                axis.imshow(panel)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(column_titles[column], fontsize=11)

        status = "FAILED" if sample.segmentation.failed else "ok"
        axes[row][0].set_ylabel(
            f"{sample.record.name[:22]}\n"
            f"{sample.segmentation.coverage:.1%} {status}\n"
            f"polarity: {sample.segmentation.polarity}",
            fontsize=7,
            rotation=0,
            ha="right",
            va="center",
            labelpad=52,
        )
        if sample.segmentation.failed:
            for column in range(4):
                for spine in axes[row][column].spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2.5)

    figure.suptitle(
        f"Segmentation sanity check - {display_name} "
        f"({n_rows} randomly drawn images)",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    return output_path


def coverage_summary(
    prepared: Dict[str, List[PreparedSample]],
) -> pd.DataFrame:
    """Summarise mask coverage and failure counts per class."""
    rows = []
    for display_name, samples in prepared.items():
        coverages = np.array([s.segmentation.coverage for s in samples], dtype=np.float64)
        failures = sum(1 for s in samples if s.segmentation.failed)
        polarities = [s.segmentation.polarity for s in samples]
        rows.append(
            {
                "class": display_name,
                "n_sampled": len(samples),
                "mean_coverage": float(np.mean(coverages)) if coverages.size else float("nan"),
                "min_coverage": float(np.min(coverages)) if coverages.size else float("nan"),
                "max_coverage": float(np.max(coverages)) if coverages.size else float("nan"),
                "n_failed": failures,
                "n_polarity_bright": polarities.count("bright"),
                "n_polarity_dark": polarities.count("dark"),
            }
        )
    return pd.DataFrame(rows)


def run(source: str, per_class: int, config: Config) -> int:
    """Execute the sanity check for one dataset source.

    Returns:
        Process exit code: 0 on success, 1 when the dataset is unavailable.
    """
    loader, source_name = SOURCES[source]
    spec = dataset_spec_for(source, config)

    print(f"Loading the {source} dataset from {getattr(config.paths, source + '_root')}")
    try:
        records = loader(config)
    except DatasetError as error:
        print(f"\nDataset unavailable.\n{error}\n")
        return 1

    print(summarise(records, spec))
    print()

    set_global_seed(config)
    drawn = sample_per_class(records, spec, per_class, config.seed)

    output_dir = config.paths.results_subdir("phase1")
    prepared: Dict[str, List[PreparedSample]] = {}
    written: List[Path] = []

    for display_name, class_records in drawn.items():
        if not class_records:
            print(f"  {display_name}: no images available, skipped")
            continue

        samples: List[PreparedSample] = []
        originals: List[np.ndarray] = []
        for record in class_records:
            raw = read_image(record.path)
            samples.append(prepare_sample(record, config=config))
            originals.append(cv2.resize(raw, config.preprocess.resize, interpolation=cv2.INTER_AREA))

        prepared[display_name] = samples
        target = output_dir / f"segmentation_{source_name}_{display_name.lower()}.png"
        written.append(render_class_grid(display_name, samples, originals, target))
        failed = sum(1 for s in samples if s.segmentation.failed)
        print(f"  {display_name}: {len(samples)} rendered, {failed} flagged as failures")

    all_samples = [sample for samples in prepared.values() for sample in samples]
    failures = collect_failures(all_samples)

    summary = coverage_summary(prepared)
    summary_path = save_dataframe(
        summary, output_dir / f"segmentation_coverage_{source_name}.csv", index=False
    )
    failure_path = save_segmentation_failures(
        failures, output_dir / f"segmentation_failures_{source_name}.csv"
    )

    print("\nCoverage summary")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nWritten:")
    for path in written + [summary_path, failure_path]:
        print(f"  {path}")

    total = len(all_samples)
    if failures:
        print(
            f"\n{len(failures)} of {total} sampled images failed the coverage check. "
            f"Inspect the red-bordered rows in the PNGs above."
        )
    else:
        print(f"\nAll {total} sampled images passed the coverage check.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the sanity check."""
    config = get_config()
    default_per_class = int(config.sanity_check.get("images_per_class", 12))

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES),
        default="primary",
        help="Which dataset root to inspect (default: primary).",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=default_per_class,
        help=f"Images to render per class (default: {default_per_class}).",
    )
    args = parser.parse_args(argv)

    return run(args.source, args.per_class, config)


if __name__ == "__main__":
    raise SystemExit(main())
