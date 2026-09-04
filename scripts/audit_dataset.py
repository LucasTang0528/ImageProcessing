"""Audit a candidate dataset for confounds before it is adopted for the study.

A comparative study is only as trustworthy as the data underneath it. A
dataset assembled by scraping the web tends to photograph each class in its
own way - unripe apples on the tree, rotten apples as studio product shots -
and a classifier can then score highly by recognising the photography rather
than the fruit. Nothing in the accuracy figures reveals this, so it has to be
measured directly, before any feature extractor is written.

The audit runs four checks and prints a verdict:

1. **Background-only classification.** The shared SVM pipeline is trained on a
   descriptor built from the border ring alone, which contains no fruit
   pixels. Near-chance accuracy means the background carries no class
   information. Well above chance means imaging style is confounded with the
   label, and every descriptor that leaks background is partly scoring on it.

2. **Segmentation behaviour by class.** Mask coverage, the polarity chosen by
   the shared segmenter, and the failure rate, broken down per class. These
   describe the segmentation *decision*, not the fruit, so they should be
   roughly constant across classes. Drift means the segmenter behaves
   differently on different classes, and any exclusion policy then removes
   images class-dependently.

3. **Background uniformity by class.** The spread of border-ring colour tells
   plain studio backdrops from cluttered scenes. If one class is mostly studio
   shots and another mostly scenes, check 1 will fail and the cause is here.

4. **Duplicate and near-duplicate images.** Scraped sets repeat images, and a
   duplicate spanning the train/test split leaks the answer. A duplicate
   spanning two *classes* additionally means the labels contradict each other.

Run from the project root::

    python scripts/audit_dataset.py                       # audit data/primary
    python scripts/audit_dataset.py --source generalisation
    python scripts/audit_dataset.py --root data/candidate --classes Unripe,Ripe,Rotten

The ``--root`` form audits a directory that is not yet wired into
``config.json``, which is how a candidate dataset is vetted before it is
adopted. Results land in ``results/audit/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402

from config import Config, DatasetSpec, get_config  # noqa: E402
from data import (  # noqa: E402
    DatasetError,
    ImageRecord,
    load_dataset,
    load_generalisation,
    load_primary,
    load_robustness,
    read_image,
    summarise,
)
from evaluate import save_dataframe  # noqa: E402
from harness import (  # noqa: E402
    build_pipeline,
    preprocess,
    segment_fruit,
    set_global_seed,
    stratified_split,
)

#: Width in pixels of the border ring treated as "definitely background".
BORDER_WIDTH = 12

#: Border-ring colour spread below which a background counts as plain.
PLAIN_BACKGROUND_STD = 20.0

#: Hamming distance between difference hashes below which two images are
#: treated as *candidate* near-duplicates of one another.
NEAR_DUPLICATE_DISTANCE = 5

#: Side length of the colour thumbnail used to confirm a candidate pair.
DUPLICATE_THUMBNAIL = 32

#: Root-mean-square difference between colour thumbnails below which a
#: candidate pair is confirmed as a genuine duplicate. Measured on this
#: study's data, true duplicates score close to 0 while distinct fruits
#: photographed the same way score above 28, so the threshold sits in a wide
#: empty gap rather than on a judgement call.
DUPLICATE_RMSE = 8.0


# --------------------------------------------------------------------------- #
# Check 1: can the background alone predict the class?
# --------------------------------------------------------------------------- #

def border_ring(image: np.ndarray, width: int = BORDER_WIDTH) -> np.ndarray:
    """Return the pixels of the outer frame of an image as an ``(N, 3)`` array.

    The ring is the part of the frame a centred fruit cannot occupy, so it is
    a conservative stand-in for "background" that needs no segmentation and
    therefore cannot be biased by a segmentation failure.
    """
    return np.concatenate(
        [
            image[:width, :, :].reshape(-1, 3),
            image[-width:, :, :].reshape(-1, 3),
            image[:, :width, :].reshape(-1, 3),
            image[:, -width:, :].reshape(-1, 3),
        ]
    )


def background_descriptor(image_bgr: np.ndarray, bins: int = 16) -> np.ndarray:
    """Describe the background ring by its HSV moments and histograms.

    Deliberately crude: the point is not to describe backgrounds well, but to
    show whether even a blunt summary of them separates the classes.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ring = border_ring(hsv).astype(np.float64)

    parts = [ring.mean(axis=0), ring.std(axis=0)]
    for channel in range(3):
        upper = 180.0 if channel == 0 else 256.0
        counts, _ = np.histogram(ring[:, channel], bins=bins, range=(0.0, upper))
        total = counts.sum()
        parts.append(counts / total if total else counts.astype(np.float64))
    return np.concatenate(parts)


def background_only_test(
    records: Sequence[ImageRecord],
    spec: DatasetSpec,
    images: Sequence[np.ndarray],
    config: Config,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Train the shared classifier on background pixels only.

    Uses the same 80:20 stratified split and the same scaler-then-SVM pipeline
    as the study itself, so the number is directly comparable with the
    accuracies the techniques will report.

    Returns:
        A dictionary of headline metrics, and a per-class recall table.
    """
    features = np.vstack([background_descriptor(image) for image in images])
    labels = np.asarray([record.label for record in records], dtype=np.int64)

    partition = stratified_split(records, config)
    train_rows, test_rows = partition.train_indices, partition.test_indices

    pipeline = build_pipeline(config)
    pipeline.fit(features[train_rows], labels[train_rows])
    predicted = pipeline.predict(features[test_rows])
    truth = labels[test_rows]

    chance = 1.0 / spec.n_classes
    accuracy = float(accuracy_score(truth, predicted))
    metrics = {
        "accuracy": accuracy,
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "chance": chance,
        "excess_over_chance": accuracy - chance,
    }

    per_class = recall_score(
        truth, predicted, labels=list(range(spec.n_classes)), average=None, zero_division=0
    )
    table = pd.DataFrame(
        {"class": list(spec.display_names), "background_only_recall": per_class}
    )
    return metrics, table


# --------------------------------------------------------------------------- #
# Checks 2 and 3: segmentation behaviour and background uniformity
# --------------------------------------------------------------------------- #

def segmentation_profile(
    records: Sequence[ImageRecord],
    images: Sequence[np.ndarray],
    config: Config,
) -> pd.DataFrame:
    """Profile the shared segmenter's behaviour on every image.

    Records statistics of the segmentation *decision* rather than of the
    fruit, so that any drift across classes can be seen.
    """
    rows = []
    for record, image in zip(records, images):
        result = segment_fruit(preprocess(image, config), config)
        ring = border_ring(image).astype(np.float64)
        rows.append(
            {
                "class": record.display_name,
                "name": record.path.name,
                "coverage": result.coverage,
                "polarity": result.polarity,
                "failed": bool(result.failed),
                "border_std": float(ring.std(axis=0).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarise_profile(profile: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Aggregate the per-image segmentation profile to one row per class."""
    grouped = profile.groupby("class")
    summary = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean_coverage": grouped["coverage"].mean(),
            "median_coverage": grouped["coverage"].median(),
            "n_failed": grouped["failed"].sum(),
            "pct_failed": 100.0 * grouped["failed"].mean(),
            "pct_polarity_dark": 100.0
            * grouped["polarity"].apply(lambda values: float((values == "dark").mean())),
            "mean_border_std": grouped["border_std"].mean(),
            "pct_plain_background": 100.0
            * grouped["border_std"].apply(
                lambda values: float((values < PLAIN_BACKGROUND_STD).mean())
            ),
        }
    )
    order = [name for name in spec.display_names if name in summary.index]
    return summary.reindex(order)


def spread(summary: pd.DataFrame, column: str) -> float:
    """Return the max-minus-min of a column across classes.

    A plain, assumption-free measure of how much a segmentation statistic
    drifts between classes. A statistic describing the segmenter rather than
    the fruit should barely move.
    """
    if summary.empty or column not in summary:
        return float("nan")
    return float(summary[column].max() - summary[column].min())


# --------------------------------------------------------------------------- #
# Check 4: duplicates and near-duplicates
# --------------------------------------------------------------------------- #

def difference_hash(image_bgr: np.ndarray, side: int = 8) -> np.ndarray:
    """Compute a 64-bit difference hash as an array of 8 bytes.

    The difference hash compares each pixel with its right-hand neighbour in a
    small greyscale thumbnail, so it is invariant to rescaling and to mild
    compression differences - exactly the variations that distinguish two
    copies of one scraped photograph.
    """
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (side + 1, side), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).astype(np.uint8).ravel()
    return np.packbits(bits)


def colour_thumbnail(image_bgr: np.ndarray, side: int = DUPLICATE_THUMBNAIL) -> np.ndarray:
    """Return a small colour thumbnail used to confirm a candidate duplicate."""
    return cv2.resize(image_bgr, (side, side), interpolation=cv2.INTER_AREA).astype(
        np.float64
    )


def thumbnail_rmse(left: np.ndarray, right: np.ndarray) -> float:
    """Root-mean-square colour difference between two thumbnails."""
    return float(np.sqrt(np.mean((left - right) ** 2)))


def duplicate_report(
    records: Sequence[ImageRecord],
    images: Sequence[np.ndarray],
    distance: int = NEAR_DUPLICATE_DISTANCE,
    rmse_limit: float = DUPLICATE_RMSE,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Find duplicate and near-duplicate image pairs.

    Detection runs in two stages, and the second stage is essential rather
    than a refinement. A difference hash keys on the luminance gradient of a
    small greyscale thumbnail, which for studio photography is dominated by
    the fruit's silhouette against a plain backdrop. Every centred apple on a
    white background therefore produces a near-identical hash, and matching on
    the hash alone reports hundreds of "duplicates" that are plainly different
    fruits - including pairs spanning two classes, which would look like
    contradictory ground truth when nothing of the sort is present.

    Each candidate pair is therefore confirmed against a colour thumbnail: a
    genuine duplicate matches in colour as well as in outline. On this study's
    data the two populations are cleanly separated, real duplicates scoring
    near zero and distinct fruits above 28.

    Args:
        records: The records being audited.
        images: Their pixel data, in the same order.
        distance: Maximum Hamming distance for a pair to be a *candidate*.
        rmse_limit: Maximum colour thumbnail difference to confirm a pair.

    Returns:
        A table of confirmed pairs, and a summary count dictionary. Pairs that
        span two different classes are listed first, because a duplicate
        carrying two different labels is a contradiction in the ground truth.
    """
    hashes = np.vstack([difference_hash(image) for image in images])
    thumbnails = [colour_thumbnail(image) for image in images]
    popcount = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)

    pairs: List[Dict[str, object]] = []
    total = len(records)
    chunk = 256
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        block = hashes[start:stop, None, :] ^ hashes[None, :, :]
        distances = popcount[block].sum(axis=2)
        rows, columns = np.nonzero(distances <= distance)
        for row, column in zip(rows, columns):
            left = start + int(row)
            right = int(column)
            if left >= right:  # Consider each unordered pair exactly once.
                continue
            colour_gap = thumbnail_rmse(thumbnails[left], thumbnails[right])
            if colour_gap > rmse_limit:  # Same outline, different fruit.
                continue
            pairs.append(
                {
                    "distance": int(distances[row, column]),
                    "colour_rmse": colour_gap,
                    "class_a": records[left].display_name,
                    "class_b": records[right].display_name,
                    "cross_class": records[left].label != records[right].label,
                    "file_a": records[left].path.name,
                    "file_b": records[right].path.name,
                }
            )

    frame = pd.DataFrame(
        pairs,
        columns=[
            "distance",
            "colour_rmse",
            "class_a",
            "class_b",
            "cross_class",
            "file_a",
            "file_b",
        ],
    )
    if not frame.empty:
        frame = frame.sort_values(
            ["cross_class", "distance", "colour_rmse"], ascending=[False, True, True]
        ).reset_index(drop=True)

    counts = {
        "n_pairs": int(len(frame)),
        # "Exact" is judged on colour, not on the hash: two images can share a
        # hash and still differ, but a near-zero colour difference cannot.
        "n_exact": int((frame["colour_rmse"] < 1.0).sum()) if not frame.empty else 0,
        "n_cross_class": int(frame["cross_class"].sum()) if not frame.empty else 0,
    }
    return frame, counts


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #

def verdict(
    background: Dict[str, float],
    summary: pd.DataFrame,
    duplicates: Dict[str, int],
    n_images: int,
) -> Tuple[bool, List[str]]:
    """Judge whether a dataset is fit to serve as the study's primary set.

    The thresholds are deliberately generous: the audit is meant to catch
    datasets that are badly confounded, not to reject every imperfection.

    Returns:
        Whether the dataset passes, and the list of findings to print.
    """
    findings: List[str] = []
    passed = True

    excess = background["excess_over_chance"]
    if excess > 0.20:
        passed = False
        findings.append(
            f"FAIL  background alone classifies at {background['accuracy']:.1%} "
            f"against a {background['chance']:.1%} chance level. Imaging style is "
            f"confounded with the label."
        )
    elif excess > 0.10:
        findings.append(
            f"WARN  background alone classifies at {background['accuracy']:.1%} "
            f"against {background['chance']:.1%}. Mild imaging-style confound; "
            f"masks must exclude background rigorously."
        )
    else:
        findings.append(
            f"PASS  background alone classifies at {background['accuracy']:.1%}, "
            f"near the {background['chance']:.1%} chance level."
        )

    failure_spread = spread(summary, "pct_failed")
    polarity_spread = spread(summary, "pct_polarity_dark")
    coverage_spread = spread(summary, "mean_coverage")

    if failure_spread > 10.0 or polarity_spread > 30.0:
        passed = False
        findings.append(
            f"FAIL  segmentation behaves differently per class: failure rate varies "
            f"by {failure_spread:.1f} points and polarity choice by "
            f"{polarity_spread:.1f} points across classes."
        )
    elif failure_spread > 5.0 or polarity_spread > 15.0:
        findings.append(
            f"WARN  segmentation drifts across classes: failure rate varies by "
            f"{failure_spread:.1f} points and polarity by {polarity_spread:.1f} points."
        )
    else:
        findings.append(
            f"PASS  segmentation behaves consistently across classes "
            f"(failure rate varies by {failure_spread:.1f} points)."
        )

    findings.append(
        f"INFO  mean mask coverage varies by {coverage_spread:.3f} across classes."
    )

    cross = duplicates["n_cross_class"]
    if cross > 0:
        passed = False
        findings.append(
            f"FAIL  {cross} duplicate pair(s) span two different classes, so the "
            f"ground truth contradicts itself."
        )
    if duplicates["n_pairs"] > 0.02 * n_images:
        findings.append(
            f"WARN  {duplicates['n_pairs']} near-duplicate pair(s) "
            f"({duplicates['n_exact']} exact) among {n_images} images; duplicates "
            f"spanning the train/test split leak the answer."
        )
    elif cross == 0:
        findings.append(
            f"PASS  {duplicates['n_pairs']} near-duplicate pair(s), none crossing a "
            f"class boundary."
        )

    return passed, findings


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def audit(
    records: Sequence[ImageRecord],
    spec: DatasetSpec,
    label: str,
    config: Config,
) -> bool:
    """Run every check over one dataset and print the report.

    Returns:
        True when the dataset passes the audit.
    """
    set_global_seed(config)
    output_dir = config.paths.results_subdir("audit")

    print(summarise(records, spec))
    print(f"\nReading and resizing {len(records)} images ...")
    images: List[np.ndarray] = []
    for index, record in enumerate(records, start=1):
        images.append(
            cv2.resize(
                read_image(record.path),
                config.preprocess.resize,
                interpolation=cv2.INTER_AREA,
            )
        )
        if index % 500 == 0:
            print(f"  {index}/{len(records)}", flush=True)

    print("\n[1/4] Background-only classification (no fruit pixels) ...")
    background, background_table = background_only_test(records, spec, images, config)
    print(
        f"  accuracy {background['accuracy']:.4f}   macro F1 {background['macro_f1']:.4f}"
        f"   chance {background['chance']:.4f}"
    )
    print(background_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n[2/4] Segmentation behaviour and [3/4] background uniformity ...")
    profile = segmentation_profile(records, images, config)
    summary = summarise_profile(profile, spec)
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n[4/4] Duplicate and near-duplicate detection ...")
    duplicates, duplicate_counts = duplicate_report(records, images)
    print(
        f"  {duplicate_counts['n_pairs']} near-duplicate pair(s), "
        f"{duplicate_counts['n_exact']} exact, "
        f"{duplicate_counts['n_cross_class']} crossing a class boundary"
    )
    if not duplicates.empty:
        print(duplicates.head(10).to_string(index=False))

    passed, findings = verdict(background, summary, duplicate_counts, len(records))

    rule = "-" * 74
    print(f"\n{rule}\nVerdict for {label}\n{rule}")
    for finding in findings:
        print(f"  {finding}")
    outcome = "SUITABLE" if passed else "NOT SUITABLE"
    print(f"\n  Overall: {outcome} as the primary dataset")

    written = [
        save_dataframe(
            background_table, output_dir / f"{label}_background_only.csv", index=False
        ),
        save_dataframe(summary, output_dir / f"{label}_segmentation_profile.csv"),
        save_dataframe(profile, output_dir / f"{label}_per_image.csv", index=False),
        save_dataframe(duplicates, output_dir / f"{label}_duplicates.csv", index=False),
    ]
    print("\nWritten:")
    for path in written:
        print(f"  {path}")
    return passed


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and audit the requested dataset."""
    config = get_config()

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        choices=["primary", "generalisation", "robustness"],
        default="primary",
        help="Audit a dataset already named in config.json (default: primary).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Audit an arbitrary directory instead, for vetting a candidate dataset.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class folder names, required with --root.",
    )
    args = parser.parse_args(argv)

    loader: Callable[[], List[ImageRecord]]
    if args.root is not None:
        if not args.classes:
            parser.error("--classes is required when --root is given")
        names = tuple(name.strip() for name in args.classes.split(",") if name.strip())
        spec = DatasetSpec(classes=names, display_names=names)
        root = (
            args.root
            if args.root.is_absolute()
            else config.paths.project_root / args.root
        )
        label = root.name

        def loader() -> List[ImageRecord]:
            """Load the candidate dataset from the directory given on the command line."""
            return load_dataset(root, spec, label, config.image_extensions)

        print(f"Auditing candidate dataset at {root}")
    else:
        spec = {
            "primary": config.primary,
            "generalisation": config.generalisation,
            "robustness": config.robustness,
        }[args.source]
        chosen = {
            "primary": load_primary,
            "generalisation": load_generalisation,
            "robustness": load_robustness,
        }[args.source]
        label = args.source

        def loader() -> List[ImageRecord]:
            """Load the configured dataset for the requested source."""
            return chosen(config)

        print(f"Auditing the {args.source} dataset")

    try:
        records = loader()
    except DatasetError as error:
        print(f"\nDataset unavailable.\n{error}\n")
        return 1

    return 0 if audit(records, spec, label, config) else 2


if __name__ == "__main__":
    raise SystemExit(main())
