"""Paint ground-truth blemish masks, for E3.4.

E3.4 compares three blemish segmentation methods - black top hat with Otsu, a
fixed intensity threshold, and the dropped hue-deviation baseline - and the
specification scores them on mean absolute blemish-ratio error against manually
annotated masks. Classification accuracy cannot stand in for that: a method can
mislabel which pixels are blemished and still feed the classifier something
separable, so accuracy would answer a different question from the one asked.

This tool produces those annotations.

Annotation happens in the **preprocessed 224 x 224 frame**, the same frame the
descriptor works in, so a painted mask lines up pixel for pixel with the
``blemish_mask`` the extractor returns and no resampling sits between the two.

Images come from ``results/annotations/subset.csv``, written by
``scripts/select_annotation_subset.py``, which draws them from the **test
partition**: the coverage measurement has to be validated on images the
pipeline was not fitted on. That is safe only while nothing is selected or
tuned on the resulting error. If a blemish method is ever chosen by its MAE,
these annotations become a test-set leak and the subset must move to the
training partition instead.

Two annotators paint the same images independently, and a merge step keeps
only the pixels both marked. A single annotator's mask is one opinion about a
boundary that is genuinely ambiguous - where exactly a bruise stops - and an
error measured against one opinion cannot be told apart from that opinion's
own noise. The per-image Jaccard index between the two says how much of the
measured error is really disagreement.

Run from the project root::

    python scripts/select_annotation_subset.py                    # choose the 60
    python scripts/annotate_blemishes.py --annotator alice        # alice paints
    python scripts/annotate_blemishes.py --annotator bob          # bob paints
    python scripts/annotate_blemishes.py --merge                  # reference + agreement
    python scripts/annotate_blemishes.py --list                   # progress

Controls
--------
==================== ======================================================
left mouse drag      paint blemish
right mouse drag     erase
``[`` / ``]``        smaller / larger brush
``c``                clear this mask
``u``                undo the last stroke
``space`` / ``n``    save and go to the next image
``b``                save and go back one image
``h``                toggle the overlay, to see the fruit underneath
``q`` / ``Esc``      save and quit
==================== ======================================================

A saved mask of all zeros is meaningful: it records a fruit judged to have no
blemishes at all, which is a real case in the Unripe class and is exactly the
case a detector most easily gets wrong. Skipping an image records nothing;
saving an empty one records a judgement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from config import Config, get_config  # noqa: E402
from data import ImageRecord, load_primary  # noqa: E402
from harness import prepare_sample, set_global_seed, stratified_split  # noqa: E402

#: Where the subset, the painted masks, the merged reference and the
#: agreement report all live. Kept under results/ rather than data/ because
#: these are produced by the study, not inputs to it.
ANNOTATION_ROOT = PROJECT_ROOT / "results" / "annotations"
INDEX_NAME = "index.csv"
SUBSET_NAME = "subset.csv"
REFERENCE_DIR = "reference"

#: Directory names that hold merged or derived output rather than one
#: annotator's work, and so must never be treated as an annotator.
RESERVED_DIRS = frozenset({REFERENCE_DIR})

WINDOW = "FreshSight - paint blemishes"
BRUSH_MIN, BRUSH_MAX = 1, 30


def annotation_path(root: Path, annotator: str, image_id: str) -> Path:
    """Where one annotator's mask for one image is stored."""
    return root / annotator / f"{image_id}.png"


def load_subset(root: Path, config: Config) -> pd.DataFrame:
    """Read the agreed annotation subset, refusing to invent one.

    The subset is a shared artefact: two annotators must paint the same
    images or their masks cannot be compared. Selecting images here, as this
    tool used to, would let each annotator silently work on a different draw.
    """
    subset = root / SUBSET_NAME
    if not subset.exists():
        raise SystemExit(
            f"no {subset}. Run scripts/select_annotation_subset.py first so that "
            f"every annotator paints the same images."
        )
    frame = pd.read_csv(subset)
    missing = [p for p in frame.path if not (PROJECT_ROOT / p).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} image(s) in the subset are not on disk, "
            f"starting with {missing[0]}"
        )
    return frame


def subset_records(frame: pd.DataFrame, config: Config) -> List[ImageRecord]:
    """Rebuild ImageRecords from the subset index, in its stored order."""
    display = list(config.primary.display_names)
    return [
        ImageRecord(
            path=PROJECT_ROOT / row.path,
            label=int(row.label),
            class_folder=str(row.class_folder),
            display_name=display[int(row.label)],
            source=str(row.source),
        )
        for row in frame.itertuples()
    ]


def sample_records(
    records: Sequence[ImageRecord],
    count: int,
    config: Config,
) -> List[ImageRecord]:
    """Draw a class-balanced sample from the training partition, reproducibly.

    Balanced because a method that only fails on rotten fruit would otherwise
    be scored mostly on the classes it handles well.
    """
    partition = stratified_split(list(records), config)
    training = partition.train
    rng = np.random.default_rng(config.seed)
    per_class = max(1, count // config.primary.n_classes)

    chosen: List[int] = []
    for label in range(config.primary.n_classes):
        pool = [i for i, record in enumerate(training) if record.label == label]
        take = min(per_class, len(pool))
        chosen.extend(int(pool[i]) for i in rng.choice(len(pool), size=take, replace=False))
    return [training[i] for i in sorted(chosen)]


class Canvas:
    """Mutable painting state for one image."""

    def __init__(self, shape: Tuple[int, int]) -> None:
        self.mask = np.zeros(shape, dtype=np.uint8)
        self.history: List[np.ndarray] = []
        self.brush = 4
        self.painting = 0  # 0 idle, 1 painting, 2 erasing

    def begin(self, erasing: bool) -> None:
        self.history.append(self.mask.copy())
        if len(self.history) > 20:
            self.history.pop(0)
        self.painting = 2 if erasing else 1

    def daub(self, x: int, y: int) -> None:
        if self.painting:
            value = 0 if self.painting == 2 else 255
            cv2.circle(self.mask, (x, y), self.brush, value, thickness=-1)

    def end(self) -> None:
        self.painting = 0

    def undo(self) -> None:
        if self.history:
            self.mask = self.history.pop()

    def clear(self) -> None:
        self.history.append(self.mask.copy())
        self.mask[:] = 0


def render(image: np.ndarray, canvas: Canvas, fruit: np.ndarray, hide: bool, caption: str):
    """Compose the annotation view: fruit outline, painted mask, status line."""
    view = image.copy()
    contours, _ = cv2.findContours(fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, (0, 200, 255), 1)

    if not hide:
        overlay = view.copy()
        overlay[canvas.mask > 0] = (0, 0, 255)
        view = cv2.addWeighted(overlay, 0.45, view, 0.55, 0)

    view = cv2.resize(view, (560, 560), interpolation=cv2.INTER_NEAREST)
    banner = np.zeros((46, 560, 3), dtype=np.uint8)
    cv2.putText(banner, caption, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(
        banner,
        f"brush {canvas.brush}  [ ] size   c clear   u undo   h hide   "
        f"space next   b back   q quit",
        (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1,
    )
    return np.vstack([view, banner])


def annotate(
    records: Sequence[ImageRecord],
    image_ids: Sequence[str],
    root: Path,
    config: Config,
    annotator: str,
    overwrite: bool = False,
) -> int:
    """Run the painting loop for one annotator. Returns masks written.

    An image this annotator has already painted is skipped unless
    ``overwrite`` is set. Silently reopening finished work invites painting
    over it by accident, and one annotator's mask is not reproducible: unlike
    every other artefact here it cannot be regenerated if lost.
    """
    root.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(WINDOW)
    written = 0
    index = 0

    while 0 <= index < len(records):
        record = records[index]
        destination = annotation_path(root, annotator, image_ids[index])
        if destination.exists() and not overwrite:
            print(f"  skipping {image_ids[index]}: already painted by {annotator} "
                  f"(--overwrite to replace)")
            index += 1
            continue

        sample = prepare_sample(record, config=config, record_index=index)
        image, fruit = sample.image, sample.mask

        canvas = Canvas(image.shape[:2])
        if destination.exists():
            existing = cv2.imread(str(destination), cv2.IMREAD_GRAYSCALE)
            if existing is not None and existing.shape == canvas.mask.shape:
                canvas.mask = ((existing > 127) * 255).astype(np.uint8)

        state = {"hide": False}

        def on_mouse(event, x, y, flags, _param, canvas=canvas):
            x = int(x * image.shape[1] / 560)
            y = int(y * image.shape[0] / 560)
            if event == cv2.EVENT_LBUTTONDOWN:
                canvas.begin(erasing=False)
                canvas.daub(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                canvas.begin(erasing=True)
                canvas.daub(x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                canvas.daub(x, y)
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                canvas.end()

        cv2.setMouseCallback(WINDOW, on_mouse)
        decision = None

        while decision is None:
            painted = int(np.count_nonzero(canvas.mask))
            fruit_px = max(int(np.count_nonzero(fruit)), 1)
            caption = (
                f"[{annotator}] {index + 1}/{len(records)}  {record.path.name}"
                f"   ratio {100.0 * painted / fruit_px:5.2f}%"
            )
            cv2.imshow(WINDOW, render(image, canvas, fruit, state["hide"], caption))
            key = cv2.waitKey(20) & 0xFF

            if key in (ord(" "), ord("n")):
                decision = "next"
            elif key == ord("b"):
                decision = "back"
            elif key in (ord("q"), 27):
                decision = "quit"
            elif key == ord("c"):
                canvas.clear()
            elif key == ord("u"):
                canvas.undo()
            elif key == ord("h"):
                state["hide"] = not state["hide"]
            elif key == ord("["):
                canvas.brush = max(BRUSH_MIN, canvas.brush - 1)
            elif key == ord("]"):
                canvas.brush = min(BRUSH_MAX, canvas.brush + 1)

        # A painted mask is only meaningful inside the fruit.
        canvas.mask[fruit == 0] = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), canvas.mask)
        written += 1

        if decision == "quit":
            break
        index += 1 if decision == "next" else -1
        index = max(index, 0)

    cv2.destroyAllWindows()
    return written


def discover_annotators(root: Path) -> List[str]:
    """Return the annotator directories present, excluding derived output."""
    if not root.exists():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and entry.name not in RESERVED_DIRS
    )


def _read_mask(path: Path) -> Optional[np.ndarray]:
    """Read one painted mask as a boolean array, or None if absent."""
    if not path.exists():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if raw is None else raw > 127


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over union of two boolean masks.

    Two annotators who both, correctly, paint nothing agree perfectly, so the
    empty-empty case is 1.0 rather than undefined. Scoring it 0 would punish
    the clean fruit that the Unripe class is full of.
    """
    union = int(np.count_nonzero(first | second))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(first & second) / union)


def merge_annotations(
    frame: pd.DataFrame,
    root: Path,
    annotators: Sequence[str],
) -> pd.DataFrame:
    """Write consensus reference masks and report inter-annotator agreement.

    A pixel enters the reference only where **every** annotator marked it.
    Intersection rather than union is the conservative choice: it yields the
    blemish extent nobody disputes, so the reference ratio is a lower bound
    and the pipeline is never charged for a pixel one annotator called peel.

    Agreement is reported alongside, because an MAE measured against masks the
    annotators themselves disagree about cannot be read without knowing how
    much they disagreed.
    """
    if len(annotators) < 2:
        raise SystemExit(
            f"merging needs at least two annotators, found {annotators or 'none'}. "
            f"Each paints with --annotator <name>."
        )

    reference_dir = root / REFERENCE_DIR
    reference_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []

    for row in frame.itertuples():
        masks = {a: _read_mask(annotation_path(root, a, row.image_id)) for a in annotators}
        present = {a: m for a, m in masks.items() if m is not None}
        if len(present) < 2:
            rows.append(
                {
                    "image_id": row.image_id,
                    "display_name": row.display_name,
                    "n_annotators": len(present),
                    "jaccard": float("nan"),
                    "reference_px": 0,
                    "status": "incomplete",
                }
            )
            continue

        shapes = {m.shape for m in present.values()}
        if len(shapes) != 1:
            raise SystemExit(
                f"{row.image_id}: annotators painted different frame sizes {shapes}"
            )

        consensus = np.logical_and.reduce(list(present.values()))
        cv2.imwrite(str(reference_dir / f"{row.image_id}.png"),
                    (consensus * 255).astype(np.uint8))

        pairs = [
            jaccard(present[a], present[b])
            for i, a in enumerate(sorted(present))
            for b in sorted(present)[i + 1:]
        ]
        rows.append(
            {
                "image_id": row.image_id,
                "display_name": row.display_name,
                "n_annotators": len(present),
                "jaccard": float(np.mean(pairs)),
                "reference_px": int(np.count_nonzero(consensus)),
                "status": "merged",
            }
        )

    agreement = pd.DataFrame(rows)
    agreement.to_csv(root / "agreement.csv", index=False)
    return agreement


def rebuild_index(
    records: Sequence[ImageRecord],
    image_ids: Sequence[str],
    root: Path,
    config: Config,
) -> pd.DataFrame:
    """Index the merged reference masks and their blemish ratios.

    The ratio uses the full fruit mask as denominator, matching the formula
    T3 reports, so the two numbers are directly comparable.
    """
    reference_dir = root / REFERENCE_DIR
    rows: List[dict] = []
    for index, (record, image_id) in enumerate(zip(records, image_ids)):
        painted = _read_mask(reference_dir / f"{image_id}.png")
        if painted is None:
            continue
        sample = prepare_sample(record, config=config, record_index=index)
        fruit_px = max(int(np.count_nonzero(sample.mask)), 1)
        blemish_px = int(np.count_nonzero(painted))
        rows.append(
            {
                "image_id": image_id,
                "path": str(record.path),
                "class_folder": record.class_folder,
                "display_name": record.display_name,
                "label": record.label,
                "fruit_px": fruit_px,
                "blemish_px": blemish_px,
                "blemish_ratio_pct": 100.0 * blemish_px / fruit_px,
            }
        )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotator",
                        help="Who is painting. Masks go to <root>/<annotator>/.")
    parser.add_argument("--root", type=Path, default=ANNOTATION_ROOT,
                        help="Where the subset, masks and reference live.")
    parser.add_argument("--merge", action="store_true",
                        help="Build consensus reference masks and report agreement.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Repaint images this annotator has already done.")
    parser.add_argument("--list", action="store_true",
                        help="Report progress and exit.")
    args = parser.parse_args(argv)

    if args.annotator and args.annotator in RESERVED_DIRS:
        parser.error(f"{args.annotator!r} is reserved for merged output")

    set_global_seed(config)
    frame = load_subset(args.root, config)
    records = subset_records(frame, config)
    image_ids = list(frame.image_id)
    annotators = discover_annotators(args.root)

    if args.list:
        print(f"Subset: {len(records)} images from the test partition.")
        if not annotators:
            print("No annotator has painted anything yet.")
            return 0
        for name in annotators:
            done = sum(annotation_path(args.root, name, i).exists() for i in image_ids)
            print(f"  {name:16s} {done:3d}/{len(image_ids)}")
        reference = sum((args.root / REFERENCE_DIR / f"{i}.png").exists() for i in image_ids)
        print(f"  {'reference':16s} {reference:3d}/{len(image_ids)}")
        return 0

    if args.merge:
        agreement = merge_annotations(frame, args.root, annotators)
        merged = agreement[agreement.status == "merged"]
        print(f"Merged {len(merged)} of {len(agreement)} images "
              f"from annotators: {', '.join(annotators)}")
        if not merged.empty:
            print(f"  mean Jaccard agreement : {merged.jaccard.mean():.3f}")
            print(f"  worst image            : {merged.jaccard.min():.3f} "
                  f"({merged.loc[merged.jaccard.idxmin(), 'image_id']})")
            print()
            print(merged.groupby("display_name")["jaccard"].agg(["count", "mean"]).round(3).to_string())
        incomplete = agreement[agreement.status != "merged"]
        if not incomplete.empty:
            print(f"\n  {len(incomplete)} image(s) still need a second annotator.")
        print(f"\n  agreement written to {args.root / 'agreement.csv'}")
    elif args.annotator:
        print(f"Annotating {len(records)} images as {args.annotator!r}.")
        print("These come from the TEST partition. Nothing may be tuned on the")
        print("resulting error, or these annotations become a test-set leak.\n")
        try:
            written = annotate(records, image_ids, args.root, config,
                               args.annotator, args.overwrite)
        except cv2.error as error:
            print(
                "OpenCV could not open a window, so annotation needs a desktop "
                f"session:\n  {error}",
                file=sys.stderr,
            )
            return 1
        print(f"\nWrote {written} mask(s) under {args.root / args.annotator}")
    else:
        parser.error("give --annotator to paint, --merge to combine, or --list")

    index = rebuild_index(records, image_ids, args.root, config)
    if index.empty:
        print("\nNo reference masks yet; run --merge once two annotators have painted.")
        return 0

    index.to_csv(args.root / INDEX_NAME, index=False)
    print(f"\n{len(index)} reference mask(s), index at {args.root / INDEX_NAME}")
    print(
        index.groupby("display_name")["blemish_ratio_pct"]
        .agg(["count", "mean", "max"]).round(3).to_string()
    )
    print("\nRe-run scripts/run_t3_experiments.py to fill the E3.4 error column.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
