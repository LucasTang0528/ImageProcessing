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

Images are drawn from the **training partition only**, with the same seed and
the same split the experiments use, so annotating cannot leak the test set.

Run from the project root::

    python scripts/annotate_blemishes.py --count 30     # paint 30 images
    python scripts/annotate_blemishes.py --list         # what is done so far

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
from typing import List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from config import Config, get_config  # noqa: E402
from data import ImageRecord, load_primary  # noqa: E402
from harness import prepare_sample, set_global_seed, stratified_split  # noqa: E402

#: Where painted masks and their index live.
ANNOTATION_ROOT = PROJECT_ROOT / "data" / "annotations" / "blemish"
INDEX_NAME = "index.csv"

WINDOW = "FreshSight - paint blemishes"
BRUSH_MIN, BRUSH_MAX = 1, 30


def annotation_path(root: Path, record: ImageRecord) -> Path:
    """Where one record's painted mask is stored."""
    return root / record.class_folder / f"{record.path.stem}.png"


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


def annotate(records: Sequence[ImageRecord], root: Path, config: Config) -> int:
    """Run the painting loop. Returns the number of masks written."""
    root.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(WINDOW)
    written = 0
    index = 0

    while 0 <= index < len(records):
        record = records[index]
        sample = prepare_sample(record, config=config, record_index=index)
        image, fruit = sample.image, sample.mask

        canvas = Canvas(image.shape[:2])
        destination = annotation_path(root, record)
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
                f"{index + 1}/{len(records)}  {record.class_folder}/{record.path.name}"
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


def rebuild_index(records: Sequence[ImageRecord], root: Path, config: Config) -> pd.DataFrame:
    """Summarise every painted mask, and the blemish ratio it implies."""
    rows = []
    for record in records:
        painted_path = annotation_path(root, record)
        if not painted_path.exists():
            continue
        painted = cv2.imread(str(painted_path), cv2.IMREAD_GRAYSCALE)
        if painted is None:
            continue
        sample = prepare_sample(record, config=config)
        fruit_px = max(int(np.count_nonzero(sample.mask)), 1)
        blemish_px = int(np.count_nonzero(painted > 127))
        rows.append(
            {
                "path": str(record.path),
                "class_folder": record.class_folder,
                "label": record.label,
                "annotation": str(painted_path),
                "fruit_px": fruit_px,
                "blemish_px": blemish_px,
                "blemish_ratio_pct": 100.0 * blemish_px / fruit_px,
            }
        )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=30,
                        help="How many images to offer, class balanced.")
    parser.add_argument("--root", type=Path, default=ANNOTATION_ROOT,
                        help="Where painted masks are stored.")
    parser.add_argument("--list", action="store_true",
                        help="Report what is annotated so far and exit.")
    args = parser.parse_args(argv)

    set_global_seed(config)
    records = sample_records(load_primary(config), args.count, config)

    if not args.list:
        print(f"Annotating {len(records)} images from the training partition.")
        print("The test split is not sampled, so annotating cannot leak it.\n")
        try:
            written = annotate(records, args.root, config)
        except cv2.error as error:
            print(
                "OpenCV could not open a window, so annotation needs a desktop "
                f"session:\n  {error}",
                file=sys.stderr,
            )
            return 1
        print(f"\nWrote {written} mask(s) under {args.root}")

    index = rebuild_index(records, args.root, config)
    if index.empty:
        print("No annotations found yet.")
        return 0

    args.root.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.root / INDEX_NAME, index=False)
    print(f"\n{len(index)} annotated image(s), index at {args.root / INDEX_NAME}")
    print(
        index.groupby("class_folder")["blemish_ratio_pct"]
        .agg(["count", "mean", "max"])
        .round(3)
        .to_string()
    )
    print("\nRe-run scripts/run_t3_experiments.py to fill the E3.4 error column.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
