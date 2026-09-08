"""Select the stratified subset of images to be hand-annotated for E3.4.

The blemish-coverage MAE compares what the pipeline measures against what a
human sees. That comparison is only meaningful on images the pipeline was not
fitted on, so the subset is drawn from the **test partition alone**: an image
the classifier trained on would let a tuned coverage threshold flatter itself.

The draw is seeded and the result is written in the dataset's own order, so
every teammate selecting the subset gets the same sixty images without
coordinating. Re-running the script is therefore safe; it either reproduces
the existing selection exactly or refuses to overwrite a different one.

Run from the project root::

    python scripts/select_annotation_subset.py
    python scripts/select_annotation_subset.py --per-class-annotate 20

Writes ``results/annotations/subset.csv`` and, beside it,
``subset_metadata.json`` recording exactly which partition the draw came from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config import Config, get_config  # noqa: E402
from data import ImageRecord, load_primary  # noqa: E402
from harness import set_global_seed, stratified_split  # noqa: E402


def image_id(record: ImageRecord) -> str:
    """Return a filesystem-safe identifier for one record.

    Annotation masks are stored as ``<annotator>/<image_id>.png``, so the
    identifier has to survive being a filename on Windows and stay stable if
    the dataset is restaged. It is derived from the source filename, which is
    already unique within the dataset, rather than from a partition position,
    which would silently change if the split ever moved.
    """
    stem = record.path.stem
    safe = "".join(character if character.isalnum() else "_" for character in stem)
    return "_".join(part for part in safe.split("_") if part)


def draw_subset(
    records: Sequence[ImageRecord],
    per_class: int,
    config: Config,
) -> List[ImageRecord]:
    """Draw ``per_class`` test records from each class, reproducibly.

    The draw is uniform within each class. It is deliberately not filtered by
    segmentation quality: excluding images the segmenter handles badly would
    measure coverage error only where coverage is already easy, which is the
    opposite of what the validation is for.
    """
    rng = np.random.default_rng(config.seed)
    chosen: List[ImageRecord] = []
    for label in range(config.primary.n_classes):
        pool = [record for record in records if record.label == label]
        if len(pool) < per_class:
            raise SystemExit(
                f"class {config.primary.display_names[label]} has only {len(pool)} "
                f"test images, fewer than the {per_class} requested"
            )
        picks = rng.choice(len(pool), size=per_class, replace=False)
        chosen.extend(pool[int(index)] for index in sorted(picks))
    return chosen


def main(argv: Sequence[str] | None = None) -> int:
    """Write the annotation subset and its provenance."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class-annotate", type=int, default=20,
                        help="Images to annotate per class (default 20, so 60 in total).")
    parser.add_argument("--out", type=Path, default=Path("results/annotations"),
                        help="Directory to write subset.csv into.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing subset that does not match.")
    args = parser.parse_args(argv)

    config = get_config()
    set_global_seed(config)

    records = load_primary(config)
    partition = stratified_split(records, config)
    subset = draw_subset(partition.test, args.per_class_annotate, config)

    frame = pd.DataFrame(
        [
            {
                "image_id": image_id(record),
                "path": record.path.relative_to(PROJECT_ROOT).as_posix(),
                "class_folder": record.class_folder,
                "display_name": record.display_name,
                "label": record.label,
                "source": record.source,
                "partition": "test",
            }
            for record in subset
        ]
    )
    duplicates = frame.image_id[frame.image_id.duplicated()].tolist()
    if duplicates:
        raise SystemExit(
            f"image_id is not unique across the subset: {duplicates}. "
            f"Annotation masks would overwrite one another."
        )

    output = PROJECT_ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    target = output / "subset.csv"

    if target.exists() and not args.force:
        existing = pd.read_csv(target)
        if existing.equals(frame):
            print(f"  subset.csv already matches this draw; nothing rewritten "
                  f"({len(frame)} images)")
        else:
            raise SystemExit(
                f"{target} exists and differs from this draw. Annotations may "
                f"already have been painted against it. Re-run with --force "
                f"only if you intend to discard that work."
            )
    else:
        frame.to_csv(target, index=False)

    (output / "subset_metadata.json").write_text(
        json.dumps(
            {
                "seed": config.seed,
                "per_class_annotate": args.per_class_annotate,
                "n_selected": int(len(frame)),
                "drawn_from": "test partition of the full primary dataset",
                "n_primary_records": len(records),
                "n_train_records": len(partition.train),
                "n_test_records": len(partition.test),
                "test_size": config.partition.test_size,
                "classes": list(config.primary.display_names),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 74)
    print("Annotation subset")
    print("=" * 74)
    print(f"  primary records : {len(records)}")
    print(f"  test partition  : {len(partition.test)}")
    print(f"  selected        : {len(frame)} "
          f"({args.per_class_annotate} per class, seed {config.seed})")
    print()
    print(frame.display_name.value_counts().to_string())
    print(f"\n  written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
