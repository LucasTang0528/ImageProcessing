"""Download the primary apple dataset and stage it under ``data/primary``.

The dataset is the Fruit Ripeness Dataset (Nurdiyansah, 2024), pulled with
``kagglehub``. Kaggle credentials are expected to be in place already at
``~/.kaggle/kaggle.json``; this script never reads, prints, copies or writes
that file, and never echoes a token.

The published folder names are not documented, so the script runs in two
stages and will not guess:

**Stage 1 - survey.** ``python scripts/fetch_dataset.py`` downloads the
dataset into the kagglehub cache and prints its directory tree with per-folder
image counts, then stops. Nothing is copied. Look at the tree and confirm the
class folders are the ones the script proposes to use.

**Stage 2 - copy.** ``python scripts/fetch_dataset.py --copy`` copies the
three apple classes into ``data/primary/<class>/`` under the exact folder
names in ``config.json``. The kagglehub cache is only ever read from, never
moved, modified or deleted.

Copying is idempotent: a file already present at the destination with the same
size is skipped, so re-running is safe and cheap.

**Staging a candidate.** The dataset originally staged here fails
``scripts/audit_dataset.py``, so replacements have to be vetted. Pass
``--slug`` to fetch a different dataset and ``--dest`` to stage it somewhere
other than ``data/primary``, which leaves the current set untouched while the
candidate is audited.

Usage::

    python scripts/fetch_dataset.py                   # survey only
    python scripts/fetch_dataset.py --copy            # survey, then copy
    python scripts/fetch_dataset.py --copy --depth 4  # deeper tree listing
    python scripts/fetch_dataset.py --copy --map UnripeApple=unripe_apple

    # Vet a candidate without disturbing data/primary:
    python scripts/fetch_dataset.py --slug hilton
    python scripts/fetch_dataset.py --slug hilton --copy --dest data/candidate_hilton
    python scripts/audit_dataset.py --root data/candidate_hilton \
        --classes UnripeApple,RipeApple,RottenApple
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import Config, get_config  # noqa: E402

#: Default dataset. Overridable with ``--slug`` so that a candidate can be
#: staged and audited without disturbing whatever currently occupies
#: ``data/primary``.
DATASET_SLUG = "dudinurdiyansah/fruit-ripeness-dataset"

#: Candidate primary datasets, by short name. The default above fails
#: ``scripts/audit_dataset.py``: its background alone classifies at 74.0%
#: against a 33.3% chance level, because unripe apples were scraped from
#: orchard photographs and rotten ones from studio catalogues. These are the
#: replacements to be vetted; none is adopted until it passes the audit.
CANDIDATE_SLUGS: Dict[str, str] = {
    "hilton": "davidhilton/apple-ripeness-levels-image-dataset",
    "leftin": "leftin/fruit-ripeness-unripe-ripe-and-rotten",
    "shawhy": "shawhy/datasets-of-fruit-ripeness-identification",
    "current": DATASET_SLUG,
}

#: Folder-name fragments that identify a split. Only ``train`` is copied: the
#: harness performs its own seeded 80:20 stratified split, and folding a
#: publisher's test set into it would silently change the fixed partition.
TRAIN_SPLIT_NAMES = {"train", "training"}
NON_TRAIN_SPLIT_NAMES = {"test", "testing", "val", "valid", "validation", "eval"}

#: Tokens that must appear in a folder name for it to be an apple class.
APPLE_TOKEN = "apple"

#: Ripeness tokens, mapped to the class folder names used in config.json.
RIPENESS_TOKENS: Dict[str, Tuple[str, ...]] = {
    "UnripeApple": ("unripe", "raw", "green", "immature"),
    "RottenApple": ("rotten", "spoiled", "decayed", "bad", "overripe"),
    "RipeApple": ("ripe", "fresh", "good", "mature"),
}


# --------------------------------------------------------------------------- #
# Survey
# --------------------------------------------------------------------------- #

def count_images(directory: Path, extensions: Sequence[str]) -> int:
    """Count image files directly inside ``directory`` (not recursively)."""
    permitted = {ext.lower() for ext in extensions}
    try:
        return sum(
            1
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() in permitted
        )
    except OSError:
        return 0


def print_tree(
    root: Path,
    extensions: Sequence[str],
    max_depth: int = 3,
) -> None:
    """Print the directory tree under ``root`` with per-folder image counts.

    Args:
        root: Directory to walk.
        extensions: Image extensions counted per folder.
        max_depth: How many levels below ``root`` to descend.
    """

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(
                (entry for entry in directory.iterdir() if entry.is_dir()),
                key=lambda p: p.name.lower(),
            )
        except OSError as error:
            print(f"{prefix}  <unreadable: {error}>")
            return

        for index, child in enumerate(children):
            last = index == len(children) - 1
            elbow = "\\-- " if last else "+-- "
            direct = count_images(child, extensions)
            try:
                subdirs = sum(1 for entry in child.iterdir() if entry.is_dir())
            except OSError:
                subdirs = 0
            annotation = []
            if direct:
                annotation.append(f"{direct} images")
            if subdirs:
                annotation.append(f"{subdirs} subfolder{'s' if subdirs != 1 else ''}")
            suffix = f"  [{', '.join(annotation)}]" if annotation else "  [empty]"
            print(f"{prefix}{elbow}{child.name}{suffix}")
            walk(child, prefix + ("    " if last else "|   "), depth + 1)

    root_images = count_images(root, extensions)
    print(f"{root}{f'  [{root_images} images]' if root_images else ''}")
    walk(root, "", 1)


def leaf_image_folders(root: Path, extensions: Sequence[str]) -> List[Path]:
    """Return every directory under ``root`` that directly contains images."""
    found: List[Path] = []
    for directory in sorted(root.rglob("*")):
        if directory.is_dir() and count_images(directory, extensions) > 0:
            found.append(directory)
    if count_images(root, extensions) > 0:
        found.insert(0, root)
    return found


# --------------------------------------------------------------------------- #
# Class identification
# --------------------------------------------------------------------------- #

def split_of(path: Path, root: Path) -> Optional[str]:
    """Return ``"train"``, ``"other"`` or ``None`` for a folder's split.

    ``None`` means no split directory appears anywhere in the path, i.e. the
    dataset is not pre-split.
    """
    parts = [part.lower() for part in path.relative_to(root).parts]
    for part in parts:
        if part in TRAIN_SPLIT_NAMES:
            return "train"
        if part in NON_TRAIN_SPLIT_NAMES:
            return "other"
    return None


def classify_folder(path: Path) -> Optional[str]:
    """Map an image folder to one of the three config class names.

    The folder's own name is inspected first, then its parent, so both
    ``.../train/rottenapples`` and ``.../rotten/apple`` are recognised.
    Non-apple fruit returns ``None`` and is ignored.

    Args:
        path: A directory that directly contains images.

    Returns:
        A class folder name from ``config.json``, or ``None``.
    """
    own = path.name.lower().replace("_", "").replace("-", "").replace(" ", "")
    parent = path.parent.name.lower().replace("_", "").replace("-", "").replace(" ", "")
    haystack = f"{parent}/{own}"

    if APPLE_TOKEN not in haystack:
        return None

    # "unripe" contains "ripe", and "overripe" should not match "ripe" either,
    # so the specific classes are tested before the general one. Dict order in
    # RIPENESS_TOKENS is deliberate: Unripe, Rotten, then Ripe.
    for class_name, tokens in RIPENESS_TOKENS.items():
        if any(token in haystack for token in tokens):
            return class_name
    return None


def folder_signature(folder: Path, extensions: Sequence[str]) -> str:
    """Return a digest of a folder's image filenames and sizes.

    Two folders holding the same files produce the same digest. Only names and
    sizes are read, never contents, so the check costs a stat per file.
    """
    permitted = {ext.lower() for ext in extensions}
    entries = sorted(
        (entry.name, entry.stat().st_size)
        for entry in folder.iterdir()
        if entry.is_file() and entry.suffix.lower() in permitted
    )
    digest = hashlib.sha1()
    for name, size in entries:
        digest.update(f"{name}:{size}\n".encode("utf-8"))
    return digest.hexdigest()


def drop_duplicate_folders(
    folders: Sequence[Path],
    extensions: Sequence[str],
) -> Tuple[List[Path], List[Path]]:
    """Remove source folders that repeat a folder already in the list.

    Published archives are often repacked with their whole tree nested inside
    itself, so that ``dataset/train/rottenapples`` and
    ``dataset/dataset/train/rottenapples`` are the same 2342 files. Copying
    both would stage every image twice under different names, and the
    duplicates would then be split across the train and test partitions,
    putting a byte-identical copy of a test image into training.

    The shallowest path wins, being the one the publisher most likely meant.

    Returns:
        The folders to copy, and the duplicates that were dropped.
    """
    keep: List[Path] = []
    dropped: List[Path] = []
    seen: Dict[str, Path] = {}

    for folder in sorted(folders, key=lambda p: (len(p.parts), str(p).lower())):
        signature = folder_signature(folder, extensions)
        if signature in seen:
            dropped.append(folder)
            continue
        seen[signature] = folder
        keep.append(folder)

    return keep, dropped


def find_apple_folders(
    root: Path,
    extensions: Sequence[str],
    overrides: Dict[str, str],
) -> Tuple[Dict[str, List[Path]], List[Path], List[Path]]:
    """Locate the apple class folders in the downloaded dataset.

    Args:
        root: The kagglehub cache path.
        extensions: Image extensions to count.
        overrides: Explicit ``class name -> folder name`` mappings from
            ``--map``, which take precedence over the automatic matching.

    Returns:
        A tuple of ``(matched, skipped_non_train, ignored)`` where ``matched``
        maps each config class name to the source folders found for it.
        Folders repeating one already matched for the same class are dropped;
        see :func:`drop_duplicate_folders`.
    """
    matched: Dict[str, List[Path]] = defaultdict(list)
    skipped_non_train: List[Path] = []
    ignored: List[Path] = []

    reverse_overrides = {
        folder.lower(): class_name for class_name, folder in overrides.items()
    }

    for folder in leaf_image_folders(root, extensions):
        split = split_of(folder, root)
        if split == "other":
            skipped_non_train.append(folder)
            continue

        class_name = reverse_overrides.get(folder.name.lower()) or classify_folder(folder)
        if class_name is None:
            ignored.append(folder)
            continue
        matched[class_name].append(folder)

    deduplicated: Dict[str, List[Path]] = {}
    for class_name, folders in matched.items():
        keep, dropped = drop_duplicate_folders(folders, extensions)
        deduplicated[class_name] = keep
        for folder in dropped:
            print(
                f"  note: ignoring {folder.relative_to(root)} - it repeats "
                f"{keep[0].relative_to(root)} file for file"
            )

    return deduplicated, skipped_non_train, ignored


# --------------------------------------------------------------------------- #
# Copying
# --------------------------------------------------------------------------- #

class CopyReport:
    """Tallies of one staging run."""

    def __init__(self) -> None:
        self.copied: Dict[str, int] = defaultdict(int)
        self.skipped: Dict[str, int] = defaultdict(int)
        self.renamed: List[Tuple[str, str]] = []
        self.duplicates: Dict[str, List[str]] = defaultdict(list)
        self.undecodable: List[Path] = []


def verify_decodes(path: Path) -> bool:
    """Return True when OpenCV can decode ``path`` as an image.

    Read through ``np.fromfile`` so that non-ASCII paths work on Windows,
    matching how :func:`data.read_image` loads images in the harness.
    """
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return False
    if buffer.size == 0:
        return False
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR) is not None


def copy_class(
    class_name: str,
    sources: Sequence[Path],
    destination: Path,
    extensions: Sequence[str],
    report: CopyReport,
) -> None:
    """Copy one class's images into ``destination``.

    Files already present with an identical size are skipped, which makes
    re-running the script cheap and idempotent. When two source folders
    contribute the same filename, the second is given a suffix rather than
    overwriting the first, and the collision is recorded.

    The suffix is derived from the source folder name - never from a counter
    that probes the destination - so a given source file always resolves to the
    same destination name. A counter would allocate a fresh ``__1``, ``__2``,
    ``__3`` on every run, quietly multiplying the dataset each time the script
    was re-run.
    """
    destination.mkdir(parents=True, exist_ok=True)
    permitted = {ext.lower() for ext in extensions}
    seen: Dict[str, Path] = {}

    for source_dir in sources:
        for source in sorted(source_dir.iterdir(), key=lambda p: p.name.lower()):
            if not source.is_file() or source.suffix.lower() not in permitted:
                continue

            name = source.name
            if name in seen:
                report.duplicates[class_name].append(name)
                name = f"{source_dir.name}__{source.name}"
                if name in seen:
                    # Two identically named source folders: disambiguate with a
                    # stable digest of the full source path.
                    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
                    name = f"{source.stem}__{digest}{source.suffix}"
                report.renamed.append((source.name, name))

            target = destination / name
            seen[name] = source

            if target.exists() and target.stat().st_size == source.stat().st_size:
                report.skipped[class_name] += 1
                continue

            shutil.copy2(source, target)
            report.copied[class_name] += 1


def verify_destination(
    destination: Path,
    extensions: Sequence[str],
    report: CopyReport,
) -> int:
    """Check every staged file decodes, recording those that do not."""
    permitted = {ext.lower() for ext in extensions}
    total = 0
    for path in sorted(destination.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in permitted:
            continue
        total += 1
        if not verify_decodes(path):
            report.undecodable.append(path)
    return total


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_map(values: Iterable[str]) -> Dict[str, str]:
    """Parse ``--map ClassName=source_folder`` arguments."""
    mapping: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--map expects ClassName=folder_name, got {value!r}")
        class_name, folder = value.split("=", 1)
        mapping[class_name.strip()] = folder.strip()
    return mapping


def download(slug: str = DATASET_SLUG) -> Path:
    """Download the dataset and return the kagglehub cache path.

    Credentials are handled entirely by ``kagglehub`` from
    ``~/.kaggle/kaggle.json``. Nothing here touches or displays them.

    Args:
        slug: The Kaggle ``owner/dataset`` identifier to fetch.
    """
    try:
        import kagglehub
    except ImportError as error:
        raise SystemExit(
            "kagglehub is not installed. Run: pip install -r requirements.txt"
        ) from error

    print(f"Downloading {slug} via kagglehub ...")
    try:
        cache_path = Path(kagglehub.dataset_download(slug))
    except Exception as error:  # noqa: BLE001 - surface any kagglehub failure plainly
        raise SystemExit(
            f"kagglehub could not download the dataset: {error}\n\n"
            "Check that ~/.kaggle/kaggle.json exists and contains a valid API "
            "token, and that you have accepted the dataset's terms on Kaggle."
        ) from error

    print(f"Cache path: {cache_path}\n")
    return cache_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Survey the dataset, and copy the apple classes when ``--copy`` is given."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the apple classes into data/primary. Without this the "
        "script only surveys the download and stops.",
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="Tree listing depth (default: 3)."
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="Class=folder",
        help="Force a class to a source folder name, e.g. "
        "--map UnripeApple=unripe_apple. Repeatable.",
    )
    parser.add_argument(
        "--slug",
        default=None,
        help="Kaggle dataset to fetch: either an owner/dataset identifier or "
        f"one of the shorthands {', '.join(sorted(CANDIDATE_SLUGS))} "
        f"(default: {DATASET_SLUG}).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Directory to stage into, relative to the project root "
        "(default: the configured data/primary). Use a separate directory to "
        "vet a candidate with audit_dataset.py before adopting it.",
    )
    args = parser.parse_args(argv)

    config: Config = get_config()
    extensions = config.image_extensions
    overrides = parse_map(args.map)

    slug = CANDIDATE_SLUGS.get(args.slug, args.slug) if args.slug else DATASET_SLUG
    if args.dest is None:
        destination_root = config.paths.primary_root
    elif args.dest.is_absolute():
        destination_root = args.dest
    else:
        destination_root = config.paths.project_root / args.dest

    cache_path = download(slug)

    print("=" * 78)
    print("DOWNLOADED DATASET STRUCTURE")
    print("=" * 78)
    print_tree(cache_path, extensions, max_depth=args.depth)
    print()

    matched, skipped_non_train, ignored = find_apple_folders(cache_path, extensions, overrides)

    print("=" * 78)
    print("APPLE CLASSES IDENTIFIED")
    print("=" * 78)
    for class_name in config.primary.classes:
        folders = matched.get(class_name, [])
        if not folders:
            print(f"  {class_name:<14} -> NOT FOUND")
            continue
        for folder in folders:
            count = count_images(folder, extensions)
            print(f"  {class_name:<14} <- {folder.relative_to(cache_path)}  ({count} images)")

    if skipped_non_train:
        print(f"\n  Skipped {len(skipped_non_train)} folder(s) in a test/validation split:")
        for folder in skipped_non_train[:10]:
            print(f"    {folder.relative_to(cache_path)}")
        if len(skipped_non_train) > 10:
            print(f"    ... and {len(skipped_non_train) - 10} more")
        print("  The harness performs its own seeded 80:20 split, so only 'train' is used.")

    if ignored:
        print(f"\n  Ignored {len(ignored)} non-apple or unrecognised folder(s):")
        for folder in ignored[:10]:
            print(f"    {folder.relative_to(cache_path)}")
        if len(ignored) > 10:
            print(f"    ... and {len(ignored) - 10} more")

    missing = [name for name in config.primary.classes if not matched.get(name)]

    if not args.copy:
        print()
        print("=" * 78)
        print("SURVEY ONLY - nothing has been copied.")
        print("=" * 78)
        if missing:
            print(
                f"\n{len(missing)} class(es) were not matched automatically: "
                f"{', '.join(missing)}.\nRe-run with an explicit mapping, for example:\n"
                f"  python scripts/fetch_dataset.py --copy --map {missing[0]}=<folder name>"
            )
        else:
            flags = f" --slug {args.slug}" if args.slug else ""
            flags += f" --dest {args.dest}" if args.dest else ""
            print(
                "\nIf the mapping above is correct, stage the files with:\n"
                f"  python scripts/fetch_dataset.py --copy{flags}"
            )
        return 0

    if missing:
        print(
            f"\nRefusing to copy: no source folder found for {', '.join(missing)}.\n"
            f"Use --map ClassName=folder to point at it explicitly."
        )
        return 1

    print()
    print("=" * 78)
    print(f"COPYING INTO {destination_root}")
    print("=" * 78)

    report = CopyReport()
    for class_name in config.primary.classes:
        destination = destination_root / class_name
        copy_class(class_name, matched[class_name], destination, extensions, report)
        print(
            f"  {class_name:<14} {report.copied[class_name]:>5} copied, "
            f"{report.skipped[class_name]:>5} already present"
        )

    print()
    print("=" * 78)
    print("VERIFICATION")
    print("=" * 78)
    grand_total = 0
    for class_name in config.primary.classes:
        destination = destination_root / class_name
        total = verify_destination(destination, extensions, report)
        grand_total += total
        print(f"  {class_name:<14} {total:>5} files staged")
    print(f"  {'TOTAL':<14} {grand_total:>5}")

    if report.duplicates:
        print("\n  Duplicate filenames across source folders (kept, suffixed):")
        for class_name, names in report.duplicates.items():
            print(f"    {class_name}: {len(names)} collision(s), e.g. {names[:3]}")
    else:
        print("\n  No duplicate filenames.")

    if report.undecodable:
        print(f"\n  {len(report.undecodable)} file(s) FAILED to decode as an image:")
        for path in report.undecodable:
            print(f"    {path}")
        print(
            "\n  Delete or replace these before running the harness - "
            "data.read_image would raise on them."
        )
        return 1

    print("\n  All staged files decode correctly.")

    # The audit comes before the visual check, and before any feature work: a
    # dataset whose imaging style predicts the label produces flattering
    # numbers that no later stage can detect or correct.
    if destination_root == config.paths.primary_root:
        print("\nNext: python scripts/audit_dataset.py")
    else:
        relative = destination_root.relative_to(config.paths.project_root)
        print(
            f"\nNext, vet this candidate before adopting it:\n"
            f"  python scripts/audit_dataset.py --root {relative.as_posix()} "
            f"--classes {','.join(config.primary.classes)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
