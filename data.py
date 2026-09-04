"""Dataset discovery and image loading for FreshSight.

The same loader serves the primary three-class apple set, the binary
generalisation set, and the robustness set: only the root directory and the
class specification differ. File listings are sorted deterministically so that
the train/test partition is reproducible on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from config import Config, DatasetSpec, get_config


@dataclass(frozen=True)
class ImageRecord:
    """A single image on disk, together with its ground-truth label.

    Attributes:
        path: Absolute path to the image file.
        label: Integer class label, fixed by the class ordering in the config.
        class_folder: Name of the folder the image was found in.
        display_name: Human-readable class name used in tables and plots.
        source: Which dataset root the image came from (``primary``,
            ``generalisation`` or ``robustness``).
    """

    path: Path
    label: int
    class_folder: str
    display_name: str
    source: str

    @property
    def name(self) -> str:
        """Filename without its directory."""
        return self.path.name


class DatasetError(RuntimeError):
    """Raised when a dataset root is missing, empty, or badly organised."""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def list_class_files(
    class_dir: Path,
    extensions: Sequence[str],
) -> List[Path]:
    """Return every image file directly inside ``class_dir``, sorted by name.

    Sorting is case-insensitive and applied to the filename only, so the
    ordering does not depend on the operating system's directory iteration
    order. This matters because the train/test split indexes into this list.

    Args:
        class_dir: Directory holding the images of one class.
        extensions: Permitted lower-case file extensions, including the dot.

    Returns:
        A deterministically ordered list of file paths.
    """
    permitted = {ext.lower() for ext in extensions}
    files = [
        entry
        for entry in class_dir.iterdir()
        if entry.is_file() and entry.suffix.lower() in permitted
    ]
    return sorted(files, key=lambda p: p.name.lower())


def load_dataset(
    root: Path,
    spec: DatasetSpec,
    source: str,
    extensions: Sequence[str],
) -> List[ImageRecord]:
    """Scan ``root`` for the class folders named in ``spec``.

    Args:
        root: Directory containing one sub-folder per class.
        spec: Class folder names and their display names.
        source: Label recording which dataset root this is.
        extensions: Permitted image file extensions.

    Returns:
        A list of :class:`ImageRecord`, ordered by class then filename.

    Raises:
        DatasetError: If the root or any class folder is missing, or if any
            class folder contains no readable images.
    """
    if not root.is_dir():
        raise DatasetError(
            f"Dataset root does not exist: {root}\n"
            f"Create it and place one sub-folder per class inside: "
            f"{', '.join(spec.classes)}"
        )

    records: List[ImageRecord] = []
    empty: List[str] = []
    missing: List[str] = []

    for class_folder, display_name in zip(spec.classes, spec.display_names):
        class_dir = root / class_folder
        if not class_dir.is_dir():
            missing.append(class_folder)
            continue
        files = list_class_files(class_dir, extensions)
        if not files:
            empty.append(class_folder)
            continue
        label = spec.label_of(class_folder)
        records.extend(
            ImageRecord(
                path=path.resolve(),
                label=label,
                class_folder=class_folder,
                display_name=display_name,
                source=source,
            )
            for path in files
        )

    if missing:
        raise DatasetError(
            f"Missing class folder(s) under {root}: {', '.join(missing)}"
        )
    if empty:
        raise DatasetError(
            f"Class folder(s) under {root} contain no images with extensions "
            f"{', '.join(extensions)}: {', '.join(empty)}"
        )
    if not records:
        raise DatasetError(f"No images found anywhere under {root}")

    return records


def load_primary(config: Config | None = None) -> List[ImageRecord]:
    """Load the primary three-class apple dataset."""
    cfg = config or get_config()
    return load_dataset(
        cfg.paths.primary_root, cfg.primary, "primary", cfg.image_extensions
    )


def load_generalisation(config: Config | None = None) -> List[ImageRecord]:
    """Load the held-out binary generalisation dataset."""
    cfg = config or get_config()
    return load_dataset(
        cfg.paths.generalisation_root,
        cfg.generalisation,
        "generalisation",
        cfg.image_extensions,
    )


def load_robustness(config: Config | None = None) -> List[ImageRecord]:
    """Load the held-out robustness dataset (varied lighting and backgrounds)."""
    cfg = config or get_config()
    return load_dataset(
        cfg.paths.robustness_root,
        cfg.robustness,
        "robustness",
        cfg.image_extensions,
    )


# --------------------------------------------------------------------------- #
# Reading pixels
# --------------------------------------------------------------------------- #

def read_image(path: Path) -> np.ndarray:
    """Read one image from disk as an 8-bit 3-channel BGR array.

    ``cv2.imread`` cannot open paths containing non-ASCII characters on
    Windows, so the bytes are read by Python and decoded in memory instead.

    Args:
        path: Path to an image file.

    Returns:
        An ``(H, W, 3)`` ``uint8`` array in BGR channel order.

    Raises:
        DatasetError: If the file cannot be read or decoded.
    """
    import cv2  # Imported lazily so that config-only tools need no OpenCV.

    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise DatasetError(f"Could not read {path}: {exc}") from exc

    if buffer.size == 0:
        raise DatasetError(f"File is empty: {path}")

    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise DatasetError(f"Could not decode {path} as an image")

    if image.ndim == 2:  # Defensive: IMREAD_COLOR should already give 3 channels.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def labels_of(records: Iterable[ImageRecord]) -> np.ndarray:
    """Return the integer labels of ``records`` as a 1-D array."""
    return np.asarray([record.label for record in records], dtype=np.int64)


def class_counts(
    records: Sequence[ImageRecord],
    spec: DatasetSpec,
) -> List[Tuple[str, int]]:
    """Count how many records belong to each class, in class order."""
    labels = labels_of(records)
    return [
        (display, int(np.count_nonzero(labels == index)))
        for index, display in enumerate(spec.display_names)
    ]


def summarise(records: Sequence[ImageRecord], spec: DatasetSpec) -> str:
    """Return a printable per-class breakdown of a loaded dataset."""
    counts = class_counts(records, spec)
    width = max(len(name) for name, _ in counts)
    lines = [f"  {name:<{width}} : {count:>5d} images" for name, count in counts]
    lines.append(f"  {'total':<{width}} : {len(records):>5d} images")
    return "\n".join(lines)


if __name__ == "__main__":
    configuration = get_config()
    for loader, dataset_spec, title in (
        (load_primary, configuration.primary, "Primary (three-class)"),
        (load_generalisation, configuration.generalisation, "Generalisation"),
        (load_robustness, configuration.robustness, "Robustness"),
    ):
        print(title)
        try:
            found = loader(configuration)
        except DatasetError as error:
            print(f"  unavailable: {error}")
        else:
            print(summarise(found, dataset_spec))
        print()
