"""Configuration loading for the FreshSight comparative study.

Every tunable value in the experiment lives in ``config.json`` at the project
root. Nothing in this codebase may hardcode an absolute path or an
experimental parameter: modules import :func:`get_config` and read what they
need from the returned :class:`Config` object.

The configuration is deliberately immutable once loaded. The three feature
extraction techniques are compared under a shared harness, and the comparison
is only valid if every stage other than feature extraction is identical, so a
technique must never be able to mutate the settings the harness runs under.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

CONFIG_FILENAME = "config.json"
ENV_CONFIG_PATH = "FRESHSIGHT_CONFIG"


# --------------------------------------------------------------------------- #
# Section dataclasses
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Paths:
    """Resolved absolute paths for every directory the study reads or writes."""

    project_root: Path
    primary_root: Path
    generalisation_root: Path
    robustness_root: Path
    results_root: Path
    cache_root: Path

    def results_subdir(self, name: str) -> Path:
        """Return ``results/<name>``, creating it if it does not yet exist."""
        target = self.results_root / name
        target.mkdir(parents=True, exist_ok=True)
        return target


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset root's class folder names and their human-readable labels."""

    classes: Tuple[str, ...]
    display_names: Tuple[str, ...]

    def label_of(self, class_folder: str) -> int:
        """Return the fixed integer label for a class folder name."""
        return self.classes.index(class_folder)

    @property
    def n_classes(self) -> int:
        """Number of classes in this dataset."""
        return len(self.classes)


@dataclass(frozen=True)
class PreprocessConfig:
    """Shared preprocessing parameters. Identical for all three techniques."""

    resize: Tuple[int, int]
    gaussian_kernel: Tuple[int, int]
    gaussian_sigma: float
    clahe_clip_limit: float
    clahe_tile_grid: Tuple[int, int]


@dataclass(frozen=True)
class SegmentationConfig:
    """Shared segmentation parameters, including failure-detection bounds.

    ``polarity`` controls which side of the Otsu threshold is treated as fruit:

    * ``"bright"`` - the fruit is brighter than the background;
    * ``"dark"``   - the fruit is darker than the background;
    * ``"auto"``   - choose per image, preferring the candidate whose largest
      component touches the frame border least. Real apple photographs appear
      on both light and dark backgrounds, so a fixed polarity inverts the mask
      on a substantial fraction of the dataset. The choice is made inside the
      shared harness, so it stays identical across all three techniques.

    ``on_failure`` decides what happens to an image whose mask falls outside
    the permitted coverage bounds:

    * ``"exclude"``  - drop the image from the experiment entirely and log it.
      Because segmentation is shared, the same images are dropped for every
      technique, so the comparison stays like-for-like.
    * ``"fallback"`` - substitute a centred elliptical mask covering the middle
      of the frame, and log it. Useful for measuring how much the failures
      cost, but the substituted mask is not a real segmentation.

    ``fill_holes`` fills interior holes so the mask is the solid fruit
    silhouette. A dark blemish on a fruit photographed against a dark
    background otherwise falls on the background side of the threshold and is
    punched out of the fruit mask, which would make T3's blemish ratio - and
    every descriptor computed over mask pixels - depend on the background
    rather than on the fruit.
    """

    close_kernel: Tuple[int, int]
    min_mask_fraction: float
    max_mask_fraction: float
    polarity: str
    on_failure: str
    fill_holes: bool


@dataclass(frozen=True)
class PartitionConfig:
    """Train/test split and cross-validation settings."""

    test_size: float
    cv_folds: int
    shuffle: bool


@dataclass(frozen=True)
class AugmentationConfig:
    """Training-only augmentation settings.

    Augmentation is applied strictly after the train/test split, and only to
    the training partition, so that no augmented variant of a test image can
    ever be seen during training.
    """

    enabled: bool
    variants_per_image: int
    horizontal_flip: bool
    rotation_degrees: float
    brightness_jitter: float


@dataclass(frozen=True)
class ClassifierConfig:
    """Shared SVM settings, applied identically to every feature vector."""

    kernel: str
    C: float
    gamma: str
    probability: bool


@dataclass(frozen=True)
class Config:
    """The complete, immutable experiment configuration."""

    seed: int
    paths: Paths
    primary: DatasetSpec
    generalisation: DatasetSpec
    robustness: DatasetSpec
    image_extensions: Tuple[str, ...]
    preprocess: PreprocessConfig
    segmentation: SegmentationConfig
    partition: PartitionConfig
    augmentation: AugmentationConfig
    classifier: ClassifierConfig
    t1_colour: Mapping[str, Any] = field(default_factory=dict)
    t2_glcm: Mapping[str, Any] = field(default_factory=dict)
    t3_lbp_blemish: Mapping[str, Any] = field(default_factory=dict)
    sanity_check: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _as_int_pair(value: Sequence[int]) -> Tuple[int, int]:
    """Coerce a two-element sequence to a tuple of two integers."""
    first, second = value
    return int(first), int(second)


def _resolve(root: Path, value: str) -> Path:
    """Resolve ``value`` against ``root`` unless it is already absolute."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (root / candidate)


def _dataset_spec(block: Mapping[str, Any]) -> DatasetSpec:
    """Build a :class:`DatasetSpec` from a ``datasets`` sub-block."""
    classes = tuple(str(name) for name in block["classes"])
    display = tuple(str(name) for name in block.get("display_names", classes))
    if len(display) != len(classes):
        raise ValueError(
            "display_names must have the same length as classes "
            f"({len(display)} vs {len(classes)})"
        )
    return DatasetSpec(classes=classes, display_names=display)


def find_config_path() -> Path:
    """Locate ``config.json``.

    The search order is the ``FRESHSIGHT_CONFIG`` environment variable, then
    ``config.json`` beside this module. No absolute path is baked in.
    """
    override = os.environ.get(ENV_CONFIG_PATH)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{ENV_CONFIG_PATH} points at {path}, which does not exist."
            )
        return path

    default = Path(__file__).resolve().parent / CONFIG_FILENAME
    if not default.is_file():
        raise FileNotFoundError(
            f"Could not find {CONFIG_FILENAME} at {default}. Either restore it "
            f"or set the {ENV_CONFIG_PATH} environment variable."
        )
    return default


def _validate_preprocess(preprocess: PreprocessConfig) -> None:
    """Raise if the preprocessing parameters are not usable by OpenCV."""
    if any(size <= 0 for size in preprocess.resize):
        raise ValueError(f"preprocess.resize must be positive; got {preprocess.resize}")
    if any(k % 2 == 0 or k <= 0 for k in preprocess.gaussian_kernel):
        raise ValueError(
            "preprocess.gaussian_kernel must contain positive odd integers; got "
            f"{preprocess.gaussian_kernel}"
        )


def load_config(path: Path | str | None = None) -> Config:
    """Read and validate the configuration file, returning a :class:`Config`.

    Args:
        path: Optional explicit path to a configuration file. When omitted the
            file is located by :func:`find_config_path`.

    Returns:
        A frozen :class:`Config` with all directory paths resolved to absolute
        locations.

    Raises:
        ValueError: If any parameter is outside its permitted range.
    """
    config_path = Path(path) if path is not None else find_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = json.load(handle)

    config_dir = config_path.resolve().parent
    project_root = _resolve(config_dir, raw["paths"].get("project_root", ".")).resolve()

    paths = Paths(
        project_root=project_root,
        primary_root=_resolve(project_root, raw["paths"]["primary_root"]),
        generalisation_root=_resolve(project_root, raw["paths"]["generalisation_root"]),
        robustness_root=_resolve(project_root, raw["paths"]["robustness_root"]),
        results_root=_resolve(project_root, raw["paths"]["results_root"]),
        cache_root=_resolve(project_root, raw["paths"]["cache_root"]),
    )

    pre = raw["preprocess"]
    preprocess = PreprocessConfig(
        resize=_as_int_pair(pre["resize"]),
        gaussian_kernel=_as_int_pair(pre["gaussian_kernel"]),
        gaussian_sigma=float(pre["gaussian_sigma"]),
        clahe_clip_limit=float(pre["clahe_clip_limit"]),
        clahe_tile_grid=_as_int_pair(pre["clahe_tile_grid"]),
    )
    _validate_preprocess(preprocess)

    seg_raw = raw["segmentation"]
    polarity = str(seg_raw.get("polarity", "auto")).lower()
    if polarity not in {"auto", "bright", "dark"}:
        raise ValueError(
            f"segmentation.polarity must be auto, bright or dark; got {polarity!r}"
        )
    on_failure = str(seg_raw.get("on_failure", "exclude")).lower()
    if on_failure not in {"exclude", "fallback"}:
        raise ValueError(
            f"segmentation.on_failure must be exclude or fallback; got {on_failure!r}"
        )
    segmentation = SegmentationConfig(
        close_kernel=_as_int_pair(seg_raw["close_kernel"]),
        min_mask_fraction=float(seg_raw["min_mask_fraction"]),
        max_mask_fraction=float(seg_raw["max_mask_fraction"]),
        polarity=polarity,
        on_failure=on_failure,
        fill_holes=bool(seg_raw.get("fill_holes", True)),
    )
    if not 0.0 <= segmentation.min_mask_fraction < segmentation.max_mask_fraction <= 1.0:
        raise ValueError(
            "segmentation bounds must satisfy 0 <= min < max <= 1; got "
            f"{segmentation.min_mask_fraction} and {segmentation.max_mask_fraction}"
        )

    part_raw = raw["partition"]
    partition = PartitionConfig(
        test_size=float(part_raw["test_size"]),
        cv_folds=int(part_raw["cv_folds"]),
        shuffle=bool(part_raw.get("shuffle", True)),
    )
    if not 0.0 < partition.test_size < 1.0:
        raise ValueError(
            f"partition.test_size must lie in (0, 1); got {partition.test_size}"
        )
    if partition.cv_folds < 2:
        raise ValueError(
            f"partition.cv_folds must be at least 2; got {partition.cv_folds}"
        )

    aug_raw = raw["augmentation"]
    augmentation = AugmentationConfig(
        enabled=bool(aug_raw["enabled"]),
        variants_per_image=int(aug_raw["variants_per_image"]),
        horizontal_flip=bool(aug_raw["horizontal_flip"]),
        rotation_degrees=float(aug_raw["rotation_degrees"]),
        brightness_jitter=float(aug_raw["brightness_jitter"]),
    )
    if augmentation.variants_per_image < 0:
        raise ValueError("augmentation.variants_per_image must not be negative")

    clf_raw = raw["classifier"]
    classifier = ClassifierConfig(
        kernel=str(clf_raw["kernel"]),
        C=float(clf_raw["C"]),
        gamma=str(clf_raw["gamma"]),
        probability=bool(clf_raw["probability"]),
    )

    datasets = raw["datasets"]
    extensions = tuple(str(ext).lower() for ext in datasets["image_extensions"])

    return Config(
        seed=int(raw["seed"]),
        paths=paths,
        primary=_dataset_spec(datasets["primary"]),
        generalisation=_dataset_spec(datasets["generalisation"]),
        robustness=_dataset_spec(datasets["robustness"]),
        image_extensions=extensions,
        preprocess=preprocess,
        segmentation=segmentation,
        partition=partition,
        augmentation=augmentation,
        classifier=classifier,
        t1_colour=dict(raw.get("t1_colour", {})),
        t2_glcm=dict(raw.get("t2_glcm", {})),
        t3_lbp_blemish=dict(raw.get("t3_lbp_blemish", {})),
        sanity_check=dict(raw.get("sanity_check", {})),
        raw=raw,
    )


_CACHED: Config | None = None


def get_config(reload: bool = False) -> Config:
    """Return the process-wide configuration, loading it on first use.

    Args:
        reload: When ``True``, discard the cached configuration and read the
            file again. Intended for tests only.
    """
    global _CACHED
    if _CACHED is None or reload:
        _CACHED = load_config()
    return _CACHED


def describe(config: Config | None = None) -> str:
    """Return a human-readable summary of the active configuration."""
    cfg = config or get_config()
    aug_state = "on" if cfg.augmentation.enabled else "off"
    lines = [
        "FreshSight configuration",
        f"  seed                 : {cfg.seed}",
        f"  project root         : {cfg.paths.project_root}",
        f"  primary dataset      : {cfg.paths.primary_root}",
        f"  generalisation set   : {cfg.paths.generalisation_root}",
        f"  robustness set       : {cfg.paths.robustness_root}",
        f"  results              : {cfg.paths.results_root}",
        f"  classes              : {', '.join(cfg.primary.display_names)}",
        f"  resize               : {cfg.preprocess.resize[0]} x {cfg.preprocess.resize[1]}",
        f"  Gaussian kernel      : {cfg.preprocess.gaussian_kernel}",
        f"  CLAHE clip limit     : {cfg.preprocess.clahe_clip_limit}",
        f"  segmentation polarity: {cfg.segmentation.polarity}",
        f"  mask bounds          : {cfg.segmentation.min_mask_fraction:.0%} - "
        f"{cfg.segmentation.max_mask_fraction:.0%} of frame",
        f"  test size            : {cfg.partition.test_size:.0%}",
        f"  CV folds             : {cfg.partition.cv_folds}",
        f"  augmentation         : {aug_state} "
        f"({cfg.augmentation.variants_per_image} variants per training image)",
        f"  classifier           : SVC(kernel={cfg.classifier.kernel}, "
        f"C={cfg.classifier.C}, gamma={cfg.classifier.gamma})",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
