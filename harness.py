"""The shared experimental harness.

Everything in this module is held constant across the three feature extraction
techniques. Preprocessing, segmentation, the train/test partition, the
cross-validation folds, the augmentation plan, and the classifier pipeline are
produced here and here only, so that the descriptor family remains the single
experimental variable.

A technique receives a preprocessed image and a fruit mask, and returns a
fixed-length vector. It is handed copies of both arrays, has no reference to
the configuration or to the harness, and therefore cannot influence any other
stage of the experiment.

Pipeline order for one sample::

    read -> (augment, training only) -> preprocess -> segment -> extract

Augmentation is deliberately applied to the raw image, before preprocessing,
so that a brightness-jittered variant is genuinely put through the same CLAHE
illumination normalisation that a real image would be.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import cv2
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import Config, get_config
from data import ImageRecord, labels_of, read_image

# A technique is any callable matching this signature. Fixed for all three.
FeatureFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@runtime_checkable
class FeatureExtractorLike(Protocol):
    """Structural type the harness requires of a feature extraction technique.

    Declared structurally, and deliberately not imported from the ``features``
    package, so that the harness has no dependency on any concrete technique
    and the coupling runs in one direction only.
    """

    short_name: str
    dim: int

    def __call__(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> np.ndarray:
        """Return a fixed-length 1-D descriptor for one segmented fruit."""
        ...


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_global_seed(config: Optional[Config] = None) -> int:
    """Seed every random source the experiment touches.

    scikit-learn estimators additionally receive ``random_state`` explicitly;
    this function covers the module-level generators that some library code
    falls back on.

    Returns:
        The seed that was applied.
    """
    cfg = config or get_config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    return cfg.seed


# --------------------------------------------------------------------------- #
# Stage 1: preprocessing
# --------------------------------------------------------------------------- #

def preprocess(image_bgr: np.ndarray, config: Optional[Config] = None) -> np.ndarray:
    """Apply the shared preprocessing chain to one BGR image.

    The three steps, in order, are:

    1. resize to the configured size (224 x 224);
    2. Gaussian blur with the configured kernel (5 x 5) for noise suppression;
    3. conversion to CIE L*a*b*, CLAHE on the L* channel only, conversion back
       to BGR. Restricting equalisation to L* normalises illumination without
       disturbing the a* and b* chromaticity that the colour descriptors read.

    Args:
        image_bgr: An ``(H, W, 3)`` ``uint8`` BGR image.
        config: Optional configuration override.

    Returns:
        A new ``(224, 224, 3)`` ``uint8`` BGR image. The input is not modified.
    """
    cfg = config or get_config()
    pre = cfg.preprocess

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) BGR image; got shape {image_bgr.shape}")

    resized = cv2.resize(image_bgr, pre.resize, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, pre.gaussian_kernel, pre.gaussian_sigma)

    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=pre.clahe_clip_limit,
        tileGridSize=pre.clahe_tile_grid,
    )
    equalised = clahe.apply(lightness)
    merged = cv2.merge((equalised, green_red, blue_yellow))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------- #
# Stage 2: segmentation
# --------------------------------------------------------------------------- #

@dataclass
class SegmentationResult:
    """The fruit mask and its derived geometry for one image.

    Attributes:
        mask: ``(H, W)`` ``uint8`` mask, 255 inside the fruit and 0 outside.
        bbox: ``(x, y, w, h)`` bounding box of the retained component.
        contour: ``(N, 1, 2)`` array of the outer contour points.
        coverage: Fraction of the frame the mask occupies, in ``[0, 1]``.
        polarity: Which side of the Otsu threshold was taken as fruit. Only
            meaningful for the ``otsu_v`` method; ``"n/a"`` otherwise.
        threshold: The Otsu threshold value chosen on the V channel, or NaN
            when the method does not threshold.
        method: The segmentation method that produced this result.
        failed: True when the mask fell outside the permitted coverage bounds.
        reason: Explanation when ``failed`` is True, otherwise an empty string.
        substituted: True when a fallback elliptical mask replaced the result.
    """

    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    contour: np.ndarray
    coverage: float
    polarity: str
    threshold: float
    method: str = "otsu_v"
    failed: bool = False
    reason: str = ""
    substituted: bool = False


def _largest_component(binary: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Keep only the largest non-background connected component.

    Args:
        binary: ``(H, W)`` ``uint8`` image with foreground at 255.

    Returns:
        A tuple of the single-component mask and its ``(x, y, w, h)`` box. When
        the input is entirely background, an all-zero mask and a zero box are
        returned.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:  # Label 0 is the background, so there is nothing to keep.
        return np.zeros_like(binary), (0, 0, 0, 0)

    areas = stats[1:, cv2.CC_STAT_AREA]
    winner = int(np.argmax(areas)) + 1
    mask = np.where(labels == winner, 255, 0).astype(np.uint8)
    box = (
        int(stats[winner, cv2.CC_STAT_LEFT]),
        int(stats[winner, cv2.CC_STAT_TOP]),
        int(stats[winner, cv2.CC_STAT_WIDTH]),
        int(stats[winner, cv2.CC_STAT_HEIGHT]),
    )
    return mask, box


def _border_touch_fraction(mask: np.ndarray) -> float:
    """Fraction of the frame border occupied by the mask.

    A correctly segmented fruit sits away from the edges, whereas an inverted
    mask (background taken as foreground) hugs the whole border. This is the
    statistic the automatic polarity choice is made on.
    """
    top, bottom = mask[0, :], mask[-1, :]
    left, right = mask[:, 0], mask[:, -1]
    border = np.concatenate([top, bottom, left, right])
    return float(np.count_nonzero(border) / border.size)


def _outer_contour(mask: np.ndarray) -> np.ndarray:
    """Return the largest external contour of ``mask``, or an empty array."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 1, 2), dtype=np.int32)
    return max(contours, key=cv2.contourArea)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes so the mask is the solid fruit silhouette.

    A dark blemish on a fruit photographed against a dark background falls on
    the background side of the Otsu threshold, and a 5 x 5 closing is far too
    small to bridge it. The blemish is then punched out of the fruit mask as a
    hole - which is precisely backwards, because T3 reports blemish ratio as
    blemished pixels over mask pixels, and pixels outside the mask are never
    examined. Left unfilled, the most severely rotten fruit would report the
    least blemishing, and the same fruit would yield a different mask on a
    light background than on a dark one.

    Filling the region enclosed by the outer contour removes that asymmetry.
    It is applied inside the shared harness, so every technique sees the same
    silhouette.
    """
    contour = _outer_contour(mask)
    if contour.size == 0:
        return mask
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return filled


def _fallback_mask(shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Build a centred elliptical mask covering the middle of the frame.

    Used only when ``segmentation.on_failure`` is ``"fallback"``. The ellipse
    spans 70% of each dimension, which is a plausible extent for a centred
    fruit but is emphatically not a real segmentation.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    centre = (width // 2, height // 2)
    axes = (int(width * 0.35), int(height * 0.35))
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, thickness=-1)
    box = (centre[0] - axes[0], centre[1] - axes[1], axes[0] * 2, axes[1] * 2)
    return mask, box


def _judge_coverage(
    mask: np.ndarray,
    box: Tuple[int, int, int, int],
    polarity: str,
    threshold: float,
    seg,
    method: str,
    degenerate_reason: str = "",
) -> SegmentationResult:
    """Apply the shared coverage check and build the result.

    Kept in one place so that every segmentation method is held to the same
    bounds and the same failure policy. A mask outside the permitted range is
    flagged rather than passed silently to a feature extractor.

    Args:
        degenerate_reason: When non-empty, the mask is flagged as a failure
            regardless of its coverage. Used by the GrabCut path, whose seeded
            foreground core means an unusable result is not necessarily an
            undersized one.
    """
    coverage = float(np.count_nonzero(mask) / mask.size)
    failed = (
        bool(degenerate_reason)
        or coverage < seg.min_mask_fraction
        or coverage > seg.max_mask_fraction
    )
    reason = degenerate_reason
    substituted = False

    if failed and not reason:
        bound = "below" if coverage < seg.min_mask_fraction else "above"
        reason = (
            f"mask covers {coverage:.1%} of the frame, {bound} the permitted "
            f"{seg.min_mask_fraction:.0%}-{seg.max_mask_fraction:.0%} range"
        )
    if failed and seg.on_failure == "fallback":
        mask, box = _fallback_mask(mask.shape)
        coverage = float(np.count_nonzero(mask) / mask.size)
        substituted = True

    return SegmentationResult(
        mask=mask,
        bbox=box,
        contour=_outer_contour(mask),
        coverage=coverage,
        polarity=polarity,
        threshold=threshold,
        method=method,
        failed=failed,
        reason=reason,
        substituted=substituted,
    )


def _core_ellipse(shape: Tuple[int, int], seg) -> np.ndarray:
    """Return the central region GrabCut is told is definitely foreground."""
    height, width = shape
    scale = seg.grabcut["core_scale"]
    core = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        core,
        (width // 2, height // 2),
        (int(width * scale / 2), int(height * scale / 2)),
        0, 0, 360, 255, thickness=-1,
    )
    return core


def _grabcut_seed(shape: Tuple[int, int], seg) -> np.ndarray:
    """Build the GrabCut label image that seeds the segmentation.

    GrabCut needs to be told roughly where the object is. Seeding it with a
    bare rectangle - the usual recipe - fails on this dataset: when the fruit
    fills the frame there is no background inside the rectangle to model, and
    the result collapses to an empty mask. Measured over the awkward cases,
    rectangle initialisation returned nothing at all for three images in
    twelve.

    Seeding with an explicit label image removes that failure. The frame
    border is marked definite background, a central core definite foreground,
    and the ellipse between them probable foreground. Because the core is
    definite, the result can never be empty, and because the border is
    definite background there is always a background distribution to fit.

    Args:
        shape: ``(height, width)`` of the image being segmented.
        seg: The active :class:`~config.SegmentationConfig`.

    Returns:
        A ``uint8`` label image of ``cv2.GC_*`` values.
    """
    height, width = shape
    centre = (width // 2, height // 2)
    scales = seg.grabcut

    labels = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    cv2.ellipse(
        labels,
        centre,
        (int(width * scales["probable_scale"] / 2), int(height * scales["probable_scale"] / 2)),
        0, 0, 360, int(cv2.GC_PR_FGD), thickness=-1,
    )
    labels[_core_ellipse(shape, seg) > 0] = cv2.GC_FGD

    border = max(1, int(round(min(height, width) * scales["border_scale"])))
    labels[:border, :] = cv2.GC_BGD
    labels[-border:, :] = cv2.GC_BGD
    labels[:, :border] = cv2.GC_BGD
    labels[:, -border:] = cv2.GC_BGD
    return labels


def _grabcut_binary(image_bgr: np.ndarray, seg, seed: int) -> np.ndarray:
    """Run seeded GrabCut and return its foreground as a binary image.

    ``cv2.grabCut`` fits its colour models with k-means, which draws its
    initial centres from OpenCV's global random number generator. Left alone
    it therefore returns a slightly different mask on every call, and two
    techniques handed "the same" image would in fact be described from
    different masks - the one difference between techniques the study is
    built to exclude. Reseeding immediately before each call makes the result
    a pure function of the image and the configuration.
    """
    cv2.setRNGSeed(seed)

    labels = _grabcut_seed(image_bgr.shape[:2], seg)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(
        image_bgr,
        labels,
        None,
        background_model,
        foreground_model,
        int(seg.grabcut["iterations"]),
        cv2.GC_INIT_WITH_MASK,
    )
    foreground = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    return np.where(foreground, 255, 0).astype(np.uint8)


def _segment_fruit_uncached(
    image_bgr: np.ndarray,
    config: Optional[Config] = None,
) -> SegmentationResult:
    """Segment the fruit from the background.

    Two methods are available, selected by ``segmentation.method``. Both end
    with the same morphological closing, largest-component selection, optional
    hole filling and coverage check, so only the way the foreground is
    proposed differs. Whichever is chosen applies identically to all three
    techniques, since the choice is made here in the shared harness.

    ``grabcut`` (default) seeds GrabCut with an explicit label image and lets
    it refine the boundary from the image's colour statistics.

    ``otsu_v`` thresholds the HSV value channel as the brief specifies, and is
    kept for the comparison reported in the write-up.

    With ``segmentation.polarity`` set to ``"auto"`` both sides of the Otsu
    threshold are evaluated and the one whose largest component touches the
    frame border least is kept, which handles light and dark backgrounds
    alike. The decision is made here, in the shared harness, so it is identical
    for every technique.

    A mask covering less than ``min_mask_fraction`` or more than
    ``max_mask_fraction`` of the frame is flagged as a segmentation failure
    rather than being passed silently to a feature extractor.

    Args:
        image_bgr: A preprocessed ``uint8`` BGR image.
        config: Optional configuration override.

    Returns:
        A :class:`SegmentationResult`.
    """
    cfg = config or get_config()
    seg = cfg.segmentation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, seg.close_kernel)

    def _finish(binary: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Close, keep the largest component, and optionally fill holes."""
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask, box = _largest_component(closed)
        return (_fill_holes(mask) if seg.fill_holes else mask), box

    if seg.method == "grabcut":
        mask, box = _finish(_grabcut_binary(image_bgr, seg, cfg.seed))

        # The seeded core is marked definite foreground, so GrabCut can never
        # return an empty mask and the minimum-coverage bound cannot catch an
        # image that holds no fruit at all. A mask that barely grew beyond that
        # seed is the equivalent signal: the colour models found nothing to
        # attach the foreground to, and the "mask" is essentially the ellipse
        # the harness drew. Flag it rather than describe a patch of background.
        #
        # Growth is measured against the seed's area rather than as an overlap,
        # because image grain lets GrabCut creep a little way past the core
        # even on a blank frame - a uniform frame grows by about 1.05x, whereas
        # a real fruit covers several times the core. Measured on flat grey,
        # flat white and grey-plus-grain frames: 1.05x, 1.05x, 1.07x.
        core_area = np.count_nonzero(_core_ellipse(image_bgr.shape[:2], seg))
        growth = (np.count_nonzero(mask) / core_area) if core_area else 0.0
        minimum_growth = float(seg.grabcut["min_seed_growth"])
        degenerate = (
            f"GrabCut grew to only {growth:.2f}x its seeded foreground core "
            f"(minimum {minimum_growth:.2f}x), so the image offers no "
            f"fruit-like region to segment"
            if growth < minimum_growth
            else ""
        )
        return _judge_coverage(
            mask, box, "n/a", float("nan"), seg, seg.method, degenerate
        )

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    threshold, bright = cv2.threshold(
        value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    dark = cv2.bitwise_not(bright)

    candidates = {
        name: _finish(binary) for name, binary in (("bright", bright), ("dark", dark))
    }

    if seg.polarity == "auto":
        def _rank(name: str) -> Tuple[int, float, int]:
            """Order candidates: plausible coverage first, then away from the border.

            The two Otsu sides are complements, so whenever one covers almost
            the whole frame the other covers almost none. Ranking plausible
            coverage first stops a degenerate empty mask winning purely by
            never touching the border.
            """
            mask = candidates[name][0]
            coverage = np.count_nonzero(mask) / mask.size
            in_bounds = seg.min_mask_fraction <= coverage <= seg.max_mask_fraction
            return (
                0 if in_bounds else 1,
                _border_touch_fraction(mask),
                -int(np.count_nonzero(mask)),
            )

        chosen = min(candidates, key=_rank)
    else:
        chosen = seg.polarity

    mask, box = candidates[chosen]
    return _judge_coverage(mask, box, chosen, float(threshold), seg, seg.method)


# --------------------------------------------------------------------------- #
# Segmentation cache
# --------------------------------------------------------------------------- #

#: Bumped whenever the cached representation itself changes shape, so that a
#: cache written by an older version is ignored rather than misread.
_CACHE_FORMAT = "v1"


def _segmentation_signature(config: Config) -> str:
    """Hash every input that can change what the segmenter returns.

    The key deliberately covers the whole ``segmentation`` block rather than
    the handful of fields the active method happens to read. Hashing only the
    live fields would let a change to ``method`` reuse masks computed under the
    other one, which is exactly the silent corruption a cache must not permit.

    The seed is included because the GrabCut path reseeds OpenCV's global
    generator from it, so it is an input to the mask like any other.
    """
    payload = json.dumps(
        {
            "format": _CACHE_FORMAT,
            "seed": config.seed,
            "segmentation": config.raw.get("segmentation", {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _image_digest(image_bgr: np.ndarray) -> str:
    """Hash the pixels the segmenter will actually see.

    Keyed on content, never on the file it came from. The harness augments a
    training image into several variants, all of which share a path but none
    of which share a mask; a path-keyed cache would hand every variant the
    original's mask and quietly destroy the augmentation.
    """
    contiguous = np.ascontiguousarray(image_bgr)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _cache_path(image_bgr: np.ndarray, config: Config) -> Path:
    """Return the file a cached result for this image would occupy."""
    digest = _image_digest(image_bgr)
    root = config.paths.cache_root / "segmentation" / _segmentation_signature(config)
    return root / digest[:2] / f"{digest}.npz"


def _load_cached(path: Path) -> Optional[SegmentationResult]:
    """Rebuild a result from disk, or return None if it cannot be read.

    Any failure - a truncated file from an interrupted run, an unreadable
    archive, a missing field - is treated as a cache miss. A cache is an
    optimisation, so it may never be the reason a run fails.
    """
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            mask = archive["mask"]
            meta = json.loads(str(archive["meta"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None

    return SegmentationResult(
        mask=mask,
        bbox=tuple(int(v) for v in meta["bbox"]),
        contour=_outer_contour(mask),
        coverage=float(meta["coverage"]),
        polarity=str(meta["polarity"]),
        threshold=float(meta["threshold"]),
        method=str(meta["method"]),
        failed=bool(meta["failed"]),
        reason=str(meta["reason"]),
        substituted=bool(meta["substituted"]),
    )


def _store_cached(path: Path, result: SegmentationResult) -> None:
    """Write a result to the cache atomically.

    The contour is not stored: it is a pure function of the mask, recomputed
    on load by the same routine that produced it. Storing a derived value
    would create a second place for it to disagree with the mask.

    The write goes to a temporary file in the destination directory and is
    renamed into place, so a run interrupted mid-write leaves either the old
    entry or none, never a half-written archive that a later run would read.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(
        {
            "bbox": [int(v) for v in result.bbox],
            "coverage": float(result.coverage),
            "polarity": result.polarity,
            "threshold": float(result.threshold),
            "method": result.method,
            "failed": bool(result.failed),
            "reason": result.reason,
            "substituted": bool(result.substituted),
        },
        sort_keys=True,
    )
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, mask=result.mask, meta=np.array(meta))
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)  # A cache write must never fail a run.


def segment_fruit(
    image_bgr: np.ndarray,
    config: Optional[Config] = None,
) -> SegmentationResult:
    """Segment the fruit, reusing a cached mask when one is available.

    Segmentation is a pure function of the preprocessed image and the
    ``segmentation`` configuration, and it is by far the most expensive stage
    in the pipeline: every technique re-derives the same masks over the same
    images, and every experiment arm re-derives them again. Caching removes
    that repetition without changing a single mask, because the cache key
    covers every input the result depends on - the image pixels themselves,
    the whole segmentation block, and the seed.

    Caching is controlled by ``segmentation.cache_masks``. With it disabled
    this function is exactly :func:`_segment_fruit_uncached`.

    Args:
        image_bgr: A preprocessed ``uint8`` BGR image.
        config: Optional configuration override.

    Returns:
        A :class:`SegmentationResult`, identical whether it was computed or
        loaded.
    """
    cfg = config or get_config()
    if not cfg.segmentation.cache_masks:
        return _segment_fruit_uncached(image_bgr, cfg)

    path = _cache_path(image_bgr, cfg)
    cached = _load_cached(path)
    if cached is not None:
        return cached

    result = _segment_fruit_uncached(image_bgr, cfg)
    _store_cached(path, result)
    return result


# --------------------------------------------------------------------------- #
# Stage 3: augmentation (training partition only)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AugmentationOp:
    """One deterministic augmentation of a single training image.

    Attributes:
        flip: Whether to mirror the image horizontally.
        angle: Rotation in degrees about the image centre.
        brightness: Multiplicative gain applied to every channel.
    """

    flip: bool
    angle: float
    brightness: float

    @property
    def is_identity(self) -> bool:
        """True when this operation would leave the image unchanged."""
        return not self.flip and self.angle == 0.0 and self.brightness == 1.0


IDENTITY_OP = AugmentationOp(flip=False, angle=0.0, brightness=1.0)


def apply_augmentation(image_bgr: np.ndarray, op: AugmentationOp) -> np.ndarray:
    """Apply one augmentation operation to a raw BGR image.

    Rotation uses replicated borders rather than zero padding, so that no
    artificial black corners are introduced for Otsu to mistake for fruit.

    Args:
        image_bgr: A ``uint8`` BGR image.
        op: The operation to apply.

    Returns:
        A new ``uint8`` BGR image of the same shape. The input is unchanged.
    """
    if op.is_identity:
        return image_bgr.copy()

    output = image_bgr
    if op.flip:
        output = cv2.flip(output, 1)

    if op.angle != 0.0:
        height, width = output.shape[:2]
        centre = (width / 2.0, height / 2.0)
        rotation = cv2.getRotationMatrix2D(centre, op.angle, 1.0)
        output = cv2.warpAffine(
            output,
            rotation,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    if op.brightness != 1.0:
        output = cv2.convertScaleAbs(output, alpha=op.brightness, beta=0)

    return np.ascontiguousarray(output)


def build_augmentation_plan(
    n_images: int,
    config: Optional[Config] = None,
) -> List[List[AugmentationOp]]:
    """Draw the augmentation operations for every training image, once.

    The plan is generated from the fixed seed in a fixed traversal order, so it
    is byte-for-byte identical every run and across all three techniques. The
    identity operation always heads each image's list, so the original image is
    always retained alongside its variants.

    Args:
        n_images: Number of images in the training partition.
        config: Optional configuration override.

    Returns:
        A list of length ``n_images``; each element lists the operations to
        apply to that image, beginning with the identity.
    """
    cfg = config or get_config()
    aug = cfg.augmentation

    plan: List[List[AugmentationOp]] = []
    if not aug.enabled or aug.variants_per_image == 0:
        return [[IDENTITY_OP] for _ in range(n_images)]

    rng = np.random.default_rng(cfg.seed)
    for _ in range(n_images):
        ops = [IDENTITY_OP]
        for _ in range(aug.variants_per_image):
            flip = bool(rng.random() < 0.5) if aug.horizontal_flip else False
            angle = float(
                rng.uniform(-aug.rotation_degrees, aug.rotation_degrees)
            ) if aug.rotation_degrees else 0.0
            brightness = float(
                rng.uniform(1.0 - aug.brightness_jitter, 1.0 + aug.brightness_jitter)
            ) if aug.brightness_jitter else 1.0
            ops.append(AugmentationOp(flip=flip, angle=angle, brightness=brightness))
        plan.append(ops)
    return plan


# --------------------------------------------------------------------------- #
# Stage 4: partition
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Partition:
    """An 80:20 stratified split of a record list.

    Attributes:
        train: Training records. Only these may be augmented.
        test: Held-out test records. Never augmented, never used for fitting.
        train_indices: Positions of the training records in the original list.
        test_indices: Positions of the test records in the original list.
    """

    train: List[ImageRecord]
    test: List[ImageRecord]
    train_indices: np.ndarray
    test_indices: np.ndarray


def stratified_split(
    records: Sequence[ImageRecord],
    config: Optional[Config] = None,
) -> Partition:
    """Split records 80:20, stratified by class, with the fixed seed.

    Args:
        records: The full, deterministically ordered record list.
        config: Optional configuration override.

    Returns:
        A :class:`Partition`. The split depends only on the record ordering and
        the seed, so every technique receives exactly the same partition.
    """
    cfg = config or get_config()
    labels = labels_of(records)
    indices = np.arange(len(records))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=cfg.partition.test_size,
        random_state=cfg.seed,
        shuffle=True,
        stratify=labels,
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    return Partition(
        train=[records[i] for i in train_idx],
        test=[records[i] for i in test_idx],
        train_indices=train_idx,
        test_indices=test_idx,
    )


def make_cv(config: Optional[Config] = None) -> StratifiedKFold:
    """Build the 5-fold stratified cross-validator used on the training set.

    The folds are never applied to the test partition. Reusing one seeded
    splitter across techniques means fold membership is identical, which is the
    precondition for the paired t-tests in Phase 4.
    """
    cfg = config or get_config()
    return StratifiedKFold(
        n_splits=cfg.partition.cv_folds,
        shuffle=cfg.partition.shuffle,
        random_state=cfg.seed,
    )


# --------------------------------------------------------------------------- #
# Stage 5: classifier
# --------------------------------------------------------------------------- #

def build_pipeline(config: Optional[Config] = None) -> Pipeline:
    """Build the shared standardise-then-SVM pipeline.

    Wrapping the scaler in a :class:`~sklearn.pipeline.Pipeline` is what keeps
    standardisation honest: when the pipeline is passed to a cross-validator,
    the scaler is fitted on each training fold alone and never sees the
    validation fold or the test partition.

    ``SVC`` is built without ``probability=True``. That flag never changed a
    prediction - ``SVC.predict`` takes the argmax of the decision function
    either way - it only fitted an extra internal Platt model on every call.
    scikit-learn deprecated it in 1.9 and removes it in 1.11, and the brief
    asks for calibrated probabilities in exactly one place, the E3 decision
    fusion, which uses :func:`build_probability_pipeline` instead.

    Returns:
        An unfitted pipeline of ``StandardScaler`` then ``SVC``.
    """
    cfg = config or get_config()
    clf = cfg.classifier
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel=clf.kernel,
                    C=clf.C,
                    gamma=clf.gamma,
                    random_state=cfg.seed,
                ),
            ),
        ]
    )


def build_probability_pipeline(config: Optional[Config] = None) -> Pipeline:
    """Build the shared pipeline in its calibrated, probability-emitting form.

    Same scaler and the same ``SVC`` hyperparameters as :func:`build_pipeline`,
    wrapped in :class:`~sklearn.calibration.CalibratedClassifierCV` so that
    ``predict_proba`` returns calibrated posteriors. The brief names this
    estimator for the E3 weighted decision-level fusion, and it is also what
    scikit-learn 1.9 directs ``SVC(probability=True)`` callers to.

    ``ensemble=False`` fits the sigmoid on out-of-fold decision values and then
    refits the ``SVC`` once on all the data handed in, so the calibrated model
    is the same single ``SVC`` the rest of the study uses rather than an
    average of five. Calibration therefore sees only the rows the pipeline is
    fitted on, which under :func:`leakage_safe_folds` is the training fold
    alone.

    Returns:
        An unfitted pipeline of ``StandardScaler`` then a calibrated ``SVC``.

    Raises:
        ValueError: If ``classifier.probability`` is disabled in the config.
            E3 and the dashboard cannot fall back to an uncalibrated decision
            function without silently reporting numbers that are not
            probabilities, so the switch fails loudly instead.
    """
    cfg = config or get_config()
    clf = cfg.classifier
    if not clf.probability:
        raise ValueError(
            "classifier.probability is false, but the E3 decision fusion and "
            "the dashboard both need calibrated class probabilities. Set it to "
            "true in config.json, or do not call this pipeline."
        )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svc",
                CalibratedClassifierCV(
                    SVC(
                        kernel=clf.kernel,
                        C=clf.C,
                        gamma=clf.gamma,
                        random_state=cfg.seed,
                    ),
                    method="sigmoid",
                    cv=cfg.partition.cv_folds,
                    ensemble=False,
                ),
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Preparing samples for a feature extractor
# --------------------------------------------------------------------------- #

@dataclass
class PreparedSample:
    """One image carried through preprocessing and segmentation.

    Attributes:
        record: The source record on disk.
        image: The preprocessed ``(224, 224, 3)`` BGR image.
        segmentation: The segmentation result, including the fruit mask.
        op: The augmentation applied before preprocessing.
        variant: 0 for the original image, 1..n for augmented variants.
        record_index: Position of the source record in the list this sample was
            prepared from. Every variant of one image shares this value, which
            is what lets the cross-validator keep an image and its variants on
            the same side of a fold.
    """

    record: ImageRecord
    image: np.ndarray
    segmentation: SegmentationResult
    op: AugmentationOp = IDENTITY_OP
    variant: int = 0
    record_index: int = 0

    @property
    def label(self) -> int:
        """Ground-truth class label."""
        return self.record.label

    @property
    def mask(self) -> np.ndarray:
        """The fruit mask."""
        return self.segmentation.mask

    @property
    def is_augmented(self) -> bool:
        """True when this sample is an augmented variant, not an original."""
        return self.variant > 0


def prepare_sample(
    record: ImageRecord,
    op: AugmentationOp = IDENTITY_OP,
    variant: int = 0,
    config: Optional[Config] = None,
    record_index: int = 0,
) -> PreparedSample:
    """Read, optionally augment, preprocess and segment one record."""
    cfg = config or get_config()
    raw = read_image(record.path)
    augmented = apply_augmentation(raw, op) if not op.is_identity else raw
    image = preprocess(augmented, cfg)
    segmentation = segment_fruit(image, cfg)
    return PreparedSample(
        record=record,
        image=image,
        segmentation=segmentation,
        op=op,
        variant=variant,
        record_index=record_index,
    )


def iter_prepared(
    records: Sequence[ImageRecord],
    augment: bool = False,
    config: Optional[Config] = None,
) -> Iterator[PreparedSample]:
    """Yield prepared samples for ``records``, optionally with augmentation.

    Args:
        records: Records to prepare. For the training partition these are the
            training records only.
        augment: When True, apply the deterministic augmentation plan. This
            must never be True for a test or validation partition.
        config: Optional configuration override.

    Yields:
        :class:`PreparedSample` objects, originals before their variants.
    """
    cfg = config or get_config()
    plan = (
        build_augmentation_plan(len(records), cfg)
        if augment
        else [[IDENTITY_OP] for _ in records]
    )
    for record_index, (record, ops) in enumerate(zip(records, plan)):
        for variant, op in enumerate(ops):
            yield prepare_sample(
                record,
                op=op,
                variant=variant,
                config=cfg,
                record_index=record_index,
            )


@dataclass
class SegmentationFailure:
    """A logged segmentation failure, written to CSV by the callers."""

    path: Path
    class_folder: str
    source: str
    variant: int
    coverage: float
    reason: str
    substituted: bool


def collect_failures(samples: Sequence[PreparedSample]) -> List[SegmentationFailure]:
    """Extract the segmentation failures from a list of prepared samples."""
    return [
        SegmentationFailure(
            path=sample.record.path,
            class_folder=sample.record.class_folder,
            source=sample.record.source,
            variant=sample.variant,
            coverage=sample.segmentation.coverage,
            reason=sample.segmentation.reason,
            substituted=sample.segmentation.substituted,
        )
        for sample in samples
        if sample.segmentation.failed
    ]


# --------------------------------------------------------------------------- #
# Building a feature matrix from any technique
# --------------------------------------------------------------------------- #

@dataclass
class FeatureMatrix:
    """The output of running one technique over one partition.

    Attributes:
        X: ``(n_samples, dim)`` feature matrix.
        y: ``(n_samples,)`` integer labels.
        technique: Short name of the technique that produced ``X``.
        extraction_times: Per-image extraction time in seconds, excluding the
            warm-up call, in the same order as the rows of ``X``.
        n_augmented: How many rows are augmented variants rather than originals.
        failures: Segmentation failures encountered while building the matrix.
        excluded: How many samples were dropped because segmentation failed.
        paths: Source file path for each row, for tracing individual errors.
        groups: ``(n_samples,)`` source-image index per row. An original and
            all of its augmented variants share one group, which is what
            :func:`leakage_safe_folds` uses to keep them on the same side of a
            cross-validation fold.
        variants: ``(n_samples,)`` variant index per row; 0 marks an original.
    """

    X: np.ndarray
    y: np.ndarray
    technique: str
    extraction_times: np.ndarray
    n_augmented: int
    failures: List[SegmentationFailure]
    excluded: int
    paths: List[Path]
    groups: np.ndarray
    variants: np.ndarray

    @property
    def originals(self) -> np.ndarray:
        """Row indices of the original, non-augmented samples."""
        return np.flatnonzero(self.variants == 0)

    @property
    def dim(self) -> int:
        """Feature dimensionality."""
        return int(self.X.shape[1]) if self.X.size else 0

    @property
    def mean_extraction_time(self) -> float:
        """Mean per-image extraction time in seconds, warm-up excluded."""
        return float(np.mean(self.extraction_times)) if self.extraction_times.size else float("nan")


def build_feature_matrix(
    records: Sequence[ImageRecord],
    extractor: "FeatureExtractorLike",
    augment: bool = False,
    config: Optional[Config] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> FeatureMatrix:
    """Run one technique over one partition, polymorphically.

    The harness knows nothing about ``extractor`` beyond the fact that it is
    callable as ``extractor(bgr_image, fruit_mask)``. Copies of the image and
    mask are handed over, so a technique cannot modify the arrays the harness
    or another technique will use.

    Timing excludes a warm-up call on the first sample, so one-off import,
    allocation and library initialisation costs are not charged to the
    algorithm.

    Args:
        records: Records to describe.
        extractor: Any object implementing the shared feature interface.
        augment: Apply the training-only augmentation plan. Must be False for
            test and validation partitions.
        config: Optional configuration override.
        progress: Optional ``(done, total)`` callback for console reporting.

    Returns:
        A :class:`FeatureMatrix`.
    """
    import time

    cfg = config or get_config()
    short_name = getattr(extractor, "short_name", extractor.__class__.__name__)

    vectors: List[np.ndarray] = []
    labels: List[int] = []
    times: List[float] = []
    paths: List[Path] = []
    groups: List[int] = []
    variants: List[int] = []
    failures: List[SegmentationFailure] = []
    excluded = 0
    n_augmented = 0
    warmed_up = False

    total = len(records) * (
        1 + (cfg.augmentation.variants_per_image if augment and cfg.augmentation.enabled else 0)
    )
    done = 0

    for sample in iter_prepared(records, augment=augment, config=cfg):
        done += 1
        if progress is not None:
            progress(done, total)

        if sample.segmentation.failed:
            failures.append(collect_failures([sample])[0])
            if not sample.segmentation.substituted:
                excluded += 1
                continue

        image = sample.image.copy()
        mask = sample.mask.copy()

        if not warmed_up:
            extractor(image, mask)  # Warm-up: result discarded, time not counted.
            warmed_up = True

        start = time.perf_counter()
        vector = extractor(image, mask)
        times.append(time.perf_counter() - start)

        vectors.append(vector)
        labels.append(sample.label)
        paths.append(sample.record.path)
        groups.append(sample.record_index)
        variants.append(sample.variant)
        if sample.is_augmented:
            n_augmented += 1

    matrix = (
        np.vstack(vectors)
        if vectors
        else np.empty((0, getattr(extractor, "dim", 0)), dtype=np.float64)
    )

    return FeatureMatrix(
        X=matrix,
        y=np.asarray(labels, dtype=np.int64),
        technique=short_name,
        extraction_times=np.asarray(times, dtype=np.float64),
        n_augmented=n_augmented,
        failures=failures,
        excluded=excluded,
        paths=paths,
        groups=np.asarray(groups, dtype=np.int64),
        variants=np.asarray(variants, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# Leakage-safe cross-validation folds
# --------------------------------------------------------------------------- #

def leakage_safe_folds(
    matrix: FeatureMatrix,
    config: Optional[Config] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build 5 stratified folds that cannot leak an augmented variant.

    Splitting the augmented feature matrix directly would be a serious error.
    Augmenting an image produces several rows that are near-duplicates of one
    another, so a plain row-wise split puts a flipped copy of a validation
    image into the training fold and the model is scored on data it has
    effectively already seen. Validation accuracy then measures memorisation
    rather than generalisation, and inflates every technique's score.

    This function instead:

    1. draws the stratified folds over **source images**, not over rows, so an
       image and all of its variants always land on the same side;
    2. puts originals *and* their augmented variants into the training side;
    3. puts **only originals** into the validation side, because a score
       measured on augmented data would not describe real-world performance.

    Fold membership depends only on the record ordering and the seed, so it is
    identical for all three techniques. That is the precondition for the Phase
    4 paired t-tests.

    Args:
        matrix: A training feature matrix built with ``augment=True``.
        config: Optional configuration override.

    Returns:
        A list of ``(train_rows, validation_rows)`` index arrays, one pair per
        fold.

    Raises:
        ValueError: If the matrix is empty, or if a fold has no validation
            rows because no original survived segmentation.
    """
    cfg = config or get_config()
    if matrix.y.size == 0:
        raise ValueError("Cannot cross-validate an empty feature matrix")

    unique_groups = np.unique(matrix.groups)
    # Every row of a group shares one label, so the first occurrence suffices.
    group_labels = np.asarray(
        [matrix.y[np.flatnonzero(matrix.groups == group)[0]] for group in unique_groups],
        dtype=np.int64,
    )

    is_original = matrix.variants == 0
    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    splitter = make_cv(cfg)
    placeholder = np.zeros((unique_groups.size, 1))
    for train_positions, validation_positions in splitter.split(placeholder, group_labels):
        train_groups = unique_groups[train_positions]
        validation_groups = unique_groups[validation_positions]

        in_train = np.isin(matrix.groups, train_groups)
        in_validation = np.isin(matrix.groups, validation_groups)

        train_rows = np.flatnonzero(in_train)
        validation_rows = np.flatnonzero(in_validation & is_original)

        if validation_rows.size == 0:
            raise ValueError(
                "A cross-validation fold has no validation rows: every original "
                "image in the fold failed segmentation."
            )
        folds.append((train_rows, validation_rows))

    return folds
