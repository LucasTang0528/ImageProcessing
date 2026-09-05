"""T3 multiscale morphological descriptors.

The extractor measures surface *geometry* from a prepared greyscale fruit
image, and nothing else. It reads neither colour nor raw intensity statistics,
so it stays disjoint from T1 (dominant colour) and T2 (GLCM) and the three
techniques can be attributed separately in the Mode A comparison.

Five blocks are emitted, in this fixed order:

============ ==================================================================
A  0..13     Granulometric pattern spectrum, openings then closings
B  14..17    Moments of the averaged spectrum
C  18..25    Black and white top-hat response statistics
D  26..27    Morphological gradient roughness
E  28..35    Connected-component blemish geometry
============ ==================================================================

Two boundary guards run before any statistic is accumulated, and between them
they are the difference between measuring the peel and measuring the
silhouette: the background is filled with the mean intensity *inside* the mask
rather than left at zero, and every statistic is accumulated over an eroded
interior mask. Without them the largest opening and closing residues sit on the
fruit outline, which is a property of the segmenter rather than of the fruit.

References
----------
Bai, X., Zhou, F., & Xue, B. (2012). Image enhancement using multi scale image
    features extracted by top hat transform. Optics & Laser Technology, 44(2),
    328-336.
Maragos, P. (1989). Pattern spectrum and multiscale shape representation.
    IEEE TPAMI, 11(7), 701-716.
Matheron, G. (1975). Random sets and integral geometry. Wiley.
Moallem, P., Serajoddin, A., & Pourghassem, H. (2017). Computer vision-based
    apple grading for golden delicious apples based on surface features.
    Information Processing in Agriculture, 4(1), 33-40.
Otsu, N. (1979). A threshold selection method from gray level histograms.
    IEEE SMC, 9(1), 62-66.
Serra, J. (1982). Image analysis and mathematical morphology. Academic Press.
Soille, P. (2004). Morphological image analysis (2nd ed.). Springer.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Any, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from skimage.measure import label as sk_label
from skimage.measure import regionprops

from features.base import FeatureExtractor, require_non_empty_mask


#: Structuring element shapes the study is allowed to sweep over (E3.2).
SE_SHAPES = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}

#: Blemish segmentation methods compared in E3.4.
BLEMISH_METHODS = ("bth_otsu", "fixed", "hue_deviation")

#: Dimensions contributed by blocks B to E, i.e. everything but the spectrum.
FIXED_DIMS = 22

DEFAULT_CONFIG: Mapping[str, Any] = {
    "r_max": 7,
    "se_shape": "ellipse",
    "tophat_radius": 9,
    "min_blemish_area": 15,
    "blemish_method": "bth_otsu",
    "fixed_threshold": 30,
    "exclude_poles": False,
    "pole_fraction": 0.15,
    # Comparator only, for E3.4. These are the thresholds the dropped LBP-era
    # design used, kept so the report can quantify what dropping it cost.
    "hue_deviation": 20.0,
    "saturation_deviation": 60.0,
    "value_deviation": 60.0,
}


def feature_names(r_max: int = 7) -> Tuple[str, ...]:
    """Names for every emitted dimension, in emitted order.

    The spectrum length follows ``r_max``, so E3.3 can sweep the radius and
    still label its columns honestly rather than padding to 36.
    """
    return (
        tuple(f"gran_open_r{radius}" for radius in range(1, r_max + 1))
        + tuple(f"gran_close_r{radius}" for radius in range(1, r_max + 1))
        + ("gran_mean_size", "gran_entropy", "gran_peak_scale", "gran_spread")
        + ("bth_mean", "bth_std", "bth_max", "bth_p95")
        + ("wth_mean", "wth_std", "wth_max", "wth_p95")
        + ("mgrad_mean", "mgrad_std")
        + (
            "blemish_ratio",
            "blemish_count",
            "blemish_area_mean",
            "blemish_area_max",
            "blemish_ecc_mean",
            "blemish_solidity_mean",
            "blemish_extent_mean",
            "blemish_eqdiam_max",
        )
    )


#: The configured 36-dimensional descriptor, for callers that do not sweep.
FEATURE_NAMES: Tuple[str, ...] = feature_names(int(DEFAULT_CONFIG["r_max"]))


def dimension_for(r_max: int = 7) -> int:
    """Length of the vector emitted at a given maximum granulometric radius."""
    return 2 * int(r_max) + FIXED_DIMS


def block_indices(r_max: int = 7) -> dict:
    """Column indices of each named block, for the E3.1 ablation.

    The blocks are computed independently of one another, so selecting columns
    after a full extraction yields exactly the vector a block-restricted
    extractor would have produced, at a third of the cost.
    """
    spectrum = 2 * int(r_max)
    return {
        "A": np.arange(0, spectrum),
        "B": np.arange(spectrum, spectrum + 4),
        "C": np.arange(spectrum + 4, spectrum + 12),
        "D": np.arange(spectrum + 12, spectrum + 14),
        "E": np.arange(spectrum + 14, spectrum + 22),
    }


# --------------------------------------------------------------------------- #
# Structuring elements
# --------------------------------------------------------------------------- #

def _disk(radius: int, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    """A structuring element of the given *radius*, i.e. of size ``2r + 1``.

    ``getStructuringElement`` takes a size rather than a radius, and confusing
    the two halves every scale in the spectrum.
    """
    size = 2 * int(radius) + 1
    return cv2.getStructuringElement(shape, (size, size))


def _resolve_shape(name: str) -> int:
    try:
        return SE_SHAPES[str(name).lower()]
    except KeyError:
        raise ValueError(
            f"se_shape must be one of {sorted(SE_SHAPES)}, got {name!r}"
        ) from None


def _resolve_method(name: str) -> str:
    method = str(name).lower()
    if method not in BLEMISH_METHODS:
        raise ValueError(
            f"blemish_method must be one of {list(BLEMISH_METHODS)}, got {name!r}"
        )
    return method


# --------------------------------------------------------------------------- #
# Blocks A and B: the granulometric pattern spectrum
# --------------------------------------------------------------------------- #

def _spectrum(
    image: np.ndarray,
    interior: np.ndarray,
    r_max: int,
    operation: int,
    shape: int,
) -> np.ndarray:
    """One normalised pattern spectrum (Maragos, 1989).

    Each bin holds the intensity mass removed by growing the structuring
    element one step, summed over the interior mask only. Openings are
    anti-extensive and closings extensive, so in exact arithmetic every bin is
    non-negative. The sieve is clamped anyway: ``getStructuringElement`` does
    not guarantee that the disc of radius r is contained in the disc of radius
    r+1 at every discrete radius, and an unclamped sieve can then emit a small
    negative bin that is an artefact of the element rather than of the image.
    """
    previous = image
    previous_sum = float(np.sum(previous[interior], dtype=np.float64))
    bins: List[float] = []

    for radius in range(1, r_max + 1):
        current = cv2.morphologyEx(image, operation, _disk(radius, shape))
        if operation == cv2.MORPH_OPEN:
            current = np.minimum(current, previous)
            current_sum = float(np.sum(current[interior], dtype=np.float64))
            difference = previous_sum - current_sum
        else:
            current = np.maximum(current, previous)
            current_sum = float(np.sum(current[interior], dtype=np.float64))
            difference = current_sum - previous_sum

        if difference < -1e-5:
            raise AssertionError(
                "morphological spectrum bin is negative; the interior mask is wrong"
            )
        bins.append(max(0.0, difference))
        previous = current
        previous_sum = current_sum

    spectrum = np.asarray(bins, dtype=np.float64)
    return spectrum / (float(spectrum.sum()) + 1e-8)


def _spectrum_moments(q: np.ndarray, r_max: int) -> List[float]:
    """Mean size, entropy, peak scale and spread of the averaged spectrum."""
    scales = np.arange(1, r_max + 1, dtype=np.float64)
    mean_size = float(np.sum(scales * q))
    return [
        mean_size,
        float(-np.sum(q * np.log2(q + 1e-12))),
        float(np.argmax(q) + 1),
        float(np.sqrt(np.sum((scales - mean_size) ** 2 * q))),
    ]


# --------------------------------------------------------------------------- #
# Blocks C and D: top-hat responses and roughness
# --------------------------------------------------------------------------- #

def _region_stats(values: np.ndarray, interior: np.ndarray) -> List[float]:
    """Mean, standard deviation, maximum and 95th percentile over the interior."""
    selected = values[interior]
    if selected.size == 0:
        return [0.0] * 4
    return [
        float(np.mean(selected)),
        float(np.std(selected)),
        float(np.max(selected)),
        float(np.percentile(selected, 95)),
    ]


# --------------------------------------------------------------------------- #
# Block E: blemish region analysis
# --------------------------------------------------------------------------- #

def _major_axis(cropped_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Centroid, unit major axis and half-extent of the fruit mask.

    All three are expressed in the **cropped** frame, which is the frame the
    component centroids live in. Mixing the two frames silently excludes the
    wrong components.
    """
    coordinates = np.argwhere(cropped_mask).astype(np.float64)
    centre = coordinates.mean(axis=0)
    if len(coordinates) < 3:
        return centre, np.array([1.0, 0.0]), 0.0

    centred = coordinates - centre
    covariance = np.cov(centred, rowvar=False)
    _, eigenvectors = np.linalg.eigh(np.atleast_2d(covariance))
    major = eigenvectors[:, -1]
    projections = centred @ major
    return centre, major, float(np.abs(projections).max())


def _is_polar(
    centroid: Sequence[float],
    centre: np.ndarray,
    major: np.ndarray,
    extent: float,
    fraction: float,
) -> bool:
    """Whether a component sits in the stem or calyx band at either pole.

    Moallem et al. (2017) note that the stem cavity and the calyx are dark
    concavities, so a defect detector keyed on darkness counts them as
    blemishes on every apple regardless of its condition. The band is measured
    along the fruit's own major axis rather than along the image axes, because
    apples are not photographed upright.
    """
    if extent <= 0.0:
        return False
    offset = np.asarray(centroid, dtype=np.float64) - centre
    return abs(float(offset @ major)) > (1.0 - fraction) * extent


def _hue_deviation_mask(
    image_bgr_crop: np.ndarray,
    cropped_mask: np.ndarray,
    options: Mapping[str, Any],
) -> np.ndarray:
    """The dropped colour-based blemish detector, kept as an E3.4 comparator.

    A pixel counts as a blemish when it departs far enough from the fruit's own
    median HSV. This is precisely why it was dropped from the technique: it
    reads the same colour evidence T1 is built on, so a comparison that used it
    could not attribute a result to morphology.
    """
    inside = cropped_mask.astype(bool)
    if not inside.any():
        return np.zeros(cropped_mask.shape, dtype=np.uint8)

    hsv = cv2.cvtColor(image_bgr_crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = (
        hsv[..., channel].astype(np.float32) for channel in range(3)
    )
    median_hue = float(np.median(hue[inside]))
    median_saturation = float(np.median(saturation[inside]))
    median_value = float(np.median(value[inside]))

    # Hue is circular on 0..179 in OpenCV, so the shorter way round is taken.
    hue_gap = np.abs(hue - median_hue)
    hue_gap = np.minimum(hue_gap, 180.0 - hue_gap)

    deviant = (
        (hue_gap > float(options["hue_deviation"]))
        | (np.abs(saturation - median_saturation) > float(options["saturation_deviation"]))
        | (np.abs(value - median_value) > float(options["value_deviation"]))
    )
    return (deviant & inside).astype(np.uint8) * 255


def _segment_blemishes(
    bth: np.ndarray,
    image_bgr_crop: np.ndarray,
    cropped_mask: np.ndarray,
    interior: np.ndarray,
    options: Mapping[str, Any],
    shape: int,
) -> np.ndarray:
    """Binary blemish mask in the cropped frame, by the configured method.

    ``bth_otsu`` is the technique's own method. The other two exist so that
    E3.4 can quantify the choice rather than assert it, and ``hue_deviation``
    in particular is the method this design replaced: reproducing it here is
    what lets the report justify the replacement with a measurement.
    """
    method = _resolve_method(options["blemish_method"])

    if method == "hue_deviation":
        binary = _hue_deviation_mask(image_bgr_crop, cropped_mask, options)
    else:
        magnitude = np.clip(bth * 255.0, 0, 255).astype(np.uint8)
        if method == "bth_otsu":
            _, binary = cv2.threshold(
                magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(
                magnitude, int(options["fixed_threshold"]), 255, cv2.THRESH_BINARY
            )

    binary = cv2.bitwise_and(binary, interior.astype(np.uint8) * 255)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, _disk(1, shape))


def _describe_components(
    binary: np.ndarray,
    cropped_mask: np.ndarray,
    options: Mapping[str, Any],
    mm_per_px: float,
) -> Tuple[np.ndarray, List[float], List[Tuple[float, ...]]]:
    """Label the blemish mask and describe every surviving component.

    Returns the relabelled component image (1..n in survival order), the areas
    in reporting units, and one descriptor tuple per surviving component.
    """
    labelled = sk_label(binary > 0, connectivity=2)
    output = np.zeros(labelled.shape, dtype=np.int32)
    if int(labelled.max()) == 0:
        return output, [], []

    minimum_area = int(options["min_blemish_area"])
    exclude_poles = bool(options["exclude_poles"])
    centre, major, extent = _major_axis(cropped_mask)
    area_scale = float(mm_per_px) ** 2

    areas: List[float] = []
    descriptors: List[Tuple[float, ...]] = []

    for region in regionprops(labelled):
        if region.area < minimum_area:
            continue
        if exclude_poles and _is_polar(
            region.centroid, centre, major, extent, float(options["pole_fraction"])
        ):
            continue

        physical_area = float(region.area) * area_scale
        descriptors.append(
            (
                physical_area,
                float(region.eccentricity),
                float(region.solidity),
                float(region.extent),
                sqrt(4.0 * physical_area / pi),
            )
        )
        areas.append(physical_area)
        output[labelled == region.label] = len(descriptors)

    return output, areas, descriptors


def _blemish_block(descriptors: Sequence[Tuple[float, ...]]) -> List[float]:
    """The seven descriptor statistics that follow ``blemish_ratio``.

    A perfectly clean apple is a real case rather than an error - the Unripe
    class has many - so an empty component list yields zeros, never NaN.
    """
    if not descriptors:
        return [0.0] * 7
    table = np.asarray(descriptors, dtype=np.float64)
    return [
        float(len(descriptors)),
        float(np.mean(table[:, 0])),
        float(np.max(table[:, 0])),
        float(np.mean(table[:, 1])),
        float(np.mean(table[:, 2])),
        float(np.mean(table[:, 3])),
        float(np.max(table[:, 4])),
    ]


# --------------------------------------------------------------------------- #
# The extractor
# --------------------------------------------------------------------------- #

def resolve_config(config: Mapping[str, Any] | None = None) -> dict:
    """Merge a partial configuration over the documented defaults."""
    options = dict(DEFAULT_CONFIG)
    options.update(config or {})
    options["r_max"] = max(1, int(options["r_max"]))
    return options


def _zero_result(shape: Tuple[int, int], r_max: int) -> Tuple[np.ndarray, dict]:
    """The documented degenerate return: a zero vector and a flag, never a raise."""
    return np.zeros(dimension_for(r_max), dtype=np.float64), {
        "blemish_mask": np.zeros(shape, dtype=np.uint8),
        "blemish_ratio_pct": 0.0,
        "labels": np.zeros(shape, dtype=np.int32),
        "n_blemish": 0,
        "areas_mm2": [],
        "degenerate": True,
    }


def extract(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    mm_per_px: float = 1.0,
    config: Mapping[str, Any] | None = None,
) -> Tuple[np.ndarray, dict]:
    """Extract the multiscale morphological descriptor for one segmented fruit.

    Args:
        image_bgr: Preprocessed ``(H, W, 3)`` ``uint8`` BGR image.
        mask: ``(H, W)`` fruit mask, non-zero inside the fruit.
        mm_per_px: Millimetres per pixel from the calibration stage. At the
            default of 1.0 no calibration is available and every area is in
            **pixels**; saying so is the caller's responsibility.
        config: Overrides for :data:`DEFAULT_CONFIG`.

    Returns:
        ``(features, aux)``. ``features`` has length ``2 * r_max + 22`` (36 at
        the default radius), is ``float64`` and free of NaN and Inf. ``aux``
        carries the blemish mask in the **full image frame**, along with the
        component labels, the surviving component count and their areas.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[:2] != mask.shape:
        raise ValueError("image_bgr and mask must describe the same image frame")

    options = resolve_config(config)
    r_max = options["r_max"]
    shape = _resolve_shape(options["se_shape"])
    _resolve_method(options["blemish_method"])

    full_mask = np.asarray(mask).astype(bool)
    if not full_mask.any():
        return _zero_result(mask.shape, r_max)

    # ---- Section 6: region preparation ---------------------------------- #
    x, y, width, height = cv2.boundingRect(full_mask.astype(np.uint8))
    window = (slice(y, y + height), slice(x, x + width))
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)[window]
    cropped_mask = full_mask[window]

    inside = grey[cropped_mask]
    if inside.size == 0:
        return _zero_result(mask.shape, r_max)

    # A zero background would create a cliff at the fruit boundary that every
    # opening and closing responds to in preference to the peel itself.
    image = grey.astype(np.float32)
    image[~cropped_mask] = float(np.mean(inside))

    interior = cv2.erode(cropped_mask.astype(np.uint8), _disk(r_max + 1, shape)) > 0
    if int(np.count_nonzero(interior)) < 100:
        return _zero_result(mask.shape, r_max)

    # ---- Blocks A and B ------------------------------------------------- #
    opening = _spectrum(image, interior, r_max, cv2.MORPH_OPEN, shape)
    closing = _spectrum(image, interior, r_max, cv2.MORPH_CLOSE, shape)
    moments = _spectrum_moments((opening + closing) / 2.0, r_max)

    # ---- Blocks C and D ------------------------------------------------- #
    # float32 throughout: closing(f) - f wraps around on uint8.
    tophat = _disk(int(options["tophat_radius"]), shape)
    bth = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, tophat) / 255.0
    wth = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, tophat) / 255.0
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, _disk(1, shape)) / 255.0
    responses = (
        _region_stats(bth, interior)
        + _region_stats(wth, interior)
        + [float(np.mean(gradient[interior])), float(np.std(gradient[interior]))]
    )

    # ---- Block E -------------------------------------------------------- #
    binary = _segment_blemishes(
        bth, image_bgr[window], cropped_mask, interior, options, shape
    )
    labels_crop, areas, descriptors = _describe_components(
        binary, cropped_mask, options, mm_per_px
    )

    blemish_full = np.zeros(mask.shape, dtype=np.uint8)
    blemish_full[window] = np.where(labels_crop > 0, 255, 0).astype(np.uint8)
    labels_full = np.zeros(mask.shape, dtype=np.int32)
    labels_full[window] = labels_crop

    # Detection runs on the interior mask, but the published ratio is defined
    # against the whole fruit. The two denominators are deliberately different.
    ratio = 100.0 * float(np.count_nonzero(blemish_full)) / float(np.count_nonzero(full_mask))

    features = np.asarray(
        list(opening) + list(closing) + moments + responses
        + [ratio] + _blemish_block(descriptors),
        dtype=np.float64,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, {
        "blemish_mask": blemish_full,
        "blemish_ratio_pct": ratio,
        "labels": labels_full,
        "n_blemish": len(descriptors),
        "areas_mm2": areas,
        "degenerate": False,
    }


@dataclass
class T3Diagnostics:
    """What one ``extract`` call did, for the degenerate-mask audit."""

    degenerate: bool
    n_blemish: int
    blemish_ratio_pct: float


class T3MorphologicalExtractor(FeatureExtractor):
    """Adapter exposing T3 through the shared two-argument harness interface.

    The harness hands every technique exactly an image and a mask, so the
    calibration factor and the experiment configuration are bound here rather
    than travelling through the shared call signature.
    """

    name = "multiscale morphological descriptors"
    short_name = "T3"

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        mm_per_px: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.config = resolve_config(config)
        # Fail on a bad configuration now, not once per image mid-run.
        _resolve_shape(self.config["se_shape"])
        _resolve_method(self.config["blemish_method"])
        self.mm_per_px = float(mm_per_px)
        # Nothing in this technique is random. The seed is carried only so that
        # the logged configuration records the run the results belong to.
        self.seed = int(seed)
        #: One :class:`T3Diagnostics` per call, in call order.
        self.diagnostics: List[T3Diagnostics] = []

    @property
    def dim(self) -> int:
        """Length of the emitted vector, fixed by ``r_max``."""
        return dimension_for(self.config["r_max"])

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Human-readable name for every dimension, in emitted order."""
        return feature_names(self.config["r_max"])

    def reset_diagnostics(self) -> None:
        """Discard the accumulated per-image diagnostics."""
        self.diagnostics.clear()

    def extract_features(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> np.ndarray:
        require_non_empty_mask(fruit_mask, technique=self.short_name)
        features, aux = extract(bgr_image, fruit_mask, self.mm_per_px, self.config)
        self.diagnostics.append(
            T3Diagnostics(
                degenerate=bool(aux.get("degenerate", False)),
                n_blemish=int(aux["n_blemish"]),
                blemish_ratio_pct=float(aux["blemish_ratio_pct"]),
            )
        )
        return features

    def extract_with_aux(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Extract, returning the auxiliary dictionary alongside the vector.

        Enhancement 2 and the dashboard consume ``aux["blemish_mask"]``, which
        the shared two-argument interface has no channel for.
        """
        return extract(bgr_image, fruit_mask, self.mm_per_px, self.config)

    @classmethod
    def from_config(cls, config: Any) -> "T3MorphologicalExtractor":
        """Build an extractor from the ``t3_morphological`` block of ``config.json``."""
        block: Mapping[str, Any] = getattr(config, "t3_morphological", {}) or {}
        return cls(
            config=block,
            mm_per_px=float(block.get("mm_per_px", 1.0)),
            seed=int(getattr(config, "seed", 42)),
        )


__all__ = [
    "BLEMISH_METHODS",
    "DEFAULT_CONFIG",
    "FEATURE_NAMES",
    "SE_SHAPES",
    "T3Diagnostics",
    "T3MorphologicalExtractor",
    "block_indices",
    "dimension_for",
    "extract",
    "feature_names",
    "resolve_config",
]
