"""T3 multiscale morphological descriptors.

The extractor measures surface geometry from a prepared greyscale fruit image:
granulometric spectra, top-hat responses, morphological roughness, and
connected-component blemish descriptors.
"""

from __future__ import annotations

from math import log2, pi, sqrt
from typing import Any, Mapping

import cv2
import numpy as np

from features.base import FeatureExtractor


FEATURE_NAMES = (
    tuple(f"gran_open_r{radius}" for radius in range(1, 8))
    + tuple(f"gran_close_r{radius}" for radius in range(1, 8))
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

DEFAULT_CONFIG = {
    "r_max": 7,
    "min_blemish_area": 15,
    "exclude_poles": False,
}


def _disk(radius: int) -> np.ndarray:
    size = 2 * int(radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _region_stats(values: np.ndarray, interior: np.ndarray) -> list[float]:
    selected = values[interior]
    if selected.size == 0:
        return [0.0] * 4
    return [
        float(np.mean(selected)),
        float(np.std(selected)),
        float(np.max(selected)),
        float(np.percentile(selected, 95)),
    ]


def _spectrum(image: np.ndarray, interior: np.ndarray, r_max: int, operation: int) -> np.ndarray:
    previous = image
    bins: list[float] = []
    previous_sum = float(np.sum(previous[interior], dtype=np.float64))
    for radius in range(1, r_max + 1):
        current = cv2.morphologyEx(image, operation, _disk(radius))
        if operation == cv2.MORPH_OPEN:
            current = np.minimum(current, previous)
        else:
            current = np.maximum(current, previous)
        current_sum = float(np.sum(current[interior], dtype=np.float64))
        difference = previous_sum - current_sum if operation == cv2.MORPH_OPEN else current_sum - previous_sum
        if difference < -1e-5:
            raise AssertionError("morphological spectrum bin is negative")
        bins.append(max(0.0, difference))
        previous = current
        previous_sum = current_sum
    spectrum = np.asarray(bins, dtype=np.float64)
    total = float(spectrum.sum())
    return spectrum / (total + 1e-8)


def _eccentricity(coords: np.ndarray) -> float:
    if len(coords) < 3:
        return 0.0
    covariance = np.cov(coords.astype(np.float64), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance))
    major, minor = float(eigenvalues[-1]), float(max(eigenvalues[0], 0.0))
    if major <= 1e-12:
        return 0.0
    return float(sqrt(max(0.0, 1.0 - minor / major)))


def _component_descriptors(
    blemish: np.ndarray,
    interior: np.ndarray,
    full_mask: np.ndarray,
    min_area: int,
    mm_per_px: float,
    exclude_poles: bool,
) -> tuple[np.ndarray, list[float], list[tuple[float, ...]]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(blemish, 8, cv2.CV_32S)
    output_labels = np.zeros_like(labels, dtype=np.int32)
    fruit_pixels = max(int(np.count_nonzero(full_mask)), 1)
    mask_centroid = np.mean(np.argwhere(full_mask), axis=0)
    candidates: list[tuple[float, ...]] = []
    areas: list[float] = []
    pixel_scale = float(mm_per_px) ** 2

    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if exclude_poles:
            row, column = centroids[component][1], centroids[component][0]
            distance = abs(float(row - mask_centroid[0])) / max(blemish.shape[0], 1)
            if distance > 0.35:
                continue
        component_mask = labels == component
        coordinates = np.argwhere(component_mask)
        contour = _largest_contour(component_mask)
        contour_area = cv2.contourArea(contour) if contour.size else 0.0
        hull = cv2.convexHull(contour) if contour.size else contour
        hull_area = cv2.contourArea(hull) if hull.size else 0.0
        x, y, width, height = stats[component, :4]
        physical_area = area * pixel_scale
        equivalent_diameter = sqrt(4.0 * physical_area / pi)
        candidates.append(
            (
                float(physical_area),
                _eccentricity(coordinates),
                float(contour_area / hull_area) if hull_area else 0.0,
                float(area / max(width * height, 1)),
                equivalent_diameter,
            )
        )
        areas.append(float(physical_area))
        output_labels[component_mask] = len(candidates)

    return output_labels, areas, candidates


def _largest_contour(component_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else np.empty((0, 1, 2), dtype=np.int32)


def _zero_result(shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    return np.zeros(36, dtype=np.float64), {
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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract the 36-dimensional T3 descriptor and diagnostic auxiliaries."""
    if image_bgr.ndim != 3 or image_bgr.shape[:2] != mask.shape:
        raise ValueError("image_bgr and mask must describe the same image frame")
    full_mask = np.asarray(mask).astype(bool)
    if not full_mask.any():
        return _zero_result(mask.shape)

    options = dict(DEFAULT_CONFIG)
    options.update(config or {})
    r_max = max(1, int(options["r_max"]))
    x, y, width, height = cv2.boundingRect(full_mask.astype(np.uint8))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)[y : y + height, x : x + width]
    cropped_mask = full_mask[y : y + height, x : x + width]
    inside_values = gray[cropped_mask]
    if inside_values.size == 0:
        return _zero_result(mask.shape)
    filled = gray.astype(np.float32)
    filled[~cropped_mask] = float(np.mean(inside_values))
    interior = cv2.erode(cropped_mask.astype(np.uint8), _disk(r_max + 1)) > 0
    if int(np.count_nonzero(interior)) < 100:
        return _zero_result(mask.shape)

    image = filled.astype(np.float32)
    opening = _spectrum(image, interior, r_max, cv2.MORPH_OPEN)
    closing = _spectrum(image, interior, r_max, cv2.MORPH_CLOSE)
    q = (opening + closing) / 2.0
    scales = np.arange(1, r_max + 1, dtype=np.float64)
    mean_size = float(np.sum(scales * q))
    entropy = float(-np.sum(q * np.log2(q + 1e-12)))
    peak_scale = float(np.argmax(q) + 1)
    spread = float(np.sqrt(np.sum((scales - mean_size) ** 2 * q)))

    structure_9 = _disk(9)
    bth = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, structure_9) / 255.0
    wth = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, structure_9) / 255.0
    mgrad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, _disk(1)) / 255.0
    response_stats = _region_stats(bth, interior) + _region_stats(wth, interior)
    response_stats += [float(np.mean(mgrad[interior])), float(np.std(mgrad[interior]))]

    bth_uint8 = np.clip(bth * 255.0, 0, 255).astype(np.uint8)
    _, blemish_crop = cv2.threshold(bth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blemish_crop = cv2.bitwise_and(blemish_crop, blemish_crop, mask=interior.astype(np.uint8) * 255)
    blemish_crop = cv2.morphologyEx(blemish_crop, cv2.MORPH_OPEN, _disk(1))
    labels_crop, areas, descriptors = _component_descriptors(
        blemish_crop,
        interior,
        full_mask,
        int(options["min_blemish_area"]),
        mm_per_px,
        bool(options["exclude_poles"]),
    )
    blemish_full = np.zeros(mask.shape, dtype=np.uint8)
    blemish_full[y : y + height, x : x + width] = np.where(labels_crop > 0, 255, 0).astype(np.uint8)
    ratio = 100.0 * float(np.count_nonzero(blemish_full)) / max(int(np.count_nonzero(full_mask)), 1)
    if descriptors:
        descriptor_array = np.asarray(descriptors, dtype=np.float64)
        blemish_stats = [
            float(len(descriptors)),
            float(np.mean(descriptor_array[:, 0])),
            float(np.max(descriptor_array[:, 0])),
            float(np.mean(descriptor_array[:, 1])),
            float(np.mean(descriptor_array[:, 2])),
            float(np.mean(descriptor_array[:, 3])),
            float(np.max(descriptor_array[:, 4])),
        ]
    else:
        blemish_stats = [0.0] * 7

    features = np.asarray(
        list(opening) + list(closing) + [mean_size, entropy, peak_scale, spread]
        + response_stats + [ratio] + blemish_stats,
        dtype=np.float64,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, {
        "blemish_mask": blemish_full,
        "blemish_ratio_pct": ratio,
        "labels": _paste_labels(labels_crop, mask.shape, x, y, width, height),
        "n_blemish": len(descriptors),
        "areas_mm2": areas,
    }


def _paste_labels(cropped: np.ndarray, shape: tuple[int, int], x: int, y: int, width: int, height: int) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    labels[y : y + height, x : x + width] = cropped
    return labels


class T3MorphologicalExtractor(FeatureExtractor):
    """Adapter exposing T3 through the shared two-argument harness interface."""

    name = "multiscale morphological descriptors"
    short_name = "T3"
    dim = 36

    def __init__(self, config: Mapping[str, Any] | None = None, mm_per_px: float = 1.0) -> None:
        self.config = dict(config or {})
        self.mm_per_px = float(mm_per_px)

    def extract_features(self, bgr_image: np.ndarray, fruit_mask: np.ndarray) -> np.ndarray:
        features, _ = extract(bgr_image, fruit_mask, self.mm_per_px, self.config)
        return features

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES


__all__ = ["FEATURE_NAMES", "T3MorphologicalExtractor", "extract"]