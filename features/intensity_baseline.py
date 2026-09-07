"""The first-order baseline E2.1 measures the GLCM descriptor against.

This is not a fourth technique. It is the comparator that makes "second-order
texture beats first-order intensity" a measured claim rather than an assumed
one, and on this dataset that claim is in genuine doubt: T2 cross-validates at
0.6235 against a background-only control of 0.7396, so it currently scores
*below* a classifier that never sees the fruit at all. E2.1 is therefore a
diagnostic as much as a requirement. Four numbers beating forty would say the
co-occurrence machinery is contributing nothing on a task whose signal is
chromatic; forty beating four would say the machinery works and the problem is
elsewhere.

The four moments
----------------

Mean, variance, skewness and kurtosis of the fruit's grey levels. They describe
the *distribution* of intensity and know nothing about arrangement: a smooth
apple and one covered in fine pitting have identical vectors here as long as
their histograms match. Every spatial fact is exactly what the GLCM adds, which
is what makes this the right thing to subtract.

Skewness is the standardised third moment ``m3 / sigma^3`` and kurtosis the
**excess** fourth moment ``m4 / sigma^4 - 3``, the conventional first-order
texture set (Gonzalez & Woods, 2018, Sec. 11.3). Note this differs from the
cube-root skewness :mod:`features.colour_histogram` uses: that follows Stricker
and Orengo's colour-moments convention, which keeps the statistic in the units
of the channel. Two baselines, two literatures, and each follows its own.

Pixel selection
---------------

The moments are taken over exactly the pixels the GLCM counts pairs over: the
fruit mask, with background excluded rather than zero-filled. Zero-filling is
the trap T2's docstring describes at length - a zeroed background becomes an
enormous uniform grey level that dominates every statistic - and it would
distort the moments just as thoroughly as it distorts a co-occurrence matrix.

References
----------
Gonzalez, R. C., & Woods, R. E. (2018). Digital image processing (4th ed.).
    Pearson.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from features.base import FeatureExtractor, FeatureExtractionError, require_non_empty_mask

#: Names of the four moments, in emitted order.
MOMENT_NAMES: Tuple[str, ...] = ("mean", "variance", "skewness", "kurtosis")


@dataclass
class IntensityDiagnostics:
    """Per-image record of what the baseline measured.

    Attributes:
        mask_pixels: Fruit pixels the moments were taken over.
        constant: True when every one of those pixels shares one grey level,
            so the distribution has no shape and the two shape moments are
            reported as zero.
    """

    mask_pixels: int
    constant: bool


def intensity_moments(values: np.ndarray) -> Tuple[float, float, float, float]:
    """Return the mean, variance, skewness and excess kurtosis of ``values``.

    Args:
        values: Grey levels of the pixels being described.

    Returns:
        ``(mean, variance, skewness, kurtosis)``.

        A constant region has zero variance, and both shape moments are then
        ``0/0``. They are reported as 0.0 rather than NaN: a flat region is a
        real input - a fruit photographed against a flat backdrop can segment
        to one - and a NaN would be trained on silently by every configuration
        downstream.
    """
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    mean = float(np.mean(values))
    deviation = values - mean
    variance = float(np.mean(deviation ** 2))
    if variance <= 0.0:
        return mean, 0.0, 0.0, 0.0

    sigma = float(np.sqrt(variance))
    skewness = float(np.mean(deviation ** 3) / sigma ** 3)
    kurtosis = float(np.mean(deviation ** 4) / sigma ** 4 - 3.0)
    return mean, variance, skewness, kurtosis


class IntensityMomentExtractor(FeatureExtractor):
    """First-order grey-level statistics over the segmented fruit.

    Deterministic by construction: no sampling, no clustering, no randomness of
    any kind, so repeated calls on one image return byte-identical vectors.
    """

    name = "First-order intensity moments"
    short_name = "T2I"

    def __init__(self, seed: int = 42) -> None:
        """Configure the baseline.

        Args:
            seed: Accepted for interface symmetry with the other extractors;
                this descriptor is deterministic and does not use it.
        """
        self.seed = int(seed)

        #: One :class:`IntensityDiagnostics` per call, in call order.
        self.diagnostics: List[IntensityDiagnostics] = []

    @property
    def dim(self) -> int:
        """Length of the emitted vector: four moments."""
        return len(MOMENT_NAMES)

    @property
    def feature_names(self) -> Sequence[str]:
        """Human-readable name for every dimension, in emitted order."""
        return [f"T2I_{name}" for name in MOMENT_NAMES]

    def reset_diagnostics(self) -> None:
        """Discard the accumulated per-image diagnostics."""
        self.diagnostics.clear()

    def extract_features(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> np.ndarray:
        """Describe one segmented fruit by its grey-level moments."""
        vector, _ = self.extract_with_diagnostics(bgr_image, fruit_mask)
        return vector

    def extract_with_diagnostics(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> Tuple[np.ndarray, IntensityDiagnostics]:
        """Extract the four moments and report what they were taken over.

        Args:
            bgr_image: A preprocessed ``uint8`` BGR image.
            fruit_mask: The fruit mask, 255 inside the fruit.

        Returns:
            The feature vector and an :class:`IntensityDiagnostics`.

        Raises:
            FeatureExtractionError: If the mask is empty.
        """
        mask = require_non_empty_mask(fruit_mask, self.short_name)
        grey = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        values = grey[mask].astype(np.float64)

        moments = intensity_moments(values)
        record = IntensityDiagnostics(
            mask_pixels=int(values.size),
            constant=bool(moments[1] == 0.0),
        )
        self.diagnostics.append(record)

        vector = np.asarray(moments, dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            raise FeatureExtractionError(
                f"{self.short_name} produced a non-finite moment; the constant-region "
                f"guard did not cover this case"
            )
        return vector, record

    @classmethod
    def from_config(cls, config: Any) -> "IntensityMomentExtractor":
        """Build the baseline from the shared seed.

        It reads nothing from the ``t2_glcm`` block, and deliberately so: it
        has no distance, no angle and no quantisation, because not having them
        is what it is for.
        """
        return cls(seed=int(getattr(config, "seed", 42)))
