"""First-order intensity baseline, the comparator for E2.1.

This is not a fourth technique. It exists so that the GLCM descriptor can be
measured against the question it is supposed to answer better: does texture
need *spatial* information at all, or would the grey-level distribution on its
own have done as well?

A GLCM counts how often two grey levels occur a fixed distance apart, so every
number it produces depends on where pixels sit relative to one another. These
four moments depend only on how many pixels hold each value. Shuffle the peel
pixels at random and this descriptor is unchanged while every GLCM property
moves. That is the whole difference between them, and it is what E2.1 prices.

The comparison matters here more than it usually would. T2 scores below the
background-only control on this dataset, so the sweep has to establish whether
the technique is misconfigured or whether texture is simply the wrong evidence
for a chromatic task. If the spatial descriptor cannot beat four moments, the
answer is the latter.

Feature layout, 4 dimensions: mean, variance, skewness, kurtosis of the masked
grey channel.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from features.base import FeatureExtractor, require_non_empty_mask


class IntensityMomentExtractor(FeatureExtractor):
    """Mean, variance, skewness and kurtosis of the masked grey channel."""

    short_name = "E2_MOM"
    name = "first-order intensity moments"
    dim = 4

    def __init__(self, seed: int = 42) -> None:
        """Configure the baseline.

        Args:
            seed: Accepted for interface symmetry. Nothing here is random.
        """
        self.seed = int(seed)

    @property
    def feature_names(self) -> Sequence[str]:
        """Names in vector order."""
        return [
            f"{self.short_name}_mean",
            f"{self.short_name}_variance",
            f"{self.short_name}_skewness",
            f"{self.short_name}_kurtosis",
        ]

    def extract_features(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> np.ndarray:
        """Describe the masked peel by the shape of its intensity histogram.

        Skewness and kurtosis are undefined on a surface of uniform intensity,
        which a heavily blown-out highlight can produce. Both fall back to
        zero rather than to a NaN: a single non-finite value would be scaled
        into every other row by the standardiser and take the whole matrix
        with it.
        """
        mask = require_non_empty_mask(fruit_mask, self.short_name)
        grey = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)[mask].astype(np.float64)

        mean = float(grey.mean())
        variance = float(grey.var())
        spread = float(np.sqrt(variance))

        if spread < 1e-12 or grey.size < 2:
            skewness = 0.0
            kurtosis = 0.0
        else:
            centred = grey - mean
            skewness = float(np.mean(centred**3) / spread**3)
            # Excess kurtosis, so a normal distribution reads 0 rather than 3.
            kurtosis = float(np.mean(centred**4) / spread**4 - 3.0)

        return np.array([mean, variance, skewness, kurtosis], dtype=np.float64)

    @classmethod
    def from_config(cls, config: Any) -> "IntensityMomentExtractor":
        """Build from the shared seed. There is nothing else to configure."""
        return cls(seed=int(getattr(config, "seed", 42)))


__all__ = ["IntensityMomentExtractor"]
