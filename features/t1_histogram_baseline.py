"""Colour-histogram baseline, the comparator for E1.1.

This is not a fourth technique. It exists so that the MPEG-7 dominant colour
descriptor can be measured against the obvious alternative: describe the peel
by the distribution of its colours rather than by a small set of representative
ones. Both read exactly the same pixels in the same colour space through the
same mask, so the comparison isolates the representation and nothing else.

The two differ in what they discard. A histogram fixes its bins in advance and
keeps how much of the fruit falls in each, so it records the shape of the
distribution but not where within a bin the colours actually sat. The dominant
colour descriptor places its bins by clustering, so it records the colours
precisely but keeps only four of them. Which loss matters more is the question
E1.1 asks.

Feature layout, 105 dimensions at the default 32 bins:

============ ==================================================
0 to 95      32-bin normalised histogram per channel, in order
96 to 104    mean, standard deviation and skewness per channel
============ ==================================================
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from features.base import FeatureExtractor, require_non_empty_mask

#: Channel ranges for each supported space, in the encoding OpenCV returns.
#: Lab and RGB are 0 to 255 on every channel; HSV hue is 0 to 179 because
#: OpenCV halves the degree scale to fit a byte.
CHANNEL_RANGES = {
    "LAB": ((0.0, 255.0), (0.0, 255.0), (0.0, 255.0)),
    "RGB": ((0.0, 255.0), (0.0, 255.0), (0.0, 255.0)),
    "HSV": ((0.0, 179.0), (0.0, 255.0), (0.0, 255.0)),
}

CONVERSIONS = {
    "LAB": cv2.COLOR_BGR2LAB,
    "HSV": cv2.COLOR_BGR2HSV,
    "RGB": cv2.COLOR_BGR2RGB,
}


def skewness(values: np.ndarray) -> float:
    """Fisher-Pearson skewness of one channel, zero when it cannot be defined.

    A channel with no spread has no skew to report. Returning zero rather than
    a NaN keeps the contract that a feature vector is always finite: a NaN here
    would be scaled into every other row by the standardiser and take the whole
    matrix with it.
    """
    if values.size < 2:
        return 0.0
    centred = values - values.mean()
    spread = float(np.sqrt(np.mean(centred**2)))
    if spread < 1e-12:
        return 0.0
    return float(np.mean(centred**3) / (spread**3))


class ColourHistogramExtractor(FeatureExtractor):
    """Per-channel colour histogram and moments over the fruit mask."""

    short_name = "E1_HIST"
    name = "colour histogram baseline"

    def __init__(
        self,
        bins: int = 32,
        space: str = "LAB",
        seed: int = 42,
    ) -> None:
        """Configure the baseline.

        Args:
            bins: Bins per channel. 32 matches the grey-level quantisation T2
                uses, so the two baselines are coarse to a comparable degree.
            space: Colour space, one of ``LAB``, ``HSV`` or ``RGB``. Should
                match the space the dominant colour descriptor is clustering
                in, or E1.1 measures two differences at once.
            seed: Accepted for interface symmetry. Nothing here is random.
        """
        if space not in CONVERSIONS:
            raise ValueError(f"space must be one of {sorted(CONVERSIONS)}, got {space!r}")
        if bins < 2:
            raise ValueError(f"bins must be at least 2, got {bins}")
        self.bins = int(bins)
        self.space = space
        self.seed = int(seed)

    @property
    def dim(self) -> int:
        """Three histograms plus three moments per channel."""
        return 3 * self.bins + 9

    @property
    def feature_names(self) -> Sequence[str]:
        """Names in vector order."""
        channels = ("c0", "c1", "c2")
        names = [
            f"{self.short_name}_{channel}_bin{index:02d}"
            for channel in channels
            for index in range(self.bins)
        ]
        names.extend(
            f"{self.short_name}_{channel}_{moment}"
            for channel in channels
            for moment in ("mean", "std", "skew")
        )
        return names

    def extract_features(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> np.ndarray:
        """Describe the masked peel by its colour distribution."""
        mask = require_non_empty_mask(fruit_mask, self.short_name)
        converted = cv2.cvtColor(bgr_image, CONVERSIONS[self.space])
        pixels = converted[mask].astype(np.float64)

        histograms: list[np.ndarray] = []
        moments: list[float] = []
        for channel in range(3):
            values = pixels[:, channel]
            low, high = CHANNEL_RANGES[self.space][channel]
            counts, _ = np.histogram(values, bins=self.bins, range=(low, high))
            total = float(counts.sum())
            # Normalised so the descriptor reports the shape of the
            # distribution rather than the size of the fruit. Without this a
            # nearer apple would read as a different colour.
            histograms.append(counts / total if total > 0 else np.zeros(self.bins))
            moments.extend([float(values.mean()), float(values.std()), skewness(values)])

        return np.concatenate([np.concatenate(histograms), np.asarray(moments)])

    @classmethod
    def from_config(cls, config: Any) -> "ColourHistogramExtractor":
        """Build from the ``t1_colour`` block, so the space matches T1's."""
        block: Mapping[str, Any] = getattr(config, "t1_colour", {}) or {}
        return cls(
            bins=int(block.get("histogram_bins", 32)),
            space=str(block.get("space", "LAB")),
            seed=int(getattr(config, "seed", 42)),
        )


__all__ = ["CHANNEL_RANGES", "ColourHistogramExtractor", "skewness"]
