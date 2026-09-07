"""The conventional colour baseline E1.1 measures the MPEG-7 descriptor against.

This is not a fourth technique. It exists so that the claim "a dominant colour
descriptor beats a colour histogram" is a measured result rather than an
assumption, and it is therefore built to be the *strongest fair* version of the
thing it stands in for, not a straw man:

* a **32-bin histogram per channel** (96 dimensions), the standard fixed-bin
  quantisation the descriptor is meant to improve on;
* **colour moments per channel** - mean, standard deviation and skewness
  (9 dimensions), following Stricker & Orengo (1995), who showed that the low
  order moments of a channel's distribution carry most of what a histogram
  carries at a fraction of its length.

105 dimensions in total, against the descriptor's 37.

What makes the comparison fair, and what makes it unfair
--------------------------------------------------------

Both arms are handed the same preprocessed image and the same fruit mask, are
computed in the same colour space, and apply the same specular exclusion with
the same fallback guard - so the only thing that differs between them is how
the surviving pixels are summarised. That is the point of the experiment.

What the baseline structurally cannot do is the interesting half of the result:

* A histogram bin is fixed in advance, so most of the 96 bins describe colours
  no apple contains. The descriptor spends its whole budget on colours the
  fruit actually has.
* Neither the histogram nor the moments know *where* a colour sits. One
  contiguous brown lesion and the same quantity of scattered speckling produce
  byte-identical vectors here; the descriptor separates them in its block B.
* In HSV the hue channel is an angle, and both the bin edges and the moments
  treat it as a line. A fruit with hues at 179 and 1 lands in opposite end
  bins with a mean in the middle. This is left uncorrected on purpose: it is
  what a conventional histogram baseline does, and correcting it here would
  quietly import one of the descriptor's ideas into the arm meant to lack it.

The extractor is deterministic - no sampling, no clustering - so repeated calls
on one image return byte-identical vectors.

References
----------
Stricker, M., & Orengo, M. (1995). Similarity of color images. Storage and
    Retrieval for Image and Video Databases III, SPIE 2420, 381 to 392.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from features.base import FeatureExtractor, FeatureExtractionError, require_non_empty_mask
from features.t1_dominant_colour import SUPPORTED_SPACES, specular_selector

#: Per-channel value ranges the histogram bins span, by colour space. OpenCV
#: stores 8-bit hue on ``[0, 180)`` to fit 360 degrees in a byte; every other
#: channel in every supported space occupies the full ``[0, 256)``.
CHANNEL_RANGES: Mapping[str, Tuple[Tuple[float, float], ...]] = {
    "LAB": ((0.0, 256.0), (0.0, 256.0), (0.0, 256.0)),
    "HSV": ((0.0, 180.0), (0.0, 256.0), (0.0, 256.0)),
    "RGB": ((0.0, 256.0), (0.0, 256.0), (0.0, 256.0)),
}

#: Channel letters per space, used only to name the dimensions.
CHANNEL_NAMES: Mapping[str, Tuple[str, str, str]] = {
    "LAB": ("L", "a", "b"),
    "HSV": ("H", "S", "V"),
    "RGB": ("R", "G", "B"),
}

#: Fewest surviving pixels for specular exclusion to be allowed to stand. The
#: same floor the descriptor applies, so neither arm can be starved by the
#: highlight guard while the other is not.
MIN_ANALYSIS_PIXELS = 32


@dataclass
class ColourHistogramDiagnostics:
    """Per-image record of what the baseline discarded and why.

    Mirrors the descriptor's diagnostics so that a run can report the two arms
    of E1.1 in the same terms.

    Attributes:
        mask_pixels: Pixels in the fruit mask.
        analysis_pixels: Pixels actually summarised, after specular exclusion.
        specular_fraction: Share of mask pixels rejected as specular.
        specular_fallback: True when exclusion would have left too few pixels
            and was therefore abandoned for this image.
    """

    mask_pixels: int
    analysis_pixels: int
    specular_fraction: float
    specular_fallback: bool


def colour_moments(values: np.ndarray) -> Tuple[float, float, float]:
    """Return the first three colour moments of one channel.

    Args:
        values: The channel's values over the analysed pixels.

    Returns:
        ``(mean, standard deviation, skewness)``. Skewness is Stricker and
        Orengo's third moment - the signed cube root of the mean cubed
        deviation - which keeps it in the same units as the channel rather
        than making it a ratio of powers.

        A channel that is exactly constant has no shape to describe, so its
        skewness is 0.0 rather than the ``0/0`` a normalised definition would
        produce. A uniform patch of colour is a real input here: a fruit
        photographed against a flat backdrop can segment to one.
    """
    if values.size == 0:
        return 0.0, 0.0, 0.0

    mean = float(np.mean(values))
    deviation = values - mean
    variance = float(np.mean(deviation ** 2))
    third = float(np.mean(deviation ** 3))

    # Cube root of a signed quantity: np.cbrt keeps the sign, which matters
    # because the sign is the whole content of a skewness.
    return mean, float(np.sqrt(variance)), float(np.cbrt(third))


class ColourHistogramExtractor(FeatureExtractor):
    """Fixed-bin colour histogram plus colour moments over the segmented fruit.

    Like the descriptor it is compared against, the extractor reads nothing
    from the configuration at call time: every parameter is fixed at
    construction, so two instances built the same way cannot diverge.
    """

    name = "Colour histogram and moments baseline"
    short_name = "T1H"

    def __init__(
        self,
        bins: int = 32,
        space: str = "LAB",
        exclude_specular: bool = True,
        specular_lightness: float = 240.0,
        specular_chroma: float = 12.0,
    ) -> None:
        """Configure the baseline.

        Args:
            bins: Histogram bins per channel.
            space: Colour space, one of ``LAB``, ``HSV`` or ``RGB``. E1.1
                holds this equal to the descriptor's space.
            exclude_specular: Drop specular highlights before summarising.
                Defaults to True to match the descriptor's default, so that
                E1.1 varies the descriptor family and nothing else.
            specular_lightness: Minimum OpenCV 8-bit ``L`` for a pixel to
                count as specular.
            specular_chroma: Maximum CIE chroma for a pixel to count as
                specular.

        Raises:
            ValueError: If any parameter is outside its permitted range.
        """
        space = space.upper()
        if space not in SUPPORTED_SPACES:
            raise ValueError(
                f"space must be one of {', '.join(SUPPORTED_SPACES)}; got {space!r}"
            )
        if bins < 1:
            raise ValueError(f"bins must be at least 1; got {bins}")

        self.bins = int(bins)
        self.space = space
        self.exclude_specular = bool(exclude_specular)
        self.specular_lightness = float(specular_lightness)
        self.specular_chroma = float(specular_chroma)

        #: One :class:`ColourHistogramDiagnostics` per call, in call order.
        self.diagnostics: List[ColourHistogramDiagnostics] = []

    # ----------------------------------------------------------------- #
    # Shape
    # ----------------------------------------------------------------- #

    @property
    def dim(self) -> int:
        """Length of the emitted vector: ``3 x bins`` histogram plus 9 moments."""
        return 3 * self.bins + 9

    @property
    def feature_names(self) -> Sequence[str]:
        """Human-readable name for every dimension, in emitted order.

        All histogram bins come first, channel by channel, then all nine
        moments. Keeping the two blocks contiguous is what lets an ablation
        select either half by slicing rather than by re-extracting.
        """
        channels = CHANNEL_NAMES[self.space]
        names: List[str] = []
        for channel in channels:
            names.extend(f"T1H_hist_{channel}_{index:02d}" for index in range(self.bins))
        for channel in channels:
            names.extend(
                [f"T1H_mean_{channel}", f"T1H_std_{channel}", f"T1H_skew_{channel}"]
            )
        return names

    def reset_diagnostics(self) -> None:
        """Discard the accumulated per-image diagnostics."""
        self.diagnostics.clear()

    # ----------------------------------------------------------------- #
    # Extraction
    # ----------------------------------------------------------------- #

    def _converted(self, bgr_image: np.ndarray) -> np.ndarray:
        """Return ``bgr_image`` in the configured colour space."""
        if self.space == "LAB":
            return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        if self.space == "HSV":
            return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    def extract_features(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> np.ndarray:
        """Describe one segmented fruit by its colour histogram and moments."""
        vector, _ = self.extract_with_diagnostics(bgr_image, fruit_mask)
        return vector

    def extract_with_diagnostics(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> Tuple[np.ndarray, ColourHistogramDiagnostics]:
        """Extract the baseline and return what was discarded alongside it.

        Args:
            bgr_image: A preprocessed ``uint8`` BGR image.
            fruit_mask: The fruit mask, 255 inside the fruit.

        Returns:
            The feature vector and a :class:`ColourHistogramDiagnostics`. The
            diagnostics are also appended to :attr:`diagnostics`.

        Raises:
            FeatureExtractionError: If the mask is empty.
        """
        mask = require_non_empty_mask(fruit_mask, self.short_name)
        mask_pixels = int(np.count_nonzero(mask))

        analysis = mask
        specular_fraction = 0.0
        specular_fallback = False

        if self.exclude_specular:
            specular = specular_selector(
                bgr_image, mask, self.specular_lightness, self.specular_chroma
            )
            specular_fraction = float(np.count_nonzero(specular) / mask_pixels)
            candidate = mask & ~specular
            if np.count_nonzero(candidate) >= MIN_ANALYSIS_PIXELS:
                analysis = candidate
            else:
                specular_fallback = True

        pixels = self._converted(bgr_image)[analysis].astype(np.float64)
        if pixels.shape[0] == 0:  # Defensive: the mask guard already rejects this.
            raise FeatureExtractionError(
                f"{self.short_name} was left with no pixels to summarise"
            )

        ranges = CHANNEL_RANGES[self.space]
        histograms: List[np.ndarray] = []
        moments: List[float] = []
        for channel in range(3):
            values = pixels[:, channel]
            counts, _ = np.histogram(values, bins=self.bins, range=ranges[channel])
            # Normalised per channel, so the vector describes the fruit's
            # colour distribution and not the size of the fruit in the frame.
            histograms.append(counts.astype(np.float64) / float(pixels.shape[0]))
            moments.extend(colour_moments(values))

        record = ColourHistogramDiagnostics(
            mask_pixels=mask_pixels,
            analysis_pixels=int(pixels.shape[0]),
            specular_fraction=specular_fraction,
            specular_fallback=specular_fallback,
        )
        self.diagnostics.append(record)
        return np.concatenate([*histograms, np.asarray(moments, dtype=np.float64)]), record

    # ----------------------------------------------------------------- #
    # Construction from configuration
    # ----------------------------------------------------------------- #

    @classmethod
    def from_config(cls, config: Any) -> "ColourHistogramExtractor":
        """Build the baseline from the ``t1_colour`` block of ``config.json``.

        The space and the specular thresholds are read from the descriptor's
        own block rather than given a block of their own. That is deliberate:
        they are the settings E1.1 holds equal across the two arms, and giving
        the baseline somewhere separate to state them would make it possible
        for the two arms to be configured to see different pixels.
        """
        block: Mapping[str, Any] = getattr(config, "t1_colour", {}) or {}
        return cls(
            bins=int(block.get("histogram_bins", 32)),
            space=str(block.get("space", "LAB")),
            exclude_specular=bool(block.get("exclude_specular", True)),
            specular_lightness=float(block.get("specular_lightness", 240.0)),
            specular_chroma=float(block.get("specular_chroma", 12.0)),
        )
