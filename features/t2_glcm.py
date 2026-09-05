"""T2 - GLCM texture descriptors.

Haralick statistics over the grey-level co-occurrence matrix. Where T1 asks
what colours the fruit is, this asks how the surface varies from one pixel to
the next: smooth skin produces co-occurrences concentrated near the diagonal,
while the pitted, wrinkled surface of decaying tissue spreads them away from
it.

The vector is ``5 properties x 2 distances x 4 angles = 40``, laid out
property-major, then distance, then angle - the order
:attr:`GLCMExtractor.feature_names` reports.

Excluding the background
------------------------

The descriptor is computed over the bounding-box crop rather than the padded
frame, but a bounding box is a rectangle and the fruit is not, so background
pixels sit inside the crop regardless. Simply zeroing them is the trap: zero
becomes an enormous, perfectly uniform grey level, and its co-occurrences with
the fruit's boundary swamp contrast and dissimilarity. The measured texture
would then be dominated by the *shape* of the mask.

Instead the fruit's grey levels are quantised into ``1..levels`` and the
background is set to ``0``, an ignore value. The co-occurrence matrix is built
with ``levels + 1`` levels, and **row and column 0 are then deleted** before
the matrix is normalised, so every pair involving a background pixel is
removed from both the numerator and the denominator rather than being counted
as texture.

Deleting the first row and column shifts every remaining grey index down by
one, which leaves all five properties unchanged: contrast, dissimilarity and
homogeneity depend only on ``|i - j|``, energy only on the probabilities, and
correlation is invariant to a shift of the marginal means. This is asserted by
``test_deleting_the_ignore_level_preserves_the_properties``.

Degenerate matrices
-------------------

Correlation divides by the standard deviations of the matrix marginals, so it
is undefined when either is zero. Two distinct cases arise and they are
treated differently, because they mean different things:

* **A uniform crop** - real pixels, all the same grey. scikit-image returns
  ``1.0``, the conventional reading that a constant surface is perfectly
  correlated with itself. That convention is kept.
* **A matrix with no pairs at all** - the crop is smaller than the offset, so
  nothing was measured. scikit-image also returns ``1.0`` here, which asserts
  perfect correlation on the strength of no evidence whatsoever. This module
  emits ``0.0`` instead and records the occurrence, because a fabricated
  extreme value is worse than a neutral one.

The harness rejects masks covering under 5% of the frame, so a crop too small
to admit a single pair at distance 2 should not reach this code. The guard
exists so that if one ever does, it produces a documented number rather than a
silent fiction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from features.base import FeatureExtractor, FeatureExtractionError, require_non_empty_mask

#: Haralick properties, in the order they appear in the vector.
PROPERTIES: Tuple[str, ...] = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
)

#: Value emitted for each property when a co-occurrence matrix holds no pairs.
#: Correlation departs from scikit-image, which returns 1.0; see the module
#: docstring.
DEGENERATE_VALUES: Mapping[str, float] = {
    "contrast": 0.0,
    "dissimilarity": 0.0,
    "homogeneity": 1.0,
    "energy": 1.0,
    "correlation": 0.0,
}


@dataclass
class GLCMDiagnostics:
    """Per-image record of the conditions the descriptor met.

    Attributes:
        crop_shape: Height and width of the bounding-box crop.
        mask_pixels: Fruit pixels inside the crop.
        background_fraction: Share of the crop that is background, and so is
            excluded from the co-occurrence counts.
        degenerate_slices: Distance-angle pairs that yielded no co-occurrences.
    """

    crop_shape: Tuple[int, int]
    mask_pixels: int
    background_fraction: float
    degenerate_slices: int


class GLCMExtractor(FeatureExtractor):
    """Grey-level co-occurrence texture descriptor over the segmented fruit."""

    name = "GLCM texture descriptors"
    short_name = "T2"

    def __init__(
        self,
        distances: Sequence[int] = (1, 2),
        angles_deg: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
        levels: int = 32,
        angle_averaged: bool = False,
        symmetric: bool = True,
        seed: int = 42,
    ) -> None:
        """Configure the descriptor.

        Args:
            distances: Pixel offsets at which co-occurrences are counted.
            angles_deg: Orientations in degrees. The defaults are the four
                Haralick directions, 0, 45, 90 and 135.
            levels: Grey levels the fruit is quantised into.
            angle_averaged: Average each property over the angles, giving a
                rotation-invariant descriptor of ``5 x len(distances)``
                dimensions instead of ``5 x len(distances) x len(angles)``.
            symmetric: Count each pair in both directions, the standard
                Haralick convention.
            seed: Accepted for interface symmetry; this descriptor is
                deterministic and does not use it.

        Raises:
            ValueError: If any parameter is outside its permitted range.
        """
        distances = tuple(int(d) for d in distances)
        angles_deg = tuple(float(a) for a in angles_deg)
        if not distances or any(d < 1 for d in distances):
            raise ValueError(f"distances must be positive integers; got {distances}")
        if not angles_deg:
            raise ValueError("at least one angle is required")
        if levels < 2:
            raise ValueError(f"levels must be at least 2; got {levels}")

        self.distances = distances
        self.angles_deg = angles_deg
        self.angles = tuple(np.deg2rad(a) for a in angles_deg)
        self.levels = int(levels)
        self.angle_averaged = bool(angle_averaged)
        self.symmetric = bool(symmetric)
        self.seed = int(seed)

        #: One :class:`GLCMDiagnostics` per call, in call order.
        self.diagnostics: List[GLCMDiagnostics] = []

    # ----------------------------------------------------------------- #
    # Shape
    # ----------------------------------------------------------------- #

    @property
    def dim(self) -> int:
        """Length of the emitted vector, fixed by the configuration."""
        per_property = len(self.distances) * (1 if self.angle_averaged else len(self.angles))
        return len(PROPERTIES) * per_property

    @property
    def feature_names(self) -> Sequence[str]:
        """Human-readable name for every dimension, in emitted order."""
        names: List[str] = []
        for prop in PROPERTIES:
            for distance in self.distances:
                if self.angle_averaged:
                    names.append(f"T2_{prop}_d{distance}_angleavg")
                else:
                    for angle in self.angles_deg:
                        names.append(f"T2_{prop}_d{distance}_a{int(round(angle))}")
        return names

    def reset_diagnostics(self) -> None:
        """Discard the accumulated per-image diagnostics."""
        self.diagnostics.clear()

    # ----------------------------------------------------------------- #
    # Preparation
    # ----------------------------------------------------------------- #

    def quantise(self, grey: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Quantise the fruit into ``1..levels``, marking background as ``0``.

        Quantisation spans the full 0-255 range rather than the fruit's own
        range. Rescaling per image would normalise away absolute brightness,
        which is a genuine difference between a pale unripe apple and a dark
        rotten one, and would make the descriptor's meaning depend on the
        image it was computed from.

        Args:
            grey: Single-channel ``uint8`` image.
            mask: Boolean array selecting the fruit.

        Returns:
            A ``uint8`` array of quantised levels, background at 0.
        """
        scaled = (grey.astype(np.int32) * self.levels) // 256
        quantised = np.clip(scaled, 0, self.levels - 1).astype(np.uint8) + 1
        quantised[~mask] = 0
        return quantised

    # ----------------------------------------------------------------- #
    # Extraction
    # ----------------------------------------------------------------- #

    def extract_features(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> np.ndarray:
        """Describe one segmented fruit by its co-occurrence texture."""
        vector, _ = self.extract_with_diagnostics(bgr_image, fruit_mask)
        return vector

    def extract_with_diagnostics(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> Tuple[np.ndarray, GLCMDiagnostics]:
        """Extract the descriptor and report the conditions it met.

        Args:
            bgr_image: A preprocessed ``uint8`` BGR image.
            fruit_mask: The fruit mask, 255 inside the fruit.

        Returns:
            The feature vector and a :class:`GLCMDiagnostics`.
        """
        mask = require_non_empty_mask(fruit_mask, self.short_name)
        grey = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
        grey_crop = grey[y : y + height, x : x + width]
        mask_crop = mask[y : y + height, x : x + width]

        quantised = self.quantise(grey_crop, mask_crop)
        matrices = graycomatrix(
            quantised,
            distances=list(self.distances),
            angles=list(self.angles),
            levels=self.levels + 1,  # +1 for the ignore level at index 0.
            symmetric=self.symmetric,
            normed=False,
        ).astype(np.float64)

        # Remove every pair involving a background pixel, from the counts and
        # from the total, then normalise what remains.
        matrices = matrices[1:, 1:, :, :]
        totals = matrices.sum(axis=(0, 1), keepdims=True)
        degenerate = totals[0, 0] <= 0.0
        normalised = np.divide(
            matrices, totals, out=np.zeros_like(matrices), where=totals > 0.0
        )

        vector = self._properties(normalised, degenerate)
        record = GLCMDiagnostics(
            crop_shape=(int(height), int(width)),
            mask_pixels=int(np.count_nonzero(mask_crop)),
            background_fraction=float(1.0 - np.count_nonzero(mask_crop) / mask_crop.size),
            degenerate_slices=int(np.count_nonzero(degenerate)),
        )
        self.diagnostics.append(record)
        return vector, record

    def _properties(self, normalised: np.ndarray, degenerate: np.ndarray) -> np.ndarray:
        """Evaluate the Haralick properties over every distance-angle pair.

        Args:
            normalised: ``(levels, levels, n_distances, n_angles)`` matrices,
                each normalised to sum to one unless it holds no pairs.
            degenerate: ``(n_distances, n_angles)`` boolean array marking the
                pairs that held no co-occurrences.

        Returns:
            The flattened feature vector.
        """
        blocks: List[np.ndarray] = []
        for prop in PROPERTIES:
            values = np.asarray(graycoprops(normalised, prop), dtype=np.float64)
            # scikit-image returns 1.0 for correlation on an empty matrix,
            # asserting perfect correlation from no measurement at all.
            values[degenerate] = DEGENERATE_VALUES[prop]
            if self.angle_averaged:
                values = values.mean(axis=1, keepdims=True)
            blocks.append(values.reshape(-1))

        vector = np.concatenate(blocks)
        if not np.all(np.isfinite(vector)):
            bad = np.flatnonzero(~np.isfinite(vector)).tolist()
            raise FeatureExtractionError(
                f"{self.short_name} produced non-finite values at {bad[:10]}; the "
                f"degenerate-matrix guard did not cover this case"
            )
        return vector

    # ----------------------------------------------------------------- #
    # Construction from configuration
    # ----------------------------------------------------------------- #

    @classmethod
    def from_config(cls, config: Any) -> "GLCMExtractor":
        """Build an extractor from the ``t2_glcm`` block of ``config.json``."""
        block: Mapping[str, Any] = getattr(config, "t2_glcm", {}) or {}
        return cls(
            distances=tuple(block.get("distances", (1, 2))),
            angles_deg=tuple(block.get("angles_deg", (0.0, 45.0, 90.0, 135.0))),
            levels=int(block.get("levels", 32)),
            angle_averaged=bool(block.get("angle_averaged", False)),
            symmetric=bool(block.get("symmetric", True)),
            seed=int(getattr(config, "seed", 42)),
        )
