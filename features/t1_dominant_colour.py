"""T1 - MPEG-7 Dominant Colour Descriptor.

The descriptor summarises the fruit's colour as a small set of dominant
colours rather than as a fixed histogram. Where a histogram spends most of its
bins on colours the fruit does not contain, this spends its whole budget on
the colours it does, which is what lets four numbers stand in for the surface
of an apple.

The vector has three blocks:

* **A - dominant colours** (``4 x 7 = 28``). For each dominant colour: its
  centroid in the clustering space (3), the share of the fruit it covers (1),
  and the per-channel variance of the pixels assigned to it (3).
* **B - spatial coherency** (``4``). For each dominant colour, the fraction of
  its pixels lying in connected components larger than 1% of the fruit. One
  contiguous brown lesion and the same quantity of scattered speckling occupy
  identical percentages of the fruit and are told apart only here.
* **C - ripeness indices** (``5``). Percentage-weighted mean a*, mean chroma,
  circular mean hue angle, the share of the fruit in the green region of the
  a*b* plane, and the share that is simultaneously dark and desaturated.

Two properties of the construction matter more than they look:

**Canonical ordering.** k-means returns its clusters in an arbitrary order
that depends on initialisation, so the same apple photographed twice would
otherwise produce the same numbers in a different arrangement. Concatenated
into a vector, that is not a permutation the classifier can see through - it
is noise in every dimension. Clusters are therefore sorted by descending
share, ties broken by ascending lightness.

**Angular hue.** The CIE hue angle ``h_ab = atan2(b*, a*)`` wraps, so its
arithmetic mean is undefined: a fruit with hues at 359 deg and 1 deg averages
to 180 deg, the opposite colour. Every hue statistic here is computed by
projecting onto the unit circle and averaging the vectors.

Colour space and units
----------------------

OpenCV's 8-bit L*a*b* is used, which is D65. It stores ``L`` as ``L* x 2.55``
in ``[0, 255]`` and offsets ``a`` and ``b`` by 128 while keeping CIE
magnitudes, so lightness is scaled between the two systems and chroma is not.
Every threshold in this module is therefore quoted in **true CIE units**, with
the 8-bit equivalent alongside wherever the implementation works in it:

===================  =====================  ============================
Threshold            True CIE               OpenCV 8-bit
===================  =====================  ============================
Specular lightness   ``L* > 94``            ``L >= 240``
Specular chroma      ``chroma < 12``        ``chroma < 12`` (unchanged)
Decay lightness      ``L* < 45``            ``L < 115``
Decay chroma         ``chroma < 25``        ``chroma < 25`` (unchanged)
===================  =====================  ============================

The two specular thresholds are only coherent as a pair in the 8-bit reading:
in true CIE units ``L*`` cannot exceed 100, so a literal ``L* > 240`` would
never fire and specular exclusion would silently do nothing.

The decay thresholds were fixed a priori, before this descriptor was run on
any real image, and have not been refitted since. They are therefore free of
any tuning-on-test concern, and equally they are not optimised: Phase 5 is the
place to fit them by cross-validation on the training folds alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from features.base import FeatureExtractor, FeatureExtractionError, require_non_empty_mask

#: Clustering spaces the descriptor can be computed in.
SUPPORTED_SPACES = ("LAB", "HSV", "RGB")

#: Offset OpenCV applies to the a* and b* channels in its 8-bit encoding.
LAB_AB_OFFSET = 128.0

#: Factor OpenCV applies to L* in its 8-bit encoding.
LAB_L_SCALE = 2.55


@dataclass
class DominantColourDiagnostics:
    """Per-image record of what the descriptor discarded and why.

    Attributes:
        mask_pixels: Pixels in the fruit mask.
        analysis_pixels: Pixels actually clustered, after specular exclusion.
        specular_fraction: Share of mask pixels rejected as specular.
        specular_fallback: True when specular exclusion would have left too
            few pixels to cluster and was therefore abandoned for this image.
        empty_clusters: Dominant colours that received no pixels.
    """

    mask_pixels: int
    analysis_pixels: int
    specular_fraction: float
    specular_fallback: bool
    empty_clusters: int


# --------------------------------------------------------------------------- #
# Angular statistics
# --------------------------------------------------------------------------- #

def circular_mean(angles: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Return the circular mean of ``angles``, in radians on ``(-pi, pi]``.

    The arithmetic mean of an angle is meaningless across the wrap point, so
    each angle is projected onto the unit circle, the vectors are averaged,
    and the angle of the resultant is taken.

    Args:
        angles: Angles in radians.
        weights: Optional non-negative weights, one per angle.

    Returns:
        The mean angle, or 0.0 when the resultant vector vanishes because the
        angles cancel exactly and no mean direction exists.
    """
    if angles.size == 0:
        return 0.0
    if weights is None:
        cosine = float(np.mean(np.cos(angles)))
        sine = float(np.mean(np.sin(angles)))
    else:
        total = float(np.sum(weights))
        if total <= 0.0:
            return 0.0
        cosine = float(np.sum(weights * np.cos(angles)) / total)
        sine = float(np.sum(weights * np.sin(angles)) / total)

    if abs(cosine) < 1e-12 and abs(sine) < 1e-12:
        return 0.0
    return float(np.arctan2(sine, cosine))


def circular_variance(angles: np.ndarray) -> float:
    """Return the circular variance of ``angles``, in ``[0, 1]``.

    Zero means every angle is identical; one means they are spread evenly
    around the circle with no preferred direction.
    """
    if angles.size == 0:
        return 0.0
    resultant = np.hypot(float(np.mean(np.cos(angles))), float(np.mean(np.sin(angles))))
    return float(np.clip(1.0 - resultant, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Canonical ordering
# --------------------------------------------------------------------------- #

def canonical_order(shares: np.ndarray, lightness: np.ndarray) -> np.ndarray:
    """Return the index order that makes the descriptor comparable.

    Clusters are ordered by descending share of the fruit, with ties broken by
    ascending lightness so that the order is total and does not depend on the
    arbitrary labels k-means assigned.

    Args:
        shares: Fraction of the fruit each cluster covers.
        lightness: Each cluster's L* value, used only as the tie-break.

    Returns:
        Indices that sort the clusters into canonical order.
    """
    # lexsort orders by the last key first, so lightness is the tie-break.
    return np.lexsort((lightness, -shares))


# --------------------------------------------------------------------------- #
# The extractor
# --------------------------------------------------------------------------- #

class DominantColourExtractor(FeatureExtractor):
    """MPEG-7 dominant colour descriptor over the segmented fruit.

    The extractor holds no reference to the harness and reads nothing from the
    configuration at call time: every parameter is fixed at construction, so
    the same instance always produces the same vector for the same input.
    """

    name = "MPEG-7 Dominant Colour Descriptor"
    short_name = "T1"

    def __init__(
        self,
        n_colours: int = 4,
        space: str = "LAB",
        exclude_specular: bool = True,
        specular_lightness: float = 240.0,
        specular_chroma: float = 12.0,
        sample_size: int = 5000,
        coherency_min_fraction: float = 0.01,
        green_a_max: float = 0.0,
        decay_chroma_max: float = 25.0,
        decay_lightness_max: float = 45.0,
        blocks: str = "ABC",
        seed: int = 42,
    ) -> None:
        """Configure the descriptor.

        Args:
            n_colours: Number of dominant colours, ``N`` in the MPEG-7 sense.
            space: Clustering space, one of ``LAB``, ``HSV`` or ``RGB``.
            exclude_specular: Drop specular highlights before clustering.
            specular_lightness: Minimum OpenCV 8-bit ``L`` for a pixel to
                count as specular. The default 240 is ``L* > 94`` in true CIE
                units; it is expressed in the 8-bit encoding because a literal
                CIE value above 100 is unreachable.
            specular_chroma: Maximum CIE chroma for a pixel to count as
                specular. A highlight is bright *and* colourless; requiring
                both stops a genuinely bright yellow fruit being discarded.
            sample_size: Pixels drawn to fit the clustering.
            coherency_min_fraction: Minimum component size, as a fraction of
                the fruit, for a component to count as coherent.
            green_a_max: Upper bound on CIE ``a*`` for a cluster to count as
                green. Zero is the neutral axis.
            decay_chroma_max: Upper bound on CIE chroma for the decay index.
                Chroma is unscaled between the two encodings, so 25 means the
                same in both.
            decay_lightness_max: Upper bound on **true CIE** ``L*`` for the
                decay index. The default 45 is ``L < 115`` in the 8-bit
                encoding, well clear of black.
            blocks: Which blocks to emit, a subset of ``"ABC"`` in order.
            seed: Seed for subsampling and for k-means.

        Raises:
            ValueError: If any parameter is outside its permitted range.
        """
        space = space.upper()
        if space not in SUPPORTED_SPACES:
            raise ValueError(
                f"space must be one of {', '.join(SUPPORTED_SPACES)}; got {space!r}"
            )
        if n_colours < 1:
            raise ValueError(f"n_colours must be at least 1; got {n_colours}")
        if sample_size < n_colours:
            raise ValueError(
                f"sample_size ({sample_size}) must be at least n_colours ({n_colours})"
            )
        blocks = "".join(dict.fromkeys(blocks.upper()))
        if not blocks or set(blocks) - set("ABC"):
            raise ValueError(f"blocks must be a non-empty subset of 'ABC'; got {blocks!r}")
        if "A" not in blocks:
            raise ValueError("block A is the descriptor itself and cannot be omitted")

        self.n_colours = int(n_colours)
        self.space = space
        self.exclude_specular = bool(exclude_specular)
        self.specular_lightness = float(specular_lightness)
        self.specular_chroma = float(specular_chroma)
        self.sample_size = int(sample_size)
        self.coherency_min_fraction = float(coherency_min_fraction)
        self.green_a_max = float(green_a_max)
        self.decay_chroma_max = float(decay_chroma_max)
        self.decay_lightness_max = float(decay_lightness_max)
        self.blocks = blocks
        self.seed = int(seed)

        #: One :class:`DominantColourDiagnostics` per call, in call order.
        self.diagnostics: List[DominantColourDiagnostics] = []

    # ----------------------------------------------------------------- #
    # Shape
    # ----------------------------------------------------------------- #

    @property
    def dim(self) -> int:
        """Length of the emitted vector, fixed by the configuration."""
        total = 0
        if "A" in self.blocks:
            total += self.n_colours * 7
        if "B" in self.blocks:
            total += self.n_colours
        if "C" in self.blocks:
            total += 5
        return total

    @property
    def feature_names(self) -> Sequence[str]:
        """Human-readable name for every dimension, in emitted order."""
        channels = {
            "LAB": ("L", "a", "b"),
            "HSV": ("H", "S", "V"),
            "RGB": ("R", "G", "B"),
        }[self.space]

        names: List[str] = []
        if "A" in self.blocks:
            for index in range(self.n_colours):
                names.extend(f"T1_c{index}_centroid_{ch}" for ch in channels)
                names.append(f"T1_c{index}_share")
                names.extend(f"T1_c{index}_variance_{ch}" for ch in channels)
        if "B" in self.blocks:
            names.extend(f"T1_c{index}_coherency" for index in range(self.n_colours))
        if "C" in self.blocks:
            names.extend(
                [
                    "T1_weighted_mean_a",
                    "T1_weighted_mean_chroma",
                    "T1_circular_mean_hue",
                    "T1_green_share",
                    "T1_decay_share",
                ]
            )
        return names

    def reset_diagnostics(self) -> None:
        """Discard the accumulated per-image diagnostics."""
        self.diagnostics.clear()

    # ----------------------------------------------------------------- #
    # Colour space handling
    # ----------------------------------------------------------------- #

    def _working_pixels(self, bgr_image: np.ndarray, selector: np.ndarray) -> np.ndarray:
        """Return the selected pixels converted into the clustering space."""
        if self.space == "LAB":
            converted = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        elif self.space == "HSV":
            converted = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        else:
            converted = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return converted[selector].astype(np.float64)

    @staticmethod
    def _lab_pixels(bgr_image: np.ndarray, selector: np.ndarray) -> np.ndarray:
        """Return the selected pixels in CIE units: L* in [0, 100], a*/b* signed.

        The ripeness indices are defined in CIE terms, so they are computed
        from this representation whatever space the clustering ran in. That is
        what keeps the indices comparable across the LAB/HSV/RGB
        sub-experiment instead of silently changing meaning with the space.
        """
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)[selector].astype(np.float64)
        return np.column_stack(
            [
                lab[:, 0] / LAB_L_SCALE,
                lab[:, 1] - LAB_AB_OFFSET,
                lab[:, 2] - LAB_AB_OFFSET,
            ]
        )

    def _specular_selector(self, bgr_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return a boolean image marking specular pixels inside the mask.

        A highlight is the camera's reflection of the light source, not the
        fruit, and it is both very bright and nearly colourless. Requiring
        both conditions is deliberate: brightness alone would discard the lit
        face of a pale fruit, and low chroma alone would discard shadow.
        """
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float64)
        chroma = np.hypot(
            lab[:, :, 1] - LAB_AB_OFFSET, lab[:, :, 2] - LAB_AB_OFFSET
        )
        specular = (lab[:, :, 0] > self.specular_lightness) & (chroma < self.specular_chroma)
        return specular & mask

    # ----------------------------------------------------------------- #
    # Extraction
    # ----------------------------------------------------------------- #

    def extract_features(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> np.ndarray:
        """Describe one segmented fruit by its dominant colours."""
        vector, _ = self.extract_with_diagnostics(bgr_image, fruit_mask)
        return vector

    def extract_with_diagnostics(
        self,
        bgr_image: np.ndarray,
        fruit_mask: np.ndarray,
    ) -> Tuple[np.ndarray, DominantColourDiagnostics]:
        """Extract the descriptor and return what was discarded alongside it.

        Args:
            bgr_image: A preprocessed ``uint8`` BGR image.
            fruit_mask: The fruit mask, 255 inside the fruit.

        Returns:
            The feature vector and a :class:`DominantColourDiagnostics`. The
            diagnostics are also appended to :attr:`diagnostics`.
        """
        mask = require_non_empty_mask(fruit_mask, self.short_name)
        mask_pixels = int(np.count_nonzero(mask))

        analysis = mask.copy()
        specular_fraction = 0.0
        specular_fallback = False

        if self.exclude_specular:
            specular = self._specular_selector(bgr_image, mask)
            specular_fraction = float(np.count_nonzero(specular) / mask_pixels)
            candidate = mask & ~specular
            # Specular exclusion must never be able to starve the clustering.
            # A fruit photographed under a hard light can be mostly highlight,
            # and describing it from a handful of survivors would be worse
            # than describing it including the highlight.
            if np.count_nonzero(candidate) >= max(self.n_colours, 32):
                analysis = candidate
            else:
                specular_fallback = True

        working = self._working_pixels(bgr_image, analysis)
        lab = self._lab_pixels(bgr_image, analysis)
        if working.shape[0] < self.n_colours:
            raise FeatureExtractionError(
                f"{self.short_name} needs at least {self.n_colours} mask pixels to form "
                f"{self.n_colours} dominant colours; got {working.shape[0]}"
            )

        labels, centres = self._cluster(working)
        blocks, empty = self._summarise(
            working=working,
            lab=lab,
            labels=labels,
            centres=centres,
            analysis=analysis,
            fruit_area=mask_pixels,
        )

        record = DominantColourDiagnostics(
            mask_pixels=mask_pixels,
            analysis_pixels=int(working.shape[0]),
            specular_fraction=specular_fraction,
            specular_fallback=specular_fallback,
            empty_clusters=empty,
        )
        self.diagnostics.append(record)
        return np.concatenate(blocks), record

    def _cluster(self, working: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit k-means on a subsample, then assign every analysed pixel.

        Fitting on a sample keeps extraction inside its time budget; assigning
        on the full mask keeps the shares exact.

        The sample is drawn from the pixels **sorted by colour**, not in the
        order they happen to occupy the array. Drawing by array position makes
        the fitted centroids depend on where pixels sit in the frame rather
        than on what colours the fruit has: the same fruit, its pixels
        rearranged, yields a different sample, a different k-means solution and
        genuinely different dominant colours - not a permutation of the same
        ones, which the canonical ordering could repair, but different numbers.

        That is not a hypothetical. The harness augments training images by
        flipping and rotating them, and every one of those operations
        rearranges pixels while leaving the fruit's colour content untouched.
        Sorting first makes the sample a function of the colour multiset
        alone, so a flipped apple is described exactly like its original.
        Spatial coherency is computed from the unsorted assignment, so it
        keeps its spatial meaning.
        """
        rng = np.random.default_rng(self.seed)
        if working.shape[0] > self.sample_size:
            ordered = working[np.lexsort(tuple(working[:, channel] for channel in (2, 1, 0)))]
            chosen = rng.choice(ordered.shape[0], size=self.sample_size, replace=False)
            sample = ordered[np.sort(chosen)]
        else:
            sample = working

        # n_init is set explicitly: scikit-learn defaults it to "auto", whose
        # meaning depends on the initialisation method and has changed across
        # releases, so leaving it unset would make the descriptor's definition
        # depend on the installed version.
        estimator = KMeans(
            n_clusters=self.n_colours,
            init="k-means++",
            n_init=10,
            random_state=self.seed,
        )
        estimator.fit(sample)
        return estimator.predict(working), estimator.cluster_centers_

    def _summarise(
        self,
        working: np.ndarray,
        lab: np.ndarray,
        labels: np.ndarray,
        centres: np.ndarray,
        analysis: np.ndarray,
        fruit_area: int,
    ) -> Tuple[List[np.ndarray], int]:
        """Build the descriptor blocks from the clustering result."""
        total = float(labels.size)
        shares = np.zeros(self.n_colours, dtype=np.float64)
        centroids = np.zeros((self.n_colours, 3), dtype=np.float64)
        variances = np.zeros((self.n_colours, 3), dtype=np.float64)
        cluster_lab = np.zeros((self.n_colours, 3), dtype=np.float64)
        empty = 0

        for index in range(self.n_colours):
            members = labels == index
            count = int(np.count_nonzero(members))
            if count == 0:
                # A cluster can end up empty because k-means was fitted on a
                # sample. Fall back to its fitted centre so the slot carries a
                # real colour rather than a zero the classifier would read as
                # "black", and leave its share and variance at zero.
                empty += 1
                centroids[index] = centres[index]
                cluster_lab[index] = 0.0
                continue

            shares[index] = count / total
            centroids[index] = self._channel_centre(working[members])
            variances[index] = self._channel_spread(working[members])
            cluster_lab[index] = self._lab_centre(lab[members])

        order = canonical_order(shares, cluster_lab[:, 0])
        shares, centroids, variances = shares[order], centroids[order], variances[order]
        cluster_lab = cluster_lab[order]

        blocks: List[np.ndarray] = []
        if "A" in self.blocks:
            blocks.append(
                np.concatenate(
                    [
                        np.concatenate([centroids[i], [shares[i] * 100.0], variances[i]])
                        for i in range(self.n_colours)
                    ]
                )
            )
        if "B" in self.blocks:
            blocks.append(
                self._coherency(labels, order, analysis, fruit_area)
            )
        if "C" in self.blocks:
            blocks.append(self._indices(shares, cluster_lab))
        return blocks, empty

    def _channel_centre(self, pixels: np.ndarray) -> np.ndarray:
        """Return the per-channel centre of ``pixels`` in the clustering space.

        In HSV the first channel is an angle, so its centre is the circular
        mean rather than the arithmetic one. OpenCV stores hue on ``[0, 180)``
        for 8-bit images, representing 0-360 degrees.
        """
        centre = pixels.mean(axis=0)
        if self.space == "HSV":
            angles = pixels[:, 0] * (2.0 * np.pi / 180.0)
            mean_angle = circular_mean(angles) % (2.0 * np.pi)
            centre[0] = mean_angle * (180.0 / (2.0 * np.pi))
        return centre

    def _channel_spread(self, pixels: np.ndarray) -> np.ndarray:
        """Return the per-channel variance of ``pixels`` in the clustering space.

        The HSV hue channel again needs circular treatment; its value is a
        circular variance on ``[0, 1]`` rather than a variance in hue units,
        which is noted because it makes that one dimension differently scaled
        from its neighbours. Standardisation in the shared pipeline absorbs
        the difference.
        """
        spread = pixels.var(axis=0)
        if self.space == "HSV":
            spread[0] = circular_variance(pixels[:, 0] * (2.0 * np.pi / 180.0))
        return spread

    @staticmethod
    def _lab_centre(pixels: np.ndarray) -> np.ndarray:
        """Return the mean CIE ``L*``, ``a*`` and ``b*`` of ``pixels``.

        ``a*`` and ``b*`` are Cartesian, so their arithmetic mean is
        well-defined; it is the hue *angle* derived from them that is not, and
        that is handled where the angle is formed.
        """
        return pixels.mean(axis=0)

    def _coherency(
        self,
        labels: np.ndarray,
        order: np.ndarray,
        analysis: np.ndarray,
        fruit_area: int,
    ) -> np.ndarray:
        """Fraction of each dominant colour lying in large connected components.

        Percentage alone cannot distinguish one contiguous lesion from the
        same area of scattered speckling, and on a fruit those mean different
        things. Components smaller than ``coherency_min_fraction`` of the
        fruit are treated as speckle.
        """
        positions = np.flatnonzero(analysis.ravel())
        minimum_area = max(1, int(round(self.coherency_min_fraction * fruit_area)))
        height, width = analysis.shape

        coherency = np.zeros(self.n_colours, dtype=np.float64)
        for slot, index in enumerate(order):
            members = labels == index
            count = int(np.count_nonzero(members))
            if count == 0:
                continue

            component_image = np.zeros(height * width, dtype=np.uint8)
            component_image[positions[members]] = 255
            component_image = component_image.reshape(height, width)

            found, _, stats, _ = cv2.connectedComponentsWithStats(
                component_image, connectivity=8
            )
            if found <= 1:
                continue
            areas = stats[1:, cv2.CC_STAT_AREA]
            coherency[slot] = float(areas[areas >= minimum_area].sum() / count)
        return coherency

    def _indices(self, shares: np.ndarray, cluster_lab: np.ndarray) -> np.ndarray:
        """Compute the five ripeness indices from the ordered clusters.

        All five are share-weighted, so a colour covering a tenth of the fruit
        contributes a tenth as much as one covering all of it.
        """
        lightness = cluster_lab[:, 0]
        a_star = cluster_lab[:, 1]
        b_star = cluster_lab[:, 2]
        chroma = np.hypot(a_star, b_star)
        hue = np.arctan2(b_star, a_star)

        weight = float(shares.sum())
        if weight <= 0.0:  # Defensive: every cluster empty cannot occur here.
            return np.zeros(5, dtype=np.float64)

        weighted_a = float(np.sum(shares * a_star) / weight)
        weighted_chroma = float(np.sum(shares * chroma) / weight)
        mean_hue = circular_mean(hue, weights=shares)

        green = shares[a_star < self.green_a_max].sum() * 100.0
        decayed = shares[
            (chroma < self.decay_chroma_max)
            & (lightness < self.decay_lightness_max)
        ].sum() * 100.0

        return np.array(
            [weighted_a, weighted_chroma, mean_hue, float(green), float(decayed)],
            dtype=np.float64,
        )

    # ----------------------------------------------------------------- #
    # Construction from configuration
    # ----------------------------------------------------------------- #

    @classmethod
    def from_config(cls, config: Any) -> "DominantColourExtractor":
        """Build an extractor from the ``t1_colour`` block of ``config.json``.

        Reading configuration here rather than inside the extraction call
        keeps the descriptor a pure function of its constructor arguments, so
        two instances built the same way cannot diverge.
        """
        block: Mapping[str, Any] = getattr(config, "t1_colour", {}) or {}
        return cls(
            n_colours=int(block.get("n_colours", 4)),
            space=str(block.get("space", "LAB")),
            exclude_specular=bool(block.get("exclude_specular", True)),
            specular_lightness=float(block.get("specular_lightness", 240.0)),
            specular_chroma=float(block.get("specular_chroma", 12.0)),
            sample_size=int(block.get("sample_size", 5000)),
            coherency_min_fraction=float(block.get("coherency_min_fraction", 0.01)),
            green_a_max=float(block.get("green_a_max", 0.0)),
            decay_chroma_max=float(block.get("decay_chroma_max", 25.0)),
            decay_lightness_max=float(block.get("decay_lightness_max", 45.0)),
            blocks=str(block.get("blocks", "ABC")),
            seed=int(getattr(config, "seed", 42)),
        )
