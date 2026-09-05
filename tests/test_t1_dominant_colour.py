"""Tests for T1, the MPEG-7 dominant colour descriptor.

The contract every technique must satisfy - documented length, no NaN or Inf,
determinism - is checked here, and so are the two properties specific to this
descriptor that would otherwise fail silently:

**Canonical ordering.** k-means labels its clusters arbitrarily. Without a
canonical order the descriptor still has the right length, the right range and
no NaN, and is still wrong: the same fruit yields the same numbers in a
different arrangement, so every dimension carries noise. Nothing but a
permutation test detects it.

**Angular hue.** The arithmetic mean of an angle is undefined across the wrap
point, and a fruit whose hue straddles it averages to the opposite colour. The
failure is invisible in the vector.

Run from the project root::

    python -m pytest tests/test_t1_dominant_colour.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import pytest  # noqa: E402

from config import get_config  # noqa: E402
from features.base import FeatureExtractionError  # noqa: E402
from features.t1_dominant_colour import (  # noqa: E402
    DominantColourExtractor,
    canonical_order,
    circular_mean,
    circular_variance,
)

CONFIG = get_config()
SIZE = 224


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def patched_apple(
    patches: Tuple[Tuple[Tuple[int, int, int], float], ...] = (
        ((40, 40, 200), 0.55),
        ((40, 160, 90), 0.25),
        ((30, 45, 60), 0.20),
    ),
    radius: int = 80,
    texture: int = 3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """A disc split into horizontal bands of known colour and known proportion.

    Returns the BGR image and its mask. Bands are contiguous, so the spatial
    coherency block should read high; the shares are set by band height.
    """
    rng = np.random.default_rng(seed)
    image = np.full((SIZE, SIZE, 3), 235, dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, (SIZE // 2, SIZE // 2), radius, 255, thickness=-1)

    top = SIZE // 2 - radius
    height = 2 * radius
    cursor = top
    for colour, proportion in patches:
        band = int(round(height * proportion))
        image[cursor : cursor + band, :] = colour
        cursor += band
    image[cursor:, :] = patches[-1][0]

    grain = rng.integers(-texture, texture + 1, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    image[mask == 0] = 235  # Background stays flat and outside the mask.
    return image, mask


def shuffle_within_mask(
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 7,
) -> np.ndarray:
    """Permute the positions of the masked pixels, leaving the multiset intact.

    The set of colours inside the mask is unchanged, so every statistic the
    descriptor derives from the colour distribution must be unchanged too.
    Only the spatial arrangement differs, which is exactly what tells the
    order-dependent parts of the descriptor from the order-independent ones.
    """
    rng = np.random.default_rng(seed)
    shuffled = image.copy()
    positions = np.flatnonzero(mask.ravel())
    flat = shuffled.reshape(-1, 3)
    flat[positions] = flat[rng.permutation(positions)]
    return flat.reshape(image.shape)


@pytest.fixture(scope="module")
def apple() -> Tuple[np.ndarray, np.ndarray]:
    """The default synthetic fruit used by most tests."""
    return patched_apple()


# --------------------------------------------------------------------------- #
# Angular statistics
# --------------------------------------------------------------------------- #

def test_circular_mean_handles_the_wrap_point():
    """The case an arithmetic mean gets exactly backwards."""
    angles = np.array([np.deg2rad(359.0), np.deg2rad(1.0)])
    assert np.rad2deg(circular_mean(angles)) == pytest.approx(0.0, abs=1e-6)
    # For contrast, the arithmetic mean of these two angles is 180 degrees.
    assert np.rad2deg(float(np.mean(angles))) == pytest.approx(180.0, abs=1e-6)


def test_circular_mean_matches_the_arithmetic_mean_away_from_the_wrap():
    """Away from the discontinuity the two agree, so nothing is lost."""
    angles = np.deg2rad(np.array([80.0, 90.0, 100.0]))
    assert np.rad2deg(circular_mean(angles)) == pytest.approx(90.0, abs=1e-6)


def test_circular_mean_is_weightable():
    """Weighting lets a dominant colour count for more than a minor one."""
    angles = np.deg2rad(np.array([0.0, 90.0]))
    heavy = np.rad2deg(circular_mean(angles, weights=np.array([0.99, 0.01])))
    assert heavy == pytest.approx(0.58, abs=0.05)


def test_circular_mean_of_cancelling_angles_is_defined():
    """Opposed angles have no mean direction; the result must not be NaN."""
    angles = np.array([0.0, np.pi])
    assert np.isfinite(circular_mean(angles))


def test_circular_variance_spans_zero_to_one():
    """Identical angles give 0; angles spread evenly give 1."""
    assert circular_variance(np.zeros(8)) == pytest.approx(0.0, abs=1e-9)
    spread = np.linspace(0.0, 2.0 * np.pi, 9)[:-1]
    assert circular_variance(spread) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Canonical ordering
# --------------------------------------------------------------------------- #

def test_canonical_order_sorts_by_descending_share():
    shares = np.array([0.1, 0.5, 0.25, 0.15])
    order = canonical_order(shares, np.zeros(4))
    assert list(shares[order]) == [0.5, 0.25, 0.15, 0.1]


def test_canonical_order_breaks_ties_by_ascending_lightness():
    """Equal shares must still produce one fixed order, not an arbitrary one."""
    shares = np.array([0.25, 0.25, 0.25, 0.25])
    lightness = np.array([70.0, 10.0, 50.0, 30.0])
    order = canonical_order(shares, lightness)
    assert list(lightness[order]) == [10.0, 30.0, 50.0, 70.0]


def test_canonical_order_is_a_permutation():
    """Every cluster must appear exactly once."""
    order = canonical_order(np.array([0.4, 0.4, 0.1, 0.1]), np.array([5.0, 5.0, 2.0, 2.0]))
    assert sorted(order.tolist()) == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# The shared contract
# --------------------------------------------------------------------------- #

def test_vector_has_the_documented_length(apple):
    """4 x (3 centroid + 1 share + 3 variance) + 4 coherency + 5 indices = 37."""
    extractor = DominantColourExtractor()
    assert extractor.dim == 37
    assert extractor(*apple).shape == (37,)


def test_vector_is_finite(apple):
    vector = DominantColourExtractor()(*apple)
    assert np.all(np.isfinite(vector))


def test_vector_is_deterministic_across_calls(apple):
    """Two calls on one input must agree exactly, not merely closely."""
    extractor = DominantColourExtractor()
    first, second = extractor(*apple), extractor(*apple)
    assert np.array_equal(first, second)


def test_two_instances_agree(apple):
    """Configuration, not instance history, must decide the vector."""
    assert np.array_equal(
        DominantColourExtractor()(*apple), DominantColourExtractor()(*apple)
    )


def test_feature_names_match_the_vector_length():
    extractor = DominantColourExtractor()
    assert len(extractor.feature_names) == extractor.dim


def test_an_empty_mask_is_refused(apple):
    """A broken mask must raise, not yield a vector of zeros to train on."""
    image, _ = apple
    with pytest.raises(FeatureExtractionError):
        DominantColourExtractor()(image, np.zeros((SIZE, SIZE), dtype=np.uint8))


def test_the_extractor_does_not_modify_its_inputs(apple):
    """The harness hands over copies; the technique must not rely on that."""
    image, mask = apple
    image_before, mask_before = image.copy(), mask.copy()
    DominantColourExtractor()(image, mask)
    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)


# --------------------------------------------------------------------------- #
# Canonical ordering, end to end
# --------------------------------------------------------------------------- #

def test_shares_are_emitted_in_descending_order(apple):
    """The ordering invariant, read straight out of the vector."""
    vector = DominantColourExtractor()(*apple)
    shares = np.array([vector[index * 7 + 3] for index in range(4)])
    assert np.all(np.diff(shares) <= 1e-9), shares


def test_shuffling_pixel_order_leaves_the_colour_statistics_unchanged(apple):
    """The test that proves canonical ordering works.

    Permuting the masked pixels leaves the multiset of colours identical, so
    the dominant colours, their shares, their variances and the ripeness
    indices must all be identical too. What changes is the order k-means
    happens to label its clusters in - and without the canonical sort, that
    permutation propagates straight into the vector.

    Spatial coherency is deliberately excluded: it measures how the pixels are
    arranged, so a spatial shuffle is supposed to change it. That is asserted
    separately below rather than glossed over.
    """
    image, mask = apple
    extractor = DominantColourExtractor()

    original = extractor(image, mask)
    shuffled = extractor(shuffle_within_mask(image, mask), mask)

    colour_block = np.concatenate([original[:28], original[32:]])
    shuffled_block = np.concatenate([shuffled[:28], shuffled[32:]])
    assert np.allclose(colour_block, shuffled_block, atol=0.5), (
        "colour statistics changed under a pixel permutation, which means the "
        "cluster ordering is not canonical"
    )


def test_shuffling_pixel_order_does_change_spatial_coherency(apple):
    """The counterpart: coherency is spatial and must react to a shuffle.

    Without this, the exclusion above could be hiding a coherency block that
    is constant - and a constant feature carries no information at all.

    The dominant cluster is exempted deliberately. Scattering it does not make
    it incoherent, because randomly placed pixels covering more than about 27%
    of the area percolate into one giant connected component on an
    8-connected grid. That is a property of the grid, not a defect in the
    descriptor, and the minor clusters below that density do collapse to zero
    as expected.
    """
    image, mask = apple
    extractor = DominantColourExtractor()

    original = extractor(image, mask)[28:32]
    shuffled = extractor(shuffle_within_mask(image, mask), mask)[28:32]

    assert original.min() > 0.9, "contiguous bands should all be highly coherent"
    assert shuffled[1:].max() < 0.1, "scattered minor clusters should not be coherent"


@pytest.mark.parametrize("flip_code, axis", [(0, "vertical"), (1, "horizontal")])
def test_a_flipped_fruit_is_described_identically(apple, flip_code, axis):
    """The invariance that matters most in practice.

    The harness augments training images by flipping them, so an original and
    its mirror are the same fruit and must receive the same descriptor. A flip
    preserves the multiset of colours exactly and maps connected components
    onto congruent ones, so *every* block - coherency included - must match,
    not merely the colour statistics.

    This is what the colour-sorted sampling buys. Drawing the k-means sample
    by array position instead makes a flip select different pixels, fit
    different centroids, and hand the classifier two different vectors for one
    fruit.
    """
    image, mask = apple
    extractor = DominantColourExtractor()

    original = extractor(image, mask)
    flipped = extractor(cv2.flip(image, flip_code), cv2.flip(mask, flip_code))

    assert np.allclose(original, flipped, atol=1e-6), (
        f"a {axis} flip changed the descriptor, so an augmented variant would "
        f"not be described like its original"
    )


# --------------------------------------------------------------------------- #
# Descriptor behaviour
# --------------------------------------------------------------------------- #

def test_dominant_colours_recover_the_bands_that_were_painted():
    """The descriptor must actually find the colours that are there."""
    image, mask = patched_apple(
        patches=(((40, 40, 200), 0.5), ((40, 170, 80), 0.5)), seed=2
    )
    vector = DominantColourExtractor()(image, mask)
    centroids = np.array([vector[i * 7 : i * 7 + 3] for i in range(4)])

    expected = cv2.cvtColor(
        np.array([[[40, 40, 200], [40, 170, 80]]], dtype=np.uint8), cv2.COLOR_BGR2LAB
    )[0].astype(np.float64)
    for target in expected:
        assert np.min(np.linalg.norm(centroids - target, axis=1)) < 12.0


def test_shares_sum_to_one_hundred(apple):
    """Shares are percentages of the analysed fruit and must account for it."""
    vector = DominantColourExtractor()(*apple)
    shares = np.array([vector[index * 7 + 3] for index in range(4)])
    assert shares.sum() == pytest.approx(100.0, abs=1e-6)


def test_specular_pixels_are_excluded_and_logged():
    """A highlight must not consume one of only four dominant colours."""
    image, mask = patched_apple(patches=(((40, 40, 200), 1.0),), seed=3)
    cv2.circle(image, (SIZE // 2 - 30, SIZE // 2 - 30), 18, (252, 252, 252), thickness=-1)

    keeping = DominantColourExtractor(exclude_specular=False)
    dropping = DominantColourExtractor(exclude_specular=True)
    keeping(image, mask)
    dropping(image, mask)

    assert dropping.diagnostics[-1].specular_fraction > 0.02
    assert keeping.diagnostics[-1].specular_fraction == 0.0
    assert dropping.diagnostics[-1].analysis_pixels < keeping.diagnostics[-1].analysis_pixels


def test_specular_exclusion_never_starves_the_clustering():
    """An almost entirely blown-out fruit must still be described.

    Excluding highlights is a refinement; it must not be able to leave so few
    pixels that the descriptor becomes meaningless, so it stands down and
    records that it did.
    """
    image, mask = patched_apple(patches=(((252, 252, 252), 1.0),), seed=4)
    extractor = DominantColourExtractor()
    vector = extractor(image, mask)

    assert np.all(np.isfinite(vector))
    assert extractor.diagnostics[-1].specular_fallback is True


def test_coherency_separates_a_lesion_from_speckling():
    """The block exists to tell these two apart; percentages cannot.

    Both fruits carry the same quantity of dark pixels on the same base
    colour. One is a single contiguous patch, the other the same area
    scattered as fine speckles.
    """
    rng = np.random.default_rng(5)
    base = (40, 40, 200)
    dark = (25, 30, 60)

    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, (SIZE // 2, SIZE // 2), 80, 255, thickness=-1)
    inside = np.flatnonzero(mask.ravel())

    lesion = np.full((SIZE, SIZE, 3), base, dtype=np.uint8)
    cv2.circle(lesion, (SIZE // 2, SIZE // 2), 32, dark, thickness=-1)
    dark_count = int(np.count_nonzero(np.all(lesion.reshape(-1, 3) == dark, axis=1)))

    speckled = np.full((SIZE, SIZE, 3), base, dtype=np.uint8)
    chosen = rng.choice(inside, size=dark_count, replace=False)
    speckled.reshape(-1, 3)[chosen] = dark

    extractor = DominantColourExtractor(n_colours=2)
    lesion_coherency = extractor(lesion, mask)[14:16]
    speckle_coherency = extractor(speckled, mask)[14:16]

    assert lesion_coherency.min() > 0.9
    assert speckle_coherency.min() < 0.3


def test_the_green_index_responds_to_an_unripe_fruit():
    """A green fruit must register in the green index and a red one must not."""
    green, mask = patched_apple(patches=(((40, 170, 80), 1.0),), seed=6)
    red, _ = patched_apple(patches=(((40, 40, 200), 1.0),), seed=6)
    extractor = DominantColourExtractor()

    assert extractor(green, mask)[35] > 90.0
    assert extractor(red, mask)[35] < 10.0


def test_the_decay_index_responds_to_dark_desaturated_tissue():
    """The decay indicator must fire on dark, colourless surface and not on ripe."""
    decayed, mask = patched_apple(patches=(((55, 58, 62), 1.0),), seed=7)
    ripe, _ = patched_apple(patches=(((40, 40, 200), 1.0),), seed=7)
    extractor = DominantColourExtractor()

    assert extractor(decayed, mask)[36] > 90.0
    assert extractor(ripe, mask)[36] < 10.0


# --------------------------------------------------------------------------- #
# Parameterisation, for the Phase 5 sub-experiments
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_colours", [3, 4, 5, 6])
def test_dimensionality_follows_the_cluster_count(apple, n_colours):
    extractor = DominantColourExtractor(n_colours=n_colours)
    assert extractor.dim == n_colours * 8 + 5
    assert extractor(*apple).shape == (extractor.dim,)


@pytest.mark.parametrize("space", ["LAB", "HSV", "RGB"])
def test_every_clustering_space_produces_a_valid_vector(apple, space):
    extractor = DominantColourExtractor(space=space)
    vector = extractor(*apple)
    assert vector.shape == (37,)
    assert np.all(np.isfinite(vector))


@pytest.mark.parametrize("blocks, expected", [("A", 28), ("AB", 32), ("ABC", 37)])
def test_block_selection_changes_the_length(apple, blocks, expected):
    """E1.5 compares block A alone, A+B and A+B+C."""
    extractor = DominantColourExtractor(blocks=blocks)
    assert extractor.dim == expected
    assert extractor(*apple).shape == (expected,)


def test_hsv_hue_statistics_survive_the_wrap_point():
    """A red fruit straddles the HSV hue wrap, where a plain mean would fail.

    Red sits at both ends of the hue range, so the arithmetic mean of its hue
    lands in the cyans - a colour nowhere on the fruit. The reported centroid
    hue must stay red.
    """
    image, mask = patched_apple(patches=(((40, 40, 200), 1.0),), seed=8)
    vector = DominantColourExtractor(space="HSV", n_colours=1)(image, mask)
    hue = vector[0]
    assert hue < 15.0 or hue > 165.0, f"hue {hue} is not red"


def test_an_invalid_space_is_rejected():
    with pytest.raises(ValueError, match="space must be one of"):
        DominantColourExtractor(space="CMYK")


def test_block_a_cannot_be_dropped():
    with pytest.raises(ValueError, match="block A"):
        DominantColourExtractor(blocks="BC")


def test_from_config_matches_the_configured_parameters():
    """The configured descriptor must be the 37-dimensional one the report quotes."""
    extractor = DominantColourExtractor.from_config(CONFIG)
    assert extractor.dim == 37
    assert extractor.space == "LAB"
    assert extractor.seed == CONFIG.seed
