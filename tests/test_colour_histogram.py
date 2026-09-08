"""Tests for the E1.1 colour-histogram baseline.

The baseline is only useful if it is a *fair* comparator, so the tests here
check two different kinds of property.

The first is the shared contract every extractor must satisfy - documented
length, no NaN or Inf, determinism, an empty mask refused rather than
described - because a baseline that quietly returned a wrong-length or
non-finite vector would make E1.1 a comparison against nothing.

The second is what makes E1.1 attributable: the baseline must see exactly the
pixels the descriptor sees. It applies the same specular predicate with the
same thresholds and the same fallback guard, and it must be measurably blind
to the two things the descriptor is built to capture - where a colour sits in
the frame, and how many bins it takes to say so. A baseline that accidentally
encoded spatial arrangement would beat the descriptor for the wrong reason.

Run from the project root::

    python -m pytest tests/test_colour_histogram.py -v
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
from features.colour_histogram import (  # noqa: E402
    MIN_ANALYSIS_PIXELS,
    ColourHistogramExtractor,
    colour_moments,
)
from features.t1_dominant_colour import DominantColourExtractor, specular_selector  # noqa: E402

CONFIG = get_config()
SIZE = 224


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def banded_apple(seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """A disc of three contiguous colour bands, plus its mask."""
    rng = np.random.default_rng(seed)
    image = np.full((SIZE, SIZE, 3), 235, dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, (SIZE // 2, SIZE // 2), 80, 255, thickness=-1)

    for index, colour in enumerate(((40, 40, 200), (40, 160, 90), (30, 45, 60))):
        image[index * SIZE // 3 : (index + 1) * SIZE // 3, :] = colour

    grain = rng.integers(-3, 4, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    image[mask == 0] = 235
    return image, mask


def shuffle_within_mask(image: np.ndarray, mask: np.ndarray, seed: int = 7) -> np.ndarray:
    """Permute the masked pixels' positions, leaving the colour multiset intact."""
    rng = np.random.default_rng(seed)
    shuffled = image.copy()
    positions = np.flatnonzero(mask.ravel())
    flat = shuffled.reshape(-1, 3)
    flat[positions] = flat[rng.permutation(positions)]
    return flat.reshape(image.shape)


@pytest.fixture(scope="module")
def apple() -> Tuple[np.ndarray, np.ndarray]:
    return banded_apple()


# --------------------------------------------------------------------------- #
# The shared contract
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bins", [8, 16, 32, 64])
def test_dimensionality_is_three_histograms_plus_nine_moments(bins: int) -> None:
    extractor = ColourHistogramExtractor(bins=bins)
    assert extractor.dim == 3 * bins + 9


@pytest.mark.parametrize("space", ["LAB", "HSV", "RGB"])
def test_vector_matches_declared_length_and_is_finite(
    apple: Tuple[np.ndarray, np.ndarray], space: str
) -> None:
    image, mask = apple
    extractor = ColourHistogramExtractor(space=space)
    vector = extractor(image, mask)
    assert vector.shape == (extractor.dim,)
    assert np.all(np.isfinite(vector))


def test_feature_names_cover_every_dimension() -> None:
    extractor = ColourHistogramExtractor()
    names = list(extractor.feature_names)
    assert len(names) == extractor.dim == 105
    assert len(set(names)) == len(names)
    assert names[0] == "T1H_hist_L_00"
    assert names[-3:] == ["T1H_mean_b", "T1H_std_b", "T1H_skew_b"]


def test_repeated_calls_return_byte_identical_vectors(
    apple: Tuple[np.ndarray, np.ndarray],
) -> None:
    image, mask = apple
    extractor = ColourHistogramExtractor()
    assert np.array_equal(extractor(image, mask), extractor(image, mask))


def test_empty_mask_is_refused_rather_than_described() -> None:
    image, _ = banded_apple()
    with pytest.raises(FeatureExtractionError):
        ColourHistogramExtractor()(image, np.zeros((SIZE, SIZE), dtype=np.uint8))


@pytest.mark.parametrize("space,bins", [("XYZ", 32), ("LAB", 0)])
def test_invalid_configuration_is_rejected_at_construction(space: str, bins: int) -> None:
    with pytest.raises(ValueError):
        ColourHistogramExtractor(space=space, bins=bins)


# --------------------------------------------------------------------------- #
# The histogram itself
# --------------------------------------------------------------------------- #

def test_each_channel_histogram_sums_to_one(apple: Tuple[np.ndarray, np.ndarray]) -> None:
    """Normalised per channel, so the vector describes colour and not fruit size."""
    image, mask = apple
    extractor = ColourHistogramExtractor(bins=32)
    vector = extractor(image, mask)
    for channel in range(3):
        assert vector[channel * 32 : (channel + 1) * 32].sum() == pytest.approx(1.0)


def test_histogram_ignores_the_size_of_the_fruit_in_the_frame() -> None:
    """The same colours over twice the area give the same normalised histogram."""
    image, mask = banded_apple()
    big = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(big, (SIZE // 2, SIZE // 2), 110, 255, thickness=-1)

    extractor = ColourHistogramExtractor(exclude_specular=False)
    small_vector = extractor(image, mask)
    big_vector = extractor(image, big)
    # Not equal - a larger disc samples the bands in different proportions -
    # but both must remain proper distributions rather than growing with area.
    for channel in range(3):
        window = slice(channel * 32, (channel + 1) * 32)
        assert small_vector[window].sum() == pytest.approx(1.0)
        assert big_vector[window].sum() == pytest.approx(1.0)


def test_hue_channel_bins_span_the_opencv_range_not_the_full_byte() -> None:
    """OpenCV packs 360 degrees of hue into ``[0, 180)``.

    Binning it over ``[0, 256)`` would leave the top 30% of every hue histogram
    permanently empty and crush the real hues into two thirds of the bins.
    """
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 255)  # Pure red: hue 0 in OpenCV's encoding.
    mask = np.full((SIZE, SIZE), 255, dtype=np.uint8)

    vector = ColourHistogramExtractor(space="HSV", exclude_specular=False)(image, mask)
    hue = vector[:32]
    assert hue[0] == pytest.approx(1.0)

    image[:, :] = (0, 255, 255)  # Pure yellow: hue 30 of 180, so bin 5 of 32.
    hue = ColourHistogramExtractor(space="HSV", exclude_specular=False)(image, mask)[:32]
    assert int(np.argmax(hue)) == 5


# --------------------------------------------------------------------------- #
# Colour moments
# --------------------------------------------------------------------------- #

def test_moments_recover_a_known_distribution() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    mean, std, skew = colour_moments(values)
    assert mean == pytest.approx(2.5)
    assert std == pytest.approx(np.std(values))
    assert skew == pytest.approx(np.cbrt(np.mean((values - 2.5) ** 3)))


def test_skewness_keeps_its_sign() -> None:
    """The sign is the whole content of a skewness, so a cube root is used."""
    right_tailed = np.array([1.0, 1.0, 1.0, 9.0])
    left_tailed = 10.0 - right_tailed
    assert colour_moments(right_tailed)[2] > 0
    assert colour_moments(left_tailed)[2] < 0


def test_constant_channel_has_zero_skewness_rather_than_nan() -> None:
    """A flat patch of colour is a real input, not a degenerate one."""
    mean, std, skew = colour_moments(np.full(64, 120.0))
    assert (mean, std, skew) == (120.0, 0.0, 0.0)

    image = np.full((SIZE, SIZE, 3), 120, dtype=np.uint8)
    mask = np.full((SIZE, SIZE), 255, dtype=np.uint8)
    vector = ColourHistogramExtractor(exclude_specular=False)(image, mask)
    assert np.all(np.isfinite(vector))


# --------------------------------------------------------------------------- #
# What makes E1.1 attributable
# --------------------------------------------------------------------------- #

def test_baseline_and_descriptor_exclude_exactly_the_same_pixels() -> None:
    """The two arms of E1.1 must see one pixel set, or the gap means nothing."""
    image, mask = banded_apple()
    image[100:110, 100:110] = 255  # A bright, colourless patch: a highlight.

    descriptor = DominantColourExtractor.from_config(CONFIG)
    baseline = ColourHistogramExtractor.from_config(CONFIG)
    assert baseline.exclude_specular is descriptor.exclude_specular
    assert baseline.specular_lightness == descriptor.specular_lightness
    assert baseline.specular_chroma == descriptor.specular_chroma
    assert baseline.space == descriptor.space

    boolean = mask.astype(bool)
    shared = specular_selector(
        image, boolean, baseline.specular_lightness, baseline.specular_chroma
    )
    _, descriptor_record = descriptor.extract_with_diagnostics(image, mask)
    _, baseline_record = baseline.extract_with_diagnostics(image, mask)

    assert baseline_record.specular_fraction == pytest.approx(
        descriptor_record.specular_fraction
    )
    assert baseline_record.analysis_pixels == int(np.count_nonzero(boolean & ~shared))
    assert baseline_record.analysis_pixels == descriptor_record.analysis_pixels


def test_specular_exclusion_never_starves_the_baseline() -> None:
    """A fruit that is almost all highlight is described including the highlight."""
    image = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, (SIZE // 2, SIZE // 2), 40, 255, thickness=-1)

    extractor = ColourHistogramExtractor()
    vector, record = extractor.extract_with_diagnostics(image, mask)
    assert record.specular_fallback is True
    assert record.analysis_pixels == record.mask_pixels
    assert record.analysis_pixels >= MIN_ANALYSIS_PIXELS
    assert np.all(np.isfinite(vector))


def test_baseline_is_blind_to_where_a_colour_sits() -> None:
    """The property that makes the descriptor's block B worth having.

    Rearranging the masked pixels leaves the colour multiset untouched, so a
    histogram and its moments cannot change. The descriptor's spatial
    coherency block does change, and E1.5 is what measures whether that
    difference is worth its four dimensions.
    """
    image, mask = banded_apple()
    extractor = ColourHistogramExtractor(exclude_specular=False)
    original = extractor(image, mask)
    shuffled = extractor(shuffle_within_mask(image, mask), mask)
    np.testing.assert_allclose(original, shuffled, rtol=0, atol=1e-12)


def test_baseline_is_longer_than_the_descriptor_it_is_compared_against() -> None:
    """E1.1 is only interesting if the descriptor wins on both axes at once."""
    assert ColourHistogramExtractor.from_config(CONFIG).dim == 105
    assert DominantColourExtractor.from_config(CONFIG).dim == 37


def test_from_config_reads_the_descriptor_block() -> None:
    """The two arms share one configuration block so they cannot be set apart."""
    baseline = ColourHistogramExtractor.from_config(CONFIG)
    assert baseline.space == str(CONFIG.t1_colour["space"]).upper()
    assert baseline.exclude_specular is bool(CONFIG.t1_colour["exclude_specular"])
    assert baseline.bins == 32
