"""Tests for the E2.1 first-order intensity baseline.

Two kinds of property are checked.

The first is the shared contract: documented length, no NaN or Inf, determinism,
an empty mask refused rather than described. A baseline that quietly returned a
non-finite vector would make E2.1 a comparison against nothing, and every
configuration downstream would train on it in silence.

The second is what makes E2.1 mean something. The baseline must describe the
*distribution* of grey levels and nothing about their arrangement, because
arrangement is precisely what the co-occurrence matrix adds and therefore what
the experiment is trying to isolate. If the baseline accidentally encoded
spatial structure it would beat the GLCM for the wrong reason, or lose to it for
one. It must also read the same pixels the GLCM pairs, with the background
excluded rather than zero-filled - zero-filling is the trap T2's own docstring
describes, and it distorts moments exactly as thoroughly as it distorts a
co-occurrence matrix.

Run from the project root::

    python -m pytest tests/test_intensity_baseline.py -v
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
from features.intensity_baseline import (  # noqa: E402
    MOMENT_NAMES,
    IntensityMomentExtractor,
    intensity_moments,
)
from features.t2_glcm import GLCMExtractor  # noqa: E402

CONFIG = get_config()
SIZE = 224


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def textured_apple(seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """A disc with a grey gradient and grain, plus its mask."""
    rng = np.random.default_rng(seed)
    gradient = np.tile(np.linspace(40, 210, SIZE, dtype=np.float64), (SIZE, 1))
    grain = rng.normal(0.0, 12.0, size=(SIZE, SIZE))
    grey = np.clip(gradient + grain, 0, 255).astype(np.uint8)

    image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, (SIZE // 2, SIZE // 2), 80, 255, thickness=-1)
    return image, mask


def shuffle_within_mask(image: np.ndarray, mask: np.ndarray, seed: int = 7) -> np.ndarray:
    """Permute the masked pixels' positions, leaving the value multiset intact."""
    rng = np.random.default_rng(seed)
    shuffled = image.copy()
    positions = np.flatnonzero(mask.ravel())
    flat = shuffled.reshape(-1, 3)
    flat[positions] = flat[rng.permutation(positions)]
    return flat.reshape(image.shape)


@pytest.fixture(scope="module")
def apple() -> Tuple[np.ndarray, np.ndarray]:
    return textured_apple()


# --------------------------------------------------------------------------- #
# The shared contract
# --------------------------------------------------------------------------- #

def test_dimensionality_is_four_moments() -> None:
    extractor = IntensityMomentExtractor()
    assert extractor.dim == len(MOMENT_NAMES) == 4


def test_vector_matches_declared_length_and_is_finite(
    apple: Tuple[np.ndarray, np.ndarray],
) -> None:
    image, mask = apple
    extractor = IntensityMomentExtractor.from_config(CONFIG)
    vector = extractor(image, mask)
    assert vector.shape == (4,)
    assert np.all(np.isfinite(vector))


def test_feature_names_cover_every_dimension() -> None:
    extractor = IntensityMomentExtractor()
    names = list(extractor.feature_names)
    assert names == ["T2I_mean", "T2I_variance", "T2I_skewness", "T2I_kurtosis"]
    assert len(names) == extractor.dim


def test_repeated_calls_return_byte_identical_vectors(
    apple: Tuple[np.ndarray, np.ndarray],
) -> None:
    image, mask = apple
    extractor = IntensityMomentExtractor()
    assert np.array_equal(extractor(image, mask), extractor(image, mask))


def test_empty_mask_is_refused_rather_than_described() -> None:
    image, _ = textured_apple()
    with pytest.raises(FeatureExtractionError):
        IntensityMomentExtractor()(image, np.zeros((SIZE, SIZE), dtype=np.uint8))


def test_diagnostics_record_one_entry_per_call(
    apple: Tuple[np.ndarray, np.ndarray],
) -> None:
    image, mask = apple
    extractor = IntensityMomentExtractor()
    extractor(image, mask)
    extractor(image, mask)
    assert len(extractor.diagnostics) == 2
    assert extractor.diagnostics[0].mask_pixels == int(np.count_nonzero(mask))
    extractor.reset_diagnostics()
    assert extractor.diagnostics == []


# --------------------------------------------------------------------------- #
# The moments themselves
# --------------------------------------------------------------------------- #

def test_moments_match_the_textbook_definitions() -> None:
    """Standardised third moment, excess fourth moment."""
    values = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    mean, variance, skewness, kurtosis = intensity_moments(values)

    deviation = values - values.mean()
    sigma = np.sqrt(np.mean(deviation ** 2))
    assert mean == pytest.approx(values.mean())
    assert variance == pytest.approx(np.mean(deviation ** 2))
    assert skewness == pytest.approx(np.mean(deviation ** 3) / sigma ** 3)
    assert kurtosis == pytest.approx(np.mean(deviation ** 4) / sigma ** 4 - 3.0)


def test_a_normal_sample_has_near_zero_excess_kurtosis() -> None:
    """Excess, not raw: a Gaussian must read about 0, not about 3."""
    values = np.random.default_rng(0).normal(128.0, 20.0, size=200_000)
    assert intensity_moments(values)[3] == pytest.approx(0.0, abs=0.05)


def test_skewness_keeps_its_sign() -> None:
    right_tailed = np.array([1.0, 1.0, 1.0, 1.0, 9.0])
    assert intensity_moments(right_tailed)[2] > 0
    assert intensity_moments(10.0 - right_tailed)[2] < 0


def test_a_constant_region_reports_zeros_rather_than_nan() -> None:
    """A flat patch of colour is a real input, not a degenerate one."""
    assert intensity_moments(np.full(64, 120.0)) == (120.0, 0.0, 0.0, 0.0)

    image = np.full((SIZE, SIZE, 3), 120, dtype=np.uint8)
    mask = np.full((SIZE, SIZE), 255, dtype=np.uint8)
    extractor = IntensityMomentExtractor()
    vector, record = extractor.extract_with_diagnostics(image, mask)
    assert record.constant is True
    assert np.all(np.isfinite(vector))


def test_an_empty_sample_reports_zeros() -> None:
    assert intensity_moments(np.empty(0)) == (0.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# What makes E2.1 attributable
# --------------------------------------------------------------------------- #

def test_the_baseline_is_blind_to_arrangement(
    apple: Tuple[np.ndarray, np.ndarray],
) -> None:
    """The property that makes the co-occurrence matrix worth having.

    Rearranging the masked pixels leaves the grey-level multiset untouched, so
    every first-order moment must be unchanged. The GLCM's answer does change,
    and the size of that change is what E2.1 is measuring.
    """
    image, mask = apple
    extractor = IntensityMomentExtractor()
    original = extractor(image, mask)
    shuffled = extractor(shuffle_within_mask(image, mask), mask)
    np.testing.assert_allclose(original, shuffled, rtol=0, atol=1e-9)


def test_the_glcm_is_not_blind_to_arrangement(
    apple: Tuple[np.ndarray, np.ndarray],
) -> None:
    """The other half of the same claim; without it the test above proves little."""
    image, mask = apple
    glcm = GLCMExtractor(distances=(1,), angles_deg=(0.0,), levels=16)
    original = glcm(image, mask)
    shuffled = glcm(shuffle_within_mask(image, mask), mask)
    assert not np.allclose(original, shuffled, rtol=0, atol=1e-6)


def test_the_baseline_reads_the_pixels_the_glcm_pairs(
    apple: Tuple[np.ndarray, np.ndarray],
) -> None:
    """Same mask, background excluded rather than zero-filled.

    A zero-filled background would drag the mean toward zero and inflate the
    variance enormously; comparing that against the GLCM would be comparing two
    different images, not two different descriptors.
    """
    image, mask = apple
    extractor = IntensityMomentExtractor()
    vector, record = extractor.extract_with_diagnostics(image, mask)

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inside = grey[mask.astype(bool)].astype(np.float64)
    assert record.mask_pixels == inside.size
    np.testing.assert_allclose(vector, np.asarray(intensity_moments(inside)))

    zero_filled = grey.astype(np.float64).copy()
    zero_filled[~mask.astype(bool)] = 0.0
    assert intensity_moments(zero_filled.ravel())[0] < vector[0] / 2


def test_the_baseline_is_far_shorter_than_the_descriptor_it_faces() -> None:
    """E2.1 is only interesting because the arms are 4 against 40."""
    assert IntensityMomentExtractor.from_config(CONFIG).dim == 4
    assert GLCMExtractor.from_config(CONFIG).dim == 40


def test_from_config_reads_only_the_shared_seed() -> None:
    """It has no distance, no angle and no quantisation, and that is the point."""
    built = IntensityMomentExtractor.from_config(CONFIG)
    assert built.seed == CONFIG.seed
    assert not hasattr(built, "levels")
    assert not hasattr(built, "distances")
