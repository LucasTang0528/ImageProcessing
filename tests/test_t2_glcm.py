"""Tests for T2, the GLCM texture descriptor.

Beyond the shared contract, two things here fail silently if unguarded and so
are tested directly:

**Background exclusion.** Zeroing masked-out pixels makes zero a huge uniform
grey level whose co-occurrences with the fruit boundary dominate contrast and
dissimilarity, so the descriptor ends up measuring the shape of the mask. The
tests check that the vector does not move when the background colour changes.

**Correlation on degenerate matrices.** Correlation divides by the marginal
standard deviations. scikit-image returns 1.0 both for a genuinely uniform
crop and for a matrix containing no pairs at all; the second is an extreme
value invented from no evidence, and is overridden here.

Run from the project root::

    python -m pytest tests/test_t2_glcm.py -v
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
from skimage.feature import graycomatrix, graycoprops  # noqa: E402

from config import get_config  # noqa: E402
from features.base import FeatureExtractionError  # noqa: E402
from features.t2_glcm import PROPERTIES, GLCMExtractor  # noqa: E402

CONFIG = get_config()
SIZE = 224


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def textured_apple(
    background: int = 235,
    period: int = 4,
    amplitude: int = 40,
    base: int = 120,
    radius: int = 80,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """A disc carrying a regular stripe texture, on a flat background.

    A deterministic texture is used rather than noise so that the co-occurrence
    statistics are predictable: stripes running down the image produce high
    contrast horizontally and near-zero contrast vertically.
    """
    rng = np.random.default_rng(seed)
    columns = np.arange(SIZE)
    stripes = base + amplitude * ((columns // period) % 2)
    grey = np.tile(stripes, (SIZE, 1)).astype(np.uint8)

    image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    grain = rng.integers(-2, 3, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, (SIZE // 2, SIZE // 2), radius, 255, thickness=-1)
    image[mask == 0] = background
    return image, mask


def uniform_apple(grey: int = 120, radius: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """A perfectly flat disc: the zero-variance case for correlation."""
    image = np.full((SIZE, SIZE, 3), 235, dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, (SIZE // 2, SIZE // 2), radius, 255, thickness=-1)
    image[mask > 0] = grey
    return image, mask


@pytest.fixture(scope="module")
def apple() -> Tuple[np.ndarray, np.ndarray]:
    """The default textured fruit used by most tests."""
    return textured_apple()


# --------------------------------------------------------------------------- #
# The shared contract
# --------------------------------------------------------------------------- #

def test_vector_has_the_documented_length(apple):
    """5 properties x 2 distances x 4 angles = 40."""
    extractor = GLCMExtractor()
    assert extractor.dim == 40
    assert extractor(*apple).shape == (40,)


def test_vector_is_finite(apple):
    assert np.all(np.isfinite(GLCMExtractor()(*apple)))


def test_vector_is_deterministic_across_calls(apple):
    extractor = GLCMExtractor()
    assert np.array_equal(extractor(*apple), extractor(*apple))


def test_two_instances_agree(apple):
    assert np.array_equal(GLCMExtractor()(*apple), GLCMExtractor()(*apple))


def test_feature_names_match_the_vector_length():
    extractor = GLCMExtractor()
    assert len(extractor.feature_names) == extractor.dim


def test_an_empty_mask_is_refused(apple):
    image, _ = apple
    with pytest.raises(FeatureExtractionError):
        GLCMExtractor()(image, np.zeros((SIZE, SIZE), dtype=np.uint8))


def test_the_extractor_does_not_modify_its_inputs(apple):
    image, mask = apple
    image_before, mask_before = image.copy(), mask.copy()
    GLCMExtractor()(image, mask)
    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)


# --------------------------------------------------------------------------- #
# Background exclusion
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("background", [0, 60, 128, 200, 255])
def test_the_background_colour_cannot_change_the_texture(background):
    """The property the ignore level exists to guarantee.

    The fruit is identical in every case and only the surrounding pixels
    differ. If background pixels reached the co-occurrence counts, contrast
    and dissimilarity would swing wildly with the backdrop - and a dataset
    whose classes are photographed against different backdrops would then be
    separable by backdrop alone.
    """
    reference, mask = textured_apple(background=235)
    variant, _ = textured_apple(background=background)

    extractor = GLCMExtractor()
    assert np.allclose(extractor(reference, mask), extractor(variant, mask), atol=1e-9)


def test_a_black_background_does_not_inflate_contrast():
    """The specific failure mode: zero as a giant artificial grey level.

    Against a black backdrop, boundary pairs would span the full grey range
    and dominate contrast. Measured against a mid-grey backdrop, the value
    must not move at all.
    """
    on_black, mask = textured_apple(background=0)
    on_grey, _ = textured_apple(background=128)
    extractor = GLCMExtractor()

    black_contrast = extractor(on_black, mask)[:8]
    grey_contrast = extractor(on_grey, mask)[:8]
    assert np.allclose(black_contrast, grey_contrast, atol=1e-9)


def test_quantisation_reserves_zero_for_the_background():
    """Fruit levels must occupy 1..levels, leaving 0 free as the ignore value."""
    extractor = GLCMExtractor(levels=32)
    grey = np.arange(256, dtype=np.uint8).reshape(16, 16)
    mask = np.ones((16, 16), dtype=bool)
    mask[0, 0] = False

    quantised = extractor.quantise(grey, mask)
    assert quantised[0, 0] == 0
    assert quantised[mask].min() >= 1
    assert quantised[mask].max() <= 32


def test_deleting_the_ignore_level_preserves_the_properties():
    """The identity the background-exclusion scheme relies on.

    Removing row and column 0 shifts every grey index down by one. Contrast,
    dissimilarity and homogeneity depend only on ``|i - j|``; energy only on
    the probabilities; correlation is invariant to a shift of the marginal
    means. If that were not so, excluding the background would silently
    change the texture measurement.
    """
    rng = np.random.default_rng(0)
    patch = rng.integers(1, 33, size=(40, 40)).astype(np.uint8)

    raw = graycomatrix(patch, [1], [0], levels=33, symmetric=True, normed=False)
    full = raw.astype(np.float64)
    full /= full.sum(axis=(0, 1), keepdims=True)

    sliced = raw[1:, 1:, :, :].astype(np.float64)
    sliced /= sliced.sum(axis=(0, 1), keepdims=True)

    for prop in PROPERTIES:
        assert graycoprops(full, prop).ravel()[0] == pytest.approx(
            graycoprops(sliced, prop).ravel()[0], rel=1e-9
        ), prop


# --------------------------------------------------------------------------- #
# Degenerate matrices
# --------------------------------------------------------------------------- #

def test_a_uniform_fruit_produces_finite_values():
    """Zero variance is the classic correlation blow-up, and must not reach NaN."""
    extractor = GLCMExtractor()
    vector = extractor(*uniform_apple())
    assert np.all(np.isfinite(vector))


def test_a_uniform_fruit_reports_the_conventional_correlation():
    """A flat surface is conventionally perfectly correlated with itself.

    scikit-image's convention is kept for this case, because the crop does
    contain real measurements - they simply do not vary.
    """
    vector = GLCMExtractor()(*uniform_apple())
    correlation = vector[32:]  # Fifth property block.
    assert np.allclose(correlation, 1.0, atol=1e-9)


def test_a_crop_with_no_pairs_reports_neutral_values_not_perfect_correlation():
    """The case where scikit-image invents an extreme value from no evidence.

    A crop narrower than the offset yields a co-occurrence matrix with no
    entries. scikit-image reports correlation 1.0 - perfect correlation, from
    nothing measured at all. A neutral 0.0 is emitted instead, and the
    occurrence is recorded.
    """
    image = np.full((SIZE, SIZE, 3), 120, dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[100:140, 100:101] = 255  # One pixel wide: no horizontal pairs exist.

    extractor = GLCMExtractor(distances=(1,), angles_deg=(0.0,))
    vector, record = extractor.extract_with_diagnostics(image, mask)

    assert record.degenerate_slices == 1
    assert np.all(np.isfinite(vector))
    correlation = vector[PROPERTIES.index("correlation")]
    assert correlation == 0.0, "an unmeasured matrix must not claim perfect correlation"


def test_degenerate_handling_does_not_fire_on_normal_fruit(apple):
    """The guard must be inert on real inputs, not quietly rewriting them."""
    extractor = GLCMExtractor()
    _, record = extractor.extract_with_diagnostics(*apple)
    assert record.degenerate_slices == 0


# --------------------------------------------------------------------------- #
# Texture behaviour
# --------------------------------------------------------------------------- #

def test_vertical_stripes_give_directional_contrast(apple):
    """The descriptor must actually respond to orientation.

    Stripes running down the image vary across columns and not down rows, so
    contrast at 0 degrees must far exceed contrast at 90 degrees. Without this
    the angle dimensions could be constant and carry nothing.
    """
    vector = GLCMExtractor()(*apple)
    # Contrast block: distance 1 at angles 0, 45, 90, 135.
    horizontal, vertical = vector[0], vector[2]
    assert horizontal > 5.0 * vertical, (horizontal, vertical)


def test_a_rough_surface_has_higher_contrast_than_a_smooth_one():
    """Contrast must order surfaces the way the descriptor claims to."""
    smooth, mask = textured_apple(amplitude=4, period=8)
    rough, _ = textured_apple(amplitude=60, period=2)
    extractor = GLCMExtractor()
    assert extractor(rough, mask)[0] > extractor(smooth, mask)[0]


def test_energy_is_higher_on_a_uniform_surface_than_a_textured_one(apple):
    """Energy measures concentration, so a flat fruit must score higher."""
    extractor = GLCMExtractor()
    uniform_energy = extractor(*uniform_apple())[24]
    textured_energy = extractor(*apple)[24]
    assert uniform_energy > textured_energy


# --------------------------------------------------------------------------- #
# Parameterisation, for the Phase 5 sensitivity study
# --------------------------------------------------------------------------- #

def test_the_angle_averaged_toggle_collapses_the_angle_axis(apple):
    """5 properties x 2 distances = 10 when angles are averaged."""
    extractor = GLCMExtractor(angle_averaged=True)
    assert extractor.dim == 10
    assert extractor(*apple).shape == (10,)


def test_angle_averaging_is_the_mean_of_the_per_angle_values(apple):
    """The toggle must average the angles, not select or reorder them."""
    full = GLCMExtractor(angle_averaged=False)(*apple)
    averaged = GLCMExtractor(angle_averaged=True)(*apple)

    for index in range(10):
        block = full[index * 4 : index * 4 + 4]
        assert averaged[index] == pytest.approx(block.mean(), rel=1e-9)


def test_angle_averaging_is_more_rotation_stable_than_the_full_descriptor(apple):
    """The reason the toggle exists, stated as a measurement.

    Averaging over the four Haralick directions removes the descriptor's
    orientation preference, so a rotated fruit should move it less than it
    moves the per-angle descriptor.
    """
    image, mask = apple
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotated_mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

    def relative_shift(extractor: GLCMExtractor) -> float:
        before = extractor(image, mask)
        after = extractor(rotated, rotated_mask)
        return float(np.linalg.norm(after - before) / (np.linalg.norm(before) + 1e-12))

    assert relative_shift(GLCMExtractor(angle_averaged=True)) < relative_shift(
        GLCMExtractor(angle_averaged=False)
    )


@pytest.mark.parametrize("levels", [8, 16, 32, 64])
def test_quantisation_levels_are_parameterised(apple, levels):
    extractor = GLCMExtractor(levels=levels)
    vector = extractor(*apple)
    assert vector.shape == (40,)
    assert np.all(np.isfinite(vector))


@pytest.mark.parametrize("distances", [(1,), (1, 2), (1, 2, 3), (1, 3, 5)])
def test_distance_sets_are_parameterised(apple, distances):
    extractor = GLCMExtractor(distances=distances)
    assert extractor.dim == 5 * len(distances) * 4
    assert extractor(*apple).shape == (extractor.dim,)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="distances"):
        GLCMExtractor(distances=(0, 1))
    with pytest.raises(ValueError, match="levels"):
        GLCMExtractor(levels=1)
    with pytest.raises(ValueError, match="angle"):
        GLCMExtractor(angles_deg=())


def test_from_config_matches_the_configured_parameters():
    """The configured descriptor must be the 40-dimensional one the report quotes."""
    extractor = GLCMExtractor.from_config(CONFIG)
    assert extractor.dim == 40
    assert extractor.levels == 32
    assert extractor.distances == (1, 2)
    assert extractor.angle_averaged is False
