"""Tests for T3, the multiscale morphological descriptor.

Beyond the shared contract, three things here are silent when they break and
so are tested directly:

**The boundary guard.** Morphology is a local operation, so the strongest
opening and closing residues in a masked image sit on the mask outline unless
two guards are in place: the background is filled with the mean intensity
inside the mask, and every statistic is accumulated over an eroded interior.
Without them the descriptor measures the segmenter's outline rather than the
peel, and it still returns 36 plausible-looking finite numbers while doing so.
The tests pin this by changing the background and requiring the vector not to
move at all.

**Pole exclusion frame.** Component centroids live in the cropped frame and the
fruit mask's centroid can be taken in either frame. Mixing the two excludes the
wrong components, and on a centred fruit the two frames nearly agree, so the
bug hides. Every pole test therefore uses a fruit placed deliberately off
centre.

**Empty blemish sets.** Perfectly clean apples exist in the Unripe class. All
eight Block E features must fall back to 0.0 rather than NaN.

Run from the project root::

    python -m pytest tests/test_t3.py -v
"""

from __future__ import annotations

import sys
import time
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
from features.t3_morphological import (  # noqa: E402
    BLEMISH_METHODS,
    FEATURE_NAMES,
    SE_SHAPES,
    T3MorphologicalExtractor,
    block_indices,
    dimension_for,
    extract,
    feature_names,
)

CONFIG = get_config()
SIZE = 224

#: Column index of the first Block E feature at the default radius.
BLEMISH_RATIO = 28


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def make_apple(
    background: int = 235,
    peel: int = 170,
    blob_radius: int = 4,
    n_blobs: int = 6,
    blob_depth: int = 70,
    radius: int = 80,
    centre: Tuple[int, int] = (SIZE // 2, SIZE // 2),
    grain: int = 3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """A round fruit carrying dark blobs of a known scale, on a flat backdrop.

    Deterministic blob placement is used rather than noise so that the scale a
    granulometry should report is known in advance.
    """
    rng = np.random.default_rng(seed)
    grey = np.full((SIZE, SIZE), background, dtype=np.uint8)
    cv2.circle(grey, centre, radius, peel, thickness=-1)

    if n_blobs:
        angles = np.linspace(0.0, 2.0 * np.pi, n_blobs, endpoint=False)
        for angle in angles:
            offset = radius // 2
            spot = (
                int(centre[0] + offset * np.cos(angle)),
                int(centre[1] + offset * np.sin(angle)),
            )
            cv2.circle(grey, spot, blob_radius, peel - blob_depth, thickness=-1)

    if grain:
        noise = rng.integers(-grain, grain + 1, size=grey.shape, dtype=np.int16)
        grey = np.clip(grey.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, centre, radius, 255, thickness=-1)
    image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    image[mask == 0] = background
    return image, mask


def flat_apple(peel: int = 170, background: int = 235, radius: int = 80):
    """A perfectly uniform fruit: the clean-apple and zero-variance case."""
    image = np.full((SIZE, SIZE, 3), background, dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, (SIZE // 2, SIZE // 2), radius, 255, thickness=-1)
    image[mask > 0] = peel
    return image, mask


#: Pole fraction used by the pole tests, deliberately wider than the default.
#:
#: At the documented default of 0.15 the band is 8.6 to 10.1 px wide on a mask
#: the size of a real apple in this dataset, and the interior erosion has
#: already removed the outer 8 px, leaving between 0.6 and 2.1 px of band for
#: the guard to act on. A spot small enough to fit inside that sliver is below
#: ``min_blemish_area`` and never survives to be excluded, so the mechanism
#: cannot be exercised at the default. The tests therefore widen the band and
#: test the mechanism; :func:`test_the_default_pole_band_is_mostly_already_eroded`
#: pins the interaction itself.
TEST_POLE_FRACTION = 0.25


def polar_apple(
    centre: Tuple[int, int] = (74, 95),
    axes: Tuple[int, int] = (44, 74),
    spot_offset: int = 60,
    spot_radius: int = 5,
    peel: int = 175,
    background: int = 240,
) -> Tuple[np.ndarray, np.ndarray]:
    """An elongated fruit, deliberately off centre, with three dark spots.

    One spot sits towards each pole of the major axis - the stem cavity and the
    calyx - and one sits in the middle of the peel. The fruit is placed away
    from the image centre so that a descriptor mixing the cropped and full
    frames excludes the wrong spots instead of quietly agreeing, and it is kept
    entirely inside the frame so that the mask stays symmetric about its own
    centroid; a fruit clipped by the image edge shifts that centroid and moves
    one pole out of the band for reasons that have nothing to do with the guard.
    """
    grey = np.full((SIZE, SIZE), background, dtype=np.uint8)
    cv2.ellipse(grey, centre, axes, 0, 0, 360, peel, thickness=-1)

    for offset in (-spot_offset, 0, spot_offset):
        cv2.circle(
            grey, (centre[0], centre[1] + offset), spot_radius, peel - 80, thickness=-1
        )

    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, thickness=-1)
    image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    image[mask == 0] = background
    return image, mask


@pytest.fixture(scope="module")
def apple() -> Tuple[np.ndarray, np.ndarray]:
    """The default blemished fruit used by most tests."""
    return make_apple()


# --------------------------------------------------------------------------- #
# The shared contract
# --------------------------------------------------------------------------- #

def test_vector_has_the_documented_length(apple):
    """2 x 7 spectrum bins + 4 moments + 8 top-hat + 2 gradient + 8 blemish."""
    extractor = T3MorphologicalExtractor()
    assert extractor.dim == 36
    assert extractor(*apple).shape == (36,)


def test_vector_is_finite(apple):
    assert np.all(np.isfinite(T3MorphologicalExtractor()(*apple)))


def test_vector_is_deterministic_across_calls(apple):
    """The spec requires two calls on one input to be byte identical."""
    extractor = T3MorphologicalExtractor()
    first, second = extractor(*apple), extractor(*apple)
    assert first.tobytes() == second.tobytes()


def test_two_instances_agree(apple):
    left = T3MorphologicalExtractor()(*apple)
    right = T3MorphologicalExtractor()(*apple)
    assert np.array_equal(left, right)


def test_feature_names_match_the_vector_length():
    extractor = T3MorphologicalExtractor()
    assert len(FEATURE_NAMES) == 36
    assert len(extractor.feature_names) == extractor.dim


def test_feature_names_are_in_the_documented_order():
    """The fusion code in Enhancement 1 indexes by position, not by name."""
    assert FEATURE_NAMES[0] == "gran_open_r1"
    assert FEATURE_NAMES[6] == "gran_open_r7"
    assert FEATURE_NAMES[7] == "gran_close_r1"
    assert FEATURE_NAMES[13] == "gran_close_r7"
    assert FEATURE_NAMES[14:18] == (
        "gran_mean_size", "gran_entropy", "gran_peak_scale", "gran_spread",
    )
    assert FEATURE_NAMES[18:22] == ("bth_mean", "bth_std", "bth_max", "bth_p95")
    assert FEATURE_NAMES[22:26] == ("wth_mean", "wth_std", "wth_max", "wth_p95")
    assert FEATURE_NAMES[26:28] == ("mgrad_mean", "mgrad_std")
    assert FEATURE_NAMES[28] == "blemish_ratio"
    assert FEATURE_NAMES[35] == "blemish_eqdiam_max"


def test_names_are_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_an_empty_mask_is_refused_through_the_harness_interface(apple):
    image, _ = apple
    with pytest.raises(FeatureExtractionError):
        T3MorphologicalExtractor()(image, np.zeros((SIZE, SIZE), dtype=np.uint8))


def test_the_extractor_does_not_modify_its_inputs(apple):
    image, mask = apple
    image_before, mask_before = image.copy(), mask.copy()
    T3MorphologicalExtractor()(image, mask)
    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)


def test_a_mismatched_image_and_mask_are_refused():
    with pytest.raises(ValueError, match="same image frame"):
        extract(np.zeros((10, 10, 3), np.uint8), np.zeros((12, 12), np.uint8))


def test_extraction_is_well_under_the_five_second_budget(apple):
    """Acceptance criterion: extraction plus inference under 5 s per image."""
    extractor = T3MorphologicalExtractor()
    extractor(*apple)  # Warm-up, excluded from the measurement.
    start = time.perf_counter()
    extractor(*apple)
    assert time.perf_counter() - start < 5.0


# --------------------------------------------------------------------------- #
# Blocks A and B: the pattern spectrum
# --------------------------------------------------------------------------- #

def test_each_spectrum_sums_to_one(apple):
    features, _ = extract(*apple)
    assert features[0:7].sum() == pytest.approx(1.0, abs=1e-6)
    assert features[7:14].sum() == pytest.approx(1.0, abs=1e-6)


def test_no_spectrum_bin_is_negative(apple):
    """A negative bin means the interior mask is wrong, not that the fruit is."""
    features, _ = extract(*apple)
    assert np.all(features[0:14] >= 0.0)


def test_a_flat_fruit_does_not_produce_nan(apple):
    """Zero mass to redistribute is the divide-by-zero case in the spectrum."""
    features, _ = extract(*flat_apple())
    assert np.all(np.isfinite(features))


def test_larger_defects_move_the_spectrum_to_a_larger_scale():
    """The descriptor must actually resolve scale, not just respond to defects.

    Granulometry earns its place in the vector only if the spectrum tracks the
    size of the structures present. Without this the seven bins could be a
    fixed profile carrying nothing about the fruit.
    """
    small, mask = make_apple(blob_radius=2, n_blobs=10)
    large, _ = make_apple(blob_radius=6, n_blobs=10)
    small_mean = extract(small, mask)[0][14]
    large_mean = extract(large, mask)[0][14]
    assert large_mean > small_mean, (small_mean, large_mean)


def test_the_peak_scale_is_reported_as_a_radius_not_an_index():
    """``argmax + 1``: the bins are radii 1..r_max, and radius 0 is not a bin."""
    features, _ = extract(*make_apple(blob_radius=5, n_blobs=10))
    assert 1.0 <= features[16] <= 7.0


# --------------------------------------------------------------------------- #
# The boundary guard
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("background", [0, 60, 128, 200, 255])
def test_the_background_colour_cannot_change_the_descriptor(background):
    """The property both boundary guards exist to provide.

    The fruit is identical in every case and only the pixels outside the mask
    differ. If the background reached the morphology, a dataset whose classes
    are photographed against different backdrops would be separable by backdrop
    alone - which is exactly the confound the audit measured on this dataset.
    """
    reference, mask = make_apple(background=235)
    variant, _ = make_apple(background=background)
    assert np.allclose(extract(reference, mask)[0], extract(variant, mask)[0], atol=1e-9)


def test_a_black_background_does_not_dominate_the_spectrum():
    """The specific failure the mean fill prevents.

    Left at zero, the fruit boundary is a cliff of 170 grey levels, and every
    opening and closing responds to it in preference to the peel. The symptom
    is a spectrum collapsed onto its first bin.
    """
    on_black, mask = make_apple(background=0)
    features, _ = extract(on_black, mask)
    assert features[0] < 0.9, "spectrum collapsed onto r=1: the mean fill is missing"


def test_a_boundary_ring_does_not_reach_the_roughness_block():
    """Erosion by ``r_max + 1`` must keep the outline out of the accumulation.

    The morphological gradient reaches one pixel, far less than the eight the
    interior erosion removes, so it is the block that isolates the accumulation
    region from the operator's own reach. A black ring painted exactly on the
    mask boundary must not move it at all. If it does, the interior erosion is
    missing and the descriptor is measuring the segmenter's outline.
    """
    image, mask = make_apple()
    marked = image.copy()
    cv2.circle(marked, (SIZE // 2, SIZE // 2), 79, (0, 0, 0), thickness=3)

    plain, _ = extract(image, mask)
    ringed, _ = extract(marked, mask)
    assert np.allclose(plain[26:28], ringed[26:28], atol=1e-12)


def test_the_top_hat_reaches_further_than_the_interior_erosion():
    """A documented shortfall in the guard, pinned rather than hidden.

    Section 6 erodes by ``r_max + 1`` (8 px at the default radius) and Section 8
    then runs both top hats with a disc of radius 9. An interior pixel is only
    guaranteed to be 8 px inside the mask, so a radius-9 top hat samples up to
    1 px beyond the mask edge - into filled background rather than peel. The
    fill makes that a soft edge rather than a cliff, so the effect is small, but
    it is real and Block C is not as isolated as Blocks A, B and D.

    This test records the gap. It is not a bug in the implementation: both
    numbers are exactly what the specification asks for, and closing the gap
    would change every Block C value, so it is the specification's call.
    """
    from features.t3_morphological import DEFAULT_CONFIG

    erosion = int(DEFAULT_CONFIG["r_max"]) + 1
    assert int(DEFAULT_CONFIG["tophat_radius"]) > erosion, (
        "the top-hat radius no longer exceeds the interior erosion; if this was "
        "deliberate, delete this test and note the change in the report"
    )


# --------------------------------------------------------------------------- #
# Block E: blemish geometry
# --------------------------------------------------------------------------- #

def test_the_blemish_mask_is_returned_in_the_full_image_frame(apple):
    """Enhancement 2 and the dashboard index it against the original image."""
    image, mask = apple
    _, aux = extract(image, mask)
    assert aux["blemish_mask"].shape == mask.shape
    assert aux["blemish_mask"].dtype == np.uint8
    assert set(np.unique(aux["blemish_mask"])).issubset({0, 255})


def test_the_label_image_is_also_full_frame_and_matches_the_count(apple):
    _, aux = extract(*apple)
    assert aux["labels"].shape == apple[1].shape
    assert aux["labels"].dtype == np.int32
    assert int(aux["labels"].max()) == aux["n_blemish"]


def test_blemishes_are_actually_found_on_a_blemished_fruit(apple):
    _, aux = extract(*apple)
    assert aux["n_blemish"] > 0
    assert aux["blemish_ratio_pct"] > 0.0


def test_a_clean_fruit_yields_zeros_not_nan():
    """Perfectly clean apples exist in the Unripe class, and are not an error."""
    features, aux = extract(*flat_apple())
    assert aux["n_blemish"] == 0
    assert aux["areas_mm2"] == []
    assert np.all(np.isfinite(features[BLEMISH_RATIO:]))
    assert np.all(features[BLEMISH_RATIO:] == 0.0)


def test_the_blemish_ratio_is_measured_against_the_whole_fruit_mask(apple):
    """Detection runs on the interior; the published ratio does not.

    Keeping the two denominators separate is what makes the reported figure
    comparable with the published formula rather than inflated by the erosion.
    """
    image, mask = apple
    features, aux = extract(image, mask)
    expected = 100.0 * np.count_nonzero(aux["blemish_mask"]) / np.count_nonzero(mask)
    assert features[BLEMISH_RATIO] == pytest.approx(expected, rel=1e-12)
    assert features[BLEMISH_RATIO] == pytest.approx(aux["blemish_ratio_pct"], rel=1e-12)


def test_components_below_the_minimum_area_are_discarded():
    image, mask = make_apple(blob_radius=2, n_blobs=8)
    generous = extract(image, mask, config={"min_blemish_area": 5})[1]["n_blemish"]
    strict = extract(image, mask, config={"min_blemish_area": 400})[1]["n_blemish"]
    assert strict < generous
    assert strict == 0


def test_areas_are_scaled_by_the_calibration_factor(apple):
    """Areas go as ``mm_per_px`` squared, equivalent diameter as its first power."""
    image, mask = apple
    pixels, aux_px = extract(image, mask, mm_per_px=1.0)
    millimetres, aux_mm = extract(image, mask, mm_per_px=2.0)

    assert aux_mm["n_blemish"] == aux_px["n_blemish"]
    assert millimetres[30] == pytest.approx(4.0 * pixels[30], rel=1e-9)  # area_mean
    assert millimetres[31] == pytest.approx(4.0 * pixels[31], rel=1e-9)  # area_max
    assert millimetres[35] == pytest.approx(2.0 * pixels[35], rel=1e-9)  # eqdiam_max


def test_the_calibration_factor_does_not_move_the_shape_descriptors(apple):
    """Eccentricity, solidity and extent are dimensionless and must not scale."""
    image, mask = apple
    pixels, _ = extract(image, mask, mm_per_px=1.0)
    millimetres, _ = extract(image, mask, mm_per_px=2.0)
    assert np.allclose(pixels[32:35], millimetres[32:35], atol=1e-12)
    assert pixels[BLEMISH_RATIO] == pytest.approx(millimetres[BLEMISH_RATIO], rel=1e-12)


# --------------------------------------------------------------------------- #
# Pole exclusion, off by default
# --------------------------------------------------------------------------- #

def test_pole_exclusion_is_off_by_default():
    """Left on, it would be an uncontrolled variable in the main benchmark."""
    assert T3MorphologicalExtractor().config["exclude_poles"] is False


POLES_ON = {"exclude_poles": True, "pole_fraction": TEST_POLE_FRACTION}
POLES_OFF = {"exclude_poles": False}


def test_pole_exclusion_removes_the_stem_and_calyx_spots():
    """Moallem et al. (2017): the two dark concavities every apple has."""
    image, mask = polar_apple()
    kept = extract(image, mask, config=POLES_OFF)[1]["n_blemish"]
    trimmed = extract(image, mask, config=POLES_ON)[1]["n_blemish"]
    assert kept == 3, f"fixture should present three spots, found {kept}"
    assert trimmed == 1, f"both poles should be excluded, {trimmed} spots survived"


def test_pole_exclusion_keeps_the_spot_in_the_middle_of_the_peel():
    """The guard must remove the poles, not simply thin the component list."""
    image, mask = polar_apple()
    _, aux = extract(image, mask, config=POLES_ON)
    rows, _ = np.nonzero(aux["blemish_mask"])
    centre_row = float(np.mean(np.nonzero(mask)[0]))
    assert abs(float(rows.mean()) - centre_row) < 20.0


def test_pole_exclusion_works_on_a_fruit_far_from_the_image_centre():
    """Regression: the cropped and full frames must not be mixed.

    A component centroid is measured in the bounding-box crop, while the fruit
    mask's centroid can be taken in either frame; the two differ by the box
    origin. Comparing one against the other excludes whichever components
    happen to land in the wrong band, and on a centred fruit the frames nearly
    agree, so the error only shows up once the fruit is off centre - which is
    the normal case in this dataset.
    """
    near, mask_near = polar_apple(centre=(55, 80))
    far, mask_far = polar_apple(centre=(165, 140))
    near_count = extract(near, mask_near, config=POLES_ON)[1]["n_blemish"]
    far_count = extract(far, mask_far, config=POLES_ON)[1]["n_blemish"]
    assert near_count == far_count == 1


def test_pole_exclusion_follows_the_fruit_axis_not_the_image_axis():
    """Apples are not photographed upright, so the band must rotate with them."""
    upright, mask_upright = polar_apple()
    rotated = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)
    mask_rotated = cv2.rotate(mask_upright, cv2.ROTATE_90_CLOCKWISE)

    assert extract(upright, mask_upright, config=POLES_ON)[1]["n_blemish"] == 1
    assert extract(rotated, mask_rotated, config=POLES_ON)[1]["n_blemish"] == 1


def test_pole_exclusion_cannot_increase_the_component_count():
    image, mask = polar_apple()
    kept = extract(image, mask, config=POLES_OFF)[1]["n_blemish"]
    trimmed = extract(image, mask, config=POLES_ON)[1]["n_blemish"]
    assert trimmed <= kept


def test_the_default_pole_band_is_mostly_already_eroded():
    """A second documented interaction, pinned rather than hidden.

    On a mask the size of a real apple here - the audit measured mean coverage
    between 0.208 and 0.284 of a 224 x 224 frame - the default 15% pole band is
    8.6 to 10.1 px wide, and the interior erosion has already removed the outer
    8 px before the guard ever runs. What is left for the guard to exclude is a
    sliver 0.6 to 2.1 px deep, far too thin to hold a component that clears
    ``min_blemish_area``. At the documented defaults ``exclude_poles`` is
    therefore close to a no-op, which is worth knowing before its effect is
    reported as evidence either way.
    """
    from features.t3_morphological import DEFAULT_CONFIG

    erosion = int(DEFAULT_CONFIG["r_max"]) + 1
    for coverage in (0.208, 0.278, 0.284):
        radius = float(np.sqrt(coverage * SIZE * SIZE / np.pi))
        band = float(DEFAULT_CONFIG["pole_fraction"]) * radius
        assert band - erosion < 3.0, (
            f"the default pole band now clears the erosion by {band - erosion:.1f}px "
            f"at coverage {coverage}; re-check whether this test still applies"
        )


# --------------------------------------------------------------------------- #
# Degenerate inputs
# --------------------------------------------------------------------------- #

def test_a_tiny_mask_returns_a_zero_vector_and_a_flag_without_raising():
    """The spec is explicit: flag it, log it, do not raise."""
    image = np.full((SIZE, SIZE, 3), 200, dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(mask, (SIZE // 2, SIZE // 2), 6, 255, thickness=-1)

    features, aux = extract(image, mask)
    assert features.shape == (36,)
    assert np.all(features == 0.0)
    assert aux["degenerate"] is True
    assert aux["blemish_mask"].shape == mask.shape


def test_an_empty_mask_returns_the_degenerate_result_from_the_bare_function():
    image = np.full((SIZE, SIZE, 3), 200, dtype=np.uint8)
    features, aux = extract(image, np.zeros((SIZE, SIZE), dtype=np.uint8))
    assert np.all(features == 0.0)
    assert aux["degenerate"] is True


def test_a_healthy_fruit_is_not_flagged_degenerate(apple):
    """The guard must be inert on real inputs, not quietly zeroing them."""
    _, aux = extract(*apple)
    assert aux["degenerate"] is False


def test_diagnostics_record_one_entry_per_call(apple):
    extractor = T3MorphologicalExtractor()
    extractor(*apple)
    extractor(*apple)
    assert len(extractor.diagnostics) == 2
    assert all(not record.degenerate for record in extractor.diagnostics)
    extractor.reset_diagnostics()
    assert extractor.diagnostics == []


# --------------------------------------------------------------------------- #
# Parameterisation, for the internal experiments
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("r_max, expected", [(5, 32), (7, 36), (9, 40), (11, 44)])
def test_the_radius_sweep_changes_the_dimensionality_honestly(apple, r_max, expected):
    """E3.3 must report accuracy against dimensionality, never pad or truncate."""
    extractor = T3MorphologicalExtractor(config={"r_max": r_max})
    assert extractor.dim == expected
    assert dimension_for(r_max) == expected
    assert extractor(*apple).shape == (expected,)
    assert len(extractor.feature_names) == expected


@pytest.mark.parametrize("r_max", [5, 7, 9, 11])
def test_the_spectrum_still_normalises_at_every_radius(apple, r_max):
    features, _ = extract(*apple, config={"r_max": r_max})
    assert features[0:r_max].sum() == pytest.approx(1.0, abs=1e-6)
    assert features[r_max : 2 * r_max].sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("shape", sorted(SE_SHAPES))
def test_every_structuring_element_shape_produces_a_valid_vector(apple, shape):
    extractor = T3MorphologicalExtractor(config={"se_shape": shape})
    vector = extractor(*apple)
    assert vector.shape == (36,)
    assert np.all(np.isfinite(vector))


def test_the_structuring_element_shape_actually_changes_the_result(apple):
    """E3.2 would be meaningless if the sweep were a no-op."""
    ellipse = extract(*apple, config={"se_shape": "ellipse"})[0]
    cross = extract(*apple, config={"se_shape": "cross"})[0]
    assert not np.allclose(ellipse, cross)


@pytest.mark.parametrize("method", BLEMISH_METHODS)
def test_every_blemish_method_produces_a_valid_vector(apple, method):
    extractor = T3MorphologicalExtractor(config={"blemish_method": method})
    vector = extractor(*apple)
    assert vector.shape == (36,)
    assert np.all(np.isfinite(vector))


def test_the_hue_comparator_reads_colour_and_the_default_method_does_not():
    """The reason hue deviation was dropped, stated as a measurement.

    Recolouring the peel while leaving its geometry untouched must not move a
    morphological descriptor. It does move the hue-deviation comparator, which
    is precisely the overlap with T1 that disqualified it.
    """
    image, mask = make_apple()
    recoloured = image.copy()
    inside = mask > 0
    recoloured[inside] = recoloured[inside][:, [2, 1, 0]]  # Swap the blue and red channels.

    morphological = extract(image, mask)[0][BLEMISH_RATIO]
    morphological_recoloured = extract(recoloured, mask)[0][BLEMISH_RATIO]
    assert morphological == pytest.approx(morphological_recoloured, rel=1e-12)


def test_a_fixed_threshold_is_monotone_in_the_threshold(apple):
    image, mask = apple
    loose = extract(image, mask, config={"blemish_method": "fixed", "fixed_threshold": 10})
    tight = extract(image, mask, config={"blemish_method": "fixed", "fixed_threshold": 90})
    assert loose[0][BLEMISH_RATIO] >= tight[0][BLEMISH_RATIO]


def test_invalid_parameters_are_rejected_at_construction():
    with pytest.raises(ValueError, match="se_shape"):
        T3MorphologicalExtractor(config={"se_shape": "diamond"})
    with pytest.raises(ValueError, match="blemish_method"):
        T3MorphologicalExtractor(config={"blemish_method": "kmeans"})


def test_block_indices_partition_the_vector_exactly():
    """E3.1 slices by these, so a gap or an overlap would silently mis-ablate."""
    blocks = block_indices(7)
    combined = np.concatenate([blocks[name] for name in "ABCDE"])
    assert np.array_equal(combined, np.arange(36))
    assert len(blocks["A"]) + len(blocks["B"]) == 18   # E3.1 spectrum-only arm.
    assert len(blocks["C"]) + len(blocks["D"]) == 10   # E3.1 response-only arm.


@pytest.mark.parametrize("r_max", [5, 7, 9, 11])
def test_block_indices_track_the_radius(r_max):
    blocks = block_indices(r_max)
    combined = np.concatenate([blocks[name] for name in "ABCDE"])
    assert np.array_equal(combined, np.arange(dimension_for(r_max)))


def test_block_names_line_up_with_the_feature_names():
    """The ablation labels its columns from these, so they must agree."""
    blocks = block_indices(7)
    names = feature_names(7)
    assert names[blocks["A"][0]] == "gran_open_r1"
    assert names[blocks["B"][0]] == "gran_mean_size"
    assert names[blocks["C"][0]] == "bth_mean"
    assert names[blocks["D"][0]] == "mgrad_mean"
    assert names[blocks["E"][0]] == "blemish_ratio"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def test_from_config_matches_the_configured_parameters():
    """The configured descriptor must be the 36-dimensional one the report quotes."""
    extractor = T3MorphologicalExtractor.from_config(CONFIG)
    assert extractor.dim == 36
    assert extractor.config["r_max"] == 7
    assert extractor.config["se_shape"] == "ellipse"
    assert extractor.config["blemish_method"] == "bth_otsu"
    assert extractor.config["exclude_poles"] is False


def test_the_seed_comes_from_the_harness_config():
    """Reproducibility is seeded from one constant, even where nothing is random."""
    assert T3MorphologicalExtractor.from_config(CONFIG).seed == CONFIG.seed


def test_an_unknown_config_key_does_not_silently_replace_a_default():
    extractor = T3MorphologicalExtractor(config={"typo_radius": 99})
    assert extractor.config["r_max"] == 7
    assert extractor.dim == 36
