"""Tests for the dataset audit in ``scripts/audit_dataset.py``.

The audit is what decides whether a candidate dataset is fit to carry the
study, so a false finding here is expensive in both directions: a missed
confound invalidates every later result, and a fabricated one would reject a
perfectly good dataset.

The duplicate detector earned most of these tests. Matching on a difference
hash alone reported hundreds of cross-class "duplicates" on the apple data
that were nothing of the kind - distinct fruits sharing a silhouette against a
plain backdrop - which read as contradictory ground truth. The colour
confirmation step exists to stop that, and the tests below pin the behaviour
in both directions.

Run from the project root::

    python -m pytest tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from audit_dataset import (  # noqa: E402
    BORDER_WIDTH,
    background_descriptor,
    border_ring,
    colour_thumbnail,
    difference_hash,
    duplicate_report,
    spread,
    summarise_profile,
    thumbnail_rmse,
    verdict,
)
from config import DatasetSpec, get_config  # noqa: E402
from data import ImageRecord  # noqa: E402

CONFIG = get_config()
SIZE = 224


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def studio_image(colour: tuple, radius: int = 60, seed: int = 0) -> np.ndarray:
    """A centred coloured disc on a white backdrop, as in a product shot.

    Two of these with different colours share an outline, so they collide
    under a difference hash while being obviously different fruits.
    """
    rng = np.random.default_rng(seed)
    image = np.full((SIZE, SIZE, 3), 245, dtype=np.uint8)
    cv2.circle(image, (SIZE // 2, SIZE // 2), radius, colour, thickness=-1)
    noise = rng.integers(-3, 4, size=image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_records(class_names: List[str]) -> List[ImageRecord]:
    """Build throwaway records with the given class names, in order."""
    unique = list(dict.fromkeys(class_names))
    return [
        ImageRecord(
            path=Path(f"{name}_{index}.jpg"),
            label=unique.index(name),
            class_folder=name,
            display_name=name,
            source="test",
        )
        for index, name in enumerate(class_names)
    ]


# --------------------------------------------------------------------------- #
# Border ring and background descriptor
# --------------------------------------------------------------------------- #

def test_border_ring_never_samples_the_centre():
    """A mark in the middle of the frame must not reach the border ring."""
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[SIZE // 2 - 20 : SIZE // 2 + 20, SIZE // 2 - 20 : SIZE // 2 + 20] = 255
    assert border_ring(image).max() == 0


def test_border_ring_has_the_expected_pixel_count():
    """The ring is four edge strips, with the corners counted twice."""
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    expected = 2 * BORDER_WIDTH * SIZE + 2 * BORDER_WIDTH * SIZE
    assert border_ring(image).shape == (expected, 3)


def test_background_descriptor_is_finite_and_fixed_length():
    """The descriptor must be usable by the classifier without cleaning."""
    vector = background_descriptor(studio_image((40, 40, 200)))
    assert vector.ndim == 1
    assert np.all(np.isfinite(vector))
    assert vector.size == background_descriptor(studio_image((0, 200, 0))).size


def test_background_descriptor_ignores_the_fruit_colour():
    """Changing only the centred disc must leave the background unchanged.

    This is what makes the background-only test honest: if the descriptor
    picked up fruit pixels, a high score would prove nothing.
    """
    red = background_descriptor(studio_image((30, 30, 220), seed=1))
    green = background_descriptor(studio_image((30, 220, 30), seed=1))
    assert np.allclose(red, green)


# --------------------------------------------------------------------------- #
# Duplicate detection
# --------------------------------------------------------------------------- #

def test_identical_images_are_reported_as_duplicates():
    """The detector must still catch the case it exists to catch."""
    image = studio_image((30, 30, 220), seed=3)
    records = make_records(["Ripe", "Ripe"])
    frame, counts = duplicate_report(records, [image, image.copy()])

    assert counts["n_pairs"] == 1
    assert counts["n_exact"] == 1
    assert counts["n_cross_class"] == 0
    assert float(frame.iloc[0]["colour_rmse"]) == pytest.approx(0.0, abs=1e-9)


def test_a_rescaled_copy_is_still_caught():
    """A duplicate that was resaved at another size must not slip through."""
    image = studio_image((30, 30, 220), seed=4)
    shrunk = cv2.resize(
        cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA),
        (SIZE, SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    _, counts = duplicate_report(make_records(["Ripe", "Ripe"]), [image, shrunk])
    assert counts["n_pairs"] == 1


def test_same_silhouette_different_colour_is_not_a_duplicate():
    """The regression this module exists for.

    A red and a green disc of the same size on the same backdrop produce
    near-identical difference hashes. Reporting them as duplicates on a
    dataset whose classes differ mainly in colour manufactures hundreds of
    fictitious cross-class label contradictions.
    """
    red = studio_image((30, 30, 220), seed=5)
    green = studio_image((30, 220, 30), seed=5)

    assert int(np.unpackbits(difference_hash(red) ^ difference_hash(green)).sum()) <= 5

    _, counts = duplicate_report(make_records(["Ripe", "Unripe"]), [red, green])
    assert counts["n_pairs"] == 0
    assert counts["n_cross_class"] == 0


def test_thumbnail_rmse_separates_copies_from_different_fruits():
    """Confirmation must leave a wide margin, not a knife-edge decision."""
    red = colour_thumbnail(studio_image((30, 30, 220), seed=6))
    same = colour_thumbnail(studio_image((30, 30, 220), seed=6))
    green = colour_thumbnail(studio_image((30, 220, 30), seed=6))

    assert thumbnail_rmse(red, same) < 1.0
    assert thumbnail_rmse(red, green) > 20.0


def test_each_duplicate_pair_is_reported_once():
    """Pairs are unordered, so a run of copies must not be double counted."""
    image = studio_image((200, 60, 60), seed=7)
    records = make_records(["Ripe", "Ripe", "Ripe"])
    frame, counts = duplicate_report(records, [image, image.copy(), image.copy()])

    assert counts["n_pairs"] == 3  # (0,1), (0,2), (1,2)
    assert len(frame) == 3
    assert not frame[["file_a", "file_b"]].duplicated().any()


def test_duplicate_report_on_distinct_images_is_empty_but_typed():
    """An empty result must still carry its columns, so callers can index it."""
    images = [studio_image((30, 30, 220), radius=40, seed=8),
              studio_image((30, 220, 30), radius=95, seed=9)]
    frame, counts = duplicate_report(make_records(["Ripe", "Unripe"]), images)

    assert counts == {"n_pairs": 0, "n_exact": 0, "n_cross_class": 0}
    assert list(frame.columns) == [
        "distance", "colour_rmse", "class_a", "class_b",
        "cross_class", "file_a", "file_b",
    ]


# --------------------------------------------------------------------------- #
# Summary and verdict
# --------------------------------------------------------------------------- #

def _profile(rows: List[dict]) -> pd.DataFrame:
    """Build a per-image profile frame from literal rows."""
    return pd.DataFrame(rows)


def test_summarise_profile_keeps_the_configured_class_order():
    """Tables are read by eye, so the class order must match the config."""
    spec = DatasetSpec(classes=("a", "b"), display_names=("Unripe", "Ripe"))
    profile = _profile(
        [
            {"class": "Ripe", "coverage": 0.3, "polarity": "dark", "failed": False, "border_std": 40.0},
            {"class": "Unripe", "coverage": 0.4, "polarity": "bright", "failed": False, "border_std": 50.0},
        ]
    )
    assert list(summarise_profile(profile, spec).index) == ["Unripe", "Ripe"]


def test_spread_measures_the_gap_between_classes():
    """``spread`` is the statistic the verdict thresholds are applied to."""
    summary = pd.DataFrame({"pct_failed": [0.0, 12.5, 4.0]})
    assert spread(summary, "pct_failed") == pytest.approx(12.5)
    assert np.isnan(spread(summary, "absent_column"))


def test_verdict_fails_a_dataset_whose_background_predicts_the_class():
    """The headline confound must be enough on its own to reject a dataset."""
    summary = pd.DataFrame(
        {"pct_failed": [0.0, 0.0], "pct_polarity_dark": [10.0, 12.0], "mean_coverage": [0.4, 0.4]}
    )
    passed, findings = verdict(
        {"accuracy": 0.74, "chance": 1 / 3, "excess_over_chance": 0.74 - 1 / 3},
        summary,
        {"n_pairs": 0, "n_exact": 0, "n_cross_class": 0},
        2400,
    )
    assert passed is False
    assert any(item.startswith("FAIL") and "confounded" in item for item in findings)


def test_verdict_fails_class_dependent_segmentation():
    """Segmentation that behaves differently per class is disqualifying."""
    summary = pd.DataFrame(
        {"pct_failed": [0.0, 12.5], "pct_polarity_dark": [13.4, 76.1], "mean_coverage": [0.41, 0.24]}
    )
    passed, findings = verdict(
        {"accuracy": 0.34, "chance": 1 / 3, "excess_over_chance": 0.01},
        summary,
        {"n_pairs": 0, "n_exact": 0, "n_cross_class": 0},
        2400,
    )
    assert passed is False
    assert any("segmentation behaves differently" in item for item in findings)


def test_verdict_passes_a_clean_dataset():
    """A dataset with no confound and stable segmentation must pass."""
    summary = pd.DataFrame(
        {"pct_failed": [1.0, 2.0], "pct_polarity_dark": [48.0, 52.0], "mean_coverage": [0.35, 0.36]}
    )
    passed, findings = verdict(
        {"accuracy": 0.36, "chance": 1 / 3, "excess_over_chance": 0.03},
        summary,
        {"n_pairs": 2, "n_exact": 1, "n_cross_class": 0},
        2400,
    )
    assert passed is True
    assert all(not item.startswith("FAIL") for item in findings)


def test_verdict_fails_on_contradictory_labels():
    """A duplicate carrying two labels makes the ground truth self-defeating."""
    summary = pd.DataFrame(
        {"pct_failed": [1.0, 2.0], "pct_polarity_dark": [48.0, 52.0], "mean_coverage": [0.35, 0.36]}
    )
    passed, findings = verdict(
        {"accuracy": 0.36, "chance": 1 / 3, "excess_over_chance": 0.03},
        summary,
        {"n_pairs": 9, "n_exact": 9, "n_cross_class": 9},
        2400,
    )
    assert passed is False
    assert any("contradicts itself" in item for item in findings)
