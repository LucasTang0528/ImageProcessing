"""Tests for the segmentation mask cache.

A cache that returns a stale or mismatched mask would corrupt every downstream
number silently and identically for all three techniques, so nothing
downstream could detect it. These pin the three ways that can happen: a wrong
result, a stale result after a configuration change, and a shared result
between an image and its augmented variant.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from config import get_config
from evaluate import (
    CrossValidationReport,
    collect_per_class,
    per_class_pass_table,
    per_class_rows,
)
from harness import (
    AugmentationOp,
    _cache_path,
    _segment_fruit_uncached,
    _segmentation_signature,
    apply_augmentation,
    segment_fruit,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _apple_frame(seed: int = 0) -> np.ndarray:
    """A synthetic fruit on a contrasting ground, with grain."""
    rng = np.random.default_rng(seed)
    image = np.full((224, 224, 3), 200, dtype=np.uint8)
    cv2.circle(image, (112, 112), 62, (40, 40, 190), thickness=-1)
    noise = rng.normal(0, 4, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def cached_config(tmp_path):
    """A config whose cache lives in a throwaway directory."""
    config = get_config()
    paths = dataclasses.replace(config.paths, cache_root=Path(tmp_path))
    segmentation = dataclasses.replace(config.segmentation, cache_masks=True)
    return dataclasses.replace(config, paths=paths, segmentation=segmentation)


def _identical(first, second) -> bool:
    """Compare two SegmentationResults field by field."""
    thresholds_match = first.threshold == second.threshold or (
        np.isnan(first.threshold) and np.isnan(second.threshold)
    )
    return (
        np.array_equal(first.mask, second.mask)
        and tuple(first.bbox) == tuple(second.bbox)
        and np.array_equal(first.contour, second.contour)
        and first.coverage == second.coverage
        and first.polarity == second.polarity
        and thresholds_match
        and first.method == second.method
        and first.failed == second.failed
        and first.reason == second.reason
        and first.substituted == second.substituted
    )


# --------------------------------------------------------------------------- #
# Segmentation cache
# --------------------------------------------------------------------------- #

def test_cached_segmentation_is_identical_to_uncached(cached_config):
    """The whole point: caching must not change a single field."""
    image = _apple_frame()
    fresh = _segment_fruit_uncached(image, cached_config)

    cold = segment_fruit(image, cached_config)  # computes and stores
    warm = segment_fruit(image, cached_config)  # reads back

    assert _identical(fresh, cold)
    assert _identical(fresh, warm)


def test_second_call_reads_from_disk_rather_than_recomputing(cached_config):
    """A cache that never hits is not a cache."""
    image = _apple_frame()
    segment_fruit(image, cached_config)
    path = _cache_path(image, cached_config)
    assert path.exists()

    # Corrupt the stored mask; a genuine cache hit must return the corruption.
    with np.load(path, allow_pickle=False) as archive:
        meta = str(archive["meta"])
    tampered = np.zeros((224, 224), dtype=np.uint8)
    tampered[:60, :60] = 255
    np.savez_compressed(path, mask=tampered, meta=np.array(meta))

    assert np.array_equal(segment_fruit(image, cached_config).mask, tampered)


def test_changing_segmentation_config_invalidates_the_cache(cached_config):
    """A stale mask under a changed configuration is the dangerous failure."""
    image = _apple_frame()
    segment_fruit(image, cached_config)

    raw = json.loads(json.dumps(cached_config.raw))
    raw["segmentation"]["fill_holes"] = not cached_config.segmentation.fill_holes
    changed = dataclasses.replace(
        cached_config,
        raw=raw,
        segmentation=dataclasses.replace(
            cached_config.segmentation,
            fill_holes=not cached_config.segmentation.fill_holes,
        ),
    )

    assert _segmentation_signature(changed) != _segmentation_signature(cached_config)
    assert _cache_path(image, changed) != _cache_path(image, cached_config)
    assert _identical(
        segment_fruit(image, changed), _segment_fruit_uncached(image, changed)
    )


def test_seed_is_part_of_the_cache_key(cached_config):
    """GrabCut reseeds OpenCV from it, so the seed is an input to the mask."""
    other = dataclasses.replace(cached_config, seed=cached_config.seed + 1)
    assert _segmentation_signature(other) != _segmentation_signature(cached_config)


def test_augmented_variant_does_not_share_its_original_mask(cached_config):
    """The trap a path-keyed cache would fall into.

    An augmented variant has the same source file but different pixels, and
    therefore a different mask. Keying on content rather than path is what
    keeps augmentation from being silently destroyed.
    """
    original = _apple_frame()
    flipped = apply_augmentation(
        original, AugmentationOp(flip=True, angle=20.0, brightness=1.0)
    )

    assert _cache_path(original, cached_config) != _cache_path(flipped, cached_config)
    assert _identical(
        segment_fruit(flipped, cached_config),
        _segment_fruit_uncached(flipped, cached_config),
    )


def test_cache_can_be_disabled(cached_config):
    """With caching off nothing is written, and results are unchanged."""
    disabled = dataclasses.replace(
        cached_config,
        segmentation=dataclasses.replace(cached_config.segmentation, cache_masks=False),
    )
    image = _apple_frame()
    assert _identical(
        segment_fruit(image, disabled), _segment_fruit_uncached(image, disabled)
    )
    assert not _cache_path(image, disabled).exists()


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(cached_config):
    """A cache is an optimisation, so it may never be why a run fails."""
    image = _apple_frame()
    path = _cache_path(image, cached_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not an npz archive")

    assert _identical(
        segment_fruit(image, cached_config), _segment_fruit_uncached(image, cached_config)
    )


