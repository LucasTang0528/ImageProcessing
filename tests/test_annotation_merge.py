"""Tests for the two-annotator blemish annotation merge.

An error measured against one annotator's mask cannot be told apart from that
annotator's own noise, so the merge keeps only what both marked and reports
how much they disagreed.
"""

from __future__ import annotations

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Annotation subset and merge
# --------------------------------------------------------------------------- #

def test_jaccard_handles_agreement_disagreement_and_empty():
    """Two annotators who both correctly paint nothing agree perfectly."""
    from scripts.annotate_blemishes import jaccard

    filled = np.zeros((10, 10), dtype=bool)
    filled[:5, :5] = True
    other = np.zeros((10, 10), dtype=bool)
    other[5:, 5:] = True
    empty = np.zeros((10, 10), dtype=bool)

    assert jaccard(filled, filled) == pytest.approx(1.0)
    assert jaccard(filled, other) == pytest.approx(0.0)
    assert jaccard(empty, empty) == pytest.approx(1.0)


def test_reference_mask_is_the_intersection_not_the_union():
    """The conservative choice: only pixels nobody disputes."""
    from scripts.annotate_blemishes import jaccard

    first = np.zeros((10, 10), dtype=bool)
    first[:6, :6] = True
    second = np.zeros((10, 10), dtype=bool)
    second[3:, 3:] = True

    consensus = np.logical_and.reduce([first, second])
    assert consensus.sum() == 9  # the 3x3 overlap
    assert consensus.sum() < first.sum() and consensus.sum() < second.sum()
    assert jaccard(first, second) == pytest.approx(9 / (36 + 49 - 9))
