"""Freeze the blemish-detection configuration while annotations live in the test split.

The E3.4 annotation subset is drawn from the **test partition**, chosen so
that blemish coverage is validated on images the pipeline was never fitted on.
That is sound under exactly one condition: nothing may be selected or tuned on
the resulting error. The moment a blemish parameter is changed because it
improved the MAE, the test partition has informed a modelling decision and
every held-out figure in the study is compromised.

A docstring cannot enforce that. This test can. It pins a hash of every
parameter that changes which pixels are called blemish, so any edit to them
fails the suite and forces the question to be asked out loud.

**This test failing is not a bug.** It means someone changed blemish detection,
which is legitimate work. It means the annotation subset must move to the
training partition first - see the instructions in the failure message.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from config import get_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Every parameter that changes which pixels T3 calls a blemish, and so
#: changes the coverage ratio the MAE is measured against.
#:
#: ``tophat_radius`` is included because the black top-hat response is what
#: gets thresholded; widening it changes the candidate mask before any
#: threshold is applied. ``hue_deviation`` and its two companions are included
#: even though they belong to the dropped comparator, because E3.4 scores that
#: comparator and tuning it would bias the comparison it exists to settle.
#:
#: ``r_max``, ``se_shape`` and ``mm_per_px`` are deliberately absent: they
#: alter the granulometric descriptor or the reported units, not which pixels
#: are judged blemished.
BLEMISH_KEYS = (
    "blemish_method",
    "exclude_poles",
    "fixed_threshold",
    "hue_deviation",
    "min_blemish_area",
    "pole_fraction",
    "saturation_deviation",
    "tophat_radius",
    "value_deviation",
)

#: Hash of the values in place when the annotation subset was drawn from the
#: test partition. Regenerate ONLY together with moving the subset.
FROZEN_DIGEST = "ec8754982dd58e1ed83d1177923011cf3af218b74a7c022e48ba350366e1a5a6"

UNFREEZE_INSTRUCTIONS = """
The blemish-detection configuration has changed while the E3.4 annotation
subset is drawn from the TEST partition.

If you are tuning blemish detection, the subset must move first:

  1. Edit scripts/select_annotation_subset.py to draw from partition.train
     rather than partition.test, and say why in its docstring.
  2. Re-run:  python scripts/select_annotation_subset.py --force
  3. Re-paint. Existing masks are for images that are no longer in the subset.
  4. Update FROZEN_DIGEST in this file to the value printed below.

If you are NOT tuning against the MAE - a rename, a comment, a default that
cannot affect which pixels are selected - update FROZEN_DIGEST and say so in
the commit message.

Do not simply update the hash to make the suite green.
"""


def blemish_config(config) -> dict:
    """Return the blemish-detection parameters, in canonical order."""
    block = config.raw.get("t3_morphological", {})
    missing = [key for key in BLEMISH_KEYS if key not in block]
    if missing:
        pytest.fail(
            f"config.json is missing blemish parameter(s) {missing}. The freeze "
            f"cannot protect a parameter that is not there; add it or remove it "
            f"from BLEMISH_KEYS with a reason."
        )
    return {key: block[key] for key in BLEMISH_KEYS}


def digest_of(payload: dict) -> str:
    """Stable hash of the blemish configuration."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_blemish_configuration_is_frozen():
    """Fail loudly if any blemish parameter moves while annotations are test-split."""
    payload = blemish_config(get_config())
    actual = digest_of(payload)
    assert actual == FROZEN_DIGEST, (
        f"{UNFREEZE_INSTRUCTIONS}\n"
        f"current values : {json.dumps(payload, sort_keys=True)}\n"
        f"new digest     : {actual}\n"
    )


def test_freeze_applies_only_while_the_subset_is_drawn_from_the_test_split():
    """The freeze exists because of where the subset comes from.

    If the subset is ever moved to the training partition the leak disappears
    and this whole file should be deleted rather than left to obstruct honest
    tuning. This test states that dependency so the two cannot drift apart.
    """
    selector = (PROJECT_ROOT / "scripts" / "select_annotation_subset.py").read_text(
        encoding="utf-8"
    )
    draws_from_test = "partition.test" in selector
    metadata = PROJECT_ROOT / "results" / "annotations" / "subset_metadata.json"

    if not draws_from_test:
        pytest.fail(
            "select_annotation_subset.py no longer draws from the test partition. "
            "The blemish freeze is then unnecessary: delete this file."
        )
    if metadata.exists():
        recorded = json.loads(metadata.read_text(encoding="utf-8"))["drawn_from"]
        assert "test" in recorded, (
            f"subset_metadata.json says the subset came from {recorded!r}, which "
            f"disagrees with the selector. One of them is stale."
        )


def test_every_frozen_key_actually_exists_in_the_config():
    """A typo in BLEMISH_KEYS would silently protect nothing."""
    block = get_config().raw.get("t3_morphological", {})
    for key in BLEMISH_KEYS:
        assert key in block, f"BLEMISH_KEYS names {key!r}, absent from config.json"


def test_digest_changes_when_a_blemish_parameter_changes():
    """The freeze has to be sensitive, or it is decoration."""
    payload = blemish_config(get_config())
    for key, altered in (
        ("blemish_method", "fixed"),
        ("min_blemish_area", 999),
        ("exclude_poles", True),
        ("tophat_radius", 3),
    ):
        tampered = dict(payload)
        tampered[key] = altered
        assert digest_of(tampered) != FROZEN_DIGEST, (
            f"changing {key} did not change the digest; the freeze would not "
            f"catch a real edit to it"
        )
