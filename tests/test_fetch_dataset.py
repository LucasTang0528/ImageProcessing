"""Tests for the staging logic in ``scripts/fetch_dataset.py``.

The fetcher decides which folders of a published archive become the study's
data, so a mistake here changes the dataset silently rather than raising. The
duplicate-folder check earned its own tests: published archives are commonly
repacked with the whole tree nested inside itself, and copying both copies
would stage every image twice and then split the copies across the train and
test partitions.

Run from the project root::

    python -m pytest tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pytest  # noqa: E402

from config import get_config  # noqa: E402
from fetch_dataset import (  # noqa: E402
    CANDIDATE_SLUGS,
    classify_folder,
    drop_duplicate_folders,
    folder_signature,
    parse_map,
    split_of,
)

EXTENSIONS = get_config().image_extensions


def write_images(directory: Path, names: List[str], payload: bytes = b"x" * 32) -> Path:
    """Create ``names`` as files of identical size inside ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(payload)
    return directory


# --------------------------------------------------------------------------- #
# Duplicate source folders
# --------------------------------------------------------------------------- #

def test_folder_signature_matches_for_identical_listings(tmp_path):
    """Same filenames at the same sizes must produce the same digest."""
    left = write_images(tmp_path / "a", ["one.jpg", "two.jpg"])
    right = write_images(tmp_path / "b", ["one.jpg", "two.jpg"])
    assert folder_signature(left, EXTENSIONS) == folder_signature(right, EXTENSIONS)


def test_folder_signature_differs_when_a_file_is_missing(tmp_path):
    """A folder holding fewer files is not the same folder."""
    left = write_images(tmp_path / "a", ["one.jpg", "two.jpg"])
    right = write_images(tmp_path / "b", ["one.jpg"])
    assert folder_signature(left, EXTENSIONS) != folder_signature(right, EXTENSIONS)


def test_folder_signature_differs_when_a_size_differs(tmp_path):
    """Same names but different contents must not be treated as a repeat."""
    left = write_images(tmp_path / "a", ["one.jpg"], payload=b"x" * 32)
    right = write_images(tmp_path / "b", ["one.jpg"], payload=b"x" * 64)
    assert folder_signature(left, EXTENSIONS) != folder_signature(right, EXTENSIONS)


def test_folder_signature_ignores_non_image_files(tmp_path):
    """A stray README must not make two otherwise identical folders differ."""
    left = write_images(tmp_path / "a", ["one.jpg"])
    right = write_images(tmp_path / "b", ["one.jpg"])
    (right / "README.txt").write_text("notes", encoding="utf-8")
    assert folder_signature(left, EXTENSIONS) == folder_signature(right, EXTENSIONS)


def test_a_nested_repack_of_the_same_folder_is_dropped(tmp_path):
    """The regression this check exists for.

    ``dataset/train/rottenapples`` and ``dataset/dataset/train/rottenapples``
    hold the same files in a real published archive. Staging both would double
    the class and scatter byte-identical copies across the train/test split.
    """
    names = ["a.jpg", "b.jpg", "c.jpg"]
    shallow = write_images(tmp_path / "dataset" / "train" / "rotten", names)
    nested = write_images(tmp_path / "dataset" / "dataset" / "train" / "rotten", names)

    keep, dropped = drop_duplicate_folders([nested, shallow], EXTENSIONS)

    assert keep == [shallow]  # The shallower path wins.
    assert dropped == [nested]


def test_genuinely_different_folders_are_both_kept(tmp_path):
    """Deduplication must not merge two folders that hold different images."""
    first = write_images(tmp_path / "one", ["a.jpg", "b.jpg"])
    second = write_images(tmp_path / "two", ["c.jpg", "d.jpg"])

    keep, dropped = drop_duplicate_folders([first, second], EXTENSIONS)

    assert sorted(keep) == sorted([first, second])
    assert dropped == []


def test_deduplication_is_order_independent(tmp_path):
    """The same folders must survive whichever order they were discovered in."""
    names = ["a.jpg", "b.jpg"]
    shallow = write_images(tmp_path / "train" / "ripe", names)
    nested = write_images(tmp_path / "outer" / "train" / "ripe", names)

    forwards, _ = drop_duplicate_folders([shallow, nested], EXTENSIONS)
    backwards, _ = drop_duplicate_folders([nested, shallow], EXTENSIONS)
    assert forwards == backwards == [shallow]


# --------------------------------------------------------------------------- #
# Class identification
# --------------------------------------------------------------------------- #

#: A parent folder carrying none of the ripeness or fruit tokens.
#:
#: ``classify_folder`` deliberately inspects the parent folder name as well as
#: the leaf, so that a ``rotten/apple`` layout resolves. That makes it
#: sensitive to whatever the enclosing directory happens to be called, and
#: pytest names its temporary directory after the running test - so a test for
#: "overripe" run under ``.../test_unripe_is_not_.../`` reads the word "unripe"
#: out of its own name and matches the wrong class. Every case below is
#: therefore nested under a neutral folder.
NEUTRAL = "images"


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("unripe apple", "UnripeApple"),
        ("unripe_apple", "UnripeApple"),
        ("rottenapples", "RottenApple"),
        ("freshapples", "RipeApple"),
        ("ripe-apples", "RipeApple"),
        ("freshbanana", None),
        ("rottenoranges", None),
        ("annotations", None),
    ],
)
def test_classify_folder_maps_names_to_classes(tmp_path, folder, expected):
    """Folder names map to config classes, and non-apple fruit is ignored."""
    assert classify_folder(tmp_path / NEUTRAL / folder) == expected


def test_unripe_is_not_mistaken_for_ripe(tmp_path):
    """"unripe" contains "ripe", so ordering inside the token map matters."""
    assert classify_folder(tmp_path / NEUTRAL / "unripe apple") == "UnripeApple"
    assert classify_folder(tmp_path / NEUTRAL / "overripe apple") == "RottenApple"


def test_the_parent_folder_is_consulted_for_the_fruit(tmp_path):
    """A ``rotten/apple`` layout names the fruit in the parent, not the leaf."""
    assert classify_folder(tmp_path / "rotten" / "apple") == "RottenApple"


def test_an_enclosing_directory_name_can_decide_the_class(tmp_path):
    """Pins the consequence of consulting the parent, so it stays deliberate.

    A class folder called ``apple`` inherits its ripeness from the folder above
    it. That is what makes ``rotten/apple`` work, and it equally means a
    dataset unpacked beneath a directory named after a class can be read the
    wrong way. Staging prints the resolved mapping for this reason: it is
    meant to be checked by eye before ``--copy``.
    """
    assert classify_folder(tmp_path / "unripe" / "apple") == "UnripeApple"
    assert classify_folder(tmp_path / "fresh" / "apple") == "RipeApple"


# --------------------------------------------------------------------------- #
# Splits, mappings and slugs
# --------------------------------------------------------------------------- #

def test_split_of_recognises_train_and_holdout_folders(tmp_path):
    """Only the publisher's train split is staged; the harness splits its own."""
    assert split_of(tmp_path / "train" / "apples", tmp_path) == "train"
    assert split_of(tmp_path / "test" / "apples", tmp_path) == "other"
    assert split_of(tmp_path / "validation" / "apples", tmp_path) == "other"
    assert split_of(tmp_path / "apples", tmp_path) is None


def test_parse_map_reads_explicit_overrides():
    """``--map`` lets a class be pointed at a folder the matcher misses."""
    assert parse_map(["UnripeApple=green_apple"]) == {"UnripeApple": "green_apple"}


def test_parse_map_rejects_a_malformed_argument():
    """A typo must stop the run rather than silently skip the override."""
    with pytest.raises(SystemExit):
        parse_map(["UnripeApple:green_apple"])


def test_candidate_slugs_are_owner_slash_dataset():
    """Every shorthand must expand to a well-formed Kaggle identifier."""
    for name, slug in CANDIDATE_SLUGS.items():
        assert slug.count("/") == 1, name
        owner, dataset = slug.split("/")
        assert owner and dataset, name
