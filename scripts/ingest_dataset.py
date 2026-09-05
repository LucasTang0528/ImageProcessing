"""Survey a source copy of the primary dataset and propose a class mapping.

This script replaces ``scripts/fetch_dataset.py``. It exists because the
Kaggle CLI is not reliably on PATH on every machine in the team, and because
one member already holds a manually downloaded, unpacked copy of the dataset.
Both cases are handled by the same code path: a source directory is surveyed,
a mapping from its folders to the class folders in ``config.json`` is
proposed, and nothing is written until that mapping has been read by a human.

Two sources, selected by flag:

``--local <path>``
    Read an already unpacked folder. With no path, the folder named by
    ``ingest.local_source`` in ``config.json`` is used. This is the default
    mode, so no network access and no credentials are needed.

``--download``
    Fetch ``ingest.kaggle_slug`` with :mod:`kagglehub`. The Kaggle CLI is
    never invoked, so a broken ``kaggle`` entry on PATH is irrelevant.
    Credentials are left entirely to kagglehub; this script never reads,
    prints or copies a token file.

Two stages, so that a wrong mapping cannot quietly stage sixteen thousand
images into the wrong class folders:

**Stage 1 - survey (the default).** Walks the source, prints the directory
tree, the per-folder image counts and extension distribution, the proposed
mapping and the copy plan it would execute. Copies nothing, so it is always
safe to run.

**Stage 2 - copy (``--copy``).** Executes that same plan. The source is only
ever read from; nothing outside ``data/primary`` is written, moved or deleted.
A class folder that already holds images is never touched without an explicit
answer, and every choice is collected *before* the first file is written, so
aborting at the prompt leaves the destination exactly as it was.

Usage::

    python scripts/ingest_dataset.py                         # survey the configured local folder
    python scripts/ingest_dataset.py --local D:/data/archive  # survey an explicit folder
    python scripts/ingest_dataset.py --download              # survey a kagglehub copy
    python scripts/ingest_dataset.py --depth 4               # deeper tree listing
    python scripts/ingest_dataset.py --map RipeApple=train/good_apple
    python scripts/ingest_dataset.py --copy                  # stage the mapped classes
    python scripts/ingest_dataset.py --copy --yes            # same, no prompts (merges)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config, get_config  # noqa: E402

#: Fallbacks for the survey's own tuning knobs only. The class mapping, the
#: source location and the dataset slug are never defaulted here - they are
#: read from ``config.json`` and their absence is an error, so that the
#: mapping always lives in configuration rather than in source.
SURVEY_DEFAULTS: Dict[str, Any] = {
    "ignore_dirs": ["test", "val", "valid", "validation"],
    "match_threshold": 0.6,
    "review_threshold": 0.3,
}

#: Split on separators and on camelCase boundaries, so that "UnripeApple",
#: "unripe_apple" and "unripe apple" all reduce to the same token set. Token
#: comparison matters here: plain substring matching reports "RipeApple" as a
#: match for the folder "UnripeApple", which is precisely the confusion this
#: dataset invites.
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])")


class IngestError(RuntimeError):
    """Raised when the survey cannot proceed and must not guess."""


# --------------------------------------------------------------------------- #
# Configuration access
# --------------------------------------------------------------------------- #

def survey_settings(config: Config) -> Dict[str, Any]:
    """Return the ``ingest`` block from ``config.json`` over the defaults."""
    settings = dict(SURVEY_DEFAULTS)
    settings.update(dict(config.raw.get("ingest", {})))
    return settings


def _required(settings: Mapping[str, Any], key: str) -> Any:
    """Fetch a setting that has no safe default, or explain what is missing."""
    if key not in settings or settings[key] in (None, ""):
        raise IngestError(
            f'config.json is missing "ingest.{key}". Add it to the "ingest" '
            f"block; this script deliberately holds no fallback for it, so that "
            f"the value stays reviewable in configuration rather than in source."
        )
    return settings[key]


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #

def resolve_local_source(explicit: Optional[str], settings: Mapping[str, Any]) -> Path:
    """Resolve the local source folder from the flag or from configuration."""
    value = explicit if explicit else _required(settings, "local_source")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.is_dir():
        raise IngestError(
            f"local source {path} is not a directory. Pass --local <path>, or "
            f'correct "ingest.local_source" in config.json.'
        )
    return path


def resolve_download_source(settings: Mapping[str, Any]) -> Path:
    """Download the dataset with kagglehub and return the cache directory.

    Credentials are handled by kagglehub alone. Nothing about a token is read
    or echoed here; only whether a credential source is visible at all, which
    is reported to make an authentication failure easier to diagnose.
    """
    slug = str(_required(settings, "kaggle_slug"))
    _report_credential_sources()
    try:
        import kagglehub
    except ImportError as error:
        raise IngestError(
            f"kagglehub is not installed ({error}). Run:\n"
            f"    python -m pip install kagglehub\n"
            f"The Kaggle CLI is never invoked by this script, so a broken "
            f"'kaggle' entry on PATH does not matter."
        ) from error

    print(f"Downloading {slug} via kagglehub ...")
    try:
        cache_path = Path(kagglehub.dataset_download(slug))
    except Exception as error:  # noqa: BLE001 - surface any kagglehub failure plainly
        raise IngestError(
            f"kagglehub could not download the dataset: {error}\n\n"
            f"Check that you have opened the dataset page once while signed in "
            f"and accepted its terms, and that your credentials are where "
            f"kagglehub expects them. If the download stays blocked, unpack the "
            f"archive by hand and use --local <path> instead."
        ) from error
    print(f"kagglehub cache: {cache_path}")
    return cache_path


def _report_credential_sources() -> None:
    """State which credential sources exist, without reading any of them."""
    kaggle_dir = Path.home() / ".kaggle"
    found: List[str] = []
    if (kaggle_dir / "kaggle.json").is_file():
        found.append("~/.kaggle/kaggle.json")
    if (kaggle_dir / "access_token").is_file():
        found.append("~/.kaggle/access_token")
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        found.append("KAGGLE_USERNAME / KAGGLE_KEY")
    if os.environ.get("KAGGLEHUB_TOKEN"):
        found.append("KAGGLEHUB_TOKEN")

    if found:
        print(f"Credential sources visible: {', '.join(found)}")
    else:
        print("No Kaggle credential source found in ~/.kaggle or the environment.")
    if found == ["~/.kaggle/access_token"]:
        print(
            "  Note: only the newer access_token file is present. Older kagglehub "
            "builds read kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY only. If the "
            "download is refused, upgrade kagglehub or fall back to --local."
        )


# --------------------------------------------------------------------------- #
# Survey
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LeafFolder:
    """A source folder that directly contains at least one image file."""

    path: Path
    relative: str
    n_images: int
    extensions: Tuple[Tuple[str, int], ...]
    n_other_files: int
    ignored_by: Optional[str] = None

    @property
    def name(self) -> str:
        """The folder's own name, which is what the mapping is matched on."""
        return self.path.name

    @property
    def extension_summary(self) -> str:
        """Extension distribution as a compact ``.jpg x800`` style string."""
        return ", ".join(f"{ext} x{count}" for ext, count in self.extensions)


def survey_source(
    root: Path,
    image_extensions: Sequence[str],
    ignore_dirs: Sequence[str],
) -> List[LeafFolder]:
    """Walk ``root`` and describe every folder holding images.

    Args:
        root: The source directory to survey.
        image_extensions: Accepted extensions, lowercase and dot-prefixed,
            taken from ``datasets.image_extensions`` in the configuration.
        ignore_dirs: Path components marking a publisher's held-out split.
            Folders beneath one are surveyed and reported, but excluded from
            the mapping - the harness draws its own seeded 80:20 split, so
            folding in a publisher's test set would change the fixed partition.

    Returns:
        Leaf folders, ordered by relative path.
    """
    accepted = {str(ext).lower() for ext in image_extensions}
    ignored = {str(name).lower() for name in ignore_dirs}
    leaves: List[LeafFolder] = []

    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        counts: Counter = Counter()
        other = 0
        for entry in directory.iterdir():
            if not entry.is_file():
                continue
            suffix = entry.suffix.lower()
            if suffix in accepted:
                counts[suffix] += 1
            else:
                other += 1
        if not counts:
            continue
        relative_path = directory.relative_to(root)
        hits = sorted({part.lower() for part in relative_path.parts} & ignored)
        leaves.append(
            LeafFolder(
                path=directory,
                relative=relative_path.as_posix(),
                n_images=sum(counts.values()),
                extensions=tuple(sorted(counts.items(), key=lambda item: -item[1])),
                n_other_files=other,
                ignored_by=hits[0] if hits else None,
            )
        )
    return leaves


def render_tree(root: Path, max_depth: int, image_extensions: Sequence[str]) -> List[str]:
    """Return the directory tree as lines, annotated with image counts."""
    accepted = {str(ext).lower() for ext in image_extensions}

    def count_images(directory: Path) -> int:
        return sum(
            1
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() in accepted
        )

    lines = [str(root)]

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        children = sorted(path for path in directory.iterdir() if path.is_dir())
        for index, child in enumerate(children):
            last = index == len(children) - 1
            label = f"{child.name}/"
            images = count_images(child)
            if images:
                label += f"  [{images} images]"
            lines.append(f"{prefix}{'`-- ' if last else '|-- '}{label}")
            walk(child, prefix + ("    " if last else "|   "), depth + 1)

    walk(root, "", 1)
    if len(lines) == 1:
        lines.append("  (no subdirectories)")
    return lines


# --------------------------------------------------------------------------- #
# Mapping proposal
# --------------------------------------------------------------------------- #

@dataclass
class Proposal:
    """One target class and the source folder the survey proposes for it."""

    target: str
    display: str
    leaf: Optional[LeafFolder] = None
    score: float = 0.0
    method: str = "no candidate"
    forced: bool = False
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def status(self) -> str:
        """One word describing how far the proposal should be trusted."""
        if self.leaf is None:
            return "UNMATCHED"
        if self.forced:
            return "FORCED"
        if self.score >= 1.0:
            return "EXACT"
        return "REVIEW"


def _tokens(name: str) -> frozenset:
    """Split a folder or class name into lowercase word tokens."""
    return frozenset(token.lower() for token in _TOKEN_SPLIT.split(name) if token)


def _normalise(name: str) -> str:
    """Reduce a name to lowercase alphanumerics for exact comparison."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def score_folder(folder_name: str, target: str, aliases: Sequence[str]) -> Tuple[float, str]:
    """Score how well a source folder name matches a target class.

    An exact match on the class name or on one of its configured aliases
    scores 1.0. Otherwise the score is the Jaccard overlap of the two token
    sets, which keeps "RipeApple" and "UnripeApple" apart where substring
    matching would merge them.

    Returns:
        The score in ``[0, 1]``, and a short description of how it was reached.
    """
    folder_norm = _normalise(folder_name)
    folder_tokens = _tokens(folder_name)
    best, method = 0.0, "no token overlap"

    candidates = [(target, "class name")] + [(alias, "alias") for alias in aliases]
    for candidate, kind in candidates:
        if folder_norm and folder_norm == _normalise(candidate):
            return 1.0, f"exact {kind} match"
        candidate_tokens = _tokens(candidate)
        if not candidate_tokens or not folder_tokens:
            continue
        overlap = len(folder_tokens & candidate_tokens) / len(folder_tokens | candidate_tokens)
        if overlap > best:
            best = overlap
            method = f"{overlap:.2f} token overlap with {kind} '{candidate}'"
    return best, method


def propose_mapping(
    leaves: Sequence[LeafFolder],
    classes: Sequence[str],
    display_names: Sequence[str],
    aliases: Mapping[str, Sequence[str]],
    overrides: Mapping[str, str],
    review_threshold: float,
) -> List[Proposal]:
    """Assign at most one source folder to each target class.

    Every (class, folder) pair is scored, then assignment runs greedily from
    the highest score down, so one folder can never be claimed by two classes
    and the strongest evidence wins. Explicit ``--map`` overrides are applied
    first and are never overruled.
    """
    proposals = {
        target: Proposal(target=target, display=display)
        for target, display in zip(classes, display_names)
    }
    by_relative = {leaf.relative: leaf for leaf in leaves}
    taken: Set[str] = set()

    for target, wanted in overrides.items():
        if target not in proposals:
            raise IngestError(
                f"--map names '{target}', which is not one of the configured "
                f"classes: {', '.join(classes)}"
            )
        wanted_norm = _normalise(wanted)
        matches = [
            leaf
            for leaf in leaves
            if wanted_norm in (_normalise(leaf.relative), _normalise(leaf.name))
        ]
        if not matches:
            raise IngestError(
                f"--map {target}={wanted} does not name any folder in the source. "
                f"Use the relative path shown in the leaf-folder table."
            )
        if len(matches) > 1:
            listed = ", ".join(leaf.relative for leaf in matches)
            raise IngestError(
                f"--map {target}={wanted} is ambiguous; it matches {listed}. "
                f"Give the full relative path."
            )
        proposal = proposals[target]
        proposal.leaf = matches[0]
        proposal.score = 1.0
        proposal.method = "forced by --map"
        proposal.forced = True
        taken.add(matches[0].relative)

    scored: List[Tuple[float, str, str, str]] = []
    for target in classes:
        if proposals[target].forced:
            continue
        for leaf in leaves:
            if leaf.ignored_by is not None:
                continue
            score, method = score_folder(leaf.name, target, list(aliases.get(target, ())))
            if score > 0.0:
                scored.append((score, target, leaf.relative, method))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))

    assigned: Set[str] = set()
    for score, target, relative, method in scored:
        proposal = proposals[target]
        if target in assigned or relative in taken:
            if proposal.leaf is not None and score >= review_threshold:
                proposal.alternatives.append((relative, score))
            continue
        proposal.leaf = by_relative[relative]
        proposal.score = score
        proposal.method = method
        assigned.add(target)
        taken.add(relative)

    return [proposals[target] for target in classes]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _rule(title: str) -> None:
    """Print a section heading."""
    print()
    print(title)
    print("=" * max(len(title), 60))


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Print a left-aligned fixed-width table."""
    if not rows:
        print("  (nothing to show)")
        return
    widths = [
        max([len(str(headers[column]))] + [len(str(row[column])) for row in rows])
        for column in range(len(headers))
    ]
    header_line = "  ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
    print("  " + header_line.rstrip())
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        line = "  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers)))
        print("  " + line.rstrip())


def report_leaves(leaves: Sequence[LeafFolder]) -> None:
    """Print every folder containing images, with counts and extensions."""
    _rule("LEAF FOLDERS CONTAINING IMAGES")
    usable = [leaf for leaf in leaves if leaf.ignored_by is None]
    _table(
        ["folder", "images", "other files", "extensions"],
        [
            [leaf.relative, leaf.n_images, leaf.n_other_files or "-", leaf.extension_summary]
            for leaf in usable
        ],
    )
    print(
        f"\n  {len(usable)} folder(s), "
        f"{sum(leaf.n_images for leaf in usable)} image(s) eligible for mapping."
    )

    skipped = [leaf for leaf in leaves if leaf.ignored_by is not None]
    if skipped:
        print(
            "\n  Excluded from the mapping because they sit under a publisher "
            "split directory. The harness draws its own seeded 80:20 split, so "
            "folding in a publisher test set would change the fixed partition:"
        )
        _table(
            ["folder", "images", "excluded by"],
            [[leaf.relative, leaf.n_images, leaf.ignored_by] for leaf in skipped],
        )

    stray = sum(leaf.n_other_files for leaf in leaves)
    if stray:
        print(
            f"\n  Note: {stray} file(s) carry an extension outside "
            f"datasets.image_extensions and would be ignored by the loader."
        )


def report_mapping(proposals: Sequence[Proposal], threshold: float) -> None:
    """Print the proposed class mapping for review."""
    _rule("PROPOSED CLASS MAPPING")
    _table(
        ["target class", "label", "source folder", "images", "status", "evidence"],
        [
            [
                proposal.target,
                proposal.display,
                proposal.leaf.relative if proposal.leaf else "-",
                proposal.leaf.n_images if proposal.leaf else 0,
                proposal.status,
                proposal.method,
            ]
            for proposal in proposals
        ],
    )

    ambiguous = [proposal for proposal in proposals if proposal.alternatives]
    if ambiguous:
        print("\n  Runner-up candidates, for context:")
        for proposal in ambiguous:
            listed = ", ".join(
                f"{name} ({score:.2f})" for name, score in proposal.alternatives[:4]
            )
            print(f"    {proposal.target}: {listed}")

    tied = [
        proposal
        for proposal in proposals
        if not proposal.forced
        and any(score >= proposal.score for _, score in proposal.alternatives)
    ]
    if tied:
        print(
            "\n  WARNING: the following class(es) had a runner-up scoring as "
            "highly as the folder chosen, which was then settled by name order "
            "alone. Pin them with --map <class>=<folder>:"
        )
        for proposal in tied:
            print(f"    {proposal.target}: chose {proposal.leaf.relative}")

    weak = [
        proposal
        for proposal in proposals
        if proposal.leaf is not None and not proposal.forced and proposal.score < threshold
    ]
    if weak:
        print(
            f"\n  WARNING: {len(weak)} class(es) matched below the {threshold:.2f} "
            f"confidence threshold. Confirm them by eye, or pin them with "
            f"--map <class>=<folder>."
        )

    unmatched = [proposal for proposal in proposals if proposal.leaf is None]
    if unmatched:
        names = ", ".join(proposal.target for proposal in unmatched)
        print(
            f"\n  WARNING: no source folder was matched to: {names}. Pin them "
            f"with --map <class>=<folder> before copying."
        )


def report_plan(config: Config, proposals: Sequence[Proposal]) -> None:
    """Print the copy plan and the current state of the destination."""
    _rule("COPY PLAN (DRY RUN - NOTHING IS WRITTEN)")
    destination_root = config.paths.primary_root
    accepted = {str(ext).lower() for ext in config.image_extensions}
    rows: List[List[Any]] = []
    occupied: List[str] = []
    total = 0

    for proposal in proposals:
        target_dir = destination_root / proposal.target
        existing = 0
        if target_dir.is_dir():
            existing = sum(
                1
                for entry in target_dir.iterdir()
                if entry.is_file() and entry.suffix.lower() in accepted
            )
        if existing:
            occupied.append(proposal.target)
        planned = proposal.leaf.n_images if proposal.leaf else 0
        total += planned
        rows.append(
            [
                proposal.leaf.relative if proposal.leaf else "-",
                "->",
                f"{_display_path(target_dir, config.paths.project_root)}/",
                planned,
                existing or "-",
            ]
        )

    _table(["from (source)", "", "to (destination)", "images", "already there"], rows)
    print(f"\n  {total} image(s) would be copied into {destination_root}")
    if occupied:
        print(
            f"\n  {', '.join(occupied)} already hold image(s); stage 2 will ask "
            f"before overwriting them."
        )


def _display_path(path: Path, root: Path) -> str:
    """Render ``path`` relative to the project root when it lies beneath it."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------- #
# Stage 2 - copy
# --------------------------------------------------------------------------- #

#: What to do with a destination class folder that already holds images.
ACTIONS = {
    "m": "merge",
    "o": "overwrite",
    "s": "skip",
    "a": "abort",
}


@dataclass
class CopyOutcome:
    """What the copy stage did to one target class folder."""

    target: str
    action: str
    copied: int = 0
    already_staged: int = 0
    renamed: List[Tuple[str, str]] = field(default_factory=list)
    deleted: int = 0
    undecodable: List[str] = field(default_factory=list)
    final_count: int = 0


def staged_images(directory: Path, accepted: Set[str]) -> List[Path]:
    """Return the image files directly inside ``directory``, sorted by name."""
    if not directory.is_dir():
        return []
    return sorted(
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in accepted
    )


def choose_actions(
    proposals: Sequence[Proposal],
    destination_root: Path,
    accepted: Set[str],
    assume_yes: bool,
) -> Dict[str, str]:
    """Decide what to do about class folders that already hold images.

    Every prompt is answered before the first file is written, so answering
    ``abort`` at any point leaves the destination untouched.

    Args:
        proposals: The confirmed mapping.
        destination_root: ``data/primary``.
        accepted: Accepted image extensions, lowercase and dot-prefixed.
        assume_yes: Skip the prompts and merge into any occupied folder.

    Returns:
        Target class name to one of :data:`ACTIONS`.

    Raises:
        IngestError: If the answer was ``abort``.
    """
    actions: Dict[str, str] = {}
    interactive = sys.stdin is not None and sys.stdin.isatty()

    for proposal in proposals:
        if proposal.leaf is None:
            actions[proposal.target] = "skip"
            continue
        existing = staged_images(destination_root / proposal.target, accepted)
        if not existing:
            actions[proposal.target] = "merge"
            continue

        if assume_yes or not interactive:
            reason = "--yes" if assume_yes else "no interactive terminal"
            print(
                f"\n  {proposal.target} already holds {len(existing)} image(s); "
                f"merging ({reason}). Nothing already staged will be deleted."
            )
            actions[proposal.target] = "merge"
            continue

        print(
            f"\n  {_display_path(destination_root / proposal.target, PROJECT_ROOT)}/ "
            f"already holds {len(existing)} image(s)."
        )
        print("    [m] merge     - keep them; copy only what is not staged yet (default)")
        print("    [o] overwrite - delete those images first, then copy")
        print("    [s] skip      - leave this class untouched")
        print("    [a] abort     - stop now, writing nothing")
        try:
            answer = input("    Choice [m/o/s/a]: ").strip().lower() or "m"
        except EOFError:
            # isatty() can report a terminal that nonetheless has no input to
            # give - a piped or wrapped shell, for instance. Merging is the
            # only safe reading of silence, since it deletes nothing.
            print("\n    (no answer available; merging, which deletes nothing)")
            actions[proposal.target] = "merge"
            continue
        action = ACTIONS.get(answer[:1])
        if action is None:
            raise IngestError(f"'{answer}' is not one of m, o, s or a. Nothing was written.")
        if action == "abort":
            raise IngestError("aborted at the prompt. Nothing was written.")
        actions[proposal.target] = action

    return actions


def copy_class(
    proposal: Proposal,
    destination_root: Path,
    action: str,
    accepted: Set[str],
) -> CopyOutcome:
    """Stage one class folder, honouring the chosen action.

    Copying is idempotent: a file already present at the destination with the
    same name and the same size is left alone rather than rewritten, so a
    repeat run is cheap and cannot multiply the dataset. A name that collides
    with a *different* file is staged under a deterministic
    ``<source folder>__<name>`` suffix rather than a counter, so repeat runs
    converge instead of accumulating.

    Args:
        proposal: The class and the source folder mapped to it.
        destination_root: ``data/primary``.
        action: One of ``merge``, ``overwrite`` or ``skip``.
        accepted: Accepted image extensions, lowercase and dot-prefixed.

    Returns:
        A :class:`CopyOutcome` describing what happened.
    """
    outcome = CopyOutcome(target=proposal.target, action=action)
    target_dir = destination_root / proposal.target

    if action == "skip" or proposal.leaf is None:
        outcome.final_count = len(staged_images(target_dir, accepted))
        return outcome

    if action == "overwrite":
        for path in staged_images(target_dir, accepted):
            path.unlink()
            outcome.deleted += 1

    target_dir.mkdir(parents=True, exist_ok=True)
    source_files = sorted(
        entry
        for entry in proposal.leaf.path.iterdir()
        if entry.is_file() and entry.suffix.lower() in accepted
    )

    for source in source_files:
        destination = target_dir / source.name
        if destination.exists():
            if destination.stat().st_size == source.stat().st_size:
                outcome.already_staged += 1
                continue
            destination = target_dir / f"{proposal.leaf.name}__{source.name}"
            if destination.exists() and destination.stat().st_size == source.stat().st_size:
                outcome.already_staged += 1
                continue
            outcome.renamed.append((source.name, destination.name))
        shutil.copy2(source, destination)
        outcome.copied += 1

    outcome.final_count = len(staged_images(target_dir, accepted))
    return outcome


def verify_decodable(paths: Sequence[Path]) -> List[Path]:
    """Return the files OpenCV cannot decode, after deleting each of them.

    A file that fails to decode would otherwise stop the harness partway
    through a run. Only files copied by this run are checked and removed, so
    the destination is left no worse than it started.
    """
    import cv2  # imported here so the survey stage needs no OpenCV

    broken: List[Path] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
        if image is None or image.size == 0:
            broken.append(path)
            path.unlink(missing_ok=True)
    return broken


def report_outcomes(
    config: Config,
    proposals: Sequence[Proposal],
    outcomes: Sequence[CopyOutcome],
) -> None:
    """Print what the copy stage did, class by class."""
    _rule("RESULT")
    by_target = {proposal.target: proposal for proposal in proposals}
    _table(
        ["target class", "action", "copied", "already staged", "deleted", "final count"],
        [
            [
                outcome.target,
                outcome.action,
                outcome.copied,
                outcome.already_staged or "-",
                outcome.deleted or "-",
                outcome.final_count,
            ]
            for outcome in outcomes
        ],
    )

    total = sum(outcome.final_count for outcome in outcomes)
    print(f"\n  {total} image(s) now staged under {config.paths.primary_root}")

    for outcome in outcomes:
        proposal = by_target[outcome.target]
        if outcome.action == "skip" or proposal.leaf is None:
            continue
        # The source count less anything that failed to decode is what a clean
        # staging should leave behind; a surplus means the folder already held
        # images from an earlier or different source.
        expected = proposal.leaf.n_images - len(outcome.undecodable)
        difference = outcome.final_count - expected
        if difference > 0:
            print(
                f"\n  Note: {outcome.target} holds {outcome.final_count} image(s), "
                f"{difference} more than its source accounts for - the folder "
                f"already held images from an earlier run or another source."
            )
        elif difference < 0:
            print(
                f"\n  Note: {outcome.target} holds {outcome.final_count} image(s), "
                f"{-difference} fewer than its source accounts for - some files "
                f"could not be staged."
            )

    renamed = [(outcome.target, pair) for outcome in outcomes for pair in outcome.renamed]
    if renamed:
        print(f"\n  {len(renamed)} filename collision(s) staged under a prefixed name:")
        for target, (original, staged) in renamed[:10]:
            print(f"    {target}: {original} -> {staged}")
        if len(renamed) > 10:
            print(f"    ... and {len(renamed) - 10} more")

    broken = [(outcome.target, name) for outcome in outcomes for name in outcome.undecodable]
    if broken:
        print(
            f"\n  WARNING: {len(broken)} newly copied file(s) could not be "
            f"decoded by OpenCV and were removed again, so the harness will "
            f"not choke on them mid-run:"
        )
        for target, name in broken[:10]:
            print(f"    {target}: {name}")
        if len(broken) > 10:
            print(f"    ... and {len(broken) - 10} more")


def run_copy(
    config: Config,
    proposals: Sequence[Proposal],
    assume_yes: bool,
) -> Tuple[List[CopyOutcome], bool]:
    """Execute the copy plan and verify what was staged.

    Returns:
        The per-class outcomes, and whether every staged file decoded.
    """
    unmatched = [proposal.target for proposal in proposals if proposal.leaf is None]
    if unmatched:
        raise IngestError(
            f"refusing to copy: no source folder is mapped to "
            f"{', '.join(unmatched)}. Pin them with --map <class>=<folder> "
            f"and re-run the survey first."
        )

    destination_root = config.paths.primary_root
    accepted = {str(ext).lower() for ext in config.image_extensions}

    _rule("STAGE 2 - COPY")
    actions = choose_actions(proposals, destination_root, accepted, assume_yes)

    outcomes: List[CopyOutcome] = []
    for proposal in proposals:
        action = actions[proposal.target]
        print(f"  {proposal.target}: {action} ...", end="", flush=True)
        before = {path.name for path in staged_images(destination_root / proposal.target, accepted)}
        outcome = copy_class(proposal, destination_root, action, accepted)
        after = staged_images(destination_root / proposal.target, accepted)
        fresh = [path for path in after if path.name not in before]
        broken = verify_decodable(fresh)
        outcome.undecodable = [path.name for path in broken]
        outcome.copied -= len(broken)
        outcome.final_count = len(staged_images(destination_root / proposal.target, accepted))
        outcomes.append(outcome)
        print(f" {outcome.copied} copied, {outcome.already_staged} already staged")

    report_outcomes(config, proposals, outcomes)
    clean = not any(outcome.undecodable for outcome in outcomes)
    return outcomes, clean


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_map_arguments(pairs: Sequence[str]) -> Dict[str, str]:
    """Parse ``--map CLASS=folder`` arguments into a dictionary."""
    overrides: Dict[str, str] = {}
    for pair in pairs:
        target, separator, folder = pair.partition("=")
        target, folder = target.strip(), folder.strip()
        if not separator or not target or not folder:
            raise IngestError(f"--map expects CLASS=folder, got '{pair}'")
        overrides[target] = folder
    return overrides


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Survey a source copy of the primary dataset and propose a class "
            "mapping. Stage 1 only: nothing is copied."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--local",
        nargs="?",
        const="",
        metavar="PATH",
        help=(
            "read an already unpacked folder; with no value, uses "
            "ingest.local_source from config.json (this is the default mode)"
        ),
    )
    source.add_argument(
        "--download",
        action="store_true",
        help="fetch ingest.kaggle_slug with kagglehub instead of reading a local folder",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="maximum directory depth shown in the tree listing (default: 3)",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="CLASS=FOLDER",
        help="pin a target class to a source folder, overriding the proposal",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="stage 2: copy the mapped folders into data/primary (survey only without it)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="answer every overwrite prompt with 'merge'; never deletes anything",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the survey and print the proposed mapping."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - redirected or older stdout
        pass

    args = build_parser().parse_args(argv)
    config = get_config()
    settings = survey_settings(config)

    try:
        overrides = parse_map_arguments(args.map)
        if args.download:
            source_root = resolve_download_source(settings)
        else:
            source_root = resolve_local_source(args.local or None, settings)

        _rule("SOURCE")
        print(f"  mode        : {'kagglehub download' if args.download else 'local folder'}")
        print(f"  source root : {source_root}")
        print(f"  destination : {config.paths.primary_root}")
        print(f"  classes     : {', '.join(config.primary.classes)}")
        print(f"  extensions  : {', '.join(config.image_extensions)}")

        _rule(f"DIRECTORY TREE (depth {args.depth})")
        for line in render_tree(source_root, args.depth, config.image_extensions):
            print("  " + line)

        leaves = survey_source(
            source_root,
            config.image_extensions,
            [str(name) for name in settings["ignore_dirs"]],
        )
        if not leaves:
            raise IngestError(
                f"no folder under {source_root} contains a file with an accepted "
                f"extension ({', '.join(config.image_extensions)}). Check that "
                f"the archive is unpacked and that the path is right."
            )
        report_leaves(leaves)

        proposals = propose_mapping(
            leaves,
            config.primary.classes,
            config.primary.display_names,
            settings.get("class_aliases", {}),
            overrides,
            float(settings["review_threshold"]),
        )
        report_mapping(proposals, float(settings["match_threshold"]))
        report_plan(config, proposals)

        if not args.copy:
            _rule("NEXT STEP")
            print(
                "  Survey only - nothing was copied.\n"
                "  Review the PROPOSED CLASS MAPPING table above. If a row is\n"
                "  wrong, re-run with --map <class>=<folder> to pin it.\n"
                "  When it is right, re-run the same command with --copy."
            )
            return 0

        _, clean = run_copy(config, proposals, args.yes)
    except IngestError as error:
        # Flush first: when stdout is piped it is block-buffered while stderr
        # is not, so an unflushed report would appear after its own error.
        sys.stdout.flush()
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        sys.stdout.flush()
        print("\nInterrupted. Re-run with --copy to finish staging.", file=sys.stderr)
        return 130

    _rule("NEXT STEP")
    print(
        "  Confirm the loader can see the staged data:\n"
        "      python data.py\n"
        "  Then check the segmentation before any feature work:\n"
        "      python scripts/sanity_check_segmentation.py"
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
