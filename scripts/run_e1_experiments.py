"""T1 internal comparative experiments (E1.1 to E1.5).

Five questions about the MPEG-7 dominant colour descriptor, each answered by
5-fold stratified cross-validation on the **training partition only**:

===== =========================================================================
E1.1  Does the descriptor beat a conventional colour histogram? 37 dimensions
      of dominant colours against a 105-dimension 32-bin histogram with
      Stricker and Orengo colour moments.
E1.2  How many dominant colours? ``N`` in {3, 4, 5, 6}.
E1.3  Which clustering space? CIE L*a*b*, HSV, RGB.
E1.4  Is specular exclusion worth it? On against off, with the share of pixels
      each arm discards reported alongside.
E1.5  What does each block contribute? A alone (28), A+B (32), A+B+C (37), and
      A+B+C without the decay-share dimension (36).
===== =========================================================================

The naming collides with Phase 5, which is unfortunate and worth stating
plainly: **E1.1 to E1.5 here are sweeps of the T1 descriptor**, and have
nothing to do with the enhancement ``scripts/run_enhancements.py`` calls E1,
which is feature-level fusion of all three techniques. Different experiment,
different file, different directory.

The test split is never read. It is drawn by the shared harness and then left
alone, so nothing tuned here can be justified by the numbers it will later be
judged against. Nothing here is tuned towards the report's acceptance targets
either: a worse number is a result.

Every variant is scored on one shared segmentation pass, exactly as
``run_t3_experiments.py`` does. Preprocessing and seeded GrabCut cost roughly
400 ms per image against the descriptor's 45 ms, and they are identical across
variants by construction - the descriptor is the only thing being changed - so
running them once and fanning out to all eight configurations is both far
faster and stricter than eight independent passes, which could otherwise drift
apart.

Run from the project root::

    # Smoke: exercises all five experiments end to end in about a minute.
    python scripts/run_e1_experiments.py --per-class 25 --no-augment --tag smoke

    # The full run the report quotes. Roughly 25 minutes.
    python scripts/run_e1_experiments.py

Results land in ``results/e1/``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config import Config, get_config  # noqa: E402
from data import ImageRecord, load_primary  # noqa: E402
from evaluate import (  # noqa: E402
    classification_metrics,
    cross_validate_technique,
    print_report,
    save_dataframe,
)
from features.base import FeatureExtractionError  # noqa: E402
from selection import (  # noqa: E402
    add_paired_differences,
    criterion_agreement,
    report_criterion_agreement,
    select_winner,
)
from features.colour_histogram import ColourHistogramExtractor  # noqa: E402
from features.t1_dominant_colour import DominantColourExtractor  # noqa: E402
from harness import (  # noqa: E402
    FeatureMatrix,
    build_pipeline,
    collect_failures,
    iter_prepared,
    set_global_seed,
    stratified_split,
)

#: Every distinct extractor configuration the five experiments need, named.
#: ``baseline`` is the descriptor as ``config.json`` configures it and is
#: shared by all five, so it is extracted once.
#:
#: E1.5 is absent from this table on purpose. Its four arms differ only in
#: which blocks are emitted, the blocks are computed independently of one
#: another, and they are concatenated in a fixed order - so an arm is obtained
#: by selecting columns from the full vector rather than by re-extracting. The
#: values are identical either way, and selecting keeps all four arms on
#: exactly the same images. :func:`assert_block_layout` checks that
#: assumption against the extractor rather than trusting it.
#: Image-processing operations each configuration performs before it emits a
#: vector, used only as the second parsimony tie-break in select_winner.
#: Counted explicitly rather than inferred: colour conversion, specular
#: exclusion, clustering and summarising are one each. Changing N or selecting
#: blocks adds no operation, so those arms carry the default's count.
PIPELINE_STAGES = {
    "baseline": 4,
    "n_colours_3": 4, "n_colours_5": 4, "n_colours_6": 4,
    "space_hsv": 4, "space_rgb": 4,
    "specular_off": 3,
    "histogram_baseline": 3,       # convert, exclude specular, histogram+moments
    "decay_green_excluded": 4,
}

DCD_VARIANTS: Dict[str, dict] = {
    "baseline": {},
    "n_colours_3": {"n_colours": 3},
    "n_colours_5": {"n_colours": 5},
    "n_colours_6": {"n_colours": 6},
    "space_hsv": {"space": "HSV"},
    "space_rgb": {"space": "RGB"},
    "specular_off": {"exclude_specular": False},
    "decay_green_excluded": {"decay_exclude_green": True},
}

#: The one arm that is not a dominant colour descriptor at all.
HISTOGRAM_VARIANT = "histogram_baseline"


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def build_dcd(config: Config, **overrides: object) -> DominantColourExtractor:
    """Build a descriptor from ``config.json``, then apply ``overrides``.

    Reading the configuration first matters more than it looks. Constructing
    the extractor bare would silently fall back to the module defaults, so an
    edit to ``config.json`` would move every other driver's T1 and leave this
    one alone - the experiment would then be sweeping around a centre nobody
    else uses, and every arm would be wrong together, which is the hardest
    kind of wrong to notice.

    Args:
        config: The loaded configuration.
        **overrides: Constructor arguments to replace for this variant.

    Returns:
        A configured :class:`DominantColourExtractor`.

    Raises:
        KeyError: If an override names a parameter the extractor does not take,
            which would otherwise be silently ignored.
    """
    accepted = set(inspect.signature(DominantColourExtractor.__init__).parameters) - {"self"}
    unknown = set(overrides) - accepted
    if unknown:
        raise KeyError(
            f"{sorted(unknown)} are not constructor arguments of "
            f"DominantColourExtractor; the variant table is out of date"
        )

    block = {
        key: value
        for key, value in dict(config.t1_colour).items()
        if key in accepted
    }
    block["seed"] = config.seed
    block.update(overrides)
    return DominantColourExtractor(**block)


def build_extractors(config: Config) -> Dict[str, object]:
    """Return every extractor configuration the five experiments need."""
    extractors: Dict[str, object] = {
        name: build_dcd(config, **overrides) for name, overrides in DCD_VARIANTS.items()
    }
    extractors[HISTOGRAM_VARIANT] = ColourHistogramExtractor.from_config(config)
    return extractors


def block_columns(extractor: DominantColourExtractor) -> Dict[str, np.ndarray]:
    """Return the column indices of blocks A, B and C in the emitted vector.

    Derived from the documented layout - ``7N`` for block A, ``N`` for B, five
    for C - and then checked against the extractor's own
    :attr:`~DominantColourExtractor.feature_names`, so a future change to the
    layout fails here rather than silently mislabelling an E1.5 arm.
    """
    n = extractor.n_colours
    return {
        "A": np.arange(0, 7 * n),
        "B": np.arange(7 * n, 8 * n),
        "C": np.arange(8 * n, 8 * n + 5),
    }


def assert_block_layout(extractor: DominantColourExtractor) -> int:
    """Check the block layout and return the decay-share column index.

    Raises:
        AssertionError: If the emitted vector is not laid out as E1.5 assumes.
    """
    names = list(extractor.feature_names)
    columns = block_columns(extractor)
    assert extractor.blocks == "ABC", (
        f"E1.5 slices the full ABC vector; config.json has blocks={extractor.blocks!r}"
    )
    assert len(names) == extractor.dim == 8 * extractor.n_colours + 5, (
        f"T1 declares {extractor.dim} dimensions and names {len(names)}"
    )
    assert all("centroid" in names[i] or "share" in names[i] or "variance" in names[i]
               for i in columns["A"]), "block A is not where E1.5 expects it"
    assert all("coherency" in names[i] for i in columns["B"]), (
        "block B is not where E1.5 expects it"
    )
    decay = int(columns["C"][-1])
    assert names[decay] == "T1_decay_share", (
        f"the last dimension of block C is {names[decay]!r}, not T1_decay_share"
    )
    return decay


# --------------------------------------------------------------------------- #
# One shared pass
# --------------------------------------------------------------------------- #

def balanced_subset(
    records: Sequence[ImageRecord],
    per_class: int,
    config: Config,
) -> List[ImageRecord]:
    """Draw ``per_class`` records from each class, reproducibly.

    Mirrors the benchmark and T3 drivers' subset so that a pilot experiment
    and a pilot benchmark describe the same images.
    """
    rng = np.random.default_rng(config.seed)
    chosen: List[int] = []
    for label in range(config.primary.n_classes):
        pool = [i for i, record in enumerate(records) if record.label == label]
        take = min(per_class, len(pool))
        chosen.extend(int(pool[i]) for i in rng.choice(len(pool), size=take, replace=False))
    return [records[i] for i in sorted(chosen)]


def extract_all_variants(
    records: Sequence[ImageRecord],
    extractors: Dict[str, object],
    augment: bool,
    config: Config,
) -> Tuple[Dict[str, FeatureMatrix], int]:
    """Build one :class:`~harness.FeatureMatrix` per variant, in a single pass.

    Row inclusion follows the harness exactly: a sample whose segmentation
    failed is dropped unless a substitute mask was supplied, and the group and
    variant provenance is carried through so that
    :func:`~evaluate.cross_validate_technique` can still draw leakage-safe
    folds.

    One thing is stricter than the T3 driver. A row is committed only once
    **every** variant has described it: the vectors are built into a scratch
    dictionary first, and if any extractor refuses the sample - the descriptor
    raises when a mask has fewer pixels than it has colours to fit, so a
    marginal mask can be describable at N=3 and not at N=6 - the row is
    dropped from all of them. Otherwise the arms would be scored on subtly
    different image sets and a difference in accuracy could be a difference in
    which apples each arm was shown.

    Returns:
        The matrices, and the number of rows dropped by an extraction refusal.
    """
    vectors: Dict[str, List[np.ndarray]] = {name: [] for name in extractors}
    times: Dict[str, List[float]] = {name: [] for name in extractors}
    warmed: Dict[str, bool] = {name: False for name in extractors}

    labels: List[int] = []
    paths: List[Path] = []
    groups: List[int] = []
    variant_ids: List[int] = []
    failures: List = []
    excluded = 0
    refused = 0
    n_augmented = 0

    total = len(records) * (
        1 + (config.augmentation.variants_per_image
             if augment and config.augmentation.enabled else 0)
    )
    done = 0

    for sample in iter_prepared(records, augment=augment, config=config):
        done += 1
        if done % 25 == 0 or done == total:
            print(f"    prepared {done}/{total}", end="\r", flush=True)

        if sample.segmentation.failed:
            failures.append(collect_failures([sample])[0])
            if not sample.segmentation.substituted:
                excluded += 1
                continue

        row: Dict[str, np.ndarray] = {}
        elapsed: Dict[str, float] = {}
        try:
            for name, extractor in extractors.items():
                image, mask = sample.image.copy(), sample.mask.copy()
                if not warmed[name]:
                    extractor(image, mask)  # Warm-up, not timed and not kept.
                    extractor.reset_diagnostics()
                    warmed[name] = True
                start = time.perf_counter()
                row[name] = extractor(image, mask)
                elapsed[name] = time.perf_counter() - start
        except FeatureExtractionError as error:
            # Roll the diagnostics back to the committed length, so that the
            # E1.4 exclusion percentages stay aligned with the matrix rows.
            for extractor in extractors.values():
                del extractor.diagnostics[len(labels):]
            refused += 1
            print(f"\n    refused {sample.record.path.name}: {error}")
            continue

        labels.append(sample.label)
        paths.append(sample.record.path)
        groups.append(sample.record_index)
        variant_ids.append(sample.variant)
        if sample.is_augmented:
            n_augmented += 1
        for name in extractors:
            vectors[name].append(row[name])
            times[name].append(elapsed[name])

    print()
    matrices = {
        name: FeatureMatrix(
            X=np.vstack(rows) if rows else np.empty((0, extractors[name].dim)),
            y=np.asarray(labels, dtype=np.int64),
            technique=name,
            extraction_times=np.asarray(times[name], dtype=np.float64),
            n_augmented=n_augmented,
            failures=failures,
            excluded=excluded,
            paths=paths,
            groups=np.asarray(groups, dtype=np.int64),
            variants=np.asarray(variant_ids, dtype=np.int64),
        )
        for name, rows in vectors.items()
    }
    return matrices, refused


def score(
    matrix: FeatureMatrix,
    arm: str,
    config: Config,
    **extra: object,
) -> dict:
    """Cross-validate one arm on the training partition and return its row."""
    report = cross_validate_technique(build_pipeline(config), matrix, technique=arm, config=config)
    print(
        f"    {arm:<24} dim {matrix.dim:>4}  "
        f"accuracy {report.mean_accuracy:.4f} +/- {report.std_accuracy:.4f}  "
        f"macro F1 {report.mean_macro_f1:.4f}"
    )
    # Read before the dict literal pops them: a dict literal evaluates its
    # values in order, so looking a key up after popping it silently yields
    # the not-found sentinel.
    config_key = str(extra.pop("config_key", ""))
    stages = PIPELINE_STAGES.get(config_key, -1)
    if config_key and stages < 0:
        raise KeyError(
            f"PIPELINE_STAGES has no entry for {config_key!r}; the parsimony "
            f"tie-break would silently rank this arm first."
        )
    display = list(config.primary.display_names)
    per_class = report.per_class_frame(display)
    recalls = (
        {f"recall_{n}": float(per_class.set_index("class").loc[n, "recall"]) for n in display}
        if not per_class.empty else {f"recall_{n}": float("nan") for n in display}
    )

    return {
        "arm": arm,
        "config_key": config_key,
        "pipeline_stages": stages,
        "dimensionality": matrix.dim,
        "cv_mean_accuracy": report.mean_accuracy,
        "cv_std_accuracy": report.std_accuracy,
        "cv_mean_macro_f1": report.mean_macro_f1,
        "cv_std_macro_f1": report.std_macro_f1,
        **{f"fold{i + 1}_accuracy": float(value) for i, value in enumerate(report.accuracy_folds)},
        **recalls,
        "extraction_mean_s": float(np.mean(matrix.extraction_times)),
        "rows": int(matrix.X.shape[0]),
        **extra,
    }


# --------------------------------------------------------------------------- #
# The five experiments
# --------------------------------------------------------------------------- #

def e1_1_baseline(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E1.1 - the descriptor against a conventional colour histogram.

    Both arms see the same images, the same masks, the same colour space and
    the same specular exclusion. The descriptor family is the only difference,
    which is what makes the gap attributable to it. The baseline is nearly
    three times longer, so a win for the descriptor is a win on both axes.
    """
    print("\n  E1.1  descriptor against histogram baseline")
    arms = {
        "dcd_37": ("baseline", "MPEG-7 dominant colour"),
        "histogram_moments_105": (HISTOGRAM_VARIANT, "32-bin histogram + colour moments"),
    }
    return add_paired_differences(pd.DataFrame(
        [
            score(matrices[key], arm, config, config_key=key, descriptor=description)
            for arm, (key, description) in arms.items()
        ]
    ), "dcd_37")


def e1_2_n_colours(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E1.2 - how many dominant colours the descriptor should carry.

    The vector length changes with ``N``, so dimensionality is reported
    alongside accuracy. Nothing is padded or truncated to make the arms
    comparable: that the arms are *not* the same length is half the finding,
    since a gain bought with sixteen extra dimensions is a different kind of
    gain from a free one.
    """
    print("\n  E1.2  number of dominant colours")
    arms = {3: "n_colours_3", 4: "baseline", 5: "n_colours_5", 6: "n_colours_6"}
    return add_paired_differences(pd.DataFrame(
        [
            score(
                matrices[key],
                f"n_colours_{n}",
                config,
                config_key=key,
                n_colours=n,
                expected_dim=8 * n + 5,
            )
            for n, key in arms.items()
        ]
    ), "n_colours_4")


def e1_3_space(matrices: Dict[str, FeatureMatrix], config: Config) -> pd.DataFrame:
    """E1.3 - which colour space the clustering should run in.

    Only the clustering changes. The five ripeness indices of block C are
    computed from CIE L*a*b* whatever space was clustered in, by construction
    in the extractor, so those five dimensions mean the same thing in all
    three arms and the comparison is genuinely about the clustering.
    """
    print("\n  E1.3  clustering colour space")
    arms = {"LAB": "baseline", "HSV": "space_hsv", "RGB": "space_rgb"}
    return add_paired_differences(pd.DataFrame(
        [
            score(matrices[key], f"space_{space.lower()}", config, config_key=key, space=space)
            for space, key in arms.items()
        ]
    ), "space_lab")


def e1_4_specular(
    matrices: Dict[str, FeatureMatrix],
    extractors: Dict[str, object],
    config: Config,
) -> pd.DataFrame:
    """E1.4 - whether excluding specular highlights helps.

    The accuracy difference alone is uninterpretable without knowing how much
    was thrown away to get it, so the share of masked pixels each arm dropped
    is reported beside it. Two numbers make the result readable:

    * ``mean_pct_pixels_excluded`` - what the arm actually discarded. Zero for
      the disabled arm by construction.
    * ``pct_images_specular_fallback`` - how often exclusion was abandoned
      because it would have left too few pixels to cluster. A large value here
      means the enabled arm is quietly behaving like the disabled one on part
      of the dataset, and the accuracy gap understates the true effect.
    """
    print("\n  E1.4  specular exclusion")
    arms = {"specular_on": "baseline", "specular_off": "specular_off"}
    rows = []
    for arm, key in arms.items():
        diagnostics = extractors[key].diagnostics
        matrix = matrices[key]
        if len(diagnostics) != matrix.X.shape[0]:
            raise AssertionError(
                f"{key}: {len(diagnostics)} diagnostics for {matrix.X.shape[0]} rows; "
                f"the two must be row-aligned for these percentages to mean anything"
            )
        fractions = np.asarray([d.specular_fraction for d in diagnostics], dtype=np.float64)
        fallbacks = np.asarray([d.specular_fallback for d in diagnostics], dtype=bool)
        excluded = np.where(fallbacks, 0.0, fractions)
        rows.append(
            score(
                matrix,
                arm,
                config,
                config_key=key,
                exclude_specular=(key == "baseline"),
                mean_pct_pixels_excluded=float(np.mean(excluded) * 100.0),
                max_pct_pixels_excluded=float(np.max(excluded, initial=0.0) * 100.0),
                mean_pct_pixels_flagged_specular=float(np.mean(fractions) * 100.0),
                pct_images_specular_fallback=float(np.mean(fallbacks) * 100.0),
                mean_analysis_pixels=float(np.mean([d.analysis_pixels for d in diagnostics])),
            )
        )
    return add_paired_differences(pd.DataFrame(rows), "specular_on")


def e1_5_blocks(
    matrices: Dict[str, FeatureMatrix],
    extractors: Dict[str, object],
    config: Config,
) -> pd.DataFrame:
    """E1.5 - what each block of the descriptor contributes.

    Four arms, the last of which is not in the original specification:

    ``A_only``
        The 28 dominant-colour dimensions alone.
    ``AB``
        Plus the four spatial coherency dimensions.
    ``ABC_full``
        Plus the five ripeness indices. The descriptor as it ships.
    ``ABC_minus_decay``
        The full vector with ``T1_decay_share`` removed, 36 dimensions.

    The fourth arm exists because the decay index appears to be inverted on
    held-out exemplars: it reads 26.5% on an Unripe apple and 0.0% on a Rotten
    one. The suspected mechanism is that the index tests a cluster for low
    chroma **and** low lightness, and a dark, low-chroma green cluster
    (L* 14.7, a* -10.4, b* 10.3) satisfies both - so on a green apple in
    shadow the index is detecting the shadow, not decay.

    This arm does not fix that. It measures whether the dimension is
    contributing signal or noise, which is the question that has to be
    answered before anyone changes the thresholds. If removing it costs
    nothing, the dimension is not doing the job it is named for; if removing
    it costs accuracy, it is carrying something real under a misleading name.
    Either way the number is reported as it comes out.
    """
    print("\n  E1.5  block ablation")
    full = matrices["baseline"]
    extractor = extractors["baseline"]
    columns = block_columns(extractor)
    decay = assert_block_layout(extractor)

    ab = np.concatenate([columns["A"], columns["B"]])
    abc = np.arange(full.dim)
    arms = {
        "A_only": (columns["A"], "A", "dominant colours"),
        "AB": (ab, "AB", "adds spatial coherency"),
        "ABC_full": (abc, "ABC", "adds the five ripeness indices"),
        "ABC_minus_decay": (
            np.delete(abc, decay),
            "ABC-decay",
            "full vector without T1_decay_share",
        ),
    }
    names = list(extractor.feature_names)
    return add_paired_differences(pd.DataFrame(
        [
            score(
                replace(full, X=full.X[:, selected], technique=arm),
                arm,
                config,
                config_key="baseline",
                blocks=label,
                description=description,
                dropped_feature="T1_decay_share" if arm == "ABC_minus_decay" else "",
                first_feature=names[int(selected[0])],
                last_feature=names[int(selected[-1])],
            )
            for arm, (selected, label, description) in arms.items()
        ]
    ), "ABC_full")


PREDICTION_FILE = (
    Path(__file__).resolve().parent.parent
    / "results" / "predictions" / "e1_6_decay_green_exclusion.txt"
)


def prediction_digest() -> str:
    """SHA-256 of the recorded prediction, or a marker if it is absent.

    Carried in the E1.6 rows so that "the prediction preceded the result" is
    checkable rather than asserted: the file cannot be edited afterwards
    without the recorded digest ceasing to match.
    """
    if not PREDICTION_FILE.exists():
        return "ABSENT"
    return hashlib.sha256(PREDICTION_FILE.read_bytes()).hexdigest()


def e1_6_decay_green_exclusion(
    matrices: Dict[str, FeatureMatrix],
    config: Config,
) -> pd.DataFrame:
    """E1.6 - removing shadowed green peel from the decay index.

    The decay index counts clusters that are dark and weakly chromatic.
    Shadowed green peel satisfies both, so on unripe fruit it counts shade as
    decay. Measured over 590 training images through the extractor itself,
    requiring the cluster also to be non-green removes 93.7% of the Unripe
    reading and exactly none of the Rotten one - the result the mechanism
    predicts, since real decay is brown and never green.

    This arm is not evidence that the index is inverted. Across the population
    it already orders the classes correctly, reading 10.97 on Unripe, 7.75 on
    Ripe and 12.97 on Rotten; the inversion reported on two held-out exemplars
    does not generalise. What is real is the contamination, and this measures
    what removing it is worth.

    The prediction was recorded before the arm was run, and is judged in the
    same paired frame as every other arm. A flat result is a redundancy
    finding - T1_green_share already identifies unripe fruit - not a failure.
    """
    print("\n  E1.6  decay index with shadowed green peel excluded")
    digest = prediction_digest()
    print(f"    prediction on file, sha256 {digest[:16]}...")
    return add_paired_differences(pd.DataFrame([
        score(matrices["baseline"], "decay_as_published", config,
              config_key="baseline", decay_exclude_green=False,
              prediction_sha256=digest),
        score(matrices["decay_green_excluded"], "decay_excluding_green", config,
              config_key="decay_green_excluded", decay_exclude_green=True,
              prediction_sha256=digest),
    ]), "decay_as_published")


def e1_6_sample_sensitivity(
    matrices: Dict[str, FeatureMatrix],
    extractors: Dict[str, object],
    config: Config,
    draws: int = 400,
) -> pd.DataFrame:
    """How the decay index's apparent strength depends on the sample size.

    A finding rather than a caveat. The same quantity reads very differently
    at different sample sizes, so an experiment that stopped early would have
    reported an effect substantially larger than the one that is there.

    The resampling is free: every training image's decay value is already in
    the extracted matrix, so subsets are drawn from those values rather than
    re-extracted. Draws are without replacement within each class, which makes
    this the sampling distribution of the ratio under the study's own design
    rather than a bootstrap approximation to it.
    """
    names = list(extractors["baseline"].feature_names)
    decay = matrices["baseline"].X[:, names.index("T1_decay_share")]
    labels = matrices["baseline"].y
    display = list(config.primary.display_names)
    unripe = decay[labels == display.index("Unripe")]
    rotten = decay[labels == display.index("Rotten")]

    print("\n  E1.6b sample-size sensitivity of the decay separation")
    rng = np.random.default_rng(config.seed)
    rows = []
    smallest = int(min(unripe.size, rotten.size))
    for per_class in (30, 45, 60, 100, 200, 400, smallest):
        if per_class > smallest:
            continue
        ratios = []
        for _ in range(draws if per_class < smallest else 1):
            u = rng.choice(unripe, size=per_class, replace=False).mean()
            r = rng.choice(rotten, size=per_class, replace=False).mean()
            if u > 0:
                ratios.append(r / u)
        values = np.asarray(ratios, dtype=np.float64)
        rows.append({
            "images_per_class": int(per_class),
            "draws": int(values.size),
            "ratio_rotten_over_unripe_mean": float(values.mean()),
            "ratio_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "ratio_p05": float(np.percentile(values, 5)),
            "ratio_p95": float(np.percentile(values, 95)),
            "unripe_decay_mean": float(unripe.mean()),
            "rotten_decay_mean": float(rotten.mean()),
            "is_full_partition": bool(per_class == smallest),
        })
        print(f"    {per_class:>4} per class   ratio {values.mean():.2f}"
              f"  (5th-95th {np.percentile(values, 5):.2f}-{np.percentile(values, 95):.2f})")
    return pd.DataFrame(rows)


def evaluate_winner(
    tables: Dict[str, pd.DataFrame],
    matrices: Dict[str, FeatureMatrix],
    extractors: Dict[str, object],
    partition,
    config: Config,
) -> pd.DataFrame:
    """Score the parsimony-selected arm on the test partition, once.

    This is a **final confirmation, not a selection criterion**. Selection is
    done entirely on cross-validation over the training partition using the
    tie band, and by the time this runs every table is closed and the winner
    is already fixed. Chapter 4 needs one held-out number and one is what this
    produces: arms_scored_on_test records that exactly one arm was evaluated,
    so a reader can confirm the test split chose nothing.
    """
    combined = (
        pd.concat([t for t in tables.values() if "arm" in t.columns], ignore_index=True)
        .drop_duplicates(subset=["arm"]).reset_index(drop=True)
    )
    winner, tied, selected_by = select_winner(combined)
    leader = combined.loc[combined.cv_mean_accuracy.idxmax()]
    key = str(winner.config_key) or "baseline"
    dropped = str(getattr(winner, "dropped_feature", "") or "")

    print("\n  Closing evaluation - one arm, one pass over the test partition")
    print(f"    highest accuracy  : {leader.arm} (cv {leader.cv_mean_accuracy:.4f})")
    print(f"    tied with it      : {len(tied)} of {len(combined)} arms")
    print(f"    selected          : {winner.arm} (dim {int(winner.dimensionality)}) "
          f"by {selected_by}")
    print(f"    extracting test features for {key} only ...")

    test_matrices, _ = extract_all_variants(
        partition.test, {key: extractors[key]}, False, config
    )
    test = test_matrices[key]
    train = matrices[key]

    # An E1.5 arm is a column selection over the baseline vector, not a
    # separate extractor, so the winner has to be rebuilt the same way it was
    # scored. Reproducing the selection here rather than trusting the recorded
    # dimensionality means a block arm and its held-out evaluation cannot
    # silently describe different feature sets.
    blocks = str(getattr(winner, "blocks", "") or "")
    if blocks and blocks != "ABC":
        extractor = extractors["baseline"]
        columns = block_columns(extractor)
        decay = assert_block_layout(extractor)
        selection_map = {
            "A": columns["A"],
            "AB": np.concatenate([columns["A"], columns["B"]]),
            "ABC-decay": np.delete(np.arange(extractor.dim), decay),
        }
        if blocks not in selection_map:
            raise KeyError(f"no column selection recorded for blocks={blocks!r}")
        chosen = selection_map[blocks]
        train = replace(train, X=train.X[:, chosen])
        test = replace(test, X=test.X[:, chosen])
    elif dropped:
        names = list(extractors[key].feature_names)
        keep = [i for i, n in enumerate(names) if n != dropped]
        train = replace(train, X=train.X[:, keep])
        test = replace(test, X=test.X[:, keep])

    if train.X.shape[1] != int(winner.dimensionality):
        raise AssertionError(
            f"rebuilt the winner at {train.X.shape[1]} dimensions but it was "
            f"scored at {int(winner.dimensionality)}; the held-out evaluation "
            f"would not be of the arm that was selected"
        )

    fitted = build_pipeline(config).fit(train.X, train.y)
    report = classification_metrics(
        test.y, fitted.predict(test.X), list(config.primary.display_names),
        technique=str(winner.arm),
    )
    print_report(report)
    return pd.DataFrame([{
        "arm": winner.arm,
        "config_key": key,
        "dropped_feature": dropped,
        "selected_by": selected_by,
        "dimensionality": int(winner.dimensionality),
        "highest_accuracy_arm": leader.arm,
        "highest_accuracy_cv": float(leader.cv_mean_accuracy),
        "winner_cv_accuracy": float(winner.cv_mean_accuracy),
        "arms_tied_with_leader": int(len(tied)),
        "arms_compared": int(len(combined)),
        "arms_scored_on_test": 1,
        "test_accuracy": report.accuracy,
        "test_macro_f1": report.macro_f1,
        "test_rows": int(test.X.shape[0]),
        "role": "final confirmation; selection was done on CV with the tie band",
    }])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=0,
                        help="Images per class; 0 uses the whole dataset.")
    parser.add_argument("--augment", dest="augment", action="store_true",
                        help="Apply the training augmentation plan. Roughly four times slower.")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="The default; accepted so the documented smoke command runs.")
    parser.add_argument("--tag", default="e1",
                        help="Sub-directory of results/ to write into.")
    parser.set_defaults(augment=False)
    args = parser.parse_args(argv)

    set_global_seed(config)
    records = load_primary(config)
    if args.per_class:
        records = balanced_subset(records, args.per_class, config)

    # The test split is drawn and then deliberately never touched again.
    partition = stratified_split(records, config)
    training = partition.train

    # Only a reduced image count makes a run a pilot. Augmentation is off by
    # default here, matching run_t3_experiments.py: these experiments tune the
    # descriptor against real images, and calling that "reduced data" would
    # misdescribe it.
    mode = "PILOT" if args.per_class else "FULL"
    extractors = build_extractors(config)

    print("=" * 74)
    print(f"T1 internal experiments - {mode}")
    print("=" * 74)
    print(f"  images          : {len(records)} ({args.per_class or 'all'} per class)")
    print(f"  training rows   : {len(training)}  (test split of {len(partition.test)} untouched)")
    print(f"  augmentation    : {'on' if args.augment else 'OFF'}")
    print(f"  variants        : {len(extractors)} extractor configurations, one segmentation pass")
    print(f"  cross-validation: {config.partition.cv_folds}-fold stratified, training partition only")
    if mode == "PILOT":
        print(f"  NOTE            : {args.per_class} images per class, not the full set. "
              f"Indicative only.")

    # Fail before the expensive pass if E1.5 cannot slice what it needs.
    assert_block_layout(extractors["baseline"])

    print("\n  Building features ...")
    started = time.perf_counter()
    matrices, refused = extract_all_variants(training, extractors, args.augment, config)
    elapsed = time.perf_counter() - started
    baseline = matrices["baseline"]
    print(f"  {baseline.X.shape[0]} rows x {len(extractors)} variants in {elapsed:.1f}s "
          f"({baseline.excluded} excluded by segmentation failure, "
          f"{refused} by extraction refusal)")

    output = config.paths.results_subdir(args.tag)
    tables = {
        "e1_1.csv": e1_1_baseline(matrices, config),
        "e1_2.csv": e1_2_n_colours(matrices, config),
        "e1_3.csv": e1_3_space(matrices, config),
        "e1_4.csv": e1_4_specular(matrices, extractors, config),
        "e1_5.csv": e1_5_blocks(matrices, extractors, config),
        "e1_6.csv": e1_6_decay_green_exclusion(matrices, config),
        "e1_6b_sample_sensitivity.csv": e1_6_sample_sensitivity(matrices, extractors, config),
    }
    comparisons = {k: v for k, v in tables.items() if "arm" in v.columns}
    final = evaluate_winner(comparisons, matrices, extractors, partition, config)
    agreement = criterion_agreement(comparisons)
    print()
    print(report_criterion_agreement(agreement))

    for filename, frame in tables.items():
        save_dataframe(frame.set_index("arm") if "arm" in frame.columns else frame,
                       output / filename)
    save_dataframe(final, output / "e1_final_test.csv", index=False)
    save_dataframe(agreement, output / "criterion_agreement.csv", index=False)
    (output / "criterion_agreement.txt").write_text(
        report_criterion_agreement(agreement) + "\n", encoding="utf-8")

    # The configuration that produced these numbers travels with them.
    log = {
        "mode": mode,
        "seed": config.seed,
        "per_class": args.per_class or None,
        "augmented": args.augment,
        "n_images": len(records),
        "n_training_records": len(training),
        "n_training_rows": int(baseline.X.shape[0]),
        "n_excluded_by_segmentation": int(baseline.excluded),
        "n_refused_by_extraction": int(refused),
        "cv_folds": config.partition.cv_folds,
        "test_split_touched": False,
        "dataset_root": str(config.paths.primary_root),
        "dataset_audit_verdict": "NOT SUITABLE - see README; 74.0% background-only "
                                 "against a 33.3% chance level. Relative comparisons "
                                 "between these arms are unaffected; absolute "
                                 "accuracies carry that ceiling.",
        "segmentation_method": config.segmentation.method,
        "t1_config": {k: v for k, v in dict(config.t1_colour).items() if k != "_comment"},
        "dcd_variants": DCD_VARIANTS,
        "histogram_baseline": {
            "bins": extractors[HISTOGRAM_VARIANT].bins,
            "space": extractors[HISTOGRAM_VARIANT].space,
            "exclude_specular": extractors[HISTOGRAM_VARIANT].exclude_specular,
            "dim": extractors[HISTOGRAM_VARIANT].dim,
        },
        "e1_5_note": "ABC_minus_decay is an addition to the original specification: "
                     "T1_decay_share reads 26.5% on an Unripe exemplar and 0.0% on a "
                     "Rotten one, consistent with a dark low-chroma green cluster "
                     "satisfying its low-chroma-low-lightness test. The arm measures "
                     "the dimension's contribution; it does not change it.",
        "elapsed_seconds": round(elapsed, 1),
    }
    (output / "run_config.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\n  Written to {output}:")
    for filename in tables:
        print(f"    {filename}")
    print("    run_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
