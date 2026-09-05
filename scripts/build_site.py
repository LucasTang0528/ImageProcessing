"""Build the local FreshSight results site from a completed benchmark run.

The site is a static page served from ``site/``. It has two parts:

* a dashboard ranking every technique on the figures the benchmark wrote, and
* an explorer that lets you pick a class, then a picture, then a technique,
  and see what that technique actually predicted for that picture.

Every gallery image is drawn from the **test** split, so the predictions shown
are held-out predictions and are consistent with the confusion matrices on the
dashboard. Nothing here re-tunes or re-selects anything; it re-fits the same
pipeline on the same partition and reports what comes out.

Run from the project root::

    python scripts/build_site.py --tag pilot3
    python -m http.server 8000 --directory site

then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config import Config, get_config  # noqa: E402
from data import ImageRecord, load_primary  # noqa: E402
from harness import (  # noqa: E402
    build_feature_matrix,
    build_pipeline,
    prepare_sample,
    set_global_seed,
    stratified_split,
)
from run_benchmarks import available_extractors, balanced_subset  # noqa: E402


#: Which descriptor values to surface per technique. A spec ending in ``*``
#: is averaged over every feature whose name starts with the prefix; anything
#: else is an exact feature name. Keeping this declarative means the page can
#: never show a number the extractor did not actually produce.
HEADLINE_FEATURES: Dict[str, List[Tuple[str, str, str]]] = {
    "T1": [
        ("Weighted mean a*", "T1_weighted_mean_a", "green negative, red positive"),
        ("Weighted mean chroma", "T1_weighted_mean_chroma", "colour intensity"),
        ("Circular mean hue", "T1_circular_mean_hue", "degrees"),
        ("Green share", "T1_green_share", "% of fruit in green clusters"),
        ("Decay share", "T1_decay_share", "% in dark, desaturated clusters"),
        ("Largest cluster share", "T1_c0_share", "% of fruit"),
    ],
    "T2": [
        ("Contrast", "T2_contrast_*", "local grey-level variation"),
        ("Dissimilarity", "T2_dissimilarity_*", "mean absolute difference"),
        ("Homogeneity", "T2_homogeneity_*", "1.0 is perfectly smooth"),
        ("Energy", "T2_energy_*", "uniformity of the co-occurrence"),
        ("Correlation", "T2_correlation_*", "linear grey dependence"),
    ],
    "T3": [
        ("Blemish area", "blemish_ratio", "% of fruit surface"),
        ("Blemish count", "blemish_count", "components above the size floor"),
        ("Largest blemish", "blemish_area_max", "px squared"),
        ("Granulometric mean size", "gran_mean_size", "structuring-element radius"),
        ("Black top-hat mean", "bth_mean", "depth of dark surface detail"),
        ("Morphological gradient", "mgrad_mean", "edge density"),
    ],
}

#: Written explanations, keyed by technique. These are claims about the
#: measured confusion matrices, so :func:`check_justifications` asserts the
#: shape of each one against the run before the page is written.
JUSTIFICATIONS: Dict[str, Dict[str, str]] = {
    "T1": {
        "headline": "Ripeness is mostly a colour progression, and this descriptor measures colour directly.",
        "body": (
            "An apple moves green, then red, then dark and desaturated. That is a path through "
            "colour space, so a descriptor built from dominant colour clusters reads the label "
            "almost off the surface. The weighted mean a* rises from strongly negative in unripe "
            "fruit to positive in ripe fruit, and chroma then falls as the fruit decays. What is "
            "left over is the Ripe/Rotten boundary: a browning apple stays red for a long time "
            "before it stops being red."
        ),
    },
    "T2": {
        "headline": "Surface roughness is not monotone in ripeness, so no threshold can order the three stages.",
        "body": (
            "Unripe skin is taut and textured, ripe skin is smooth and glossy, and rotten skin is "
            "broken and rough again. Texture therefore falls and then rises across the ripeness "
            "sequence, which puts the middle class in the middle of the feature space with the "
            "other two on either side of it. A classifier working from texture alone absorbs both "
            "extremes into Ripe, which is why Ripe recall looks respectable while Ripe precision "
            "does not."
        ),
    },
    "T3": {
        "headline": "Rot breaks the surface, and granulometry measures exactly that - but an intact apple is an intact apple.",
        "body": (
            "Decay collapses and pits the skin, which changes surface geometry at a measurable "
            "scale, so this is the strongest of the three descriptors at identifying rotten fruit "
            "specifically. Its weakness is the other end of the sequence: an unripe apple and a "
            "ripe apple both have smooth, undamaged surfaces, and geometry cannot tell them apart "
            "when the thing that separates them is colour. One block also actively costs it - the "
            "blemish detector thresholds the black top-hat with Otsu, and Otsu always splits an "
            "image, so it reports a similar blemish fraction whether or not there is a lesion to "
            "find. Those dimensions carry noise into an otherwise sound descriptor."
        ),
    },
}

CLASS_ORDER = ("Unripe", "Ripe", "Rotten")


def resolve_feature(names: Sequence[str], values: np.ndarray, spec: str) -> float:
    """Return one displayable number for ``spec`` against a feature vector."""
    if spec.endswith("*"):
        prefix = spec[:-1]
        picked = [v for n, v in zip(names, values) if n.startswith(prefix)]
        if not picked:
            raise KeyError(f"no feature starts with {prefix!r}")
        return float(np.mean(picked))
    index = list(names).index(spec)
    return float(values[index])


def contour_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw the segmentation boundary over a dimmed copy of the image."""
    dimmed = image_bgr.copy()
    outside = mask == 0
    dimmed[outside] = (dimmed[outside] * 0.28).astype(np.uint8)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(dimmed, contours, -1, (90, 220, 120), 2)
    return dimmed


def write_jpeg(image_bgr: np.ndarray, path: Path, quality: int = 88) -> None:
    """Write a BGR image, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    path.write_bytes(buffer.tobytes())


def choose_gallery(
    records: Sequence[ImageRecord],
    per_class: int,
    config: Config,
) -> List[ImageRecord]:
    """Pick a reproducible spread of test images from each class."""
    rng = np.random.default_rng(config.seed + 1)
    chosen: List[ImageRecord] = []
    for name in CLASS_ORDER:
        pool = [r for r in records if r.display_name == name]
        take = min(per_class, len(pool))
        picks = rng.choice(len(pool), size=take, replace=False)
        chosen.extend(pool[int(i)] for i in sorted(picks))
    return chosen


def load_run_metrics(results_dir: Path, techniques: Sequence[str]) -> Dict[str, dict]:
    """Read back exactly the figures the benchmark wrote to disk."""
    matrix = pd.read_csv(results_dir / "benchmark_matrix.csv").set_index("technique")
    metrics: Dict[str, dict] = {}
    for short in techniques:
        row = matrix.loc[short]
        per_class = pd.read_csv(results_dir / f"{short}_per_class.csv", index_col=0)
        confusion = pd.read_csv(
            results_dir / f"{short}_confusion_normalised.csv", index_col=0
        )
        folds = [
            float(row[c]) for c in matrix.columns if c.startswith("cv_fold")
        ]
        metrics[short] = {
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1"]),
            "weighted_f1": float(row["weighted_f1"]),
            "cv_mean": float(row["cv_mean_accuracy"]),
            "cv_std": float(row["cv_std_accuracy"]),
            "folds": folds,
            "dims": int(row["dimensionality"]),
            "extract_ms": float(row["extraction_mean_s"]) * 1000.0,
            "infer_ms": float(row["inference_mean_s"]) * 1000.0,
            "per_class": {
                name: {
                    "precision": float(per_class.loc[name, "precision"]),
                    "recall": float(per_class.loc[name, "recall"]),
                    "f1": float(per_class.loc[name, "f1"]),
                    "support": int(per_class.loc[name, "support"]),
                }
                for name in CLASS_ORDER
            },
            "confusion": {
                name: [float(v) for v in confusion.loc[name]] for name in CLASS_ORDER
            },
        }
    return metrics


def check_justifications(metrics: Dict[str, dict]) -> List[str]:
    """Verify each written claim against the run, returning any that fail.

    The explanations on the page assert specific shapes in the confusion
    matrices. If a future run contradicts one, the page should not quietly
    keep asserting it, so the mismatch is reported to the console.
    """
    problems: List[str] = []
    if "T2" in metrics:
        per_class = metrics["T2"]["per_class"]["Ripe"]
        if not per_class["recall"] > per_class["precision"]:
            problems.append(
                "T2: the page claims Ripe recall exceeds Ripe precision, but this run "
                f"has recall {per_class['recall']:.3f} and precision {per_class['precision']:.3f}"
            )
    if "T1" in metrics:
        confusion = metrics["T1"]["confusion"]
        if confusion["Unripe"][2] > 0.05:
            problems.append(
                "T1: the page treats Unripe/Rotten confusion as negligible, but this run "
                f"sends {confusion['Unripe'][2]:.1%} of Unripe to Rotten"
            )
    if "T3" in metrics:
        per_class = metrics["T3"]["per_class"]
        best = max(CLASS_ORDER, key=lambda c: per_class[c]["precision"])
        if best != "Rotten":
            problems.append(
                "T3: the page claims Rotten is its strongest class, but this run makes "
                f"{best} the most precise one"
            )
    return problems


def build_payload(
    config: Config,
    results_dir: Path,
    gallery_size: int,
) -> dict:
    """Fit every technique, predict the gallery, and assemble the page data."""
    metadata = json.loads((results_dir / "run_metadata.json").read_text(encoding="utf-8"))
    set_global_seed(config)

    records = load_primary(config)
    per_class = metadata["per_class"]
    if per_class != "all":
        records = balanced_subset(records, int(per_class), config)
    partition = stratified_split(records, config)
    augment = bool(metadata["augmentation"])

    extractors = available_extractors(config)
    techniques = [t for t in metadata["techniques"] if t in extractors]
    metrics = load_run_metrics(results_dir, techniques)

    gallery_records = choose_gallery(partition.test, gallery_size, config)
    gallery_paths = {str(r.path) for r in gallery_records}

    print(f"  fitting {len(techniques)} technique(s) on {len(partition.train)} training images")
    predictions: Dict[str, dict] = {}
    for short in techniques:
        extractor = extractors[short]
        print(f"    {short}: train ...", end="", flush=True)
        train = build_feature_matrix(
            partition.train, extractor, augment=augment, config=config
        )
        pipeline = build_pipeline(config)
        pipeline.fit(train.X, train.y)
        print(" test ...", end="", flush=True)
        test = build_feature_matrix(partition.test, extractor, augment=False, config=config)
        predicted = pipeline.predict(test.X)
        proba = pipeline.predict_proba(test.X)
        names = list(extractor.feature_names)
        per_image: Dict[str, dict] = {}
        for row, path in enumerate(test.paths):
            key = str(path)
            if key not in gallery_paths:
                continue
            per_image[key] = {
                "predicted": CLASS_ORDER[int(predicted[row])],
                "proba": [float(p) for p in proba[row]],
                "values": [
                    {
                        "label": label,
                        "value": resolve_feature(names, test.X[row], spec),
                        "note": note,
                    }
                    for label, spec, note in HEADLINE_FEATURES[short]
                ],
            }
        predictions[short] = per_image
        print(" done")

    site_images = Path("images")
    gallery: Dict[str, List[dict]] = {name: [] for name in CLASS_ORDER}
    print(f"  rendering {len(gallery_records)} gallery images")
    for record in gallery_records:
        sample = prepare_sample(record, config=config)
        stem = record.path.stem.replace(" ", "_").replace("(", "").replace(")", "")
        original = site_images / f"{stem}.jpg"
        overlay = site_images / f"{stem}_seg.jpg"
        key = str(record.path)
        entry = {
            "file": record.path.name,
            "original": original.as_posix(),
            "segmented": overlay.as_posix(),
            "coverage": round(float(np.count_nonzero(sample.mask)) / sample.mask.size * 100.0, 1),
            "truth": record.display_name,
            "by_technique": {
                short: predictions[short].get(key) for short in techniques
            },
        }
        if any(v is None for v in entry["by_technique"].values()):
            continue  # Segmentation failed for this image under some technique.
        gallery[record.display_name].append(entry)
        entry["_render"] = (sample.image, sample.mask, original, overlay)

    ranked = sorted(techniques, key=lambda t: metrics[t]["macro_f1"], reverse=True)
    ttest = None
    if len(ranked) >= 2:
        best, runner = metrics[ranked[0]]["folds"], metrics[ranked[1]]["folds"]
        if len(best) == len(runner) and len(best) > 1:
            statistic, pvalue = stats.ttest_rel(best, runner)
            ttest = {
                "t": float(statistic),
                "p": float(pvalue),
                "against": [ranked[0], ranked[1]],
                "diff": float(np.mean(best) - np.mean(runner)),
            }

    return {
        "ttest": ttest,
        "meta": {
            "mode": metadata["mode"],
            "n_images": metadata["n_images"],
            "n_train": metadata["n_train_images"],
            "n_test": metadata["n_test_images"],
            "augmentation": augment,
            "seed": metadata["seed"],
            "segmenter": metadata["segmentation_method"],
            "per_class": per_class,
        },
        "classes": list(CLASS_ORDER),
        "techniques": techniques,
        "labels": {
            "T1": "Dominant colour",
            "T2": "GLCM texture",
            "T3": "Morphology",
        },
        "descriptions": {
            "T1": "MPEG-7 dominant colour descriptor",
            "T2": "Grey-level co-occurrence texture",
            "T3": "Multiscale morphological descriptors",
        },
        "metrics": metrics,
        "justifications": JUSTIFICATIONS,
        "gallery": gallery,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Generate ``site/`` from a completed benchmark run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="pilot3", help="results/<tag> to read.")
    parser.add_argument("--gallery", type=int, default=8, help="Pictures offered per class.")
    parser.add_argument("--out", default="site", help="Directory to write the site into.")
    args = parser.parse_args(argv)

    config = get_config()
    results_dir = config.paths.results_subdir(args.tag)
    if not (results_dir / "benchmark_matrix.csv").exists():
        parser.error(
            f"no benchmark in {results_dir}. Run scripts/run_benchmarks.py --tag {args.tag} first."
        )

    print("=" * 74)
    print(f"Building site from results/{args.tag}")
    print("=" * 74)

    payload = build_payload(config, results_dir, args.gallery)

    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for entries in payload["gallery"].values():
        for entry in entries:
            image, mask, original, overlay = entry.pop("_render")
            write_jpeg(image, out / original)
            write_jpeg(contour_overlay(image, mask), out / overlay)

    problems = check_justifications(payload["metrics"])
    for problem in problems:
        print(f"  WARNING  {problem}")

    (out / "data.js").write_text(
        "window.FRESHSIGHT = " + json.dumps(payload, indent=1) + ";\n", encoding="utf-8"
    )
    template = Path(__file__).resolve().parent / "site_template.html"
    (out / "index.html").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")

    ranked = sorted(
        payload["techniques"], key=lambda t: payload["metrics"][t]["macro_f1"], reverse=True
    )
    print(f"\n  techniques : {', '.join(ranked)} (best first)")
    print(f"  pictures   : {sum(len(v) for v in payload['gallery'].values())}")
    print(f"  written to : {out}")
    print(f"\nServe it with:\n  python -m http.server 8000 --directory {args.out}")
    print("then open http://localhost:8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
