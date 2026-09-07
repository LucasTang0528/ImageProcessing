"""Render what each technique actually sees, for the results report.

The benchmark tables say which descriptor family scores best. They do not show
*why*, and on this study the why is visual: T1 reduces the peel to four
colours, T2 throws colour away and reads a 32-level grey surface, T3 ignores
both and measures geometry. Put side by side on the same apple, the three are
obviously looking at different evidence, which is the whole premise of a Mode A
comparison.

This script builds that strip for a few held-out apples and writes it to
``results/visuals/panels.json`` as base64 PNG/JPEG, so the report stays a
single self-contained file with no image directory beside it.

Run from the project root::

    python scripts/build_visuals.py                  # 1 apple per class
    python scripts/build_visuals.py --per-class 2

Images are drawn from the **test partition**, with the same seed and split
every experiment uses. Nothing here fits a model or produces a metric; the
feature values shown beside each panel are read straight out of the extractor
that the benchmark ran.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import Config, get_config  # noqa: E402
from data import ImageRecord, load_primary, read_image  # noqa: E402
from features.t1_dominant_colour import (  # noqa: E402
    LAB_AB_OFFSET,
    LAB_L_SCALE,
    DominantColourExtractor,
)
from features.t2_glcm import GLCMExtractor  # noqa: E402
from features.t3_morphological import T3MorphologicalExtractor  # noqa: E402
from harness import preprocess, segment_fruit, set_global_seed, stratified_split  # noqa: E402


def encode(image: np.ndarray, lossy: bool = True) -> str:
    """Return a ``data:`` URI for one BGR or greyscale image.

    Photographs go out as JPEG, which is roughly six times smaller than PNG at
    this size and indistinguishable at 224 px. Masks and quantised levels go
    out as PNG, because JPEG ringing on a hard edge would misrepresent exactly
    the thing the panel is there to show.
    """
    if lossy:
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
        mime = "image/jpeg"
    else:
        ok, buffer = cv2.imencode(".png", image)
        mime = "image/png"
    if not ok:
        raise RuntimeError("OpenCV could not encode a panel image")
    return f"data:{mime};base64,{base64.b64encode(buffer.tobytes()).decode('ascii')}"


def lab8_to_hex(centroid: Sequence[float]) -> str:
    """Convert one OpenCV 8-bit L*a*b* centroid to an sRGB hex string."""
    pixel = np.array([[list(centroid)]], dtype=np.uint8)
    bgr = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return "#{:02X}{:02X}{:02X}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))


def mask_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dim everything outside the fruit and trace the retained silhouette.

    Every descriptor is restricted to this region, so a reader who doubts a
    result should be able to see the region first.
    """
    inside = mask.astype(bool)
    out = image_bgr.copy()
    out[~inside] = (out[~inside] * 0.22 + 26).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (90, 220, 255), 2)
    return out


def blemish_overlay(image_bgr: np.ndarray, mask: np.ndarray, blemish: np.ndarray) -> np.ndarray:
    """Paint T3's detected blemish pixels over the fruit."""
    out = image_bgr.copy()
    out[~mask.astype(bool)] = (out[~mask.astype(bool)] * 0.22 + 26).astype(np.uint8)
    hit = blemish.astype(bool)
    if hit.any():
        tint = np.zeros_like(out)
        tint[hit] = (60, 60, 235)  # BGR red
        out = np.where(hit[..., None], (out * 0.35 + tint * 0.65).astype(np.uint8), out)
    return out


def quantised_panel(extractor: GLCMExtractor, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The exact array T2 builds its co-occurrence matrix from.

    ``quantise`` returns ``1..levels`` inside the fruit and ``0`` outside, the
    ignore level whose row and column are deleted before the matrix is
    normalised. Rescaling for display keeps that 0 at black, so the region
    excluded from every co-occurrence count is visible as such.
    """
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    boolean = mask.astype(bool)
    quantised = extractor.quantise(grey, boolean)
    return (quantised.astype(np.float32) * (255.0 / extractor.levels)).astype(np.uint8)


def describe_one(
    record: ImageRecord,
    config: Config,
    t1: DominantColourExtractor,
    t2: GLCMExtractor,
    t3: T3MorphologicalExtractor,
) -> Optional[Dict[str, Any]]:
    """Build every panel and headline figure for a single apple."""
    raw = read_image(record.path)
    resized = cv2.resize(raw, config.preprocess.resize, interpolation=cv2.INTER_AREA)
    image = preprocess(raw, config)
    segmentation = segment_fruit(image, config)
    if segmentation.failed:
        return None

    mask = segmentation.mask

    v1 = np.asarray(t1(image.copy(), mask.copy()), dtype=np.float64)
    v2 = np.asarray(t2(image.copy(), mask.copy()), dtype=np.float64)
    v3, aux = t3.extract_with_aux(image.copy(), mask.copy())

    names1 = list(t1.feature_names)
    names2 = list(t2.feature_names)
    names3 = list(t3.feature_names)

    def pick(names: List[str], vector: np.ndarray, key: str) -> float:
        return float(vector[names.index(key)])

    swatches = []
    for index in range(t1.n_colours):
        centroid = [pick(names1, v1, f"T1_c{index}_centroid_{ch}") for ch in ("L", "a", "b")]
        swatches.append({
            "hex": lab8_to_hex(centroid),
            "share": pick(names1, v1, f"T1_c{index}_share"),
            "coherency": pick(names1, v1, f"T1_c{index}_coherency"),
            "lab": [
                round(centroid[0] / LAB_L_SCALE, 1),
                round(centroid[1] - LAB_AB_OFFSET, 1),
                round(centroid[2] - LAB_AB_OFFSET, 1),
            ],
        })

    return {
        "class": record.display_name,
        "file": record.name,
        "coverage": float(segmentation.coverage),
        "panels": {
            "input": encode(resized),
            "preprocessed": encode(image),
            # The two overlays are photographs with a tint on top, so JPEG
            # costs nothing visible. The quantised panel is genuinely
            # posterised into 32 flat levels and stays PNG: JPEG would invent
            # gradients across exactly the flat regions the panel exists to
            # show, and its black ignore level has to stay pure black.
            "mask": encode(mask_overlay(image, mask)),
            "quantised": encode(quantised_panel(t2, image, mask), lossy=False),
            "blemish": encode(blemish_overlay(image, mask, aux["blemish_mask"])),
        },
        "t1": {
            "swatches": swatches,
            "green_share": pick(names1, v1, "T1_green_share"),
            "decay_share": pick(names1, v1, "T1_decay_share"),
            "mean_chroma": pick(names1, v1, "T1_weighted_mean_chroma"),
        },
        "t2": {
            "contrast": pick(names2, v2, "T2_contrast_d1_a0"),
            "homogeneity": pick(names2, v2, "T2_homogeneity_d1_a0"),
            "energy": pick(names2, v2, "T2_energy_d1_a0"),
            "correlation": pick(names2, v2, "T2_correlation_d1_a0"),
            "levels": t2.levels,
        },
        "t3": {
            "blemish_ratio": float(aux["blemish_ratio_pct"]),
            "blemish_count": int(aux["n_blemish"]),
            "peak_scale": pick(names3, v3, "gran_peak_scale"),
            "entropy": pick(names3, v3, "gran_entropy"),
            "mgrad_mean": pick(names3, v3, "mgrad_mean"),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Render the panels and write them to ``results/visuals/panels.json``."""
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-class", type=int, default=1,
                        help="How many held-out apples to render per class.")
    parser.add_argument("--out", default=None,
                        help="Destination JSON. Defaults to results/visuals/panels.json.")
    args = parser.parse_args(argv)

    set_global_seed(config)
    records = load_primary(config)
    partition = stratified_split(records, config)

    t1 = DominantColourExtractor.from_config(config)
    t2 = GLCMExtractor.from_config(config)
    t3 = T3MorphologicalExtractor.from_config(config)

    rng = np.random.default_rng(config.seed)
    chosen: List[ImageRecord] = []
    for label in range(config.primary.n_classes):
        pool = [r for r in partition.test if r.label == label]
        if not pool:
            continue
        order = rng.permutation(len(pool))
        chosen.extend(pool[int(i)] for i in order[: max(1, args.per_class) * 3])

    entries: List[Dict[str, Any]] = []
    per_class_done = {name: 0 for name in config.primary.display_names}
    for record in chosen:
        if per_class_done[record.display_name] >= args.per_class:
            continue
        print(f"  rendering {record.display_name:<7} {record.name} ...", flush=True)
        entry = describe_one(record, config, t1, t2, t3)
        if entry is None:
            print("    segmentation failed; skipping")
            continue
        entries.append(entry)
        per_class_done[record.display_name] += 1

    entries.sort(key=lambda e: list(config.primary.display_names).index(e["class"]))

    destination = Path(args.out) if args.out else (
        config.paths.results_root / "visuals" / "panels.json"
    )
    if not destination.is_absolute():
        destination = config.paths.project_root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"seed": config.seed, "source": "test partition", "images": entries}),
        encoding="utf-8",
    )

    size_kb = destination.stat().st_size / 1024
    print(f"\nWrote {destination} ({size_kb:.0f} KB, {len(entries)} apples)")
    print("Rebuild the report to include them: python scripts/build_report.py --tag phase5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
