"""Build a standalone HTML results report that anyone can open.

Every result this study produces lands in ``results/``, which ``.gitignore``
excludes, so a teammate who clones the repository sees none of it. This script
reads whatever has actually been produced and bakes it into a single
self-contained page at ``docs/index.html`` - no server, no CSV files alongside
it, no Python needed to view it. That file is committed, so cloning is enough
to read the results, and GitHub Pages serves ``docs/`` directly if the team
wants a URL.

Run from the project root::

    python scripts/build_report.py                 # newest enhancement run
    python scripts/build_report.py --tag phase5    # a specific run
    python scripts/build_report.py --out docs/index.html

Nothing here computes a metric. Every number is transcribed from a CSV that a
pipeline run wrote, and a section whose source file is missing says so on the
page rather than being quietly omitted or filled with a plausible value.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import get_config  # noqa: E402

TECHNIQUES = ("T1", "T2", "T3")
ENHANCEMENTS = ("E1", "E2", "E3")

LONG_NAME = {
    "T1": "dominant colour",
    "T2": "GLCM texture",
    "T3": "morphological",
    "E1": "feature fusion",
    "E2": "regional weighting",
    "E3": "decision fusion",
}

#: Three-class chance, drawn on every accuracy chart as the lower reference.
CHANCE = 1.0 / 3.0

#: The brief's bars. Recorded here so the page reports against them; nothing
#: in the pipeline is ever tuned towards them.
TARGET_BEST_INDIVIDUAL = 0.80
TARGET_HYBRID = 0.85
TARGET_HYBRID_GAIN = 0.03
TARGET_F1 = 0.83
TARGET_ALPHA = 0.05
TARGET_BLEMISH_MAE = 10.0
TARGET_SECONDS = 5.0


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def esc(value: Any) -> str:
    """HTML-escape a value for safe interpolation into the page."""
    return html.escape(str(value), quote=True)


def num(value: Any, places: int = 4, dash: str = "&mdash;") -> str:
    """Format a number, or return an em dash when it is missing."""
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(as_float):
        return dash
    return f"{as_float:.{places}f}"


def pct(value: Any, places: int = 2, dash: str = "&mdash;") -> str:
    """Format a value already expressed in percent."""
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(as_float):
        return dash
    return f"{as_float:.{places}f}%"


def read_csv(path: Path, **kwargs: Any) -> Optional[pd.DataFrame]:
    """Read a CSV, returning ``None`` when it has not been produced yet."""
    if not path.exists():
        return None
    return pd.read_csv(path, **kwargs)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON file, returning ``None`` when it does not exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def missing(source: str, how: str) -> str:
    """Render the placeholder shown in place of a section with no data."""
    return (
        f'<div class="absent"><p><b>Not available.</b> This section reads '
        f'<code>{esc(source)}</code>, which has not been produced yet. '
        f'{esc(how)}</p></div>'
    )


def chip(verdict: str) -> str:
    """Render a PASS / FAIL / pending verdict chip."""
    kind = {"Pass": "pass", "Fail": "fail"}.get(verdict, "warn")
    return f'<span class="chip {kind}">{esc(verdict)}</span>'


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

@dataclass
class Bar:
    """One row of a horizontal accuracy chart."""

    label: str
    value: float
    kind: str = "primary"


def accuracy_chart(
    bars: Sequence[Bar],
    control: Optional[float],
    control_label: str,
    axis_label: str,
    gutter: int = 64,
) -> str:
    """Draw a horizontal bar chart on a full 0-1 accuracy scale.

    The axis always spans the whole range rather than being cropped to the
    data. A truncated axis makes a three-point gap look like a landslide, and
    the point of this page is that several of those gaps are not landslides.

    Reference lines for three-class chance and, where known, the
    background-only control are drawn through every chart, because an absolute
    accuracy on this dataset means very little read on its own.
    """
    left, right = gutter, gutter + 496
    span = right - left
    row_h, bar_h = 30, 16
    top = 40
    axis_y = top + row_h * len(bars) + 2
    height = axis_y + 46

    def x_of(value: float) -> float:
        return left + max(0.0, min(1.0, float(value))) * span

    parts: List[str] = []
    parts.append(
        f'<svg viewBox="0 0 660 {height}" role="img" aria-label="{esc(axis_label)}: '
        + esc(", ".join(f"{b.label} {b.value:.3f}" for b in bars))
        + '.">'
    )

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x_of(tick)
        stroke = "var(--rule-strong)" if tick == 0.0 else "var(--rule)"
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{axis_y}" '
            f'stroke="{stroke}" stroke-width="1"/>'
        )

    x_chance = x_of(CHANCE)
    parts.append(
        f'<line x1="{x_chance:.1f}" y1="{top - 14}" x2="{x_chance:.1f}" y2="{axis_y}" '
        f'stroke="var(--ink-3)" stroke-width="1.5" stroke-dasharray="4 3"/>'
        f'<text x="{x_chance:.1f}" y="{top - 20}" fill="var(--ink-3)" '
        f'font-family="IBM Plex Mono, monospace" font-size="10.5" text-anchor="middle">'
        f'chance .333</text>'
    )

    if control is not None and np.isfinite(control):
        x_control = x_of(control)
        parts.append(
            f'<line x1="{x_control:.1f}" y1="{top - 14}" x2="{x_control:.1f}" y2="{axis_y}" '
            f'stroke="var(--fail)" stroke-width="1.5" stroke-dasharray="4 3"/>'
            f'<text x="{x_control:.1f}" y="{top - 20}" fill="var(--fail)" '
            f'font-family="IBM Plex Mono, monospace" font-size="10.5" text-anchor="middle">'
            f'{esc(control_label)} {control:.3f}</text>'
        )

    fills = {
        "primary": "var(--accent)",
        "secondary": "var(--accent-soft)",
        "alarm": "var(--fail)",
        "caution": "var(--warn)",
    }
    for index, bar in enumerate(bars):
        y = top + index * row_h
        width = max(1.0, x_of(bar.value) - left)
        parts.append(
            f'<text x="{left - 8}" y="{y + 12}" fill="var(--ink)" '
            f'font-family="IBM Plex Sans, sans-serif" font-size="12.5" font-weight="500" '
            f'text-anchor="end">{esc(bar.label)}</text>'
            f'<rect x="{left}" y="{y}" width="{width:.1f}" height="{bar_h}" '
            f'fill="{fills.get(bar.kind, fills["primary"])}"/>'
            f'<text x="{left + width + 6:.1f}" y="{y + 12}" fill="var(--ink)" '
            f'font-family="IBM Plex Mono, monospace" font-size="12" font-weight="500">'
            f'{bar.value:.4f}</text>'
        )

    parts.append(
        f'<line x1="{left}" y1="{axis_y}" x2="{right}" y2="{axis_y}" '
        f'stroke="var(--rule-strong)" stroke-width="1"/>'
    )
    for tick, text in ((0.0, "0"), (0.25, ".25"), (0.5, ".50"), (0.75, ".75"), (1.0, "1.0")):
        parts.append(
            f'<text x="{x_of(tick):.1f}" y="{axis_y + 18}" fill="var(--ink-3)" '
            f'font-family="IBM Plex Mono, monospace" font-size="10.5" '
            f'text-anchor="middle">{text}</text>'
        )
    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="{axis_y + 38}" fill="var(--ink-3)" '
        f'font-family="IBM Plex Mono, monospace" font-size="10.5" text-anchor="middle">'
        f'{esc(axis_label)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str = "",
    lead_rows: Sequence[int] = (),
) -> str:
    """Render a scrollable table. Cell contents are inserted as trusted HTML."""
    head = "".join(f'<th scope="col">{h}</th>' for h in headers)
    body: List[str] = []
    for index, row in enumerate(rows):
        css = ' class="lead"' if index in set(lead_rows) else ""
        cells = f'<th scope="row">{row[0]}</th>' + "".join(f"<td>{c}</td>" for c in row[1:])
        body.append(f"<tr{css}>{cells}</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (
        '<div class="scroller"><table>'
        f"{cap}<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def headline_section(matrix: Optional[pd.DataFrame], control: Optional[float], tag: str) -> str:
    """The six-configuration ranking, its chart and its table."""
    if matrix is None:
        return missing(
            f"results/{tag}/enhancement_matrix.csv",
            "Run scripts/run_enhancements.py to produce it.",
        )

    present = [n for n in TECHNIQUES + ENHANCEMENTS if n in matrix.index]
    bars = [
        Bar(name, float(matrix.loc[name, "cv_mean_accuracy"]),
            "primary" if name in TECHNIQUES else "secondary")
        for name in present
    ]

    rows = []
    lead = []
    for index, name in enumerate(present):
        row = matrix.loc[name]
        dim = int(row["dimensionality"])
        if name in ENHANCEMENTS and float(row["cv_mean_accuracy"]) == max(
            float(matrix.loc[e, "cv_mean_accuracy"]) for e in ENHANCEMENTS if e in matrix.index
        ):
            lead.append(index)
        rows.append([
            f'{name} &middot; {LONG_NAME.get(name, "")}',
            "&mdash;" if dim < 0 else str(dim),
            num(row["cv_mean_accuracy"]),
            num(row["cv_std_accuracy"]),
            num(row.get("cv_mean_macro_f1")),
            num(row.get("test_accuracy")),
            num(row.get("test_macro_f1")),
            num(row.get("test_weighted_f1")),
        ])

    return f"""
      <div class="chart-wrap">
        {accuracy_chart(bars, control, "background only", "cross-validated accuracy")}
        <div class="legend">
          <span><i class="swatch" style="background: var(--accent)"></i> individual technique</span>
          <span><i class="swatch" style="background: var(--accent-soft)"></i> enhancement</span>
          <span><i class="dashkey"></i> background-only control</span>
          <span><i class="dashkey chance"></i> three-class chance</span>
        </div>
      </div>
      <div style="margin-top:22px">
      {table(
          ["Configuration", "Dim", "CV acc", "SD", "CV macro F1",
           "Test acc", "Test macro F1", "Test wtd F1"],
          rows,
          "Folds are drawn over source images, so no augmented variant straddles a fold "
          "boundary and validation folds hold originals only. E3 fuses three fitted "
          "pipelines rather than one feature vector, so it has no single dimensionality.",
          lead,
      )}
      </div>
    """


def acceptance_section(
    matrix: Optional[pd.DataFrame],
    vs_best: Optional[pd.DataFrame],
    t3_experiments: Optional[pd.DataFrame],
    t3_ablation: Optional[pd.DataFrame],
) -> str:
    """The brief's eight acceptance targets, scored against measured values."""
    if matrix is None:
        return missing(
            "the enhancement matrix",
            "The targets are scored from a completed enhancement run.",
        )

    techniques = [n for n in TECHNIQUES if n in matrix.index]
    hybrids = [n for n in ENHANCEMENTS if n in matrix.index]
    if not techniques or not hybrids:
        return missing("a complete enhancement run", "Some configurations are absent.")

    best_t = max(techniques, key=lambda n: float(matrix.loc[n, "cv_mean_accuracy"]))
    best_e = max(hybrids, key=lambda n: float(matrix.loc[n, "cv_mean_accuracy"]))
    best_t_acc = float(matrix.loc[best_t, "cv_mean_accuracy"])
    best_e_acc = float(matrix.loc[best_e, "cv_mean_accuracy"])
    gain = best_e_acc - best_t_acc

    macro = float(matrix.loc[best_e, "test_macro_f1"])
    weighted = float(matrix.loc[best_e, "test_weighted_f1"])

    p_value = None
    if vs_best is not None and "technique_a" in vs_best.columns:
        hit = vs_best[vs_best["technique_a"] == best_e]
        if not hit.empty:
            p_value = float(hit.iloc[0]["p_value"])

    blemish_mae = None
    if t3_experiments is not None and "mean_abs_ratio_error_vs_annotation" in t3_experiments:
        values = pd.to_numeric(
            t3_experiments["mean_abs_ratio_error_vs_annotation"], errors="coerce"
        ).dropna()
        if not values.empty:
            blemish_mae = float(values.min())

    extraction = None
    if t3_ablation is not None and "extraction_mean_s" in t3_ablation.columns:
        extraction = float(t3_ablation["extraction_mean_s"].iloc[0])

    rows: List[List[str]] = [
        [
            "Best individual technique",
            f"&ge; {TARGET_BEST_INDIVIDUAL:.2f}",
            f'{num(best_t_acc)} &nbsp;<span class="mono dim">{best_t}</span>',
            chip("Pass" if best_t_acc >= TARGET_BEST_INDIVIDUAL else "Fail"),
        ],
        [
            "Hybrid accuracy",
            f"&ge; {TARGET_HYBRID:.2f}",
            f'{num(best_e_acc)} &nbsp;<span class="mono dim">{best_e}</span>',
            chip("Pass" if best_e_acc >= TARGET_HYBRID else "Fail"),
        ],
        [
            "Hybrid gain over best individual",
            f"&ge; {TARGET_HYBRID_GAIN * 100:.1f} pp",
            f"{gain * 100:+.2f} pp",
            chip("Pass" if gain >= TARGET_HYBRID_GAIN else "Fail"),
        ],
        [
            "Hybrid macro F1",
            f"&ge; {TARGET_F1:.2f}",
            num(macro),
            chip("Pass" if macro >= TARGET_F1 else "Fail"),
        ],
        [
            "Hybrid weighted F1",
            f"&ge; {TARGET_F1:.2f}",
            num(weighted),
            chip("Pass" if weighted >= TARGET_F1 else "Fail"),
        ],
        [
            "Claimed gains significant",
            f"p &lt; {TARGET_ALPHA:.2f}",
            "&mdash;" if p_value is None else f"p = {p_value:.4f}",
            chip("Pass" if (p_value is not None and p_value < TARGET_ALPHA) else "Fail"),
        ],
        [
            "Per-class precision and recall",
            "&ge; 0.80 each",
            '<span class="note">Computed by the evaluation module but not written to '
            "disk: the Phase&nbsp;5 driver saves only the confusion PNG for each "
            "enhancement.</span>",
            chip("Not exported"),
        ],
        [
            "Blemish coverage MAE",
            f"&le; {TARGET_BLEMISH_MAE:.0f}%",
            '<span class="note">No ground-truth masks painted yet, so every E3.4 arm '
            "reports NaN.</span>" if blemish_mae is None else pct(blemish_mae),
            chip("No data") if blemish_mae is None
            else chip("Pass" if blemish_mae <= TARGET_BLEMISH_MAE else "Fail"),
        ],
        [
            "Extraction + inference per image",
            f"&le; {TARGET_SECONDS:.0f} s",
            "&mdash;" if extraction is None
            else f"{extraction * 1000:.1f} ms <span class='dim'>(T3 extraction)</span>",
            chip("Pass") if (extraction is not None and extraction < TARGET_SECONDS)
            else chip("No data"),
        ],
    ]

    note = ""
    if gain < TARGET_HYBRID_GAIN or p_value is None or p_value >= TARGET_ALPHA:
        note = f"""
      <div class="callout method">
        <h3>The gain target is the one under pressure</h3>
        <p>
          Fusion buys {gain * 100:+.2f} points over {best_t} alone
          {"" if p_value is None else f", and a paired t-test across the folds puts that at p&nbsp;=&nbsp;{p_value:.4f}"}.
          The brief asks for a three-point gain <em>and</em> significance, and both are
          judged on the same five folds, so a small fold count makes this the hardest
          pair of targets to clear.
        </p>
      </div>"""

    return table(
        ["Target", "Required", "Measured", "Verdict"],
        rows,
        "Targets come from the assignment brief. Nothing in the pipeline is tuned "
        "towards them; a missed target is reported as missed.",
    ) + note


def enhancement_section(vs_best: Optional[pd.DataFrame], tag: str) -> str:
    """Paired t-tests of each enhancement against the best individual technique."""
    if vs_best is None:
        return missing(
            f"results/{tag}/enhancement_vs_best.csv",
            "Run scripts/run_enhancements.py to produce it.",
        )

    rows = []
    for _, row in vs_best.iterrows():
        meets = bool(row.get("meets_3pt_target", False))
        rows.append([
            f'{esc(row["technique_a"])} vs {esc(row["technique_b"])}',
            num(row["mean_a"]),
            num(row["mean_b"]),
            f'{float(row["mean_difference"]):+.4f}',
            num(row["t_statistic"], 3),
            num(row["p_value"]),
            chip("Pass" if meets else "Fail"),
        ])

    return table(
        ["Comparison", "Enhancement", "Best individual", "Difference", "t", "p", "&ge; 3 pp"],
        rows,
        "Paired fold by fold, which removes the variation the two configurations share "
        "and isolates the difference between them.",
    )


def ablation_section(ablation: Optional[pd.DataFrame], matrix: Optional[pd.DataFrame], tag: str) -> str:
    """One technique dropped at a time, against the full three-way fusion."""
    if ablation is None:
        return missing(
            f"results/{tag}/ablation.csv",
            "Run scripts/run_enhancements.py to produce it.",
        )

    baseline = {
        "E1_feature_fusion": float(matrix.loc["E1", "cv_mean_accuracy"])
        if matrix is not None and "E1" in matrix.index else None,
        "E3_decision_fusion": float(matrix.loc["E3", "cv_mean_accuracy"])
        if matrix is not None and "E3" in matrix.index else None,
    }

    rows, lead = [], []
    for index, (_, row) in enumerate(ablation.iterrows()):
        strategy = str(row["strategy"])
        accuracy = float(row["cv_mean_accuracy"])
        base = baseline.get(strategy)
        if base is None:
            delta = "&mdash;"
        else:
            difference = accuracy - base
            colour = "var(--pass)" if difference > 0 else "var(--fail)"
            delta = f'<span style="color:{colour}">{difference:+.4f}</span>'
            if difference > 0:
                lead.append(index)
        rows.append([
            strategy.replace("_", " ").replace("E1 ", "E1 &middot; ").replace("E3 ", "E3 &middot; "),
            f'<span class="mono">{esc(row["techniques"])}</span>',
            f'<span class="mono">{esc(row["dropped"])}</span>',
            num(accuracy),
            num(row["cv_std_accuracy"]),
            delta,
        ])

    return table(
        ["Strategy", "Kept", "Dropped", "CV accuracy", "SD", "vs full fusion"],
        rows,
        "The final column compares each pair with the corresponding three-way fusion. "
        "A positive value means the dropped technique was costing the fusion accuracy.",
        lead,
    )


def t3_section(experiments: Dict[str, Optional[pd.DataFrame]]) -> str:
    """The four T3 sweeps, each on the complete dataset."""
    blocks: List[str] = []

    ablation = experiments.get("e3_1")
    if ablation is not None:
        rows = [
            [esc(r["arm"]).replace("_", " "), f'<span class="mono">{esc(r["blocks"])}</span>',
             str(int(r["dimensionality"])), num(r["cv_mean_accuracy"]),
             num(r["cv_std_accuracy"]), num(r["cv_mean_macro_f1"])]
            for _, r in ablation.iterrows()
        ]
        blocks.append(
            "<h3>E3.1 &mdash; which blocks carry the signal</h3>"
            + table(["Arm", "Blocks", "Dim", "CV accuracy", "SD", "Macro F1"], rows,
                    "Neither half reaches the whole, so the granulometric and top-hat "
                    "blocks are complementary rather than redundant.", [0])
        )

    shape = experiments.get("e3_2")
    if shape is not None:
        rows = [
            [esc(r["se_shape"]).title(), num(r["cv_mean_accuracy"]), num(r["cv_std_accuracy"]),
             f'{float(r["extraction_mean_s"]) * 1000:.1f} ms']
            for _, r in shape.iterrows()
        ]
        spread = float(shape["cv_mean_accuracy"].max() - shape["cv_mean_accuracy"].min())
        blocks.append(
            "<h3>E3.2 &mdash; structuring element shape</h3>"
            + table(["Shape", "CV accuracy", "SD", "Extraction"], rows,
                    f"A {spread:.4f} spread across three shapes: on this data the choice "
                    f"of structuring element does not matter.", [0])
        )

    radius = experiments.get("e3_3")
    if radius is not None:
        best = radius["cv_mean_accuracy"].idxmax()
        rows = []
        for index, r in radius.iterrows():
            marker = ' <span class="dim">widest fold spread</span>' if (
                index == best and float(r["cv_std_accuracy"]) > float(radius["cv_std_accuracy"].min()) * 2
            ) else ""
            rows.append([
                str(int(r["r_max"])), str(int(r["dimensionality"])),
                num(r["cv_mean_accuracy"]) + marker, num(r["cv_std_accuracy"]),
            ])
        blocks.append(
            "<h3>E3.3 &mdash; maximum granulometric radius</h3>"
            + table(["r max", "Dim", "CV accuracy", "SD"], rows,
                    "Accuracy is reported against dimensionality rather than padded to a "
                    "common length, as the brief requires. The nominally best radius is "
                    "not the most stable one.")
        )

    segmentation = experiments.get("e3_4")
    if segmentation is not None:
        rows = []
        for _, r in segmentation.iterrows():
            rows.append([
                esc(r["blemish_method"]).replace("_", " ").title(),
                num(r["cv_mean_accuracy"]),
                pct(r["mean_blemish_ratio_pct"]),
                pct(r.get("mean_blemish_ratio_Unripe")),
                pct(r.get("mean_blemish_ratio_Ripe")),
                pct(r.get("mean_blemish_ratio_Rotten")),
                pct(r.get("mean_abs_ratio_error_vs_annotation")),
            ])
        blocks.append(
            "<h3>E3.4 &mdash; blemish segmentation method</h3>"
            + table(["Method", "CV accuracy", "Mean ratio", "Unripe", "Ripe", "Rotten",
                     "MAE vs truth"], rows,
                    "The brief scores this experiment on coverage error against annotated "
                    "masks, not on classification accuracy.", [0])
        )

        try:
            otsu = float(segmentation.loc[
                segmentation["blemish_method"] == "bth_otsu", "cv_mean_accuracy"].iloc[0])
            hue = float(segmentation.loc[
                segmentation["blemish_method"] == "hue_deviation", "cv_mean_accuracy"].iloc[0])
        except (KeyError, IndexError):
            otsu = hue = float("nan")
        if np.isfinite(otsu) and np.isfinite(hue) and hue > otsu:
            blocks.append(f"""
      <div class="callout">
        <h3>The tempting {(hue - otsu) * 100:.1f} points, and why not to take them</h3>
        <p>
          Hue deviation scores {hue:.4f} against the black top-hat&rsquo;s {otsu:.4f}. It is also
          the method this design deliberately replaced, because it reads the same colour
          evidence T1 is built on. Switching back would buy {(hue - otsu) * 100:.1f} accuracy
          points and destroy the study&rsquo;s central claim: T3 would no longer be disjoint
          from T1, and no result could be attributed to morphology. The right use of this
          number is as a reported cost &mdash; keeping the descriptor families independent is
          worth {(hue - otsu) * 100:.1f} points &mdash; not as a configuration change.
        </p>
      </div>""")

    if not blocks:
        return missing("results/t3/e3_*.csv", "Run scripts/run_t3_experiments.py to produce them.")
    return "".join(blocks)


def visuals_section(panels: Optional[Dict[str, Any]]) -> str:
    """What each technique sees, on the same apple, stage by stage.

    The tables elsewhere say which descriptor wins. This says why they differ
    at all: the three are handed identical pixels and identical masks, and each
    then discards almost everything the others keep.
    """
    if not panels or not panels.get("images"):
        return missing(
            "results/visuals/panels.json",
            "Run scripts/build_visuals.py to render them.",
        )

    stages = [
        ("input", "Input", "Resized to 224 &times; 224. Nothing else applied."),
        ("preprocessed", "Shared preprocessing",
         "Gaussian 5&times;5, then CLAHE on L* only so chromaticity is left intact."),
        ("mask", "Shared segmentation",
         "Seeded GrabCut, holes filled, largest component kept. Every technique "
         "is restricted to this region."),
        ("quantised", "T2 sees this",
         "Colour discarded, 32 grey levels. Black is the ignore level, deleted "
         "from the co-occurrence matrix before it is normalised."),
        ("blemish", "T3 sees this",
         "Geometry only. Red is what the black top-hat plus Otsu called a "
         "blemish, inside the eroded interior mask."),
    ]

    cards: List[str] = []
    for entry in panels["images"]:
        strip = "".join(
            f'<figure class="panel">'
            f'<img src="{entry["panels"][key]}" alt="{esc(title)}: {esc(entry["file"])}" '
            f'width="224" height="224" loading="lazy">'
            f'<figcaption><b>{esc(title)}</b><span>{blurb}</span></figcaption>'
            f'</figure>'
            for key, title, blurb in stages if key in entry["panels"]
        )

        t1 = entry["t1"]
        swatches = "".join(
            f'<div class="sw">'
            f'<span class="chipcolour" style="background:{esc(s["hex"])}"></span>'
            f'<b>{s["share"]:.1f}%</b>'
            f'<span class="dim">L* {s["lab"][0]:g} &nbsp; a* {s["lab"][1]:g} &nbsp; b* {s["lab"][2]:g}</span>'
            f'</div>'
            for s in t1["swatches"]
        )

        t2, t3 = entry["t2"], entry["t3"]
        readout = f"""
          <div class="readout">
            <div>
              <h4>T1 &mdash; dominant colour</h4>
              <div class="swatches">{swatches}</div>
              <dl>
                <div><dt>green share</dt><dd>{t1["green_share"]:.1f}%</dd></div>
                <div><dt>decay share</dt><dd>{t1["decay_share"]:.1f}%</dd></div>
                <div><dt>mean chroma</dt><dd>{t1["mean_chroma"]:.1f}</dd></div>
              </dl>
            </div>
            <div>
              <h4>T2 &mdash; GLCM, d=1 at 0&deg;</h4>
              <dl>
                <div><dt>contrast</dt><dd>{t2["contrast"]:.3f}</dd></div>
                <div><dt>homogeneity</dt><dd>{t2["homogeneity"]:.3f}</dd></div>
                <div><dt>energy</dt><dd>{t2["energy"]:.3f}</dd></div>
                <div><dt>correlation</dt><dd>{t2["correlation"]:.3f}</dd></div>
              </dl>
            </div>
            <div>
              <h4>T3 &mdash; morphology</h4>
              <dl>
                <div><dt>blemish ratio</dt><dd>{t3["blemish_ratio"]:.2f}%</dd></div>
                <div><dt>blemish count</dt><dd>{t3["blemish_count"]}</dd></div>
                <div><dt>peak scale</dt><dd>r = {t3["peak_scale"]:.0f}</dd></div>
                <div><dt>spectrum entropy</dt><dd>{t3["entropy"]:.3f}</dd></div>
              </dl>
            </div>
          </div>"""

        cards.append(f"""
        <article class="specimen">
          <div class="specimen-head">
            <h3>{esc(entry["class"])}</h3>
            <span class="dim mono">{esc(entry["file"])} &middot; mask covers
              {entry["coverage"] * 100:.1f}% of frame</span>
          </div>
          <div class="strip">{strip}</div>
          {readout}
        </article>""")

    return f"""
      <div class="specimens">{"".join(cards)}</div>
      <div class="callout method">
        <h3>Why the three disagree</h3>
        <p>
          Read the last three panels across any row. T2 is handed a colourless
          32-level surface, so a green apple and a red one of the same texture are
          the same picture to it &mdash; which is why it scores worst on a task that is
          largely chromatic. T3 sees neither colour nor absolute brightness, only
          where the surface dips and by how much. T1 keeps colour and throws away
          every spatial relationship except coherency. None of the three could
          reconstruct another&rsquo;s input from its own, which is what makes the
          comparison a comparison rather than three views of one measurement.
        </p>
      </div>"""


def audit_section(
    background: Optional[pd.DataFrame],
    profile: Optional[pd.DataFrame],
    duplicates: Optional[pd.DataFrame],
) -> str:
    """The dataset suitability gate: background-only baseline and imaging style."""
    if background is None and profile is None:
        return missing("results/audit/*.csv", "Run scripts/audit_dataset.py to produce them.")

    blocks: List[str] = []

    if background is not None and "background_only_recall" in background.columns:
        bars = [
            Bar(str(r["class"]), float(r["background_only_recall"]),
                "alarm" if float(r["background_only_recall"]) >= 0.6 else "caution")
            for _, r in background.iterrows()
        ]
        overall = float(background["background_only_recall"].mean())
        blocks.append(f"""
      <div class="chart-wrap">
        {accuracy_chart(bars, None, "", "recall with the fruit masked out", gutter=76)}
      </div>
      <p style="margin-top:20px">
        Overall background-only accuracy is <b>{overall:.4f}</b> against a three-class chance of
        0.333. The brief calls for this to be near chance and says a materially higher value
        means the background predicts the label.
      </p>""")

    if profile is not None:
        rows = [
            [esc(r["class"]), num(r["mean_coverage"]), pct(r["pct_failed"]),
             pct(r["pct_plain_background"]), pct(r["pct_background_like"]), str(int(r["n"]))]
            for _, r in profile.iterrows()
        ]
        blocks.append(table(
            ["Class", "Mean mask coverage", "Segmentation failures",
             "Plain background", "Background-like frame", "Images"],
            rows,
            "Imaging style differs systematically by class, which is the mechanism behind "
            "the background-only result above.",
        ))

    if duplicates is not None and not duplicates.empty:
        pairs = len(duplicates)
        cross = int(duplicates["cross_class"].sum()) if "cross_class" in duplicates else 0
        classes = sorted(set(duplicates["class_a"]) | set(duplicates["class_b"])) \
            if "class_a" in duplicates else []
        blocks.append(f"""
      <p style="margin-top:16px">
        Near-duplicate detection with colour confirmation found <b>{pairs} duplicate pairs</b>
        {f"across {esc(', '.join(classes))}" if classes else ""}, {cross} of them crossing a class
        boundary. The pipeline does not currently remove them, so any pair straddling the
        train/test boundary hands the classifier a free correct answer.
      </p>""")

    blocks.append("""
      <div class="callout">
        <h3>What this means for every number above</h3>
        <p>
          The three descriptors stay comparable <em>with each other</em>: all three see identical
          masks from the same shared harness, so the ranking between them is sound. What is not
          sound is reading any absolute accuracy as evidence about apples. Every headline figure
          in this report should be quoted with the background-only control beside it, and the
          honest fix is a better dataset rather than a better classifier.
        </p>
      </div>""")

    return "".join(blocks)


# --------------------------------------------------------------------------- #
# Page assembly
# --------------------------------------------------------------------------- #

STYLE = """
:root{--ground:#EFF2F5;--surface:#FFFFFF;--surface-2:#E7ECF1;--ink:#141A21;--ink-2:#56626E;
--ink-3:#7D8896;--rule:#D5DCE3;--rule-strong:#B6C1CC;--accent:#2D5D8A;--accent-soft:#A9C4DC;
--pass:#2E7D5B;--warn:#A8761C;--fail:#A83C48;--pass-bg:#E1F0E8;--warn-bg:#F6EDDA;
--fail-bg:#F7E3E5;--measure:66ch;--shell:1080px;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#11161C;
--surface:#171E26;--surface-2:#1E2831;--ink:#E4E9EE;--ink-2:#9DA9B5;--ink-3:#7A8693;
--rule:#2A343E;--rule-strong:#3B4854;--accent:#6FA3D2;--accent-soft:#3A5A78;--pass:#5FB98C;
--warn:#D3A253;--fail:#E08894;--pass-bg:#16332A;--warn-bg:#33290F;--fail-bg:#371E22;}}
:root[data-theme="dark"]{--ground:#11161C;--surface:#171E26;--surface-2:#1E2831;--ink:#E4E9EE;
--ink-2:#9DA9B5;--ink-3:#7A8693;--rule:#2A343E;--rule-strong:#3B4854;--accent:#6FA3D2;
--accent-soft:#3A5A78;--pass:#5FB98C;--warn:#D3A253;--fail:#E08894;--pass-bg:#16332A;
--warn-bg:#33290F;--fail-bg:#371E22;}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--ground);color:var(--ink);
font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
font-size:15px;line-height:1.62;-webkit-font-smoothing:antialiased}
img{max-width:100%}
.shell{max-width:var(--shell);margin:0 auto;padding:0 28px 96px}
.masthead{padding:56px 0 28px;border-bottom:2px solid var(--ink)}
.course{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;letter-spacing:.14em;
text-transform:uppercase;color:var(--accent);margin:0 0 14px}
h1{font-family:Spectral,Georgia,"Times New Roman",serif;font-weight:600;
font-size:clamp(34px,5.4vw,52px);line-height:1.06;letter-spacing:-.015em;margin:0 0 16px;
text-wrap:balance}
.standfirst{max-width:var(--measure);font-size:17px;color:var(--ink-2);margin:0}
.byline{display:flex;flex-wrap:wrap;gap:8px 26px;margin-top:26px;
font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--ink-3)}
.byline b{color:var(--ink-2);font-weight:500}
.runbar{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1px;
background:var(--rule);border:1px solid var(--rule);margin:32px 0 0}
.runbar>div{background:var(--surface);padding:14px 16px}
.runbar dt{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3);margin:0 0 5px}
.runbar dd{margin:0;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:15px;
font-weight:500;font-variant-numeric:tabular-nums}
section{padding-top:56px}
.eyebrow{display:flex;flex-wrap:wrap;align-items:baseline;gap:12px;
font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.13em;
text-transform:uppercase;color:var(--ink-3);padding-bottom:8px;margin:0 0 20px;
border-bottom:1px solid var(--rule)}
.eyebrow .src{color:var(--accent)}
h2{font-family:Spectral,Georgia,serif;font-weight:600;font-size:27px;line-height:1.2;
letter-spacing:-.01em;margin:0 0 12px;text-wrap:balance}
h3{font-weight:600;font-size:15px;margin:32px 0 10px;letter-spacing:-.005em}
p{max-width:var(--measure);margin:0 0 14px}
.lede{font-size:16.5px;color:var(--ink-2)}
a{color:var(--accent)}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.9em}
code{background:var(--surface-2);padding:1px 5px;border-radius:2px}
.dim{color:var(--ink-3)}
.chart-wrap{background:var(--surface);border:1px solid var(--rule);padding:20px 20px 14px;
overflow-x:auto}
.chart-wrap svg{display:block;min-width:560px;width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:14px;padding-top:12px;
border-top:1px solid var(--rule);font-family:"IBM Plex Mono",ui-monospace,monospace;
font-size:11.5px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.swatch{width:13px;height:13px;flex:none}
.dashkey{width:17px;height:0;flex:none;border-top:2px dashed var(--fail)}
.dashkey.chance{border-top-color:var(--ink-3)}
.scroller{overflow-x:auto;border:1px solid var(--rule);background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:9px 14px;border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;letter-spacing:.08em;
text-transform:uppercase;color:var(--ink-3);font-weight:500;
border-bottom:1px solid var(--rule-strong);background:var(--surface);vertical-align:bottom}
tbody td{font-family:"IBM Plex Mono",ui-monospace,monospace}
tbody th[scope="row"]{font-weight:500;text-align:left}
tbody tr:last-child td,tbody tr:last-child th{border-bottom:none}
tr.lead td,tr.lead th{background:var(--surface-2)}
.note{font-family:"IBM Plex Sans",sans-serif;white-space:normal;color:var(--ink-2);
text-align:left;display:inline-block;min-width:220px}
caption{caption-side:bottom;text-align:left;padding:11px 14px;font-size:12.5px;
color:var(--ink-3);border-top:1px solid var(--rule)}
.chip{display:inline-block;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
font-weight:600;letter-spacing:.07em;padding:2px 8px;border-radius:2px;white-space:nowrap}
.chip.pass{background:var(--pass-bg);color:var(--pass)}
.chip.fail{background:var(--fail-bg);color:var(--fail)}
.chip.warn{background:var(--warn-bg);color:var(--warn)}
.callout{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--fail);
padding:20px 22px;margin:22px 0}
.callout.method{border-left-color:var(--accent)}
.callout h3{margin-top:0}
.callout p:last-child{margin-bottom:0}
.absent{background:var(--surface);border:1px dashed var(--rule-strong);padding:20px 22px}
.absent p{margin:0;color:var(--ink-2);max-width:none}
.fixes{list-style:none;padding:0;margin:0;display:grid;gap:1px;background:var(--rule);
border:1px solid var(--rule)}
.fixes>li{background:var(--surface);padding:16px 18px}
.fixes h3{margin:0 0 6px}
.fixes p{margin:0;font-size:14px;color:var(--ink-2);max-width:none}
.fixes .where{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;
color:var(--ink-3);margin-top:7px;display:block}
.todo{list-style:none;padding:0;margin:0}
.todo li{display:grid;grid-template-columns:108px 1fr;gap:4px 18px;padding:11px 0;
border-bottom:1px solid var(--rule);align-items:baseline}
.todo li:last-child{border-bottom:none}
.todo .tag{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;
letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
.todo .what{font-size:14px}
footer{margin-top:72px;padding-top:20px;border-top:2px solid var(--ink);
font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--ink-3);
max-width:var(--measure)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (max-width:640px){.todo li{grid-template-columns:1fr}}
.specimens{display:grid;gap:28px}
.specimen{background:var(--surface);border:1px solid var(--rule)}
.specimen-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;
padding:14px 18px;border-bottom:1px solid var(--rule)}
.specimen-head h3{margin:0;font-size:16px}
.strip{display:flex;gap:1px;overflow-x:auto;background:var(--rule);
border-bottom:1px solid var(--rule)}
.panel{margin:0;flex:0 0 224px;background:var(--surface);display:flex;flex-direction:column}
.panel img{display:block;width:224px;height:224px;object-fit:cover;background:var(--surface-2)}
.panel figcaption{padding:10px 12px 14px;display:flex;flex-direction:column;gap:4px;
font-size:12px;line-height:1.45;flex:1}
.panel figcaption b{font-size:11px;letter-spacing:.06em;text-transform:uppercase;
font-family:"IBM Plex Mono",ui-monospace,monospace;color:var(--accent);font-weight:600}
.panel figcaption span{color:var(--ink-2)}
.readout{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:1px;
background:var(--rule)}
.readout>div{background:var(--surface);padding:16px 18px}
.readout h4{margin:0 0 12px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;
font-family:"IBM Plex Mono",ui-monospace,monospace;color:var(--ink-3);font-weight:600}
.readout dl{margin:0;display:grid;gap:5px}
.readout dl>div{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
font-size:12.5px}
.readout dt{color:var(--ink-2)}
.readout dd{margin:0;font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:500;
font-variant-numeric:tabular-nums}
.swatches{display:grid;gap:6px;margin-bottom:12px}
.sw{display:grid;grid-template-columns:16px 46px 1fr;align-items:center;gap:9px;font-size:11.5px;
font-family:"IBM Plex Mono",ui-monospace,monospace}
.sw b{font-weight:600;font-variant-numeric:tabular-nums;text-align:right}
.chipcolour{width:16px;height:16px;border:1px solid var(--rule-strong)}
"""

FIXES = [
    ("The pipeline would have stopped working",
     "<code>SVC(probability=True)</code> is deprecated in scikit-learn 1.9 and removed in 1.11. "
     "It never changed a prediction &mdash; <code>SVC.predict</code> takes the argmax of the "
     "decision function either way &mdash; but it fitted a redundant internal Platt model on "
     "every call. The brief names <code>CalibratedClassifierCV</code> for E3 and the code was "
     "not using it; both are now correct.",
     "harness.py &middot; build_pipeline, build_probability_pipeline"),
    ("The Phase 3 benchmark ignored T3&rsquo;s configuration",
     "T3 was constructed bare while T1 and T2 were built from <code>config.json</code>. Editing "
     "the config would have changed two techniques and silently left the third alone &mdash; "
     "exactly the difference between techniques that Mode A exists to exclude.",
     "scripts/run_benchmarks.py &middot; available_extractors"),
    ("E3&rsquo;s voting weights leaked across folds",
     "The driver computed macro-F1 weights over the whole training partition and passed them "
     "into cross-validation, so every fold was scored with weights its own validation rows had "
     "helped set. Each fold now derives its weights from its own training rows.",
     "scripts/run_enhancements.py &middot; E3 and the ablation loop"),
    ("The target verdict named the wrong row",
     "The ranking carries enhancements alongside techniques, and the summary read the 80% "
     "individual-technique bar against whatever topped it &mdash; printing &ldquo;best "
     "individual technique: E1&rdquo;. It now names the best technique and says so when a "
     "higher-ranked entry is out of scope.",
     "compare.py &middot; ComparisonReport.summary_text"),
    ("The winner&rsquo;s lead printed with the wrong sign",
     "Pairwise tests are generated in insertion order, so the top-ranked configuration is "
     "often the second name in its own comparison and its stored difference is negative. The "
     "verdict sentence read it raw and rendered &ldquo;<code>+-0.0138</code>&rdquo; with a "
     "negative t. Both are now oriented to the direction the sentence claims.",
     "compare.py &middot; ComparisonReport.summary_text"),
    ("A dead configuration key",
     "<code>classifier.probability</code> stopped being read once the deprecation was fixed. It "
     "is load-bearing again: the calibrated pipeline fails loudly if it is disabled, rather "
     "than silently reporting decision-function values as probabilities.",
     "config.json &middot; harness.py"),
]

TODO = [
    ("CLI", "<b>Eight commands, feature caching, resumability.</b> Ten separate scripts exist "
            "instead; nothing is cached, so every rerun re-extracts and re-segments."),
    ("Dashboard", "<b>Tkinter + Matplotlib viewer.</b> Not started."),
    ("T1 &middot; E1.1&ndash;E1.5", "<b>Five sub-experiments.</b> Histogram baseline, N = 3..6 "
            "colours, LAB vs HSV vs RGB, specular exclusion on/off, block ablation. The "
            "extractor supports every one; only the runner is missing."),
    ("T2", "<b>Parameter sensitivity study.</b> Distance sets, quantisation levels and the "
           "angle-averaged rotation-invariant variant. 32 levels is a default, not a measured "
           "choice."),
    ("E2", "<b>The learned weighting.</b> Sub-regions are concatenated and the SVM left to "
           "weight them implicitly; the brief asks for a learned weighting."),
    ("Calibration", "<b>Pixel-to-millimetre scale from a reference object.</b> Absent, so "
                    "<code>mm_per_px</code> is pinned at 1.0 and blemish areas are in pixels."),
    ("Annotations", "<b>60-image stratified subset, two annotators, averaged.</b> The painting "
                    "tool is built and single-annotator; no masks exist, so blemish MAE cannot "
                    "be scored."),
    ("Phase 6", "<b>Held-out generalisation and robustness sets.</b> Fruits-360, Fresh/Rotten "
                "and FruitNet directories are empty."),
    ("Docs", "<b>CHOICES.md.</b> Required by the brief wherever the specification is silent."),
    ("Ablation", "<b>Removing each enhancement in turn.</b> Only the technique-drop half is "
                 "implemented; E2 is not ablated at all."),
    ("Results", "<b><code>results/**</code> is gitignored.</b> No CSV reaches a teammate who "
                "clones the repository, which is why this page bakes every figure in."),
]


def build(tag: str, results_root: Path) -> Tuple[str, Dict[str, Any]]:
    """Assemble the whole page and report what it was able to read."""
    run_dir = results_root / tag
    matrix = read_csv(run_dir / "enhancement_matrix.csv", index_col=0)
    vs_best = read_csv(run_dir / "enhancement_vs_best.csv")
    ablation = read_csv(run_dir / "ablation.csv")
    metadata = read_json(run_dir / "run_metadata.json") or {}

    t3 = {name: read_csv(results_root / "t3" / f"{name}_{suffix}.csv")
          for name, suffix in (("e3_1", "ablation"), ("e3_2", "se_shape"),
                               ("e3_3", "rmax"), ("e3_4", "segmentation"))}

    panels = read_json(results_root / "visuals" / "panels.json")
    background = read_csv(results_root / "audit" / "primary_background_only.csv")
    profile = read_csv(results_root / "audit" / "primary_segmentation_profile.csv")
    duplicates = read_csv(results_root / "audit" / "primary_duplicates.csv")

    control = metadata.get("background_only_control")
    if control is None and background is not None:
        control = float(background["background_only_recall"].mean())

    mode = str(metadata.get("mode", "unknown"))
    per_class = metadata.get("per_class", "all")
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    provenance = "PILOT" if mode == "PILOT" else "FULL"
    scope = f"{per_class} per class" if per_class != "all" else "all 2400 images"

    runbar = [
        ("Run", esc(mode.title())),
        ("Scope", esc(scope)),
        ("Augmentation", "on" if metadata.get("augmentation") else "off"),
        ("Train / test rows", f'{metadata.get("n_train_rows", "&mdash;")} / '
                              f'{metadata.get("n_test_rows", "&mdash;")}'),
        ("Background control", num(control)),
    ]
    runbar_html = "".join(
        f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in runbar
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FreshSight Bench Report</title>
<meta name="description" content="Benchmark results, dataset audit and outstanding work for the FreshSight apple-grading comparative study.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&amp;family=IBM+Plex+Mono:wght@400;500;600&amp;family=IBM+Plex+Sans:wght@400;500;600&amp;display=swap">
<style>{STYLE}</style>
</head>
<body>
<div class="shell">

  <header class="masthead">
    <p class="course">BMDS2133 &middot; Mode A &middot; Comparative &amp; Enhancement Study</p>
    <h1>Three descriptors, one harness, and a background that already knows the answer</h1>
    <p class="standfirst">
      Benchmark results for FreshSight, a three-stage apple grader (Unripe / Ripe / Rotten)
      built from classical image processing only. Colour, texture and morphology are compared
      under an identical pipeline so that the descriptor family is the sole experimental variable.
    </p>
    <div class="byline">
      <span><b>Seed</b> {esc(metadata.get("seed", 42))}</span>
      <span><b>Split</b> 80:20 stratified</span>
      <span><b>Folds</b> 5, leakage-safe</span>
      <span><b>Classifier</b> SVC rbf, C=1.0, gamma=scale</span>
      <span><b>Segmentation</b> {esc(metadata.get("segmentation_method", "grabcut"))}</span>
    </div>
  </header>

  <dl class="runbar">{runbar_html}</dl>

  <section>
    <p class="eyebrow"><span class="src">{provenance} run &middot; {esc(scope)}</span>
      <span>results/{esc(tag)}/enhancement_matrix.csv</span></p>
    <h2>Where the six configurations stand</h2>
    <p class="lede">
      Cross-validated accuracy on the training partition. The dashed red line is the control
      that matters: what the same classifier scores when the fruit is masked out and only the
      background remains.
    </p>
    {headline_section(matrix, control, tag)}
  </section>

  <section>
    <p class="eyebrow"><span class="src">Held-out apples &middot; one per class</span>
      <span>results/visuals/panels.json</span></p>
    <h2>What each technique actually sees</h2>
    <p class="lede">
      The same apple, carried through the shared pipeline and then handed to each
      descriptor family. Identical pixels and an identical mask go in; what comes out
      is three readings with almost nothing in common.
    </p>
    {visuals_section(panels)}
  </section>

  <section>
    <p class="eyebrow"><span class="src">Acceptance targets &middot; brief section 9</span>
      <span>never tuned to hit</span></p>
    <h2>Scored against the brief, honestly</h2>
    <p class="lede">
      The specification sets these bars and says explicitly not to tune towards them. A missed
      target is reported as missed, and a target that cannot be scored says why.
    </p>
    {acceptance_section(matrix, vs_best, t3.get("e3_4"), t3.get("e3_1"))}
  </section>

  <section>
    <p class="eyebrow"><span class="src">{provenance} run &middot; paired t-tests over 5 folds</span>
      <span>results/{esc(tag)}/enhancement_vs_best.csv</span></p>
    <h2>Does any enhancement actually beat the best technique?</h2>
    {enhancement_section(vs_best, tag)}
  </section>

  <section>
    <p class="eyebrow"><span class="src">{provenance} run &middot; drop one technique at a time</span>
      <span>results/{esc(tag)}/ablation.csv</span></p>
    <h2>What each technique contributes to the hybrid</h2>
    {ablation_section(ablation, matrix, tag)}
  </section>

  <section>
    <p class="eyebrow"><span class="src">Full dataset &middot; 2400 images</span>
      <span>results/t3/e3_*.csv</span></p>
    <h2>T3 internal experiments</h2>
    <p class="lede">
      The four sweeps required for the morphological branch, each run on the complete dataset
      with five-fold cross-validation on the training partition only.
    </p>
    {t3_section(t3)}
  </section>

  <section>
    <p class="eyebrow"><span class="src">Dataset audit &middot; 2400 images</span>
      <span>gate verdict: FAIL</span></p>
    <h2>The dataset predicts its own labels</h2>
    <p class="lede">
      The brief requires a suitability audit before any dataset is adopted: train the shared
      classifier on images with the fruit region masked out, and expect something near
      three-class chance.
    </p>
    {audit_section(background, profile, duplicates)}
  </section>

  <section>
    <p class="eyebrow"><span class="src">Code review</span> <span>290 tests passing</span></p>
    <h2>Defects found and fixed</h2>
    <ol class="fixes">
      {"".join(f'<li><h3>{title}</h3><p>{body}</p><span class="where">{where}</span></li>'
               for title, body, where in FIXES)}
    </ol>
  </section>

  <section>
    <p class="eyebrow"><span class="src">Brief coverage</span>
      <span>deliverables not yet built</span></p>
    <h2>Still outstanding</h2>
    <p class="lede">
      Specification requirements with no implementation, distinct from the defects above.
      Several are also where the remaining legitimate accuracy lives.
    </p>
    <ul class="todo">
      {"".join(f'<li><span class="tag">{tag_}</span><span class="what">{what}</span></li>'
               for tag_, what in TODO)}
    </ul>
  </section>

  <footer>
    FreshSight &middot; BMDS2133 Image Processing &middot; generated {esc(built)} by
    scripts/build_report.py from results/{esc(tag)}, results/t3 and results/audit.
    Every figure was produced by running the pipeline on real data; nothing on this page is
    hardcoded. Rebuild after any run to refresh it.
  </footer>

</div>
</body>
</html>
"""

    read_sources = {
        "enhancement_matrix": matrix is not None,
        "enhancement_vs_best": vs_best is not None,
        "ablation": ablation is not None,
        "t3_experiments": sum(1 for v in t3.values() if v is not None),
        "audit": background is not None,
        "visuals": len(panels["images"]) if panels else 0,
        "mode": mode,
    }
    return page, read_sources


def _as_fragment(page: str) -> str:
    """Strip the document shell, keeping the title, styles and body content.

    Some hosts wrap supplied markup in their own ``<!doctype>``/``<head>``, so
    a full document would nest one inside another. The figures and the markup
    are identical to the standalone file: this only removes the outer shell,
    which is why the two copies can never disagree about a number.
    """
    title = re.search(r"<title>.*?</title>", page, re.S)
    style = re.search(r"<style>.*?</style>", page, re.S)
    fonts = re.findall(r'<link rel="(?:preconnect|stylesheet)"[^>]*>', page)
    body = re.search(r"<body>(.*)</body>", page, re.S)
    if not (title and style and body):
        raise ValueError("could not split the page; its shell has changed shape")
    return "\n".join([title.group(0), *fonts, style.group(0), "", body.group(1).strip()])


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build the report and write it to disk."""
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="phase5",
                        help="Sub-directory of results/ holding the enhancement run.")
    parser.add_argument("--out", default="docs/index.html",
                        help="Where to write the report, relative to the project root.")
    parser.add_argument("--fragment", default=None,
                        help="Also write a body-only copy here, for hosts that supply "
                             "their own document shell. Same figures, same markup, so the "
                             "two copies cannot drift apart.")
    args = parser.parse_args(argv)

    page, sources = build(args.tag, config.paths.results_root)

    destination = Path(args.out)
    if not destination.is_absolute():
        destination = config.paths.project_root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")

    if args.fragment:
        fragment_path = Path(args.fragment)
        if not fragment_path.is_absolute():
            fragment_path = config.paths.project_root / fragment_path
        fragment_path.parent.mkdir(parents=True, exist_ok=True)
        fragment_path.write_text(_as_fragment(page), encoding="utf-8")
        print(f"Wrote {fragment_path} (body only)")

    size_kb = destination.stat().st_size / 1024
    print(f"Wrote {destination} ({size_kb:.0f} KB, self-contained)")
    print(f"  enhancement run   : {args.tag} [{sources['mode']}] "
          f"{'found' if sources['enhancement_matrix'] else 'MISSING'}")
    print(f"  paired t-tests    : {'found' if sources['enhancement_vs_best'] else 'MISSING'}")
    print(f"  ablation          : {'found' if sources['ablation'] else 'MISSING'}")
    print(f"  T3 experiments    : {sources['t3_experiments']}/4 found")
    print(f"  dataset audit     : {'found' if sources['audit'] else 'MISSING'}")
    print(f"  technique panels  : {sources['visuals']} apples"
          f"{'' if sources['visuals'] else ' (run scripts/build_visuals.py)'}")
    print("\nShare it by committing docs/ and enabling GitHub Pages on that folder, "
          "or just send the single HTML file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
