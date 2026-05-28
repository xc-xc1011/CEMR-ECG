from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Wedge, Circle
import pandas as pd


ROOT = Path(__file__).resolve().parent
PRIMARY_DIR = ROOT / "els-cas-templates" / "figures"
BACKUP_DIR = ROOT / "results" / "cemr_ecg_bspc_figures"
SOURCE_DIR = BACKUP_DIR / "source_data"

for path in (PRIMARY_DIR, BACKUP_DIR, SOURCE_DIR):
    path.mkdir(parents=True, exist_ok=True)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.6,
    }
)


COL = {
    "ink": "#222222",
    "muted": "#686868",
    "line": "#B8BDC7",
    "soft": "#F6F7F9",
    "beat": "#EAF2FF",
    "evidence": "#E8F6F3",
    "encoder": "#EAF0FB",
    "adapter": "#FFF4E4",
    "decoder": "#FDEDEC",
    "out": "#ECF7EE",
    "blue": "#245A9C",
    "teal": "#159A8C",
    "orange": "#CF7A1C",
    "red": "#BF4C42",
    "green": "#2E8B57",
}


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def add_label(ax, x, y, text, size=7, weight="normal", color=None, ha="center", va="center"):
    ax.text(x, y, text, fontsize=size, fontweight=weight, color=color or COL["ink"], ha=ha, va=va)


def rounded_box(ax, x, y, w, h, text, fc, ec="#8C96A6", lw=0.9, size=7, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    add_label(ax, x + w / 2, y + h / 2, text, size=size, weight=weight)
    return patch


def arrow(ax, x1, y1, x2, y2, color="#667085", lw=1.0, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def save_pub(fig, stem: str, width_mm: float, height_mm: float, dpi: int = 600):
    fig.set_size_inches(mm_to_in(width_mm), mm_to_in(height_mm))
    for out_dir in (PRIMARY_DIR, BACKUP_DIR):
        for ext in (".svg", ".pdf", ".tiff", ".png"):
            target = out_dir / f"{stem}{ext}"
            if target.exists():
                target.unlink()

    for out_dir in (PRIMARY_DIR, BACKUP_DIR):
        base = out_dir / stem
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
        fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def draw_ecg_trace(ax, x0, y0, w, h, color=COL["blue"]):
    xs = []
    ys = []
    n = 280
    for i in range(n):
        t = i / (n - 1)
        x = x0 + t * w
        base = y0 + h * 0.48
        y = base + 0.02 * h * math.sin(10 * math.pi * t)
        for c, amp, width in [(0.22, 0.08, 0.025), (0.47, -0.13, 0.012), (0.50, 0.42, 0.008), (0.53, -0.18, 0.012), (0.74, 0.11, 0.035)]:
            y += amp * h * math.exp(-((t - c) ** 2) / (2 * width**2))
        xs.append(x)
        ys.append(y)
    ax.plot(xs, ys, color=color, linewidth=1.4)


def fig1_framework():
    fig, ax = plt.subplots(figsize=(mm_to_in(183), mm_to_in(108)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_label(ax, 0.02, 0.965, "(a)", size=8, weight="bold", ha="left")
    add_label(ax, 0.52, 0.965, "(b)", size=8, weight="bold", ha="left")
    add_label(ax, 0.765, 0.965, "(c)", size=8, weight="bold", ha="left")
    add_label(ax, 0.765, 0.185, "(d)", size=8, weight="bold", ha="left")

    rounded_box(ax, 0.035, 0.62, 0.17, 0.18, "Annotated beat", COL["beat"], size=7, weight="bold")
    draw_ecg_trace(ax, 0.058, 0.625, 0.125, 0.055)

    evidence = [
        ("Morphology\namplitude, width", 0.29, 0.79),
        ("RR rhythm\nlocal intervals", 0.29, 0.61),
        ("Lead-aware\nwaveform shape", 0.29, 0.43),
        ("Derivative\nshape summary", 0.29, 0.25),
        ("Prototype\nclass proximity", 0.29, 0.07),
    ]
    fan_starts = [(0.205, 0.75), (0.205, 0.72), (0.205, 0.69), (0.205, 0.66), (0.205, 0.63)]
    for idx, (text, x, y) in enumerate(evidence):
        rounded_box(ax, x, y, 0.18, 0.105, text, COL["evidence"], ec="#6CB6AC", size=6.4)
        arrow(ax, fan_starts[idx][0], fan_starts[idx][1], x, y + 0.052, color=COL["teal"], lw=0.75, rad=0.02)

    rounded_box(ax, 0.55, 0.50, 0.18, 0.17, "Clinical-prototype\nevidence encoder\nsingle probability layer", COL["encoder"], ec="#788AB8", size=6.5, weight="bold")
    ax.plot([0.49, 0.49], [0.12, 0.84], color="#C9D1E5", linewidth=1.1)
    for _, x, y in evidence:
        arrow(ax, x + 0.18, y + 0.052, 0.49, y + 0.052, color="#788AB8", lw=0.65)
    arrow(ax, 0.49, 0.585, 0.55, 0.585, color="#788AB8", lw=1.0)

    rounded_box(ax, 0.55, 0.21, 0.18, 0.12, "Raw backbone\nprobabilities", "#F1F2F6", ec="#8E95A3", size=6.5)
    ax.plot([0.12, 0.12, 0.52], [0.62, 0.18, 0.18], color="#8E95A3", linewidth=0.8)
    arrow(ax, 0.52, 0.18, 0.55, 0.25, color="#8E95A3", lw=0.8, rad=0.0)

    rounded_box(ax, 0.79, 0.48, 0.16, 0.17, "Backbone/evidence\nadapter\nvalidation selected", COL["adapter"], ec="#D39B48", size=6.4, weight="bold")
    arrow(ax, 0.73, 0.585, 0.79, 0.565, color=COL["orange"], lw=1.0)
    arrow(ax, 0.73, 0.27, 0.79, 0.515, color=COL["orange"], lw=1.0, rad=0.18)

    rounded_box(ax, 0.79, 0.22, 0.16, 0.14, "BioAdaptive\ndecoder\nbounded recovery", COL["decoder"], ec="#C86A61", size=6.4, weight="bold")
    arrow(ax, 0.87, 0.48, 0.87, 0.36, color=COL["red"], lw=1.0)

    rounded_box(ax, 0.79, 0.045, 0.16, 0.105, "AAMI output\nN / S / V / F / Q", COL["out"], ec="#5CA56D", size=6.6, weight="bold")
    arrow(ax, 0.87, 0.22, 0.87, 0.15, color=COL["green"], lw=1.0)

    add_label(ax, 0.12, 0.565, "shared evidence extraction", size=6.2, color=COL["muted"])
    add_label(ax, 0.70, 0.735, "method-specific\nprobability interface", size=6.2, color=COL["muted"])
    add_label(ax, 0.69, 0.13, "test labels are used\nonly for reporting", size=6.2, color=COL["muted"])

    ax.add_patch(Rectangle((0.015, 0.02), 0.97, 0.91, linewidth=0.6, edgecolor="#E0E3E8", facecolor="none"))
    return fig


def graphical_abstract():
    fig, ax = plt.subplots(figsize=(mm_to_in(180), mm_to_in(75)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_label(ax, 0.04, 0.86, "CEMR-ECG", size=13, weight="bold", ha="left", color=COL["blue"])
    add_label(ax, 0.04, 0.68, "Classifier-agnostic evidence framework", size=7.6, ha="left", color=COL["ink"])
    add_label(ax, 0.04, 0.59, "for imbalanced heartbeat classification", size=7.6, ha="left", color=COL["ink"])

    rounded_box(ax, 0.045, 0.30, 0.15, 0.20, "ECG beat", COL["beat"], ec="#7FA7D9", size=7, weight="bold")
    draw_ecg_trace(ax, 0.065, 0.325, 0.11, 0.055)

    steps = [
        ("Named evidence\nmorphology + rhythm", 0.25, COL["evidence"], "#6CB6AC"),
        ("Evidence encoder\nclass probabilities", 0.44, COL["encoder"], "#788AB8"),
        ("BioAdaptive decoder\nbounded recovery", 0.63, COL["decoder"], "#C86A61"),
        ("AAMI output\nN S V F Q", 0.82, COL["out"], "#5CA56D"),
    ]
    prev_x = 0.195
    prev_y = 0.40
    for text, x, fill, edge in steps:
        rounded_box(ax, x, 0.30, 0.15, 0.20, text, fill, ec=edge, size=6.7, weight="bold")
        arrow(ax, prev_x, prev_y, x, 0.40, color=edge, lw=1.0)
        prev_x = x + 0.15

    center = (0.51, 0.82)
    radius = 0.105
    ring = [
        ("Morphology", 0, 72, COL["teal"]),
        ("Rhythm", 72, 144, COL["orange"]),
        ("Lead shape", 144, 216, COL["blue"]),
        ("Prototype", 216, 288, COL["green"]),
        ("Validation", 288, 360, COL["red"]),
    ]
    for label, start, end, color in ring:
        ax.add_patch(Wedge(center, radius, start, end, width=0.035, facecolor=color, edgecolor="white", linewidth=0.7))
    ax.add_patch(Circle(center, 0.058, facecolor="white", edgecolor="#D5D9E2", linewidth=0.7))
    add_label(ax, center[0], center[1], "evidence\nfunnel", size=6.2, weight="bold")

    add_label(ax, 0.50, 0.12, "20 backbones x 3 datasets x 5 seeds: paired raw-versus-framework evaluation", size=7, color=COL["muted"])
    add_label(ax, 0.84, 0.80, "+11.48 pp\nM-F1(4)", size=9, weight="bold", color=COL["green"])
    add_label(ax, 0.84, 0.62, "minority-sensitive gain\nwith bounded claims", size=6.4, color=COL["muted"])
    return fig


def write_sources():
    nodes = pd.DataFrame(
        [
            {"figure": "fig1", "module": "ECG beat", "role": "input"},
            {"figure": "fig1", "module": "Panel (a)", "role": "evidence extraction"},
            {"figure": "fig1", "module": "Panel (b)", "role": "evidence encoder"},
            {"figure": "fig1", "module": "Panel (c)", "role": "backbone/evidence adapter and decoder"},
            {"figure": "fig1", "module": "Panel (d)", "role": "AAMI output and reporting"},
            {"figure": "fig1", "module": "Morphology evidence", "role": "evidence"},
            {"figure": "fig1", "module": "RR rhythm evidence", "role": "evidence"},
            {"figure": "fig1", "module": "Lead-aware waveform shape", "role": "evidence"},
            {"figure": "fig1", "module": "Derivative shape summary", "role": "evidence"},
            {"figure": "fig1", "module": "Prototype class proximity", "role": "evidence"},
            {"figure": "fig1", "module": "Clinical-prototype evidence encoder", "role": "encoder"},
            {"figure": "fig1", "module": "Raw backbone probabilities", "role": "backbone"},
            {"figure": "fig1", "module": "Backbone/evidence adapter", "role": "adapter"},
            {"figure": "fig1", "module": "BioAdaptive decoder", "role": "decoder"},
            {"figure": "fig1", "module": "AAMI output N/S/V/F/Q", "role": "output"},
        ]
    )
    ga = pd.DataFrame(
        [
            {"figure": "graphical_abstract", "step": "ECG beat", "role": "input"},
            {"figure": "graphical_abstract", "step": "Named evidence", "role": "evidence"},
            {"figure": "graphical_abstract", "step": "Evidence encoder", "role": "encoder"},
            {"figure": "graphical_abstract", "step": "BioAdaptive decoder", "role": "decoder"},
            {"figure": "graphical_abstract", "step": "AAMI output", "role": "output"},
            {"figure": "graphical_abstract", "step": "20 backbones x 3 datasets x 5 seeds", "role": "evaluation"},
        ]
    )
    nodes.to_csv(SOURCE_DIR / "fig1_python_framework_source_data.csv", index=False)
    ga.to_csv(SOURCE_DIR / "fig_graphical_abstract_python_source_data.csv", index=False)


def main():
    write_sources()
    save_pub(fig1_framework(), "fig1_cemr_ecg_framework", 183, 108)
    save_pub(graphical_abstract(), "fig_graphical_abstract", 180, 75)
    print(f"Wrote Python schematic figures to {PRIMARY_DIR} and {BACKUP_DIR}")


if __name__ == "__main__":
    main()
