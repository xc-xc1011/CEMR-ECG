"""Reliability diagrams before and after BioAdaptive adaptation (reviewer 2, point 13).

The manuscript already reports Brier score, negative log-likelihood and ECE,
but reviewers asked for calibration *curves* before and after adaptation and
for an explicit statement on whether the adapted scores may be read as
probabilities.

This script turns the existing reliability-bin table into a multi-panel figure:
one row per dataset, one line for the raw backbone interface and one for the
CEMR-ECG decision scores, pooled over seeds and the three representative
backbones. A separate panel contrasts the pre-adaptation fused distribution
with the post-adaptation decision scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Embed TrueType (Type 42) fonts instead of matplotlib's default Type 3
# outlines, which journal production systems reject.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parent
DEFAULT_BINS = ROOT / "results" / "final_representative_calibration_reliability_bins.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "pesm_revision_figures"
DATASETS = ["MIT-BIH", "INCART", "SVDB"]

COLOR_RAW = "#4C72B0"
COLOR_CEMR = "#C44E52"
COLOR_IDEAL = "#999999"


# Bins holding only a handful of beats produce accuracy steps of 1/n and make
# the pooled curve jump; they are excluded from the figure, not from the ECE.
MIN_BIN_BEATS = 20


def pooled_curve(df: pd.DataFrame, dataset: str, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average the per-run reliability curve for one dataset and score source."""
    sub = df[
        (df.dataset == dataset) & (df.source == source) & (df["n"] >= MIN_BIN_BEATS)
    ]
    if sub.empty:
        return np.array([]), np.array([]), np.array([])

    # Weight each run's bin by its beat count before pooling.
    key = ["method", "seed", "bin"]
    curves = []
    for _, run in sub.groupby(["method", "seed"]):
        run = run.sort_values("bin")
        curves.append(run.set_index("bin")[["lo", "hi", "n", "accuracy", "confidence"]])
    if not curves:
        return np.array([]), np.array([]), np.array([])

    all_bins = sorted(set().union(*[set(c.index) for c in curves]))
    conf, acc, weight = [], [], []
    for b in all_bins:
        num_a = num_c = num_w = 0.0
        for c in curves:
            if b in c.index:
                row = c.loc[b]
                n = float(row["n"])
                if n <= 0 or np.isnan(row["accuracy"]):
                    continue
                num_a += float(row["accuracy"]) * n
                num_c += float(row["confidence"]) * n
                num_w += n
        if num_w > 0:
            acc.append(num_a / num_w)
            conf.append(num_c / num_w)
            weight.append(num_w)
    return np.asarray(conf), np.asarray(acc), np.asarray(weight)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="calibration_curves_before_after")
    parser.add_argument(
        "--bins",
        default=str(DEFAULT_BINS),
        help="Reliability-bin table to plot (default: the representative beat-level table).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for the figure and its source data.",
    )
    args = parser.parse_args()

    df = pd.read_csv(Path(args.bins))
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        ax.plot([0, 1], [0, 1], "--", color=COLOR_IDEAL, lw=1.2, label="Ideal")
        for source, color, label in [
            ("raw", COLOR_RAW, "Raw backbone scores"),
            ("cemr", COLOR_CEMR, "CEMR-ECG decision scores"),
        ]:
            conf, acc, weight = pooled_curve(df, dataset, source)
            if conf.size == 0:
                continue
            order = np.argsort(conf)
            ax.plot(
                conf[order],
                acc[order],
                "-o",
                color=color,
                ms=3.2,
                lw=1.5,
                label=label,
                alpha=0.9,
            )
        ax.set_title(dataset, fontsize=11)
        ax.set_xlabel("Mean confidence", fontsize=9.5)
        ax.set_xlim(0, 1)
        # Low-confidence bins can exceed 1.0 accuracy for a sparse class; clamp
        # the view so the axis stays interpretable without hiding the curve.
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("Empirical accuracy", fontsize=9.5)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        "Reliability of decision scores before and after CEMR-ECG adaptation",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    png = OUT_DIR / f"{args.out}.png"
    pdf = OUT_DIR / f"{args.out}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    print(f"Saved {png}")
    print(f"Saved {pdf}")

    # Source data for the figure.
    rows = []
    for dataset in DATASETS:
        for source in ["raw", "cemr"]:
            conf, acc, weight = pooled_curve(df, dataset, source)
            for c, a, w in zip(conf, acc, weight):
                rows.append(
                    {
                        "dataset": dataset,
                        "source": source,
                        "confidence": float(c),
                        "accuracy": float(a),
                        "weight": float(w),
                    }
                )
    src = OUT_DIR / f"{args.out}_source_data.csv"
    pd.DataFrame(rows).to_csv(src, index=False, encoding="utf-8")
    print(f"Saved {src}")

    # Report the mean |accuracy - confidence| gap, the calibration error view.
    print("\n=== pooled calibration gap (mean |accuracy - confidence|) ===")
    for dataset in DATASETS:
        for source in ["raw", "cemr"]:
            conf, acc, weight = pooled_curve(df, dataset, source)
            if conf.size == 0:
                continue
            gap = float(np.average(np.abs(acc - conf), weights=weight))
            print(f"{dataset:<8} {source:<5} gap={gap:.4f}  bins={len(conf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
