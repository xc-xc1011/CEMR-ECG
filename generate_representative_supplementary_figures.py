from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PRIMARY_DIR = ROOT / "els-cas-templates" / "figures"
BACKUP_DIR = RESULTS / "cemr_ecg_bspc_figures"
SOURCE_DIR = BACKUP_DIR / "source_data"

CAL_SUMMARY = RESULTS / "final_representative_calibration_audit_summary.csv"
RELIABILITY = RESULTS / "final_representative_calibration_reliability_bins.csv"
SENSITIVITY = RESULTS / "final_class_weight_sensitivity_summary.csv"

STEMS = ("fig10_probability_quality_robustness", "figS_representative_calibration_sensitivity")


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)


COL = {
    "raw": "#8F98A8",
    "cemr": "#3A8F73",
    "blue": "#496D9D",
    "green": "#3A8F73",
    "orange": "#D0A15C",
    "red": "#B76E68",
    "ink": "#222222",
    "muted": "#667085",
    "grid": "#E7E8EC",
    "light": "#F5F7FA",
}


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def require_inputs() -> None:
    missing = [p for p in (CAL_SUMMARY, RELIABILITY, SENSITIVITY) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files: " + ", ".join(str(p) for p in missing))
    for p in (PRIMARY_DIR, BACKUP_DIR, SOURCE_DIR):
        p.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        stale = SOURCE_DIR / f"{stem}_split_audit_source_data.csv"
        if stale.exists():
            stale.unlink()


def percentage(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return x * 100.0


def write_source_for_stems(df: pd.DataFrame, suffix: str) -> None:
    for stem in STEMS:
        df.to_csv(SOURCE_DIR / f"{stem}_{suffix}_source_data.csv", index=False)


def build_calibration_source(cal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    overall = cal[cal["dataset"].eq("Overall")].copy()
    method_order = ["CAT-Net", "ExtraTrees_raw", "TimeMixer", "All representatives"]
    overall["method"] = pd.Categorical(overall["method"], method_order, ordered=True)
    overall = overall.sort_values("method")
    for _, row in overall.iterrows():
        method = str(row["method"])
        for metric, label in (("brier", "Brier score"), ("ece", "ECE")):
            rows.append(
                {
                    "method": method,
                    "metric": label,
                    "source": "Raw",
                    "value": row[f"raw_{metric}_mean"],
                    "std": row[f"raw_{metric}_std"],
                    "n_seeds": row["n_seeds"],
                }
            )
            rows.append(
                {
                    "method": method,
                    "metric": label,
                    "source": "CEMR-ECG",
                    "value": row[f"cemr_{metric}_mean"],
                    "std": row[f"cemr_{metric}_std"],
                    "n_seeds": row["n_seeds"],
                }
            )
    out = pd.DataFrame(rows)
    write_source_for_stems(out, "calibration_summary")
    return out


def build_reliability_source(rel: pd.DataFrame) -> pd.DataFrame:
    rel = rel.dropna(subset=["n", "accuracy", "confidence"]).copy()
    rel = rel[rel["n"] > 0].copy()
    grouped = []
    for (source, bin_id), g in rel.groupby(["source", "bin"], sort=True):
        n = g["n"].sum()
        if n <= 0:
            continue
        grouped.append(
            {
                "source": "Raw" if source == "raw" else "CEMR-ECG",
                "bin": int(bin_id),
                "n": int(n),
                "confidence": float(np.average(g["confidence"], weights=g["n"])),
                "accuracy": float(np.average(g["accuracy"], weights=g["n"])),
                "abs_gap": float(np.average(g["abs_gap"], weights=g["n"])),
            }
        )
    out = pd.DataFrame(grouped)
    write_source_for_stems(out, "reliability_bins")
    return out


def build_sensitivity_source(sens: pd.DataFrame) -> pd.DataFrame:
    out = sens[
        sens["dataset"].eq("Overall") & sens["method"].eq("All representatives")
    ].copy()
    order = ["no_cost", "balanced", "moderate_tail", "default_cost", "strong_tail"]
    labels = {
        "no_cost": "No cost",
        "balanced": "Balanced",
        "moderate_tail": "Moderate tail",
        "default_cost": "Default",
        "strong_tail": "Strong tail",
    }
    out["class_cost_profile"] = pd.Categorical(out["class_cost_profile"], order, ordered=True)
    out = out.sort_values("class_cost_profile")
    out["profile_label"] = out["class_cost_profile"].astype(str).map(labels)
    write_source_for_stems(out, "sensitivity")
    return out


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.075,
        1.095,
        label,
        transform=ax.transAxes,
        fontsize=7.4,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def draw_panel_a(ax, cal_src: pd.DataFrame) -> None:
    plot = cal_src[cal_src["method"].ne("All representatives")].copy()
    methods = ["CAT-Net", "ExtraTrees_raw", "TimeMixer"]
    method_labels = ["CAT-Net", "ExtraTrees", "TimeMixer"]
    metrics = ["Brier score", "ECE"]
    metric_x = {"Brier score": 0.0, "ECE": 1.0}
    colors = {"Raw": COL["raw"], "CEMR-ECG": COL["cemr"]}
    jitter = {"CAT-Net": -0.11, "ExtraTrees_raw": 0.0, "TimeMixer": 0.11}
    markers = {"CAT-Net": "o", "ExtraTrees_raw": "s", "TimeMixer": "^"}
    for method, label in zip(methods, method_labels):
        for metric in metrics:
            raw = plot[
                plot["method"].eq(method) & plot["metric"].eq(metric) & plot["source"].eq("Raw")
            ].iloc[0]
            cemr = plot[
                plot["method"].eq(method) & plot["metric"].eq(metric) & plot["source"].eq("CEMR-ECG")
            ].iloc[0]
            x = metric_x[metric] + jitter[method]
            ax.plot([x, x], [raw["value"], cemr["value"]], color="#B8BFCB", linewidth=0.9, zorder=1)
            ax.errorbar(
                x,
                raw["value"],
                yerr=raw["std"],
                fmt=markers[method],
                markersize=4.1,
                color=colors["Raw"],
                markeredgecolor="white",
                markeredgewidth=0.35,
                elinewidth=0.6,
                capsize=1.5,
                zorder=3,
            )
            ax.errorbar(
                x,
                cemr["value"],
                yerr=cemr["std"],
                fmt=markers[method],
                markersize=4.1,
                color=colors["CEMR-ECG"],
                markeredgecolor="white",
                markeredgewidth=0.35,
                elinewidth=0.6,
                capsize=1.5,
                zorder=4,
            )
    ax.set_xticks([metric_x[m] for m in metrics])
    ax.set_xticklabels(metrics)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Lower-is-better score")
    ax.set_title("Representative paired score diagnostics", fontsize=7.1, fontweight="bold")
    ax.grid(axis="y", color=COL["grid"], linewidth=0.6)
    from matplotlib.lines import Line2D

    source_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors["Raw"], label="Raw", markersize=4.3),
        Line2D([0], [0], marker="o", linestyle="", color=colors["CEMR-ECG"], label="CEMR-ECG", markersize=4.3),
    ]
    method_handles = [
        Line2D([0], [0], marker=markers[m], linestyle="", color=COL["muted"], label=l, markersize=4.0)
        for m, l in zip(methods, method_labels)
    ]
    leg1 = ax.legend(handles=source_handles, loc="upper right", fontsize=5.8, handlelength=1.0)
    ax.add_artist(leg1)
    ax.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.45, 0.99),
        ncol=3,
        fontsize=5.25,
        columnspacing=0.55,
        handlelength=0.75,
    )
    panel_label(ax, "a")


def draw_panel_b(ax, rel_src: pd.DataFrame) -> None:
    colors = {"Raw": COL["raw"], "CEMR-ECG": COL["cemr"]}
    ax.plot([0, 1], [0, 1], color="#B7BDC9", linestyle="--", linewidth=0.85, label="Ideal")
    for source in ("Raw", "CEMR-ECG"):
        g = rel_src[rel_src["source"].eq(source)].sort_values("confidence")
        ax.plot(
            g["confidence"],
            g["accuracy"],
            marker="o",
            markersize=3.0,
            linewidth=1.15,
            color=colors[source],
            label=source,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Pooled reliability curve", fontsize=7.1, fontweight="bold")
    ax.grid(color=COL["grid"], linewidth=0.6)
    ax.legend(loc="lower right", fontsize=5.9)
    panel_label(ax, "b")


def draw_panel_c(ax, sens_src: pd.DataFrame) -> None:
    labels = sens_src["profile_label"].tolist()
    vals = percentage(sens_src["macro_f1_4_mean"].to_numpy())
    errs = percentage(sens_src["macro_f1_4_std"].to_numpy())
    colors = []
    for profile in sens_src["class_cost_profile"].astype(str):
        if profile == "default_cost":
            colors.append(COL["blue"])
        elif profile in {"no_cost", "moderate_tail", "strong_tail"}:
            colors.append(COL["green"])
        elif profile == "default_cost":
            colors.append(COL["blue"])
        elif profile == "balanced":
            colors.append("#D6A05C")
        else:
            colors.append("#AAB2C1")
    x = np.arange(len(labels))
    ax.axhspan(59.74, 60.07, color=COL["green"], alpha=0.10, zorder=0)
    ax.errorbar(
        x,
        vals,
        yerr=errs,
        fmt="o",
        markersize=4.4,
        color=COL["muted"],
        ecolor="#A9B1BE",
        elinewidth=0.75,
        capsize=1.8,
        zorder=2,
    )
    ax.plot(x, vals, color="#B8BFCB", linewidth=0.85, zorder=1)
    ax.scatter(x, vals, s=28, c=colors, edgecolors="white", linewidths=0.35, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("M-F1(4), %")
    ax.set_title("Class-cost sensitivity", fontsize=7.1, fontweight="bold")
    ax.grid(axis="y", color=COL["grid"], linewidth=0.6)
    ax.set_ylim(50, 70)
    default_idx = labels.index("Default")
    ax.text(
        default_idx,
        vals[default_idx] + errs[default_idx] + 0.65,
        "selected",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=COL["blue"],
        fontweight="bold",
    )
    ax.text(
        0.625,
        0.14,
        "nearby profiles",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.9,
        color=COL["green"],
    )
    panel_label(ax, "c")


def update_manifest() -> None:
    rows = [
        {
            "figure": "Fig. 10",
            "file_stem": "fig10_probability_quality_robustness",
            "role": "Representative probability-quality, reliability and class-cost sensitivity diagnostics",
            "source": "final_representative_calibration_audit_summary.csv; final_representative_calibration_reliability_bins.csv; final_class_weight_sensitivity_summary.csv; probability_quality_direction_audit_summary.csv",
        },
        {
            "figure": "Supplementary Fig. S1",
            "file_stem": "figS_representative_calibration_sensitivity",
            "role": "Representative probability-quality, reliability and class-cost sensitivity diagnostics",
            "source": "final_representative_calibration_audit_summary.csv; final_representative_calibration_reliability_bins.csv; final_class_weight_sensitivity_summary.csv; probability_quality_direction_audit_summary.csv",
        },
    ]
    for manifest_csv in (PRIMARY_DIR / "figure_manifest.csv", BACKUP_DIR / "figure_manifest.csv"):
        if manifest_csv.exists():
            df = pd.read_csv(manifest_csv)
            for row in rows:
                mask = df["file_stem"].eq(row["file_stem"])
                if mask.any():
                    for key, value in row.items():
                        df.loc[mask, key] = value
                else:
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame(rows)
        df.to_csv(manifest_csv, index=False)

    md_lines = [
        "# CEMR-ECG BSPC Figure Manifest",
        "",
        "Fig. 1, Fig. 10, Supplementary Fig. S1 and the graphical abstract were generated in Python/matplotlib; Fig. 2--9 were generated in R/ggplot2. All figures include editable SVG/PDF, PNG previews and 600 dpi TIFF exports.",
        "Main-result figures use the corrected 20-method, three-dataset, five-seed per-method CEMR-ECG tables.",
        "",
        "| Figure | File stem | Role | Source |",
        "| --- | --- | --- | --- |",
    ]
    manifest_df = pd.read_csv(PRIMARY_DIR / "figure_manifest.csv")
    for _, r in manifest_df.iterrows():
        md_lines.append(f"| {r['figure']} | {r['file_stem']} | {r['role']} | {r['source']} |")
    for manifest_md in (PRIMARY_DIR / "figure_manifest.md", BACKUP_DIR / "figure_manifest.md"):
        manifest_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure) -> None:
    for out_dir in (PRIMARY_DIR, BACKUP_DIR):
        for stem in STEMS:
            for ext in (".svg", ".pdf", ".tiff", ".png"):
                target = out_dir / f"{stem}{ext}"
                if target.exists():
                    target.unlink()
    for stem in STEMS:
        base = PRIMARY_DIR / stem
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
        for ext in (".svg", ".pdf", ".tiff", ".png"):
            shutil.copy2(base.with_suffix(ext), BACKUP_DIR / f"{stem}{ext}")


def main() -> None:
    require_inputs()
    cal = pd.read_csv(CAL_SUMMARY)
    rel = pd.read_csv(RELIABILITY)
    sens = pd.read_csv(SENSITIVITY)

    cal_src = build_calibration_source(cal)
    rel_src = build_reliability_source(rel)
    sens_src = build_sensitivity_source(sens)

    fig = plt.figure(figsize=(mm_to_in(183), mm_to_in(98)))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86], hspace=0.52, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    draw_panel_a(ax_a, cal_src)
    draw_panel_b(ax_b, rel_src)
    draw_panel_c(ax_c, sens_src)
    fig.subplots_adjust(left=0.073, right=0.985, top=0.915, bottom=0.155)
    save_figure(fig)
    plt.close(fig)
    update_manifest()
    print(f"Wrote {', '.join(STEMS)} to {PRIMARY_DIR} and {BACKUP_DIR}")


if __name__ == "__main__":
    main()
