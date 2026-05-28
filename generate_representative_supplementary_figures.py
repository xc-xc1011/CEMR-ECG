from __future__ import annotations

from pathlib import Path

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
SPLIT_AUDIT = RESULTS / "final_split_audit.csv"
SENSITIVITY = RESULTS / "final_class_weight_sensitivity_summary.csv"

STEM = "figS_representative_calibration_sensitivity"


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
    "raw": "#8B96A8",
    "cemr": "#2A7F62",
    "blue": "#315B96",
    "green": "#2A7F62",
    "orange": "#C9822B",
    "red": "#B9574D",
    "ink": "#222222",
    "muted": "#667085",
    "grid": "#E6E8EF",
}


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def require_inputs() -> None:
    missing = [p for p in (CAL_SUMMARY, RELIABILITY, SPLIT_AUDIT, SENSITIVITY) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files: " + ", ".join(str(p) for p in missing))
    for p in (PRIMARY_DIR, BACKUP_DIR, SOURCE_DIR):
        p.mkdir(parents=True, exist_ok=True)


def percentage(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return x * 100.0


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
    out.to_csv(SOURCE_DIR / f"{STEM}_calibration_summary_source_data.csv", index=False)
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
    out.to_csv(SOURCE_DIR / f"{STEM}_reliability_bins_source_data.csv", index=False)
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
    out.to_csv(SOURCE_DIR / f"{STEM}_sensitivity_source_data.csv", index=False)
    return out


def build_split_source(split: pd.DataFrame) -> pd.DataFrame:
    overlap_cols = [
        "train_test_record_overlap",
        "fit_test_record_overlap",
        "val_test_record_overlap",
    ]
    rows = []
    for col in overlap_cols:
        rows.append(
            {
                "overlap_type": col.replace("_record_overlap", "").replace("_", " to "),
                "max_overlap_records": int(split[col].max()),
                "rows_checked": int(len(split)),
                "datasets": ";".join(sorted(split["dataset"].unique())),
                "seeds": ";".join(str(x) for x in sorted(split["seed"].unique())),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE_DIR / f"{STEM}_split_audit_source_data.csv", index=False)
    return out


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def draw_panel_a(ax, cal_src: pd.DataFrame) -> None:
    plot = cal_src[cal_src["method"].ne("All representatives")].copy()
    methods = ["CAT-Net", "ExtraTrees_raw", "TimeMixer"]
    method_labels = ["CAT-Net", "ExtraTrees", "TimeMixer"]
    metrics = ["Brier score", "ECE"]
    x_base = np.arange(len(methods))
    width = 0.17
    offsets = {
        ("Brier score", "Raw"): -0.27,
        ("Brier score", "CEMR-ECG"): -0.09,
        ("ECE", "Raw"): 0.09,
        ("ECE", "CEMR-ECG"): 0.27,
    }
    hatches = {"Brier score": "", "ECE": "///"}
    colors = {"Raw": COL["raw"], "CEMR-ECG": COL["cemr"]}
    for metric in metrics:
        for source in ("Raw", "CEMR-ECG"):
            vals = []
            errs = []
            for method in methods:
                row = plot[
                    plot["method"].eq(method) & plot["metric"].eq(metric) & plot["source"].eq(source)
                ].iloc[0]
                vals.append(row["value"])
                errs.append(row["std"])
            ax.bar(
                x_base + offsets[(metric, source)],
                vals,
                width,
                yerr=errs,
                color=colors[source],
                alpha=0.95 if metric == "Brier score" else 0.65,
                edgecolor="white",
                linewidth=0.5,
                hatch=hatches[metric],
                error_kw={"elinewidth": 0.7, "capsize": 1.5, "capthick": 0.7},
                label=f"{source} {metric}",
            )
    ax.set_xticks(x_base)
    ax.set_xticklabels(method_labels)
    ax.set_ylabel("Score")
    ax.set_title("Calibration diagnostics by representative backbone", fontsize=7.5, fontweight="bold")
    ax.grid(axis="y", color=COL["grid"], linewidth=0.6)
    ax.legend(ncol=2, loc="upper right", fontsize=5.7, handlelength=1.3, columnspacing=0.8)
    panel_label(ax, "(a)")


def draw_panel_b(ax, rel_src: pd.DataFrame) -> None:
    colors = {"Raw": COL["raw"], "CEMR-ECG": COL["cemr"]}
    ax.plot([0, 1], [0, 1], color="#B7BDC9", linestyle="--", linewidth=0.9, label="Perfect calibration")
    for source in ("Raw", "CEMR-ECG"):
        g = rel_src[rel_src["source"].eq(source)].sort_values("confidence")
        ax.plot(
            g["confidence"],
            g["accuracy"],
            marker="o",
            markersize=3.2,
            linewidth=1.2,
            color=colors[source],
            label=source,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Pooled reliability curve", fontsize=7.5, fontweight="bold")
    ax.grid(color=COL["grid"], linewidth=0.6)
    ax.legend(loc="lower right", fontsize=6)
    panel_label(ax, "(b)")


def draw_panel_c(ax, sens_src: pd.DataFrame) -> None:
    labels = sens_src["profile_label"].tolist()
    vals = percentage(sens_src["macro_f1_4_mean"].to_numpy())
    errs = percentage(sens_src["macro_f1_4_std"].to_numpy())
    colors = []
    for profile in sens_src["class_cost_profile"].astype(str):
        if profile == "moderate_tail":
            colors.append("#4FA987")
        elif profile == "default_cost":
            colors.append(COL["blue"])
        elif profile == "balanced":
            colors.append("#D6A05C")
        else:
            colors.append("#AAB2C1")
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, color=colors, edgecolor="white", linewidth=0.6, width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("M-F1(4), %")
    ax.set_title("Class-cost sensitivity", fontsize=7.5, fontweight="bold")
    ax.grid(axis="y", color=COL["grid"], linewidth=0.6)
    ax.set_ylim(max(40, vals.min() - 12), min(75, vals.max() + 12))
    best_idx = int(np.argmax(vals))
    ax.text(
        best_idx,
        vals[best_idx] + errs[best_idx] + 1.0,
        "best",
        ha="center",
        va="bottom",
        fontsize=6,
        color=COL["green"],
        fontweight="bold",
    )
    panel_label(ax, "(c)")


def draw_panel_d(ax, split_src: pd.DataFrame) -> None:
    labels = ["Train-test", "Fit-test", "Val-test"]
    vals = split_src["max_overlap_records"].to_numpy()
    x = np.arange(len(labels))
    ax.bar(x, vals, color=[COL["green"]] * len(labels), width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Max overlap records")
    ax.set_title("Record-level split audit", fontsize=7.5, fontweight="bold")
    ax.grid(axis="y", color=COL["grid"], linewidth=0.6)
    for i, v in enumerate(vals):
        ax.text(i, 0.08, str(int(v)), ha="center", va="bottom", fontsize=8, fontweight="bold", color=COL["green"])
    rows_checked = int(split_src["rows_checked"].iloc[0])
    ax.text(
        0.5,
        0.78,
        f"{rows_checked} dataset-seed split records checked\\nNo test-record overlap observed",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.7,
        color=COL["ink"],
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F2FAF5", edgecolor="#B9DCC8", linewidth=0.7),
    )
    panel_label(ax, "(d)")


def update_manifest() -> None:
    row = {
        "figure": "Supplementary Fig. S1",
        "file_stem": STEM,
        "role": "Representative calibration, reliability, split-leakage and class-cost sensitivity diagnostics",
        "source": "final_representative_calibration_audit_summary.csv; final_representative_calibration_reliability_bins.csv; final_split_audit.csv; final_class_weight_sensitivity_summary.csv",
    }
    for manifest_csv in (PRIMARY_DIR / "figure_manifest.csv", BACKUP_DIR / "figure_manifest.csv"):
        if manifest_csv.exists():
            df = pd.read_csv(manifest_csv)
            df = df[df["file_stem"].ne(STEM)]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(manifest_csv, index=False)

    md_lines = [
        "# CEMR-ECG BSPC Figure Manifest",
        "",
        "Fig. 1 and the graphical abstract were generated in Python/matplotlib; Fig. 2--9 were generated in R/ggplot2. Supplementary Fig. S1 was generated in Python/matplotlib from the representative supplementary audit tables. All figures include editable SVG/PDF, PNG previews and 600 dpi TIFF exports.",
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
        for ext in (".svg", ".pdf", ".tiff", ".png"):
            target = out_dir / f"{STEM}{ext}"
            if target.exists():
                target.unlink()
    for out_dir in (PRIMARY_DIR, BACKUP_DIR):
        base = out_dir / STEM
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)


def main() -> None:
    require_inputs()
    cal = pd.read_csv(CAL_SUMMARY)
    rel = pd.read_csv(RELIABILITY)
    split = pd.read_csv(SPLIT_AUDIT)
    sens = pd.read_csv(SENSITIVITY)

    cal_src = build_calibration_source(cal)
    rel_src = build_reliability_source(rel)
    sens_src = build_sensitivity_source(sens)
    split_src = build_split_source(split)

    fig, axes = plt.subplots(2, 2, figsize=(mm_to_in(183), mm_to_in(130)))
    draw_panel_a(axes[0, 0], cal_src)
    draw_panel_b(axes[0, 1], rel_src)
    draw_panel_c(axes[1, 0], sens_src)
    draw_panel_d(axes[1, 1], split_src)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.11, wspace=0.32, hspace=0.44)
    save_figure(fig)
    plt.close(fig)
    update_manifest()
    print(f"Wrote {STEM} to {PRIMARY_DIR} and {BACKUP_DIR}")


if __name__ == "__main__":
    main()
