from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path("results")
DEFAULT_DETAIL = RESULTS / "datasetwise_multimethod_permethod_bioadaptive_detail.csv"
DEFAULT_SUMMARY_OUT = RESULTS / "datasetwise_multimethod_permethod_bioadaptive_summary_no_outliers.csv"
DEFAULT_OUTLIERS_OUT = RESULTS / "datasetwise_multimethod_permethod_bioadaptive_removed_outliers.csv"

GROUP_COLUMNS = ["dataset", "family", "method"]
OUTLIER_METRIC = "macro_f1_4"
MIN_RETAINED_SEEDS = 3
EXPECTED_DETAIL_ROWS = 300
EXPECTED_GROUPS = 60
EXPECTED_SEEDS_PER_GROUP = 5

METRIC_COLUMNS = [
    "accuracy",
    "macro_f1_5",
    "macro_f1_4",
    "macro_f1_3_nsv",
    "Se_N",
    "Se_S",
    "Se_V",
    "Se_F",
    "Se_Q",
    "Pr_N",
    "Pr_S",
    "Pr_V",
    "Pr_F",
    "Pr_Q",
    "F1_N",
    "F1_S",
    "F1_V",
    "F1_F",
    "F1_Q",
]

SUMMARY_NUMERIC_COLUMNS = [
    *METRIC_COLUMNS,
    *[f"raw_{m}" for m in METRIC_COLUMNS],
    *[f"delta_{m}" for m in METRIC_COLUMNS],
    "train_time",
    "predict_time",
    "backbone_train_time",
    "backbone_predict_time",
    "evidence_train_time",
    "evidence_predict_time",
    "best_val_macro_f1_4",
    "epochs_run",
    "train_size",
    "val_size",
    "test_size",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an outlier-removed CEMR-BioAdaptive summary. Outliers are "
            "detected within each dataset/family/method group using 1.5*IQR "
            "on macro_f1_4, then the whole seed row is removed."
        )
    )
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--outliers-out", type=Path, default=DEFAULT_OUTLIERS_OUT)
    return parser.parse_args()


def unique_join(values: pd.Series) -> str:
    cleaned = [x for x in values.dropna().unique().tolist()]
    if not cleaned:
        return ""
    try:
        cleaned = sorted(cleaned)
    except TypeError:
        cleaned = sorted(str(x) for x in cleaned)
    return " ".join(str(x) for x in cleaned)


def seed_join(values: pd.Series) -> str:
    seeds = sorted(int(x) for x in values.dropna().unique().tolist())
    return " ".join(str(x) for x in seeds)


def validate_input(detail: pd.DataFrame) -> None:
    required = set(GROUP_COLUMNS + ["seed", OUTLIER_METRIC])
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(detail) != EXPECTED_DETAIL_ROWS:
        raise ValueError(f"Expected {EXPECTED_DETAIL_ROWS} detail rows, found {len(detail)}")

    group_count = detail.groupby(GROUP_COLUMNS, dropna=False).ngroups
    if group_count != EXPECTED_GROUPS:
        raise ValueError(f"Expected {EXPECTED_GROUPS} dataset/family/method groups, found {group_count}")

    seed_counts = detail.groupby(GROUP_COLUMNS, dropna=False)["seed"].nunique()
    bad = seed_counts[seed_counts != EXPECTED_SEEDS_PER_GROUP]
    if not bad.empty:
        raise ValueError(
            "Every group must have exactly "
            f"{EXPECTED_SEEDS_PER_GROUP} seeds. Bad groups:\n{bad.to_string()}"
        )


def outlier_decisions(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept_parts = []
    removed_rows = []

    for keys, group in detail.groupby(GROUP_COLUMNS, dropna=False, sort=False):
        dataset, family, method = keys
        metric = group[OUTLIER_METRIC].astype(float)
        q1 = float(metric.quantile(0.25))
        q3 = float(metric.quantile(0.75))
        iqr = float(q3 - q1)
        lower = float(q1 - 1.5 * iqr)
        upper = float(q3 + 1.5 * iqr)
        median = float(metric.median())

        preliminary_mask = (metric < lower) | (metric > upper)
        preliminary_removed = set(group.loc[preliminary_mask, "seed"].astype(int).tolist())
        keep_mask = ~preliminary_mask
        guard_applied = int(keep_mask.sum()) < MIN_RETAINED_SEEDS

        if guard_applied:
            ranked = group.assign(
                _distance_to_median=(group[OUTLIER_METRIC].astype(float) - median).abs()
            ).sort_values(["_distance_to_median", "seed"], ascending=[True, True])
            retained = set(ranked.head(MIN_RETAINED_SEEDS)["seed"].astype(int).tolist())
            final_keep_mask = group["seed"].astype(int).isin(retained)
        else:
            final_keep_mask = keep_mask

        kept_parts.append(group.loc[final_keep_mask].copy())
        removed = group.loc[~final_keep_mask].copy()
        for _, row in removed.iterrows():
            seed = int(row["seed"])
            reason = "macro_f1_4_iqr_outlier"
            if guard_applied:
                reason = f"{reason};minimum_retained_guard"
            if seed not in preliminary_removed:
                reason = f"{reason};removed_by_distance_to_median"
            removed_rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "method": method,
                    "seed": seed,
                    "macro_f1_4": float(row[OUTLIER_METRIC]),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower": lower,
                    "upper": upper,
                    "median": median,
                    "n_seeds_before": int(group["seed"].nunique()),
                    "n_seeds_after": int(final_keep_mask.sum()),
                    "reason": reason,
                }
            )

    cleaned = pd.concat(kept_parts, ignore_index=True) if kept_parts else detail.iloc[0:0].copy()
    removed_df = pd.DataFrame(
        removed_rows,
        columns=[
            "dataset",
            "family",
            "method",
            "seed",
            "macro_f1_4",
            "q1",
            "q3",
            "iqr",
            "lower",
            "upper",
            "median",
            "n_seeds_before",
            "n_seeds_after",
            "reason",
        ],
    )
    return cleaned, removed_df


def summarize(cleaned: pd.DataFrame, removed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    removed_by_group = {
        keys: grp for keys, grp in removed.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    }

    for keys, group in cleaned.groupby(GROUP_COLUMNS, dropna=False, sort=False):
        dataset, family, method = keys
        removed_group = removed_by_group.get(keys)
        removed_seeds = "" if removed_group is None else seed_join(removed_group["seed"])

        row = {
            "dataset": dataset,
            "family": family,
            "method": method,
            "framework_method": str(group["framework_method"].iloc[0])
            if "framework_method" in group.columns
            else f"{method}+CEMR-BioAdaptive",
            "n_seeds_before": EXPECTED_SEEDS_PER_GROUP,
            "n_seeds_after": int(group["seed"].nunique()),
            "removed_seeds": removed_seeds,
            "seeds_after": seed_join(group["seed"]),
            "outlier_rule": "macro_f1_4_1.5xIQR_by_dataset_family_method",
            "protocol": str(group["protocol"].iloc[0]) if "protocol" in group.columns else "",
            "selected_alpha_values": unique_join(group["selected_alpha"])
            if "selected_alpha" in group.columns
            else "",
            "selected_decoder_values": unique_join(group["selected_decoder"])
            if "selected_decoder" in group.columns
            else "",
            "source": "outlier-removed method-specific CEMR summary; original full 5-seed table is preserved",
        }
        for column in SUMMARY_NUMERIC_COLUMNS:
            if column in group.columns:
                row[f"{column}_mean"] = float(group[column].mean())
                row[f"{column}_std"] = float(group[column].std(ddof=0))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ["dataset", "macro_f1_4_mean", "delta_macro_f1_4_mean", "accuracy_mean"],
        ascending=[True, False, False, False],
    )
    return summary


def validate_outputs(detail: pd.DataFrame, cleaned: pd.DataFrame, removed: pd.DataFrame, summary: pd.DataFrame) -> None:
    if len(summary) != EXPECTED_GROUPS:
        raise ValueError(f"Expected {EXPECTED_GROUPS} summary rows, found {len(summary)}")

    seed_counts_after = cleaned.groupby(GROUP_COLUMNS, dropna=False)["seed"].nunique()
    too_small = seed_counts_after[seed_counts_after < MIN_RETAINED_SEEDS]
    if not too_small.empty:
        raise ValueError(
            f"Every group must retain at least {MIN_RETAINED_SEEDS} seeds. "
            f"Bad groups:\n{too_small.to_string()}"
        )

    if len(cleaned) + len(removed) != len(detail):
        raise ValueError(
            f"Cleaned + removed row mismatch: {len(cleaned)} + {len(removed)} != {len(detail)}"
        )

    key_cols = GROUP_COLUMNS + ["seed"]
    cleaned_keys = set(map(tuple, cleaned[key_cols].to_numpy()))
    removed_keys = set(map(tuple, removed[key_cols].to_numpy())) if not removed.empty else set()
    if cleaned_keys & removed_keys:
        raise ValueError("A seed appears in both cleaned and removed tables.")


def print_comparison(original_summary: pd.DataFrame, new_summary: pd.DataFrame, removed: pd.DataFrame) -> None:
    merged = original_summary.merge(
        new_summary,
        on=GROUP_COLUMNS,
        suffixes=("_full5", "_no_outliers"),
    )
    print("Input groups:", len(original_summary))
    print("Output groups:", len(new_summary))
    print("Removed seed rows:", len(removed))
    print("Groups with removed seeds:", int((new_summary["removed_seeds"].astype(str) != "").sum()))
    print()
    print("Mean change after outlier removal, in percentage points:")
    for metric in ["accuracy", "macro_f1_4", "F1_S", "F1_F"]:
        before = merged[f"{metric}_mean_full5"]
        after = merged[f"{metric}_mean_no_outliers"]
        print(f"  {metric}: {(after - before).mean() * 100.0:+.2f}")

    print()
    print("Dataset-level M-F1(4) full5 -> no_outliers:")
    for dataset, group in merged.groupby("dataset", sort=True):
        before = group["macro_f1_4_mean_full5"].mean() * 100.0
        after = group["macro_f1_4_mean_no_outliers"].mean() * 100.0
        print(f"  {dataset}: {before:.2f} -> {after:.2f} ({after - before:+.2f})")


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.detail)
    validate_input(detail)

    cleaned, removed = outlier_decisions(detail)
    summary = summarize(cleaned, removed)
    validate_outputs(detail, cleaned, removed, summary)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.outliers_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False, encoding="utf-8")
    removed.to_csv(args.outliers_out, index=False, encoding="utf-8")

    original_summary = summarize(detail, detail.iloc[0:0].copy())
    print_comparison(original_summary, summary, removed)
    print()
    print(f"Saved {args.summary_out}")
    print(f"Saved {args.outliers_out}")


if __name__ == "__main__":
    main()
