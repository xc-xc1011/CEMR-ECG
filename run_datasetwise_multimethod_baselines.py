import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

import config
from metrics_eval import metrics_row
from run_datasetwise_cemr_framework_experiments import (
    fit_deep_with_oom_retry,
    load_cache,
    make_scenario,
    raw_dual,
    train_val_split,
)
from run_external_raw_baselines import specs as ml_specs
from run_modern_deep_baselines import model_specs


warnings.filterwarnings("ignore")

RESULTS = Path("results")
DETAIL = RESULTS / "datasetwise_multimethod_baselines_detail.csv"
SUMMARY = RESULTS / "datasetwise_multimethod_baselines_summary.csv"
REPORT = RESULTS / "datasetwise_multimethod_baselines_report.md"
CONFUSION = RESULTS / "datasetwise_multimethod_baselines_confusion.json"

SEEDS = [303, 1303, 2303, 3303, 4303]
DATASETS = ["MIT-BIH", "INCART", "SVDB"]
CLASS_NAMES = ["N", "S", "V", "F", "Q"]

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

DETAIL_COLUMNS = [
    "dataset",
    "protocol",
    "family",
    "method",
    "seed",
    "train_data",
    "test_data",
    "train_size",
    "val_size",
    "test_size",
    "train_counts",
    "val_counts",
    "test_counts",
    "feature_set",
    "input_mode",
    "source",
    "model_config",
    "batch_size",
    "epochs_config",
    "epochs_run",
    "best_val_macro_f1_4",
    "train_time",
    "predict_time",
    *METRIC_COLUMNS,
]


def counts_text(y):
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=config.N_CLASSES)
    return json.dumps({name: int(v) for name, v in zip(CLASS_NAMES, counts)}, ensure_ascii=False)


def completed_keys():
    if not DETAIL.exists():
        return set()
    df = pd.read_csv(DETAIL)
    return set(zip(df["dataset"].astype(str), df["family"].astype(str), df["method"].astype(str), df["seed"].astype(int)))


def load_confusion():
    if CONFUSION.exists():
        return json.loads(CONFUSION.read_text(encoding="utf-8"))
    return {}


def save_confusion(conf):
    CONFUSION.write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")


def append_detail(row):
    df = pd.DataFrame([row])
    for col in DETAIL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[DETAIL_COLUMNS]
    df.to_csv(DETAIL, mode="a", header=not DETAIL.exists(), index=False, encoding="utf-8")


def run_ml_one(data, dataset, spec, seed):
    scenario = make_scenario(data, dataset, seed)
    x_train = raw_dual(scenario["b0_tr"], scenario["b1_tr"])
    x_test = raw_dual(scenario["b0_te"], scenario["b1_te"])
    model = spec["estimator"]

    start = time.time()
    model.fit(x_train, scenario["y_tr"])
    train_time = time.time() - start

    pred_start = time.time()
    pred = np.asarray(model.predict(x_test)).ravel().astype(np.int64)
    predict_time = time.time() - pred_start

    row = {
        "dataset": dataset,
        "protocol": scenario["protocol"],
        "family": "raw_machine_learning",
        "method": spec["model"],
        "seed": int(seed),
        "train_data": scenario["train_record_policy"],
        "test_data": scenario["test_record_policy"],
        "train_size": int(len(scenario["y_tr"])),
        "val_size": 0,
        "test_size": int(len(scenario["y_te"])),
        "train_counts": counts_text(scenario["y_tr"]),
        "val_counts": "",
        "test_counts": counts_text(scenario["y_te"]),
        "feature_set": "raw_dual_wave_flattened",
        "input_mode": "dual_flat",
        "source": "classical raw waveform baseline; no CEMR module",
        "model_config": spec["config"],
        "batch_size": 0,
        "epochs_config": 0,
        "epochs_run": 0,
        "best_val_macro_f1_4": np.nan,
        "train_time": float(train_time),
        "predict_time": float(predict_time),
        **metrics_row(scenario["y_te"], pred),
    }
    cm = confusion_matrix(scenario["y_te"], pred, labels=list(range(config.N_CLASSES))).astype(int).tolist()
    return row, cm


def run_deep_one(data, dataset, spec, seed, device, smoke=False):
    scenario = make_scenario(data, dataset, seed)
    train_idx, val_idx = train_val_split(scenario["y_tr"], seed)
    result = fit_deep_with_oom_retry(spec, scenario, train_idx, val_idx, seed, device, smoke=smoke)
    pred = np.asarray(result["proba_test"]).argmax(axis=1).astype(np.int64)

    row = {
        "dataset": dataset,
        "protocol": scenario["protocol"],
        "family": "raw_deep_or_time_series",
        "method": spec.name,
        "seed": int(seed),
        "train_data": scenario["train_record_policy"],
        "test_data": scenario["test_record_policy"],
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(scenario["y_te"])),
        "train_counts": counts_text(scenario["y_tr"][train_idx]),
        "val_counts": counts_text(scenario["y_tr"][val_idx]),
        "test_counts": counts_text(scenario["y_te"]),
        "feature_set": "raw_waveform",
        "input_mode": result["input_mode"],
        "source": spec.source,
        "model_config": result["backbone_config"],
        "batch_size": int(result["batch_size"]),
        "epochs_config": int(spec.epochs),
        "epochs_run": int(result["epochs_run"]),
        "best_val_macro_f1_4": float(result["best_val_macro_f1_4"]),
        "train_time": float(result["train_time"]),
        "predict_time": float(result["predict_time"]),
        **metrics_row(scenario["y_te"], pred),
    }
    cm = confusion_matrix(scenario["y_te"], pred, labels=list(range(config.N_CLASSES))).astype(int).tolist()
    return row, cm


def summarize():
    if not DETAIL.exists():
        return pd.DataFrame()
    detail = pd.read_csv(DETAIL)
    rows = []
    group_cols = ["dataset", "family", "method"]
    numeric_metrics = [
        *METRIC_COLUMNS,
        "train_time",
        "predict_time",
        "best_val_macro_f1_4",
        "epochs_run",
        "train_size",
        "val_size",
        "test_size",
    ]
    for (dataset, family, method), g in detail.groupby(group_cols, dropna=False):
        row = {
            "dataset": dataset,
            "family": family,
            "method": method,
            "n_seeds": int(g["seed"].nunique()),
            "seeds": " ".join(str(int(s)) for s in sorted(g["seed"].unique())),
            "protocol": str(g["protocol"].iloc[0]),
            "feature_set": str(g["feature_set"].iloc[0]),
            "input_mode": str(g["input_mode"].iloc[0]),
            "source": str(g["source"].iloc[0]),
            "model_config": str(g["model_config"].iloc[0]),
        }
        for metric in numeric_metrics:
            if metric in g.columns:
                row[f"{metric}_mean"] = float(g[metric].mean())
                row[f"{metric}_std"] = float(g[metric].std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["dataset", "macro_f1_4_mean", "accuracy_mean"],
        ascending=[True, False, False],
    )
    summary.to_csv(SUMMARY, index=False, encoding="utf-8")
    return summary


def fmt_pct(x):
    if pd.isna(x):
        return ""
    return f"{float(x) * 100:.2f}"


def write_report():
    summary = summarize()
    lines = [
        "# Dataset-wise Multi-method Baselines",
        "",
        "Protocol: MIT-BIH uses the fixed DS1 to DS2 split. INCART and SVDB use seed-specific record-level train/test splits from the same dataset. These runs do not use CEMR-ECG modules.",
        "",
    ]
    if summary.empty:
        lines.append("_No completed runs._")
    else:
        for dataset, g in summary.groupby("dataset", sort=False):
            top = g.sort_values(["macro_f1_4_mean", "accuracy_mean"], ascending=False)
            lines += [
                f"## {dataset}",
                "",
                "| Rank | Method | Family | Seeds | Acc | M-F1(4) | F1_S | F1_V | F1_F | Train time |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for i, (_, r) in enumerate(top.iterrows(), 1):
                lines.append(
                    f"| {i} | {r['method']} | {r['family']} | {int(r['n_seeds'])} | "
                    f"{fmt_pct(r.get('accuracy_mean'))} | {fmt_pct(r.get('macro_f1_4_mean'))} | "
                    f"{fmt_pct(r.get('F1_S_mean'))} | {fmt_pct(r.get('F1_V_mean'))} | "
                    f"{fmt_pct(r.get('F1_F_mean'))} | {float(r.get('train_time_mean', np.nan)):.1f} |"
                )
            lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def method_allowed(name, wanted):
    return not wanted or name.lower() in wanted


def run_all(args):
    RESULTS.mkdir(exist_ok=True)
    data = load_cache()
    datasets = args.datasets or DATASETS
    seeds = args.seeds or SEEDS
    families = set(args.families or ["ml", "deep"])
    wanted = {m.lower() for m in (args.methods or [])}
    done = completed_keys()
    conf = load_confusion()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if "ml" in families:
        for dataset in datasets:
            for seed in seeds:
                for spec in ml_specs(int(seed)):
                    method = spec["model"]
                    key = (dataset, "raw_machine_learning", method, int(seed))
                    if not method_allowed(method, wanted):
                        continue
                    if key in done and not args.force:
                        print(f"Skip completed {dataset} {method} seed={seed}", flush=True)
                        continue
                    print(f"Running {dataset} {method} seed={seed}", flush=True)
                    row, cm = run_ml_one(data, dataset, spec, int(seed))
                    append_detail(row)
                    conf[f"{dataset}|raw_machine_learning|{method}|{seed}"] = cm
                    save_confusion(conf)
                    done.add(key)
                    write_report()
                    print(
                        f"[{dataset} {method} seed={seed}] Acc={row['accuracy'] * 100:.2f}% "
                        f"M-F1(4)={row['macro_f1_4'] * 100:.2f}% "
                        f"F1_S={row['F1_S'] * 100:.2f}% F1_F={row['F1_F'] * 100:.2f}% "
                        f"train={row['train_time']:.1f}s",
                        flush=True,
                    )

    if "deep" in families:
        specs = model_specs()
        for dataset in datasets:
            for spec in specs:
                if not method_allowed(spec.name, wanted):
                    continue
                for seed in seeds:
                    key = (dataset, "raw_deep_or_time_series", spec.name, int(seed))
                    if key in done and not args.force and not args.smoke:
                        print(f"Skip completed {dataset} {spec.name} seed={seed}", flush=True)
                        continue
                    print(f"Running {dataset} {spec.name} seed={seed}", flush=True)
                    row, cm = run_deep_one(data, dataset, spec, int(seed), device, smoke=args.smoke)
                    if not args.smoke:
                        append_detail(row)
                        conf[f"{dataset}|raw_deep_or_time_series|{spec.name}|{seed}"] = cm
                        save_confusion(conf)
                        done.add(key)
                        write_report()
                    print(
                        f"[{dataset} {spec.name} seed={seed}] Acc={row['accuracy'] * 100:.2f}% "
                        f"M-F1(4)={row['macro_f1_4'] * 100:.2f}% "
                        f"F1_S={row['F1_S'] * 100:.2f}% F1_F={row['F1_F'] * 100:.2f}% "
                        f"epochs={row['epochs_run']} train={row['train_time']:.1f}s",
                        flush=True,
                    )

    write_report()
    summary = summarize()
    print(f"Saved {DETAIL}", flush=True)
    print(f"Saved {SUMMARY}", flush=True)
    print(f"Saved {REPORT}", flush=True)
    print(f"Saved {CONFUSION}", flush=True)
    if not summary.empty:
        print(
            summary[["dataset", "family", "method", "n_seeds", "accuracy_mean", "macro_f1_4_mean", "F1_S_mean", "F1_V_mean", "F1_F_mean"]]
            .to_string(index=False),
            flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dataset-wise raw baselines for multiple methods and five seeds.")
    parser.add_argument("--datasets", nargs="*", default=None, choices=DATASETS)
    parser.add_argument("--families", nargs="*", default=None, choices=["ml", "deep"])
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run deep models for one epoch and do not save them.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for deep models.")
    parser.add_argument("--force", action="store_true", help="Rerun completed keys.")
    parser.add_argument("--rebuild-report", action="store_true", help="Only rebuild summary/report from existing detail CSV.")
    args = parser.parse_args(argv)
    if args.rebuild_report:
        write_report()
        return
    run_all(args)


if __name__ == "__main__":
    main()
