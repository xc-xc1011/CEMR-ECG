import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from feature_engineering import add_proto_features
from metrics_eval import metrics_row
from run_datasetwise_cemr_framework_experiments import load_cache, make_scenario


RESULTS = Path("results")
DETAIL = RESULTS / "cemr_bio_evidence_core_detail.csv"
SUMMARY = RESULTS / "cemr_bio_evidence_core_summary.csv"
REPORT = RESULTS / "cemr_bio_evidence_core_report.md"
CONFUSION = RESULTS / "cemr_bio_evidence_core_confusion.json"

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
DEFAULT_DATASETS = ["MIT-BIH", "INCART", "SVDB"]
DEFAULT_SEEDS = [303, 1303]
EPS = 1e-12


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPS, None)
    return p / np.maximum(p.sum(axis=1, keepdims=True), EPS)


def power_calibrate(p, power):
    p = normalize(p)
    if abs(float(power) - 1.0) < 1e-12:
        return p
    return normalize(np.power(p, float(power)))


def align_proba(classes, proba):
    out = np.zeros((proba.shape[0], config.N_CLASSES), dtype=np.float64)
    for i, cls in enumerate(classes):
        out[:, int(cls)] = proba[:, i]
    missing = out.sum(axis=1) <= 0
    if np.any(missing):
        out[missing] = 1.0 / config.N_CLASSES
    return normalize(out)


def counts_dict(y):
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=config.N_CLASSES)
    return {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)}


def raw_dual(b0, b1):
    return np.concatenate([b0, b1], axis=1).astype(np.float32)


def lead_summary_features(b0, b1):
    diff = b0 - b1
    summ = b0 + b1
    rows = []
    for arr in [diff, summ]:
        rows.extend(
            [
                arr.mean(axis=1),
                arr.std(axis=1),
                np.mean(np.abs(arr), axis=1),
                np.max(arr, axis=1),
                np.min(arr, axis=1),
                np.ptp(arr, axis=1),
                np.mean(np.diff(arr, axis=1) ** 2, axis=1),
            ]
        )
    return np.stack(rows, axis=1).astype(np.float32)


def build_evidence_features(scenario, feature_set):
    x_tr, x_te = add_proto_features(
        scenario["x_tr_base"],
        scenario["y_tr"],
        scenario["x_te_base"],
        scenario["b0_tr"],
        scenario["b1_tr"],
        scenario["b0_te"],
        scenario["b1_te"],
    )
    if feature_set == "clinical_proto":
        return x_tr, x_te
    if feature_set == "clinical_proto_lead":
        return (
            np.concatenate([x_tr, lead_summary_features(scenario["b0_tr"], scenario["b1_tr"])], axis=1).astype(np.float32),
            np.concatenate([x_te, lead_summary_features(scenario["b0_te"], scenario["b1_te"])], axis=1).astype(np.float32),
        )
    if feature_set == "raw_clinical_proto":
        return (
            np.concatenate([raw_dual(scenario["b0_tr"], scenario["b1_tr"]), x_tr], axis=1).astype(np.float32),
            np.concatenate([raw_dual(scenario["b0_te"], scenario["b1_te"]), x_te], axis=1).astype(np.float32),
        )
    raise ValueError(f"Unknown feature_set: {feature_set}")


def prior_ratio(y_train):
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=config.N_CLASSES).astype(np.float64)
    ref = max(counts[0], 1.0)
    return ref / np.maximum(counts, 1.0)


def class_weight_value(y_train, mode):
    if mode == "none":
        return None
    if mode == "balanced":
        return "balanced"
    if mode == "legacy_cost":
        return {0: 1.0, 1: 3.0, 2: 1.5, 3: 12.0, 4: 1.0}
    ratio = prior_ratio(y_train)
    if mode == "smooth_cost":
        return {
            0: 1.0,
            1: float(min(ratio[1] ** 0.22, 4.0)),
            2: float(min(ratio[2] ** 0.08, 1.8)),
            3: float(min(ratio[3] ** 0.28, 12.0)),
            4: 1.0,
        }
    if mode == "tail_cost":
        return {
            0: 0.95,
            1: float(min(ratio[1] ** 0.28, 5.0)),
            2: float(min(ratio[2] ** 0.10, 2.0)),
            3: float(min(ratio[3] ** 0.35, 16.0)),
            4: 1.0,
        }
    raise ValueError(mode)


def decision_multiplier(y_train, mode):
    ratio = prior_ratio(y_train)
    mult = np.ones(config.N_CLASSES, dtype=np.float64)
    if mode == "none":
        return mult
    if mode == "stable_prior":
        mult[1] = min(ratio[1] ** 0.25, 4.0)
        mult[2] = min(ratio[2] ** 0.03, 1.5)
        mult[3] = min(ratio[3] ** 0.20, 4.0)
    elif mode == "mild_prior":
        mult[1] = min(ratio[1] ** 0.20, 3.0)
        mult[2] = min(ratio[2] ** 0.02, 1.3)
        mult[3] = min(ratio[3] ** 0.15, 3.0)
    elif mode == "tail_prior":
        mult[0] = 0.95
        mult[1] = min(ratio[1] ** 0.28, 4.5)
        mult[2] = min(ratio[2] ** 0.05, 1.7)
        mult[3] = min(ratio[3] ** 0.25, 5.0)
    else:
        raise ValueError(mode)
    mult[4] = 1.0
    return mult


def candidate_configs():
    configs = []
    for feature_set in ["clinical_proto", "clinical_proto_lead"]:
        configs.extend(
            [
                {
                    "name": f"BioCore_legacy_stable_{feature_set}",
                    "feature_set": feature_set,
                    "class_weight_mode": "legacy_cost",
                    "decoder": "stable_prior",
                    "n_estimators": 700,
                    "max_features": 0.35,
                    "min_samples_leaf": 2,
                    "proba_power": 0.90,
                },
                {
                    "name": f"BioCore_smooth_stable_{feature_set}",
                    "feature_set": feature_set,
                    "class_weight_mode": "smooth_cost",
                    "decoder": "stable_prior",
                    "n_estimators": 700,
                    "max_features": 0.35,
                    "min_samples_leaf": 2,
                    "proba_power": 0.90,
                },
                {
                    "name": f"BioCore_tail_tail_{feature_set}",
                    "feature_set": feature_set,
                    "class_weight_mode": "tail_cost",
                    "decoder": "tail_prior",
                    "n_estimators": 700,
                    "max_features": 0.35,
                    "min_samples_leaf": 2,
                    "proba_power": 0.90,
                },
                {
                    "name": f"BioCore_legacy_mild_{feature_set}",
                    "feature_set": feature_set,
                    "class_weight_mode": "legacy_cost",
                    "decoder": "mild_prior",
                    "n_estimators": 700,
                    "max_features": 0.35,
                    "min_samples_leaf": 2,
                    "proba_power": 1.00,
                },
            ]
        )
    return configs


def fit_raw_rbfsvm(scenario, seed):
    model = make_pipeline(
        StandardScaler(),
        SVC(C=1.0, kernel="rbf", gamma="scale", probability=False, random_state=seed),
    )
    x_tr = raw_dual(scenario["b0_tr"], scenario["b1_tr"])
    x_te = raw_dual(scenario["b0_te"], scenario["b1_te"])
    t0 = time.time()
    model.fit(x_tr, scenario["y_tr"])
    train_time = time.time() - t0
    t1 = time.time()
    pred = model.predict(x_te).astype(np.int64)
    predict_time = time.time() - t1
    return pred, train_time, predict_time


def fit_predict_evidence(scenario, seed, cfg):
    x_tr, x_te = build_evidence_features(scenario, cfg["feature_set"])
    y_tr = scenario["y_tr"]
    cw = class_weight_value(y_tr, cfg["class_weight_mode"])
    model = ExtraTreesClassifier(
        n_estimators=int(cfg["n_estimators"]),
        max_features=float(cfg["max_features"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        class_weight=cw,
        bootstrap=False,
        random_state=seed,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(x_tr, y_tr)
    train_time = time.time() - t0
    t1 = time.time()
    proba = align_proba(model.classes_, model.predict_proba(x_te))
    predict_time = time.time() - t1
    p_dec = power_calibrate(proba, cfg["proba_power"])
    mult = decision_multiplier(y_tr, cfg["decoder"])
    pred_argmax = proba.argmax(axis=1).astype(np.int64)
    pred = np.argmax(p_dec * mult[None, :], axis=1).astype(np.int64)
    full_cfg = dict(cfg)
    full_cfg["class_weight"] = cw
    full_cfg["decision_multiplier"] = mult.tolist()
    full_cfg["train_counts"] = counts_dict(y_tr)
    return pred_argmax, pred, full_cfg, train_time, predict_time


def transition_text(before, after):
    before = np.asarray(before, dtype=np.int64)
    after = np.asarray(after, dtype=np.int64)
    changed = before != after
    pairs = {}
    for src, dst in zip(before[changed], after[changed]):
        key = f"{CLASS_NAMES[int(src)]}->{CLASS_NAMES[int(dst)]}"
        pairs[key] = pairs.get(key, 0) + 1
    return json.dumps(dict(sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))), ensure_ascii=False)


def add_row(rows, conf, dataset, seed, stage, cfg_name, y, pred, ref, cfg, train_time, predict_time):
    row = {
        "dataset": dataset,
        "seed": int(seed),
        "stage": stage,
        "config_name": cfg_name,
        "selection": "train-prior formula; full training split refit",
        "changed_samples": int(np.sum(np.asarray(pred) != np.asarray(ref))),
        "changed_pairs": transition_text(ref, pred),
        "config": json.dumps(cfg, ensure_ascii=False),
        "train_time": float(train_time),
        "predict_time": float(predict_time),
        **metrics_row(y, pred),
    }
    rows.append(row)
    conf[f"{dataset}|{seed}|{stage}|{cfg_name}"] = confusion_matrix(
        y, pred, labels=list(range(config.N_CLASSES))
    ).astype(int).tolist()


def summarize(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metrics = [
        "accuracy",
        "macro_f1_5",
        "macro_f1_4",
        "macro_f1_3_nsv",
        "F1_N",
        "F1_S",
        "F1_V",
        "F1_F",
        "F1_Q",
        "Se_S",
        "Se_F",
        "Pr_S",
        "Pr_F",
        "changed_samples",
        "train_time",
        "predict_time",
    ]
    out = []
    for (dataset, stage, config_name), g in df.groupby(["dataset", "stage", "config_name"]):
        row = {
            "dataset": dataset,
            "stage": stage,
            "config_name": config_name,
            "n_seeds": int(g["seed"].nunique()),
            "seeds": " ".join(str(int(s)) for s in sorted(g["seed"].unique())),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=0))
        out.append(row)
    summary = pd.DataFrame(out)
    return summary.sort_values(["dataset", "macro_f1_4_mean", "accuracy_mean"], ascending=[True, False, False])


def write_outputs(rows, conf):
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(DETAIL, index=False, encoding="utf-8")
    summary = summarize(rows)
    summary.to_csv(SUMMARY, index=False, encoding="utf-8")
    CONFUSION.write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# CEMR-ECG Bio-Evidence Core",
        "",
        "This run tests a paper-safe core framework: RR/morphology/prototype ECG evidence, cost-sensitive evidence learning, and a train-prior decision decoder. No test labels are used for configuration selection.",
        "",
        "| dataset | stage | config | n | Acc | M-F1(4) | F1_S | F1_V | F1_F | changed |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not summary.empty:
        for _, r in summary.iterrows():
            lines.append(
                f"| {r['dataset']} | {r['stage']} | {r['config_name']} | {int(r['n_seeds'])} | "
                f"{r['accuracy_mean']*100:.2f} | {r['macro_f1_4_mean']*100:.2f} | "
                f"{r['F1_S_mean']*100:.2f} | {r['F1_V_mean']*100:.2f} | {r['F1_F_mean']*100:.2f} | "
                f"{r['changed_samples_mean']:.1f} |"
            )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def load_existing(force):
    if force or not DETAIL.exists():
        return [], {}
    rows = pd.read_csv(DETAIL).to_dict("records")
    conf = json.loads(CONFUSION.read_text(encoding="utf-8")) if CONFUSION.exists() else {}
    return rows, conf


def completed(rows):
    done = set()
    for r in rows:
        done.add((str(r.get("dataset")), int(r.get("seed")), str(r.get("stage")), str(r.get("config_name"))))
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description="CEMR-ECG bio-evidence core experiments")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--configs", nargs="*", default=None, help="Optional config-name filters.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    data = load_cache()
    rows, conf = load_existing(args.force)
    done = completed(rows)
    configs = candidate_configs()
    if args.configs:
        filters = [x.lower() for x in args.configs]
        configs = [c for c in configs if any(f in c["name"].lower() for f in filters)]

    for dataset in args.datasets:
        for seed in args.seeds:
            scenario = make_scenario(data, dataset, int(seed))
            y = scenario["y_te"]
            raw_key = (dataset, int(seed), "raw", "RbfSVM_full_train")
            if raw_key not in done:
                print(f"Running {dataset} seed={seed} raw RbfSVM full train", flush=True)
                raw_pred, tr, pr = fit_raw_rbfsvm(scenario, int(seed))
                add_row(rows, conf, dataset, seed, "raw", "RbfSVM_full_train", y, raw_pred, raw_pred, {}, tr, pr)
                write_outputs(rows, conf)
                done.add(raw_key)
            else:
                raw_rows = [
                    r for r in rows
                    if str(r.get("dataset")) == dataset
                    and int(r.get("seed")) == int(seed)
                    and str(r.get("stage")) == "raw"
                    and str(r.get("config_name")) == "RbfSVM_full_train"
                ]
                raw_pred = None

            if raw_pred is None:
                # Recompute the raw reference only for transition accounting; it is not added again.
                raw_pred, _, _ = fit_raw_rbfsvm(scenario, int(seed))

            for cfg in configs:
                key = (dataset, int(seed), "CEMR-BioCore", cfg["name"])
                arg_key = (dataset, int(seed), "evidence_argmax", cfg["name"])
                if key in done and arg_key in done:
                    print(f"Skip completed {dataset} seed={seed} {cfg['name']}", flush=True)
                    continue
                print(f"Running {dataset} seed={seed} {cfg['name']}", flush=True)
                pred_argmax, pred, full_cfg, tr, pr = fit_predict_evidence(scenario, int(seed), cfg)
                if arg_key not in done:
                    add_row(rows, conf, dataset, seed, "evidence_argmax", cfg["name"], y, pred_argmax, raw_pred, full_cfg, tr, pr)
                if key not in done:
                    add_row(rows, conf, dataset, seed, "CEMR-BioCore", cfg["name"], y, pred, raw_pred, full_cfg, tr, pr)
                write_outputs(rows, conf)
                done.add(arg_key)
                done.add(key)
                m = metrics_row(y, pred)
                print(
                    f"[{dataset} seed={seed} {cfg['name']}] "
                    f"Acc={m['accuracy']*100:.2f}% M-F1(4)={m['macro_f1_4']*100:.2f}% "
                    f"F1_S={m['F1_S']*100:.2f}% F1_F={m['F1_F']*100:.2f}%",
                    flush=True,
                )

    write_outputs(rows, conf)
    if SUMMARY.exists():
        print(pd.read_csv(SUMMARY).to_string(index=False), flush=True)
    print(f"Saved {DETAIL}", flush=True)
    print(f"Saved {SUMMARY}", flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
