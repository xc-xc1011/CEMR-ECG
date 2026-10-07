import argparse
import copy
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

import config
from feature_engineering import add_proto_features
from metrics_eval import metrics_row
from record_aware_split import record_aware_train_val_split, split_audit
from run_cemr_bio_adaptive_decoder import decoder_candidates, predict_with_decoder
from run_cemr_bio_evidence_core import class_weight_value, counts_dict
from run_datasetwise_cemr_framework_experiments import (
    CLASS_NAMES,
    align_proba,
    align_scores_to_proba,
    fit_deep_with_oom_retry,
    load_cache,
    make_scenario,
    model_specs,
    raw_dual,
    train_val_split,
)
from run_external_raw_baselines import specs as ml_specs


warnings.filterwarnings("ignore")

RESULTS = Path("results")

# Internal validation protocol.
#   "beat"   -> original submission: class-stratified beat sampling, which lets
#               beats from one record appear in both the fitting and validation
#               sets (record overlap > 0).
#   "record" -> record-level holdout: no record is shared between fitting,
#               validation and test, so alpha and the decoder policy are
#               selected without subject-level leakage.
# Set by --val-protocol in main(); the default preserves prior behaviour.
VAL_PROTOCOL = "beat"

# Output stems are suffixed per protocol so both result sets can coexist.
_STEM = "datasetwise_multimethod_permethod_bioadaptive"
_STEM_RECORD = f"{_STEM}_valrec"


def output_paths() -> dict[str, Path]:
    """Return the result paths for the active validation protocol."""
    stem = _STEM_RECORD if VAL_PROTOCOL == "record" else _STEM
    if RUN_TAG:
        stem = f"{stem}_{RUN_TAG}"
    return {
        "detail": RESULTS / f"{stem}_detail.csv",
        "summary": RESULTS / f"{stem}_summary.csv",
        "report": RESULTS / f"{stem}_report.md",
        "confusion": RESULTS / f"{stem}_confusion.json",
        "evidence_cache": RESULTS / f"{stem}_evidence_cache",
        "split_audit": RESULTS / f"{stem}_split_audit.csv",
    }


# Resolved once in run() from VAL_PROTOCOL; see output_paths().
PATHS: dict[str, Path] = {}

# Optional suffix used to give concurrent runs (for example one process per
# dataset) their own result files and evidence cache.
RUN_TAG = ""


def set_val_protocol(protocol: str, run_tag: str = "") -> None:
    """Activate a validation protocol and resolve its output paths."""
    global VAL_PROTOCOL, RUN_TAG
    VAL_PROTOCOL = protocol
    RUN_TAG = run_tag or ""
    PATHS.update(output_paths())


DATASETS = ["MIT-BIH", "INCART", "SVDB"]
SEEDS = [303, 1303, 2303, 3303, 4303]
EPS = 1e-12

BASE_ENCODER = {
    "n_estimators": 700,
    "max_features": 0.35,
    "min_samples_leaf": 2,
    "class_weight_mode": "legacy_cost",
}

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


def counts_text(y):
    return json.dumps(counts_dict(y), ensure_ascii=False)


def model_classes(model):
    if hasattr(model, "classes_"):
        return model.classes_
    if hasattr(model, "named_steps"):
        for step in reversed(model.named_steps.values()):
            if hasattr(step, "classes_"):
                return step.classes_
    raise AttributeError("Fitted model does not expose classes_.")


def sklearn_proba(model, x):
    if hasattr(model, "predict_proba"):
        return align_proba(model_classes(model), model.predict_proba(x))
    if hasattr(model, "decision_function"):
        return align_scores_to_proba(model_classes(model), model.decision_function(x))
    pred = np.asarray(model.predict(x)).astype(np.int64)
    out = np.full((len(pred), config.N_CLASSES), 1e-6, dtype=np.float64)
    out[np.arange(len(pred)), pred] = 1.0
    return out / out.sum(axis=1, keepdims=True)


def safe_log(p):
    return np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0))


def fuse_backbone_evidence(p_backbone, p_evidence, alpha):
    score = (1.0 - float(alpha)) * safe_log(p_backbone) + float(alpha) * safe_log(p_evidence)
    score = score - score.max(axis=1, keepdims=True)
    exp = np.exp(score)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)


def build_features_for_fit_target(scenario, fit_idx, target_idx):
    return add_proto_features(
        scenario["x_tr_base"][fit_idx],
        scenario["y_tr"][fit_idx],
        scenario["x_tr_base"][target_idx],
        scenario["b0_tr"][fit_idx],
        scenario["b1_tr"][fit_idx],
        scenario["b0_tr"][target_idx],
        scenario["b1_tr"][target_idx],
    )


def build_features_for_test(scenario, fit_idx):
    return add_proto_features(
        scenario["x_tr_base"][fit_idx],
        scenario["y_tr"][fit_idx],
        scenario["x_te_base"],
        scenario["b0_tr"][fit_idx],
        scenario["b1_tr"][fit_idx],
        scenario["b0_te"],
        scenario["b1_te"],
    )


def fit_encoder_predict(x_fit, y_fit, x_target, seed):
    model = ExtraTreesClassifier(
        n_estimators=int(BASE_ENCODER["n_estimators"]),
        max_features=float(BASE_ENCODER["max_features"]),
        min_samples_leaf=int(BASE_ENCODER["min_samples_leaf"]),
        class_weight=class_weight_value(y_fit, BASE_ENCODER["class_weight_mode"]),
        bootstrap=False,
        random_state=int(seed),
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(x_fit, y_fit)
    train_time = time.time() - t0
    t1 = time.time()
    proba = align_proba(model.classes_, model.predict_proba(x_target))
    predict_time = time.time() - t1
    return proba, train_time, predict_time


def make_fit_val_split(scenario, seed):
    """Build the fitting/validation indices under the active validation protocol.

    ``beat``   reproduces the original beat-level stratified sampling, which
               permits records to be shared between fitting and validation.
    ``record`` holds out whole records from the training pool so that fitting
               and validation never share a subject.
    """
    if VAL_PROTOCOL == "record":
        return record_aware_train_val_split(scenario["y_tr"], scenario["pid_tr"], seed)
    return train_val_split(scenario["y_tr"], seed)


def load_or_build_evidence(data, dataset, seed, force=False):
    evidence_cache = PATHS["evidence_cache"]
    evidence_cache.mkdir(parents=True, exist_ok=True)
    cache_path = evidence_cache / f"{dataset}_{int(seed)}_evidence.npz"
    meta_path = evidence_cache / f"{dataset}_{int(seed)}_evidence_meta.json"
    scenario = make_scenario(data, dataset, seed)
    train_idx, val_idx = make_fit_val_split(scenario, seed)

    # The evidence encoder depends only on (dataset, seed) under a given
    # protocol, so a run-tagged job may read the untagged cache instead of
    # refitting 700 trees twice per seed. Writes always go to this run's own
    # directory, so a concurrent untagged job is never disturbed.
    read_path, read_meta = cache_path, meta_path
    if not read_path.exists() and RUN_TAG and VAL_PROTOCOL == "record":
        shared_cache = RESULTS / f"{_STEM_RECORD}_evidence_cache"
        candidate_npz = shared_cache / f"{dataset}_{int(seed)}_evidence.npz"
        candidate_meta = shared_cache / f"{dataset}_{int(seed)}_evidence_meta.json"
        if candidate_npz.exists() and candidate_meta.exists():
            print(
                f"Reusing shared evidence cache {candidate_npz.name} for tag '{RUN_TAG}'",
                flush=True,
            )
            read_path, read_meta = candidate_npz, candidate_meta

    if read_path.exists() and read_meta.exists() and not force:
        cached = np.load(read_path, allow_pickle=True)
        meta = json.loads(read_meta.read_text(encoding="utf-8"))
        return scenario, train_idx, val_idx, {
            "p_evidence_val": cached["p_evidence_val"].astype(np.float64),
            "p_evidence_test": cached["p_evidence_test"].astype(np.float64),
            "evidence_train_time": float(meta.get("evidence_train_time", 0.0)),
            "evidence_predict_time": float(meta.get("evidence_predict_time", 0.0)),
            "cache_path": str(read_path),
        }

    y_fit = scenario["y_tr"][train_idx]
    x_fit, x_val = build_features_for_fit_target(scenario, train_idx, val_idx)
    p_val, val_train_time, val_predict_time = fit_encoder_predict(x_fit, y_fit, x_val, seed)
    x_fit_full, x_test = build_features_for_test(scenario, train_idx)
    p_test, test_train_time, test_predict_time = fit_encoder_predict(x_fit_full, y_fit, x_test, seed)
    meta = {
        "dataset": dataset,
        "seed": int(seed),
        "evidence_train_time": float(val_train_time + test_train_time),
        "evidence_predict_time": float(val_predict_time + test_predict_time),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(scenario["y_te"])),
    }
    np.savez_compressed(cache_path, p_evidence_val=p_val, p_evidence_test=p_test)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return scenario, train_idx, val_idx, {
        "p_evidence_val": p_val,
        "p_evidence_test": p_test,
        "evidence_train_time": meta["evidence_train_time"],
        "evidence_predict_time": meta["evidence_predict_time"],
        "cache_path": str(cache_path),
    }


def compact_validation_table(candidates):
    out = []
    for item in sorted(candidates, key=lambda x: x["key"], reverse=True)[:8]:
        d = item["decoder"]
        m = item["metrics"]
        out.append(
            {
                "alpha": float(item["alpha"]),
                "decoder": d["decoder"],
                "power": float(d["power"]),
                "multiplier": [float(x) for x in d["multiplier"]],
                "val_macro_f1_4": float(m["macro_f1_4"]),
                "val_accuracy": float(m["accuracy"]),
                "val_F1_S": float(m["F1_S"]),
                "val_F1_F": float(m["F1_F"]),
                "pred_counts": item["pred_counts"],
                "val_counts": item["val_counts"],
            }
        )
    return out


def select_adapter(dataset, p_backbone_val, p_evidence_val, y_val, y_fit):
    candidates = []
    # Keep every CEMR run method-specific. alpha=1.0 collapses to evidence-only
    # and can make different backbones report identical +CEMR values, which is
    # not valid evidence for a plug-in framework.
    for alpha in [0.20, 0.35, 0.50, 0.65, 0.80]:
        p_fused = fuse_backbone_evidence(p_backbone_val, p_evidence_val, alpha)
        for decoder in decoder_candidates(y_fit):
            pred = predict_with_decoder(p_fused, decoder)
            metrics = metrics_row(y_val, pred)
            pred_counts = np.bincount(pred, minlength=config.N_CLASSES).astype(np.float64)
            val_counts = np.bincount(y_val, minlength=config.N_CLASSES).astype(np.float64)
            absent_penalty = float(np.sum((val_counts[:4] == 0) * pred_counts[:4])) / max(1.0, len(y_val))
            key = (
                float(metrics["macro_f1_4"]) - 0.25 * absent_penalty,
                float(metrics["F1_S"]) + float(metrics["F1_F"]),
                float(metrics["accuracy"]),
            )
            candidates.append(
                {
                    "alpha": float(alpha),
                    "decoder": decoder,
                    "metrics": metrics,
                    "key": key,
                    "pred_counts": pred_counts.astype(int).tolist(),
                    "val_counts": val_counts.astype(int).tolist(),
                }
            )
    selected = max(candidates, key=lambda x: x["key"])
    if dataset == "MIT-BIH":
        # Inter-patient DS1->DS2 transfer has a strong validation-to-test shift.
        # Keep the same transparent prevalence decoder but let alpha remain
        # validation-selected for each backbone.
        clinical = [d for d in decoder_candidates(y_fit) if d["decoder"] == "clinical_default"][0]
        selected = dict(selected)
        selected["decoder"] = clinical
        selected["domain_shift_policy"] = "MIT-BIH uses clinical_default decoder with method-specific validation alpha"
    return selected, candidates


def fit_ml_backbone(scenario, spec, train_idx, val_idx, seed):
    x_all = raw_dual(scenario["b0_tr"], scenario["b1_tr"])
    x_test = raw_dual(scenario["b0_te"], scenario["b1_te"])
    model = clone(spec["estimator"])
    t0 = time.time()
    model.fit(x_all[train_idx], scenario["y_tr"][train_idx])
    train_time = time.time() - t0
    t1 = time.time()
    p_val = sklearn_proba(model, x_all[val_idx])
    p_test = sklearn_proba(model, x_test)
    predict_time = time.time() - t1
    return {
        "proba_val": p_val,
        "proba_test": p_test,
        "train_time": float(train_time),
        "predict_time": float(predict_time),
        "epochs_run": 0,
        "best_val_macro_f1_4": np.nan,
        "batch_size": 0,
        "input_mode": "dual_flat",
        "backbone_config": spec["config"],
        "method": spec["model"],
        "family": "raw_machine_learning",
        "feature_set": "raw_dual_wave_flattened",
    }


def fit_deep_backbone(scenario, spec, train_idx, val_idx, seed, device, smoke=False):
    result = fit_deep_with_oom_retry(spec, scenario, train_idx, val_idx, seed, device, smoke=smoke)
    return {
        "proba_val": result["proba_val"],
        "proba_test": result["proba_test"],
        "train_time": float(result["train_time"]),
        "predict_time": float(result["predict_time"]),
        "epochs_run": int(result["epochs_run"]),
        "best_val_macro_f1_4": float(result["best_val_macro_f1_4"]),
        "batch_size": int(result["batch_size"]),
        "input_mode": result["input_mode"],
        "backbone_config": result["backbone_config"],
        "method": spec.name,
        "family": "raw_deep_or_time_series",
        "feature_set": "raw_waveform",
    }


def run_one(data, dataset, method_spec, family_kind, seed, device, smoke=False, evidence_force=False):
    scenario, train_idx, val_idx, evidence = load_or_build_evidence(data, dataset, seed, force=evidence_force)
    if family_kind == "ml":
        backbone = fit_ml_backbone(scenario, method_spec, train_idx, val_idx, seed)
    else:
        backbone = fit_deep_backbone(scenario, method_spec, train_idx, val_idx, seed, device, smoke=smoke)

    y_fit = scenario["y_tr"][train_idx]
    y_val = scenario["y_tr"][val_idx]
    p_evidence_val = evidence["p_evidence_val"]
    selected, candidates = select_adapter(dataset, backbone["proba_val"], p_evidence_val, y_val, y_fit)

    p_evidence_test = evidence["p_evidence_test"]
    p_fused_test = fuse_backbone_evidence(backbone["proba_test"], p_evidence_test, selected["alpha"])
    pred_raw = backbone["proba_test"].argmax(axis=1).astype(np.int64)
    pred_cemr = predict_with_decoder(p_fused_test, selected["decoder"])
    raw_metrics = metrics_row(scenario["y_te"], pred_raw)
    cemr_metrics = metrics_row(scenario["y_te"], pred_cemr)

    row = {
        "dataset": dataset,
        "protocol": scenario["protocol"],
        "family": backbone["family"],
        "method": backbone["method"],
        "framework_method": f"{backbone['method']}+CEMR-BioAdaptive",
        "seed": int(seed),
        "train_data": scenario["train_record_policy"],
        "test_data": scenario["test_record_policy"],
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(scenario["y_te"])),
        "train_counts": counts_text(y_fit),
        "val_counts": counts_text(y_val),
        "test_counts": counts_text(scenario["y_te"]),
        "feature_set": backbone["feature_set"],
        "input_mode": backbone["input_mode"],
        "source": "method-specific raw probability interface plus CEMR-BioAdaptive",
        "model_config": backbone["backbone_config"],
        "selected_alpha": float(selected["alpha"]),
        "selected_decoder": selected["decoder"]["decoder"],
        "selected_power": float(selected["decoder"]["power"]),
        "selected_multiplier": json.dumps([float(x) for x in selected["decoder"]["multiplier"]], ensure_ascii=False),
        "selected_rationale": selected["decoder"].get("rationale", ""),
        "domain_shift_policy": selected.get("domain_shift_policy", ""),
        "validation_table": json.dumps(compact_validation_table(candidates), ensure_ascii=False),
        "evidence_cache": evidence["cache_path"],
        "batch_size": int(backbone["batch_size"]),
        "epochs_run": int(backbone["epochs_run"]),
        "best_val_macro_f1_4": backbone["best_val_macro_f1_4"],
        "backbone_train_time": float(backbone["train_time"]),
        "evidence_train_time": float(evidence["evidence_train_time"]),
        "train_time": float(backbone["train_time"] + evidence["evidence_train_time"]),
        "backbone_predict_time": float(backbone["predict_time"]),
        "evidence_predict_time": float(evidence["evidence_predict_time"]),
        "predict_time": float(backbone["predict_time"] + evidence["evidence_predict_time"]),
    }
    row.update(cemr_metrics)
    for k, v in raw_metrics.items():
        row[f"raw_{k}"] = float(v)
        row[f"delta_{k}"] = float(row[k] - v)

    cm_raw = confusion_matrix(scenario["y_te"], pred_raw, labels=list(range(config.N_CLASSES))).astype(int).tolist()
    cm_cemr = confusion_matrix(scenario["y_te"], pred_cemr, labels=list(range(config.N_CLASSES))).astype(int).tolist()
    return row, cm_raw, cm_cemr


def completed_keys():
    detail = PATHS["detail"]
    if not detail.exists():
        return set()
    df = pd.read_csv(detail)
    return set(zip(df["dataset"].astype(str), df["family"].astype(str), df["method"].astype(str), df["seed"].astype(int)))


def load_confusion():
    confusion = PATHS["confusion"]
    if confusion.exists():
        return json.loads(confusion.read_text(encoding="utf-8"))
    return {}


def append_detail(row):
    detail = PATHS["detail"]
    pd.DataFrame([row]).to_csv(detail, mode="a", header=not detail.exists(), index=False, encoding="utf-8")


def summarize():
    detail_path = PATHS["detail"]
    if not detail_path.exists():
        return pd.DataFrame()
    detail = pd.read_csv(detail_path)
    metrics = [
        *METRIC_COLUMNS,
        *[f"raw_{m}" for m in METRIC_COLUMNS],
        *[f"delta_{m}" for m in METRIC_COLUMNS],
        "train_time",
        "predict_time",
        "backbone_train_time",
        "evidence_train_time",
        "best_val_macro_f1_4",
        "epochs_run",
        "train_size",
        "val_size",
        "test_size",
    ]
    rows = []
    for (dataset, family, method), g in detail.groupby(["dataset", "family", "method"], dropna=False):
        row = {
            "dataset": dataset,
            "family": family,
            "method": method,
            "framework_method": f"{method}+CEMR-BioAdaptive",
            "n_seeds": int(g["seed"].nunique()),
            "seeds": " ".join(str(int(s)) for s in sorted(g["seed"].unique())),
            "protocol": str(g["protocol"].iloc[0]),
            "selected_alpha_values": " ".join(str(x) for x in sorted(g["selected_alpha"].unique())),
            "selected_decoder_values": " ".join(str(x) for x in sorted(g["selected_decoder"].unique())),
            "source": "valid method-specific CEMR run; +CEMR values are not shared across methods",
        }
        for metric in metrics:
            if metric in g.columns:
                row[f"{metric}_mean"] = float(g[metric].mean())
                row[f"{metric}_std"] = float(g[metric].std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["dataset", "macro_f1_4_mean", "delta_macro_f1_4_mean", "accuracy_mean"],
        ascending=[True, False, False, False],
    )
    summary.to_csv(PATHS["summary"], index=False, encoding="utf-8")
    return summary


def fmt_pct(x):
    return f"{100.0 * float(x):.2f}" if pd.notna(x) else ""


def write_report():
    summary = summarize()
    lines = [
        "# Dataset-wise Method-specific CEMR-BioAdaptive Framework",
        "",
        "Each row is a true method-specific run: the raw model is trained for the given dataset and seed, its own validation/test probabilities are fused with CEMR evidence, and the BioAdaptive adapter is selected on that method's validation split. This file replaces the deprecated shared-output diagnostic table.",
        "",
    ]
    if summary.empty:
        lines.append("_No completed runs._")
    else:
        for dataset, g in summary.groupby("dataset", sort=False):
            lines += [
                f"## {dataset}",
                "",
                "| Rank | Method | Family | Seeds | Raw Acc | +CEMR Acc | Delta Acc | Raw M-F1(4) | +CEMR M-F1(4) | Delta M-F1(4) | Raw F1_S | +CEMR F1_S | Raw F1_F | +CEMR F1_F |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            ranked = g.sort_values(["macro_f1_4_mean", "delta_macro_f1_4_mean"], ascending=False)
            for i, (_, r) in enumerate(ranked.iterrows(), 1):
                lines.append(
                    f"| {i} | {r['method']} | {r['family']} | {int(r['n_seeds'])} | "
                    f"{fmt_pct(r['raw_accuracy_mean'])} | {fmt_pct(r['accuracy_mean'])} | {fmt_pct(r['delta_accuracy_mean'])} | "
                    f"{fmt_pct(r['raw_macro_f1_4_mean'])} | {fmt_pct(r['macro_f1_4_mean'])} | {fmt_pct(r['delta_macro_f1_4_mean'])} | "
                    f"{fmt_pct(r['raw_F1_S_mean'])} | {fmt_pct(r['F1_S_mean'])} | "
                    f"{fmt_pct(r['raw_F1_F_mean'])} | {fmt_pct(r['F1_F_mean'])} |"
                )
            lines.append("")
    PATHS["report"].write_text("\n".join(lines), encoding="utf-8")


def method_allowed(name, wanted):
    return not wanted or name.lower() in wanted


def run(args):
    RESULTS.mkdir(exist_ok=True)
    set_val_protocol(args.val_protocol, getattr(args, "run_tag", ""))
    print(f"Validation protocol: {VAL_PROTOCOL}", flush=True)
    print(f"Outputs: {PATHS['summary'].name}", flush=True)
    if args.force:
        for key in ["detail", "summary", "report", "confusion"]:
            path = PATHS[key]
            if path.exists():
                path.unlink()

    data = load_cache()
    done = completed_keys()
    conf = load_confusion()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    families = set(args.families or ["ml", "deep"])
    wanted = {m.lower() for m in (args.methods or [])}

    specs = []
    if "ml" in families:
        for seed in args.seeds:
            for spec in ml_specs(int(seed)):
                if method_allowed(spec["model"], wanted):
                    specs.append(("ml", int(seed), spec["model"], spec))
    if "deep" in families:
        for spec in model_specs():
            if method_allowed(spec.name, wanted):
                for seed in args.seeds:
                    specs.append(("deep", int(seed), spec.name, spec))

    for dataset in args.datasets:
        for family_kind, seed, method_name, spec in specs:
            family = "raw_machine_learning" if family_kind == "ml" else "raw_deep_or_time_series"
            key = (dataset, family, method_name, int(seed))
            if key in done and not args.force:
                print(f"Skip completed {dataset} {method_name} seed={seed}", flush=True)
                continue
            print(f"Running {dataset} {method_name} seed={seed} method-specific CEMR", flush=True)
            row, cm_raw, cm_cemr = run_one(
                data,
                dataset,
                copy.deepcopy(spec),
                family_kind,
                int(seed),
                device,
                smoke=args.smoke,
                evidence_force=args.force_evidence,
            )
            if args.smoke:
                print(
                    f"[SMOKE {dataset} {method_name} seed={seed}] raw={row['raw_macro_f1_4']*100:.2f} "
                    f"CEMR={row['macro_f1_4']*100:.2f}",
                    flush=True,
                )
                continue
            append_detail(row)
            conf[f"{dataset}|{row['family']}|{method_name}|{seed}|raw"] = cm_raw
            conf[f"{dataset}|{row['family']}|{method_name}|{seed}|CEMR-BioAdaptive"] = cm_cemr
            PATHS["confusion"].write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")
            done.add(key)
            write_report()
            print(
                f"[{dataset} {method_name} seed={seed}] raw={row['raw_macro_f1_4']*100:.2f} "
                f"CEMR={row['macro_f1_4']*100:.2f} delta={row['delta_macro_f1_4']*100:.2f} "
                f"Acc={row['accuracy']*100:.2f}",
                flush=True,
            )
    write_report()
    print(f"Saved {PATHS['detail']}", flush=True)
    print(f"Saved {PATHS['summary']}", flush=True)
    print(f"Saved {PATHS['report']}", flush=True)


def write_split_audit(args):
    """Record the fitting/validation/test record overlap for every dataset-seed pair."""
    set_val_protocol(args.val_protocol, getattr(args, "run_tag", ""))

    data = load_cache()
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            scenario = make_scenario(data, dataset, int(seed))
            fit_idx, val_idx = make_fit_val_split(scenario, int(seed))
            row = split_audit(
                scenario["y_tr"],
                scenario["pid_tr"],
                fit_idx,
                val_idx,
                dataset,
                int(seed),
                test_pid=scenario["pid_te"],
            )
            row["val_protocol"] = VAL_PROTOCOL
            rows.append(row)
            print(
                f"{dataset:<8} seed={seed:<5} fit={row['fit_size']:>6} val={row['val_size']:>5} "
                f"valF={row['val_counts']['F']:>3} fit_val_overlap={row['fit_val_record_overlap']} "
                f"fit_test_overlap={row.get('fit_test_record_overlap')} | {row['audit_interpretation']}",
                flush=True,
            )

    out = PATHS["split_audit"]
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved {out}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="True method-specific CEMR-BioAdaptive experiments.")
    parser.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS)
    parser.add_argument("--families", nargs="*", default=None, choices=["ml", "deep"])
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-evidence", action="store_true", help="Rebuild cached CEMR evidence probabilities.")
    parser.add_argument(
        "--val-protocol",
        choices=["beat", "record"],
        default="beat",
        help="Internal validation splitting. 'record' removes subject-level leakage "
        "between the fitting and validation sets.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Suffix for result files and the evidence cache, so concurrent "
        "runs (for example one per dataset) do not overwrite each other.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rebuild-report", action="store_true")
    parser.add_argument(
        "--write-split-audit",
        action="store_true",
        help="Write the fitting/validation/test record-overlap audit and exit.",
    )
    args = parser.parse_args(argv)
    if args.rebuild_report:
        # write_report() needs PATHS resolved for the requested protocol.
        set_val_protocol(args.val_protocol, getattr(args, "run_tag", ""))
        write_report()
        return
    if args.write_split_audit:
        write_split_audit(args)
        return
    run(args)


if __name__ == "__main__":
    main()
