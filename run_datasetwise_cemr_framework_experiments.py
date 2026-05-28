import argparse
import copy
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

import config
from feature_engineering import add_proto_features
from metrics_eval import classification_metrics, metrics_row
from optimize_external_datasets import cap_augmented_training, split_by_record
from run_modern_deep_baselines import (
    BeatDataset,
    forward_with_aux,
    make_loader,
    make_loss,
    make_optimizer,
    model_specs,
    normalize_from_train,
    set_seed,
    smote_tomek_waveforms,
)


warnings.filterwarnings("ignore")

RESULTS = Path("results")
CACHE = RESULTS / "external_base_features_cache.npz"
DETAIL = RESULTS / "datasetwise_cemr_framework_detail.csv"
SUMMARY = RESULTS / "datasetwise_cemr_framework_summary.csv"
REPORT = RESULTS / "datasetwise_cemr_framework_report.md"
CONFUSION = RESULTS / "datasetwise_cemr_framework_confusion.json"

SEEDS = [303, 1303, 2303, 3303, 4303]
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
BACKBONES = ["RbfSVM", "CAT-Net", "TimeMixer"]


def raw_dual(b0, b1):
    return np.concatenate([b0, b1], axis=1).astype(np.float32)


def counts_text(y):
    return json.dumps(
        {name: int(v) for name, v in zip(CLASS_NAMES, np.bincount(y, minlength=config.N_CLASSES))},
        ensure_ascii=False,
    )


def train_val_split(y, seed, val_fraction=0.15):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    y = np.asarray(y, dtype=np.int64)
    for cls in range(config.N_CLASSES):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        if len(idx) <= 2:
            train_idx.extend(idx.tolist())
            continue
        n_val = max(1, int(round(len(idx) * val_fraction)))
        n_val = min(n_val, len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    if not val_idx:
        all_idx = np.arange(len(y))
        rng.shuffle(all_idx)
        n_val = max(1, int(round(len(y) * val_fraction)))
        val_idx = all_idx[:n_val].tolist()
        train_idx = all_idx[n_val:].tolist()
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def load_cache():
    if not CACHE.exists():
        raise FileNotFoundError(f"Missing cache: {CACHE}")
    return np.load(CACHE, allow_pickle=True)


def make_scenario(data, dataset, seed):
    if dataset == "MIT-BIH":
        return {
            "dataset": dataset,
            "protocol": "MIT-BIH DS1->DS2",
            "train_record_policy": "fixed DS1 records",
            "test_record_policy": "fixed DS2 records",
            "x_tr_base": data["mit_tr_x"].astype(np.float32),
            "y_tr": data["mit_tr_y"].astype(np.int64),
            "pid_tr": data["mit_tr_pid"],
            "b0_tr": data["mit_tr_b0"].astype(np.float32),
            "b1_tr": data["mit_tr_b1"].astype(np.float32),
            "x_te_base": data["mit_te_x"].astype(np.float32),
            "y_te": data["mit_te_y"].astype(np.int64),
            "pid_te": data["mit_te_pid"],
            "b0_te": data["mit_te_b0"].astype(np.float32),
            "b1_te": data["mit_te_b1"].astype(np.float32),
        }

    if dataset == "INCART":
        prefix = "inc"
        full_name = "INCART"
    elif dataset == "SVDB":
        prefix = "sv"
        full_name = "SVDB"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    y = data[f"{prefix}_y"].astype(np.int64)
    pid = data[f"{prefix}_pid"]
    train_mask, test_mask = split_by_record(y, pid, seed)
    x_tr = data[f"{prefix}_x"][train_mask].astype(np.float32)
    y_tr = y[train_mask]
    pid_tr = pid[train_mask]
    b0_tr = data[f"{prefix}_b0"][train_mask].astype(np.float32)
    b1_tr = data[f"{prefix}_b1"][train_mask].astype(np.float32)
    x_tr, y_tr, b0_tr, b1_tr, pid_tr = cap_augmented_training(
        x_tr, y_tr, b0_tr, b1_tr, pid_tr, seed=seed, n_cap=50000
    )
    return {
        "dataset": dataset,
        "protocol": f"{full_name} record-level train/test split",
        "train_record_policy": "seed-specific 67% record split; N capped at 50000 for tractable training",
        "test_record_policy": "held-out records from the same dataset",
        "x_tr_base": x_tr,
        "y_tr": y_tr,
        "pid_tr": pid_tr,
        "b0_tr": b0_tr,
        "b1_tr": b1_tr,
        "x_te_base": data[f"{prefix}_x"][test_mask].astype(np.float32),
        "y_te": y[test_mask],
        "pid_te": pid[test_mask],
        "b0_te": data[f"{prefix}_b0"][test_mask].astype(np.float32),
        "b1_te": data[f"{prefix}_b1"][test_mask].astype(np.float32),
    }


def align_proba(classes, proba):
    out = np.zeros((proba.shape[0], config.N_CLASSES), dtype=np.float64)
    for i, cls in enumerate(classes):
        out[:, int(cls)] = proba[:, i]
    row_sum = out.sum(axis=1, keepdims=True)
    missing = row_sum.squeeze(1) <= 0
    if np.any(missing):
        out[missing] = 1.0 / config.N_CLASSES
        row_sum = out.sum(axis=1, keepdims=True)
    return out / np.maximum(row_sum, 1e-12)


def align_scores_to_proba(classes, scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    out = np.full((scores.shape[0], config.N_CLASSES), -1e6, dtype=np.float64)
    for i, cls in enumerate(classes):
        out[:, int(cls)] = scores[:, i]
    out = out - np.max(out, axis=1, keepdims=True)
    exp = np.exp(out)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def fit_rbfsvm_backbone(scenario, train_idx, val_idx, seed):
    x_raw = raw_dual(scenario["b0_tr"], scenario["b1_tr"])
    x_test = raw_dual(scenario["b0_te"], scenario["b1_te"])
    model = make_pipeline(
        StandardScaler(),
        SVC(C=1.0, kernel="rbf", gamma="scale", probability=False, random_state=seed),
    )
    start = time.time()
    model.fit(x_raw[train_idx], scenario["y_tr"][train_idx])
    train_time = time.time() - start
    pred_start = time.time()
    svc = model.named_steps["svc"]
    p_val = align_scores_to_proba(svc.classes_, model.decision_function(x_raw[val_idx]))
    p_test = align_scores_to_proba(svc.classes_, model.decision_function(x_test))
    predict_time = time.time() - pred_start
    return {
        "proba_val": p_val,
        "proba_test": p_test,
        "train_time": train_time,
        "predict_time": predict_time,
        "epochs_run": 0,
        "best_val_macro_f1_4": np.nan,
        "batch_size": 0,
        "input_mode": "raw_dual_waveform",
        "backbone_config": "StandardScaler + SVC(C=1.0, kernel=rbf, gamma=scale); decision_function softmax for CEMR probability interface.",
    }


def deep_input_arrays(scenario, input_mode):
    if input_mode == "single":
        x_all = scenario["b0_tr"][:, None, :].astype(np.float32)
        x_test = scenario["b0_te"][:, None, :].astype(np.float32)
    elif input_mode == "dual":
        x_all = np.stack([scenario["b0_tr"], scenario["b1_tr"]], axis=1).astype(np.float32)
        x_test = np.stack([scenario["b0_te"], scenario["b1_te"]], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    return x_all, x_test


@torch.no_grad()
def predict_proba_deep(model, x, y, batch_size, device):
    pin_memory = os.environ.get("CEMR_PIN_MEMORY", "0") == "1"
    loader = DataLoader(
        BeatDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    model.eval()
    probas = []
    targets = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probas.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(targets), np.concatenate(probas).astype(np.float64)


def fit_deep_backbone(spec, scenario, train_idx, val_idx, seed, device, smoke=False, batch_size_override=None):
    set_seed(seed)
    x_all, x_test = deep_input_arrays(scenario, spec.input_mode)
    y_all = scenario["y_tr"]
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    x_train, x_val, x_test = normalize_from_train(x_train, x_val, x_test)

    used_resampling = ""
    if spec.resample_mode == "smote_tomek":
        x_res, y_res, ok = smote_tomek_waveforms(x_train, y_train, seed)
        if ok:
            x_train, y_train = x_res, y_res
            used_resampling = "smote_tomek"
        else:
            used_resampling = "sampler_fallback"

    batch_size = batch_size_override or spec.batch_size
    epochs = 1 if smoke else spec.epochs
    patience = 1 if smoke else spec.patience
    train_loader = make_loader(
        x_train,
        y_train,
        batch_size,
        train=True,
        sampler_mode=spec.sampler_mode,
        augment=spec.augment,
    )
    val_loader = make_loader(x_val, y_val, batch_size, train=False)

    model = spec.make_model().to(device)
    loss_fn = make_loss(spec, y_train, device)
    optimizer = make_optimizer(spec, model)
    use_amp = os.environ.get("CEMR_USE_AMP", "0") == "1"
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and use_amp)

    best_state = None
    best_val = -1.0
    best_epoch = 0
    bad_epochs = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda" and use_amp):
                logits, aux = forward_with_aux(model, xb, spec)
                loss = loss_fn(logits, yb)
                if spec.ib_beta:
                    loss = loss + spec.ib_beta * aux["ib_kl"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(yb)

        y_val_true, p_val_epoch = predict_proba_deep(model, x_val, y_val, batch_size, device)
        val_pred = p_val_epoch.argmax(axis=1)
        val_score = classification_metrics(y_val_true, val_pred)["macro_f1_4"]
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"[{scenario['dataset']} {spec.name} seed={seed}] epoch={epoch:03d} "
            f"loss={total_loss / max(1, len(y_train)):.4f} val_MF1(4)={val_score:.4f} best={best_val:.4f}",
            flush=True,
        )
        if bad_epochs >= patience:
            break

    train_time = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    pred_start = time.time()
    _, p_val = predict_proba_deep(model, x_val, y_val, batch_size, device)
    _, p_test = predict_proba_deep(model, x_test, scenario["y_te"], batch_size, device)
    predict_time = time.time() - pred_start

    return {
        "proba_val": p_val,
        "proba_test": p_test,
        "train_time": train_time,
        "predict_time": predict_time,
        "epochs_run": best_epoch,
        "best_val_macro_f1_4": best_val,
        "batch_size": batch_size,
        "input_mode": spec.input_mode,
        "backbone_config": f"{spec.params} resample={used_resampling or spec.resample_mode or ''}".strip(),
    }


def fit_deep_with_oom_retry(spec, scenario, train_idx, val_idx, seed, device, smoke=False):
    batch_size = int(os.environ.get("CEMR_BATCH_SIZE_OVERRIDE", spec.batch_size))
    while True:
        try:
            return fit_deep_backbone(
                spec,
                scenario,
                train_idx,
                val_idx,
                seed,
                device,
                smoke=smoke,
                batch_size_override=batch_size,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg and device.type == "cuda" and batch_size > 16:
                torch.cuda.empty_cache()
                batch_size = max(16, batch_size // 2)
                print(f"[{scenario['dataset']} {spec.name} seed={seed}] CUDA OOM; retry batch_size={batch_size}", flush=True)
                continue
            raise


def build_cemr_features(scenario, train_idx, val_idx):
    x_tr = scenario["x_tr_base"][train_idx]
    y_tr = scenario["y_tr"][train_idx]
    b0_tr = scenario["b0_tr"][train_idx]
    b1_tr = scenario["b1_tr"][train_idx]
    x_val = scenario["x_tr_base"][val_idx]
    b0_val = scenario["b0_tr"][val_idx]
    b1_val = scenario["b1_tr"][val_idx]
    x_te = scenario["x_te_base"]
    b0_te = scenario["b0_te"]
    b1_te = scenario["b1_te"]

    x_ev_tr, x_ev_val = add_proto_features(x_tr, y_tr, x_val, b0_tr, b1_tr, b0_val, b1_val)
    _, x_ev_te = add_proto_features(x_tr, y_tr, x_te, b0_tr, b1_tr, b0_te, b1_te)
    return x_ev_tr, x_ev_val, x_ev_te


def fit_evidence_encoder(scenario, train_idx, val_idx, seed):
    x_ev_tr, x_ev_val, x_ev_te = build_cemr_features(scenario, train_idx, val_idx)
    model = ExtraTreesClassifier(
        n_estimators=500,
        max_features=0.35,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    start = time.time()
    model.fit(x_ev_tr, scenario["y_tr"][train_idx])
    train_time = time.time() - start
    pred_start = time.time()
    p_val = align_proba(model.classes_, model.predict_proba(x_ev_val))
    p_test = align_proba(model.classes_, model.predict_proba(x_ev_te))
    predict_time = time.time() - pred_start
    return p_val, p_test, train_time, predict_time


def safe_log(p):
    return np.log(np.clip(p, 1e-8, 1.0))


def cemr_predict(p_backbone, p_evidence, cfg):
    log_b = safe_log(p_backbone)
    log_e = safe_log(p_evidence)
    scores = (1.0 - cfg["alpha"]) * log_b + cfg["alpha"] * log_e + np.asarray(cfg["bias"], dtype=np.float64)[None, :]
    if cfg["boundary_eta"] > 0.0:
        sorted_scores = np.sort(scores, axis=1)
        margin = sorted_scores[:, -1] - sorted_scores[:, -2]
        uncertainty = np.exp(-margin / max(cfg["boundary_tau"], 1e-6))
        evidence_centered = log_e - log_e.mean(axis=1, keepdims=True)
        scores = scores + cfg["boundary_eta"] * uncertainty[:, None] * evidence_centered
    return scores.argmax(axis=1).astype(np.int64)


def prior_bias(y_train, strength):
    counts = np.bincount(y_train, minlength=config.N_CLASSES).astype(np.float64) + 1.0
    prior = counts / counts.sum()
    target = np.ones(config.N_CLASSES, dtype=np.float64) / config.N_CLASSES
    bias = strength * np.log(target / prior)
    return bias - bias.mean()


def fit_cemr_adapter(p_backbone_val, p_evidence_val, y_val, y_train):
    best = None
    start = time.time()
    configs = []
    for alpha in [0.25, 0.50, 0.75]:
        for prior_strength in [0.0, 0.25, 0.50, 0.75, 1.0]:
            bias = prior_bias(y_train, prior_strength)
            configs.append({
                "alpha": alpha,
                "prior_strength": prior_strength,
                "boundary_tau": 0.0,
                "boundary_eta": 0.0,
                "bias": bias.tolist(),
            })
            for tau in [0.05, 0.10, 0.20]:
                for eta in [0.05, 0.10, 0.20]:
                    configs.append({
                        "alpha": alpha,
                        "prior_strength": prior_strength,
                        "boundary_tau": tau,
                        "boundary_eta": eta,
                        "bias": bias.tolist(),
                    })

    for cfg in configs:
        pred = cemr_predict(p_backbone_val, p_evidence_val, cfg)
        m = metrics_row(y_val, pred)
        key = (m["macro_f1_4"], m["accuracy"], m["F1_S"] + m["F1_F"])
        if best is None or key > best["key"]:
            best = {"key": key, "config": cfg, "metrics": m}
    return best["config"], best["metrics"], time.time() - start


def run_one(data, dataset, backbone, seed, device, smoke=False):
    scenario = make_scenario(data, dataset, seed)
    train_idx, val_idx = train_val_split(scenario["y_tr"], seed)

    if backbone == "RbfSVM":
        backbone_result = fit_rbfsvm_backbone(scenario, train_idx, val_idx, seed)
        group = "machine_learning"
    else:
        spec_by_name = {s.name: s for s in model_specs()}
        spec = spec_by_name[backbone]
        backbone_result = fit_deep_with_oom_retry(spec, scenario, train_idx, val_idx, seed, device, smoke=smoke)
        group = "deep_learning" if backbone == "CAT-Net" else "modern_time_series"

    p_evidence_val, p_evidence_test, ev_train_time, ev_predict_time = fit_evidence_encoder(
        scenario, train_idx, val_idx, seed
    )
    cfg, val_metrics, adapter_time = fit_cemr_adapter(
        backbone_result["proba_val"],
        p_evidence_val,
        scenario["y_tr"][val_idx],
        scenario["y_tr"][train_idx],
    )

    pred_raw = backbone_result["proba_test"].argmax(axis=1).astype(np.int64)
    pred_cemr = cemr_predict(backbone_result["proba_test"], p_evidence_test, cfg)

    raw_metrics = metrics_row(scenario["y_te"], pred_raw)
    cemr_metrics = metrics_row(scenario["y_te"], pred_cemr)
    row = {
        "dataset": dataset,
        "protocol": scenario["protocol"],
        "method": f"{backbone}+CEMR-ECG",
        "backbone": backbone,
        "backbone_group": group,
        "seed": int(seed),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(scenario["y_te"])),
        "train_counts": counts_text(scenario["y_tr"][train_idx]),
        "val_counts": counts_text(scenario["y_tr"][val_idx]),
        "test_counts": counts_text(scenario["y_te"]),
        "input_mode": backbone_result["input_mode"],
        "backbone_config": backbone_result["backbone_config"],
        "cemr_evidence_encoder": "single ExtraTrees evidence encoder on morphology-rhythm + prototype features",
        "cemr_adapter": json.dumps({k: v for k, v in cfg.items() if k != "bias"}, ensure_ascii=False),
        "cemr_bias": json.dumps(cfg["bias"], ensure_ascii=False),
        "val_macro_f1_4_after_cemr": float(val_metrics["macro_f1_4"]),
        "best_val_macro_f1_4_backbone": float(backbone_result["best_val_macro_f1_4"])
        if not pd.isna(backbone_result["best_val_macro_f1_4"]) else np.nan,
        "epochs_run": int(backbone_result["epochs_run"]),
        "batch_size": int(backbone_result["batch_size"]),
        "train_time": float(backbone_result["train_time"] + ev_train_time + adapter_time),
        "predict_time": float(backbone_result["predict_time"] + ev_predict_time),
        "backbone_train_time": float(backbone_result["train_time"]),
        "evidence_train_time": float(ev_train_time),
        "adapter_fit_time": float(adapter_time),
        "backbone_predict_time": float(backbone_result["predict_time"]),
        "evidence_predict_time": float(ev_predict_time),
    }
    row.update(cemr_metrics)
    for k, v in raw_metrics.items():
        row[f"raw_same_run_{k}"] = v
    row["delta_accuracy"] = row["accuracy"] - row["raw_same_run_accuracy"]
    row["delta_macro_f1_4"] = row["macro_f1_4"] - row["raw_same_run_macro_f1_4"]
    row["delta_F1_S"] = row["F1_S"] - row["raw_same_run_F1_S"]
    row["delta_F1_F"] = row["F1_F"] - row["raw_same_run_F1_F"]
    cm_final = confusion_matrix(scenario["y_te"], pred_cemr, labels=list(range(config.N_CLASSES))).astype(int).tolist()
    cm_raw = confusion_matrix(scenario["y_te"], pred_raw, labels=list(range(config.N_CLASSES))).astype(int).tolist()
    return row, cm_raw, cm_final


def completed_keys():
    if not DETAIL.exists():
        return set()
    df = pd.read_csv(DETAIL)
    return set(zip(df["dataset"].astype(str), df["backbone"].astype(str), df["seed"].astype(int)))


def load_confusion():
    if CONFUSION.exists():
        return json.loads(CONFUSION.read_text(encoding="utf-8"))
    return {}


def save_confusion(conf):
    CONFUSION.write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")


def append_detail(row):
    pd.DataFrame([row]).to_csv(DETAIL, mode="a", header=not DETAIL.exists(), index=False, encoding="utf-8")


def mean_std_summary(detail):
    metrics = [
        "accuracy", "macro_f1_5", "macro_f1_4", "macro_f1_3_nsv",
        "Se_N", "Se_S", "Se_V", "Se_F", "Se_Q",
        "Pr_N", "Pr_S", "Pr_V", "Pr_F", "Pr_Q",
        "F1_N", "F1_S", "F1_V", "F1_F", "F1_Q",
        "raw_same_run_accuracy", "raw_same_run_macro_f1_4", "raw_same_run_F1_S", "raw_same_run_F1_F",
        "delta_accuracy", "delta_macro_f1_4", "delta_F1_S", "delta_F1_F",
        "train_time", "predict_time", "backbone_train_time", "evidence_train_time",
    ]
    rows = []
    for (dataset, backbone), g in detail.groupby(["dataset", "backbone"], dropna=False):
        row = {
            "dataset": dataset,
            "backbone": backbone,
            "method": f"{backbone}+CEMR-ECG",
            "backbone_group": str(g["backbone_group"].iloc[0]),
            "protocol": str(g["protocol"].iloc[0]),
            "n_seeds": int(g["seed"].nunique()),
            "seeds": " ".join(str(int(s)) for s in sorted(g["seed"].unique())),
            "input_mode": str(g["input_mode"].iloc[0]),
            "train_size_mean": float(g["train_size"].mean()),
            "test_size_mean": float(g["test_size"].mean()),
        }
        for metric in metrics:
            if metric in g.columns:
                row[f"{metric}_mean"] = float(g[metric].mean())
                row[f"{metric}_std"] = float(g[metric].std(ddof=0))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["dataset", "macro_f1_4_mean", "accuracy_mean"],
        ascending=[True, False, False],
    )


def add_legacy_raw_columns(summary):
    if summary.empty:
        return summary
    out = summary.copy()
    out["legacy_raw_accuracy_mean"] = np.nan
    out["legacy_raw_macro_f1_4_mean"] = np.nan
    out["legacy_raw_protocol"] = ""
    for idx, row in out.iterrows():
        dataset = row["dataset"]
        backbone = row["backbone"]
        if backbone == "RbfSVM":
            method_name = "RbfSVM_raw"
            path = RESULTS / "comparison_methods_summary.csv" if dataset == "MIT-BIH" else RESULTS / "external_raw_baselines_summary.csv"
        else:
            method_name = backbone
            path = RESULTS / "modern_deep_baselines_summary.csv" if dataset == "MIT-BIH" else RESULTS / "external_deep_baselines_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if dataset == "MIT-BIH":
            hit = df[df["model"].astype(str) == method_name]
            protocol = "MIT-BIH DS1->DS2; same dataset protocol"
        else:
            ext_name = f"{dataset}_all"
            hit = df[(df["dataset"].astype(str) == ext_name) & (df["model"].astype(str) == method_name)]
            protocol = "legacy MIT-BIH-train external-test table; not used as same-split delta"
        if hit.empty:
            continue
        hit = hit.iloc[0]
        out.loc[idx, "legacy_raw_accuracy_mean"] = float(hit.get("accuracy_mean", np.nan))
        out.loc[idx, "legacy_raw_macro_f1_4_mean"] = float(hit.get("macro_f1_4_mean", np.nan))
        out.loc[idx, "legacy_raw_protocol"] = protocol
    return out


def add_paper_alias_columns(summary):
    """Add stable, paper-facing column names without changing the raw summary fields."""
    if summary.empty:
        return summary
    out = summary.copy()
    aliases = {
        "cemr_accuracy_mean": "accuracy_mean",
        "cemr_accuracy_std": "accuracy_std",
        "cemr_macro_f1_5_mean": "macro_f1_5_mean",
        "cemr_macro_f1_5_std": "macro_f1_5_std",
        "cemr_macro_f1_4_mean": "macro_f1_4_mean",
        "cemr_macro_f1_4_std": "macro_f1_4_std",
        "cemr_macro_f1_3_nsv_mean": "macro_f1_3_nsv_mean",
        "cemr_macro_f1_3_nsv_std": "macro_f1_3_nsv_std",
        "raw_baseline_accuracy_mean": "raw_same_run_accuracy_mean",
        "raw_baseline_accuracy_std": "raw_same_run_accuracy_std",
        "raw_baseline_macro_f1_4_mean": "raw_same_run_macro_f1_4_mean",
        "raw_baseline_macro_f1_4_std": "raw_same_run_macro_f1_4_std",
        "raw_baseline_F1_S_mean": "raw_same_run_F1_S_mean",
        "raw_baseline_F1_S_std": "raw_same_run_F1_S_std",
        "raw_baseline_F1_F_mean": "raw_same_run_F1_F_mean",
        "raw_baseline_F1_F_std": "raw_same_run_F1_F_std",
        "Delta_Acc_mean": "delta_accuracy_mean",
        "Delta_Acc_std": "delta_accuracy_std",
        "Delta_M_F1_4_mean": "delta_macro_f1_4_mean",
        "Delta_M_F1_4_std": "delta_macro_f1_4_std",
    }
    for alias, source in aliases.items():
        if source in out.columns:
            out[alias] = out[source]
    out["raw_baseline_source"] = (
        "same-run backbone predictions before CEMR-ECG; no additional raw training run"
    )
    return out


def summarize():
    if not DETAIL.exists():
        return pd.DataFrame()
    detail = pd.read_csv(DETAIL)
    summary = add_paper_alias_columns(add_legacy_raw_columns(mean_std_summary(detail)))
    summary.to_csv(SUMMARY, index=False, encoding="utf-8")
    return summary


def fmt_pct(x):
    if pd.isna(x):
        return ""
    return f"{100.0 * float(x):.2f}"


def markdown_table(df, cols):
    if df is None or df.empty:
        return "_No data._"
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def recall_from_cm(cm):
    cm = np.asarray(cm, dtype=np.float64)
    return np.diag(cm) / np.maximum(cm.sum(axis=1), 1e-12)


def confusion_diagnostics():
    if not CONFUSION.exists():
        return pd.DataFrame()
    conf = load_confusion()
    rows = []
    for key, cm in conf.items():
        parts = key.split("|")
        if len(parts) != 4:
            continue
        dataset, backbone, seed, stage = parts
        rows.append({
            "dataset": dataset,
            "backbone": backbone,
            "seed": int(seed),
            "stage": stage,
            "cm": cm,
        })
    by_key = {}
    for row in rows:
        by_key[(row["dataset"], row["backbone"], row["seed"], row["stage"])] = row["cm"]
    out = []
    for (dataset, backbone, seed, stage), raw_cm in by_key.items():
        if stage != "raw_same_run":
            continue
        cemr_key = (dataset, backbone, seed, "CEMR-ECG")
        if cemr_key not in by_key:
            continue
        raw_recall = recall_from_cm(raw_cm)
        cemr_recall = recall_from_cm(by_key[cemr_key])
        item = {
            "dataset": dataset,
            "backbone": backbone,
            "seed": seed,
        }
        for i, name in enumerate(CLASS_NAMES):
            item[f"raw_Se_{name}"] = float(raw_recall[i])
            item[f"cemr_Se_{name}"] = float(cemr_recall[i])
            item[f"delta_Se_{name}"] = float(cemr_recall[i] - raw_recall[i])
        out.append(item)
    if not out:
        return pd.DataFrame()
    detail = pd.DataFrame(out)
    summary_rows = []
    for (dataset, backbone), g in detail.groupby(["dataset", "backbone"], dropna=False):
        row = {
            "dataset": dataset,
            "backbone": backbone,
        }
        for name in CLASS_NAMES:
            row[f"raw_Se_{name}_mean"] = float(g[f"raw_Se_{name}"].mean())
            row[f"cemr_Se_{name}_mean"] = float(g[f"cemr_Se_{name}"].mean())
            row[f"delta_Se_{name}_mean"] = float(g[f"delta_Se_{name}"].mean())
        delta_pairs = [(name, row[f"delta_Se_{name}_mean"]) for name in CLASS_NAMES]
        cemr_pairs = [(name, row[f"cemr_Se_{name}_mean"]) for name in CLASS_NAMES]
        row["largest_recall_gain_class"] = max(delta_pairs, key=lambda x: x[1])[0]
        row["largest_recall_gain"] = max(delta_pairs, key=lambda x: x[1])[1]
        row["hardest_recall_class"] = min(cemr_pairs, key=lambda x: x[1])[0]
        row["hardest_recall"] = min(cemr_pairs, key=lambda x: x[1])[1]
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def build_report():
    summary = summarize()
    if summary.empty:
        REPORT.write_text("# Dataset-wise CEMR-ECG Framework Experiments\n\n_No completed runs._\n", encoding="utf-8")
        return
    diag = confusion_diagnostics()

    show = summary.copy()
    for col in [
        "accuracy_mean", "macro_f1_5_mean", "macro_f1_4_mean", "macro_f1_3_nsv_mean",
        "raw_same_run_accuracy_mean", "raw_same_run_macro_f1_4_mean",
        "delta_accuracy_mean", "delta_macro_f1_4_mean", "delta_F1_S_mean", "delta_F1_F_mean",
        "cemr_accuracy_mean", "cemr_macro_f1_5_mean", "cemr_macro_f1_4_mean", "cemr_macro_f1_3_nsv_mean",
        "raw_baseline_accuracy_mean", "raw_baseline_macro_f1_4_mean",
        "Delta_Acc_mean", "Delta_M_F1_4_mean",
        "legacy_raw_accuracy_mean", "legacy_raw_macro_f1_4_mean",
    ]:
        if col in show.columns:
            show[col] = show[col].map(lambda x: fmt_pct(x) if not pd.isna(x) else "")
    lines = [
        "# Dataset-wise CEMR-ECG Framework Experiments",
        "",
        "Protocol: each dataset is trained and tested within itself. MIT-BIH uses the fixed DS1 to DS2 split; INCART and SVDB use seed-specific record-level splits.",
        "",
        "Primary deltas use the same-run raw backbone predictions before CEMR-ECG is applied. This keeps the dataset-wise protocol aligned and does not require an extra raw-baseline training run.",
        "",
        "CEMR-ECG is evaluated as a plug-in framework on three backbone families: RbfSVM for classical machine learning, CAT-Net for ECG-specific deep learning, and TimeMixer for modern time-series modeling.",
        "",
        "## Summary",
        "",
        markdown_table(
            show,
            [
                "dataset", "backbone", "n_seeds",
                "raw_baseline_accuracy_mean", "cemr_accuracy_mean", "Delta_Acc_mean",
                "raw_baseline_macro_f1_4_mean", "cemr_macro_f1_4_mean", "Delta_M_F1_4_mean",
                "delta_F1_S_mean", "delta_F1_F_mean", "train_time_mean",
            ],
        ),
        "",
        "Metric note: values in the summary table are percentages except train_time_mean, which is reported in seconds. M-F1(4) is macro-F1 over N/S/V/F, excluding Q because Q is extremely sparse in these datasets.",
        "",
        "## Recall Diagnostics From Confusion Matrices",
        "",
        "The table below summarizes average sensitivity changes from the saved raw and CEMR-ECG confusion matrices. It is intended for error analysis, not as a separate optimization target.",
        "",
        markdown_table(
            diag.assign(
                largest_recall_gain=diag["largest_recall_gain"].map(fmt_pct) if not diag.empty else [],
                hardest_recall=diag["hardest_recall"].map(fmt_pct) if not diag.empty else [],
            ) if not diag.empty else diag,
            [
                "dataset", "backbone", "largest_recall_gain_class", "largest_recall_gain",
                "hardest_recall_class", "hardest_recall",
            ],
        ),
        "",
        "## Legacy Raw Tables",
        "",
        "Legacy raw columns are included for traceability. For INCART and SVDB they come from the older MIT-BIH-train external-test protocol, so same-split improvement should be read from the raw same-run columns instead.",
        "",
        markdown_table(
            show,
            [
                "dataset", "backbone", "legacy_raw_macro_f1_4_mean", "legacy_raw_accuracy_mean", "legacy_raw_protocol",
            ],
        ),
        "",
        "## Per-dataset Notes",
        "",
    ]
    for dataset, g in summary.groupby("dataset"):
        best = g.sort_values(["macro_f1_4_mean", "accuracy_mean"], ascending=False).iloc[0]
        dataset_diag = diag[diag["dataset"] == dataset] if not diag.empty else pd.DataFrame()
        if dataset_diag.empty:
            gain_sentence = "Confusion-matrix diagnostics are unavailable."
        else:
            best_diag = dataset_diag[dataset_diag["backbone"] == best["backbone"]]
            best_diag = best_diag.iloc[0] if not best_diag.empty else dataset_diag.iloc[0]
            gain_sentence = (
                f"For the best backbone, the largest sensitivity gain is in class "
                f"{best_diag['largest_recall_gain_class']} ({fmt_pct(best_diag['largest_recall_gain'])} points), "
                f"while class {best_diag['hardest_recall_class']} remains the hardest "
                f"(mean Se={fmt_pct(best_diag['hardest_recall'])}%)."
            )
        lines.extend([
            f"### {dataset}",
            "",
            f"Best +CEMR backbone: {best['backbone']} with M-F1(4)={fmt_pct(best['macro_f1_4_mean'])}% and Acc={fmt_pct(best['accuracy_mean'])}%.",
            f"Mean same-run delta: M-F1(4)={fmt_pct(best['delta_macro_f1_4_mean'])} percentage points, F1_S={fmt_pct(best['delta_F1_S_mean'])}, F1_F={fmt_pct(best['delta_F1_F_mean'])}.",
            gain_sentence,
            "",
        ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run_all(args):
    RESULTS.mkdir(exist_ok=True)
    data = load_cache()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    datasets = args.datasets or ["MIT-BIH", "INCART", "SVDB"]
    backbones = args.backbones or BACKBONES
    seeds = args.seeds or SEEDS
    done = completed_keys()
    conf = load_confusion()
    for dataset in datasets:
        for backbone in backbones:
            for seed in seeds:
                key = (dataset, backbone, int(seed))
                if key in done and not args.force and not args.no_save:
                    print(f"Skip completed {dataset} {backbone} seed={seed}", flush=True)
                    continue
                print(f"Running {dataset} {backbone}+CEMR-ECG seed={seed}", flush=True)
                row, cm_raw, cm_final = run_one(data, dataset, backbone, int(seed), device, smoke=args.smoke)
                if not args.no_save:
                    append_detail(row)
                    conf[f"{dataset}|{backbone}|{seed}|raw_same_run"] = cm_raw
                    conf[f"{dataset}|{backbone}|{seed}|CEMR-ECG"] = cm_final
                    save_confusion(conf)
                    build_report()
                print(
                    f"[{dataset} {backbone} seed={seed}] "
                    f"raw_MF1(4)={row['raw_same_run_macro_f1_4'] * 100:.2f}% "
                    f"CEMR_MF1(4)={row['macro_f1_4'] * 100:.2f}% "
                    f"delta={row['delta_macro_f1_4'] * 100:.2f}pp",
                    flush=True,
                )
    if not args.no_save:
        build_report()
        print(f"Saved {DETAIL}", flush=True)
        print(f"Saved {SUMMARY}", flush=True)
        print(f"Saved {REPORT}", flush=True)
        print(f"Saved {CONFUSION}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dataset-wise CEMR-ECG framework experiments")
    parser.add_argument("--datasets", nargs="*", default=None, choices=["MIT-BIH", "INCART", "SVDB"])
    parser.add_argument("--backbones", nargs="*", default=None, choices=BACKBONES)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one epoch for deep models")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--force", action="store_true", help="Rerun completed keys")
    parser.add_argument("--no-save", action="store_true", help="Do not append outputs; useful for smoke checks")
    parser.add_argument("--rebuild-report", action="store_true", help="Only rebuild summary/report from detail CSV")
    args = parser.parse_args(argv)
    if args.rebuild_report:
        build_report()
        return
    run_all(args)


if __name__ == "__main__":
    main()
