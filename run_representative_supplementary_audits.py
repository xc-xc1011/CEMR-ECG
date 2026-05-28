from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from metrics_eval import metrics_row
from run_external_raw_baselines import specs as ml_specs
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


RESULTS = Path("results")
ARCHIVE_CACHE = Path("archive/generated_cleanup_20260527/results_legacy/external_base_features_cache.npz")
ACTIVE_CACHE = RESULTS / "external_base_features_cache.npz"
EVIDENCE_CACHE = RESULTS / "representative_supplementary_evidence_cache"
PROBA_CACHE = RESULTS / "representative_supplementary_probability_cache"

CAL_DETAIL = RESULTS / "final_representative_calibration_audit_detail.csv"
CAL_SUMMARY = RESULTS / "final_representative_calibration_audit_summary.csv"
CAL_BINS = RESULTS / "final_representative_calibration_reliability_bins.csv"
SPLIT_AUDIT = RESULTS / "final_split_audit.csv"
SENS_DETAIL = RESULTS / "final_class_weight_sensitivity_detail.csv"
SENS_SUMMARY = RESULTS / "final_class_weight_sensitivity_summary.csv"
REPORT = RESULTS / "final_representative_supplementary_audit_report.md"
CONFUSION = RESULTS / "final_representative_supplementary_confusion.json"

DATASETS = ["MIT-BIH", "INCART", "SVDB"]
SEEDS = [303, 1303, 2303, 3303, 4303]
REPRESENTATIVE_METHODS = ["ExtraTrees_raw", "CAT-Net", "TimeMixer"]
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
EPS = 1e-12


def cache_path() -> Path:
    if ACTIVE_CACHE.exists():
        return ACTIVE_CACHE
    if ARCHIVE_CACHE.exists():
        return ARCHIVE_CACHE
    raise FileNotFoundError(f"Missing external feature cache. Tried {ACTIVE_CACHE} and {ARCHIVE_CACHE}.")


def counts_dict(y: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=config.N_CLASSES)
    return {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)}


def counts_text(y: np.ndarray) -> str:
    return json.dumps(counts_dict(y), ensure_ascii=False)


def normalize_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPS, None)
    return p / np.maximum(p.sum(axis=1, keepdims=True), EPS)


def safe_log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0))


def fuse_backbone_evidence(p_backbone: np.ndarray, p_evidence: np.ndarray, alpha: float) -> np.ndarray:
    score = (1.0 - float(alpha)) * safe_log(p_backbone) + float(alpha) * safe_log(p_evidence)
    score = score - score.max(axis=1, keepdims=True)
    exp = np.exp(score)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)


def power_calibrate(p: np.ndarray, power: float) -> np.ndarray:
    p = normalize_prob(p)
    if abs(float(power) - 1.0) < 1e-12:
        return p
    return normalize_prob(np.power(p, float(power)))


def decoder_scores(proba: np.ndarray, decoder: dict) -> np.ndarray:
    p = power_calibrate(proba, float(decoder["power"]))
    return normalize_prob(p * np.asarray(decoder["multiplier"], dtype=np.float64)[None, :])


def predict_with_decoder(proba: np.ndarray, decoder: dict) -> np.ndarray:
    return decoder_scores(proba, decoder).argmax(axis=1).astype(np.int64)


def cap(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


def prior_ratio(y_train: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=config.N_CLASSES).astype(np.float64)
    ref = max(counts[0], 1.0)
    return ref / np.maximum(counts, 1.0)


def decoder_candidates(y_train: np.ndarray) -> list[dict]:
    ratio = prior_ratio(y_train)
    return [
        {
            "decoder": "evidence_argmax",
            "power": 1.0,
            "multiplier": np.ones(config.N_CLASSES, dtype=np.float64),
            "rationale": "direct evidence/backbone mixture argmax",
        },
        {
            "decoder": "clinical_default",
            "power": 1.0,
            "multiplier": np.asarray(
                [
                    0.85,
                    cap(ratio[1] ** 0.23, 1.0, 2.60),
                    cap(ratio[2] ** -0.02, 0.90, 1.05),
                    cap(ratio[3] ** 0.15, 1.0, 2.20),
                    1.0,
                ],
                dtype=np.float64,
            ),
            "rationale": "dominant-normal damping plus moderate S/F prevalence recovery",
        },
        {
            "decoder": "stable_prior",
            "power": 0.90,
            "multiplier": np.asarray(
                [1.0, cap(ratio[1] ** 0.25, 1.0, 4.0), cap(ratio[2] ** 0.03, 1.0, 1.5), cap(ratio[3] ** 0.20, 1.0, 4.0), 1.0],
                dtype=np.float64,
            ),
            "rationale": "long-tail prior decoder for under-detected S/F classes",
        },
        {
            "decoder": "s_recovery",
            "power": 0.95,
            "multiplier": np.asarray(
                [0.90, cap(ratio[1] ** 0.28, 1.0, 4.5), 1.0, cap(ratio[3] ** 0.08, 1.0, 1.6), 1.0],
                dtype=np.float64,
            ),
            "rationale": "S-focused recovery",
        },
        {
            "decoder": "f_recovery",
            "power": 0.95,
            "multiplier": np.asarray(
                [0.90, cap(ratio[1] ** 0.08, 1.0, 1.6), 0.98, cap(ratio[3] ** 0.22, 1.0, 4.0), 1.0],
                dtype=np.float64,
            ),
            "rationale": "F-focused recovery when support is sufficient",
        },
        {
            "decoder": "mild_prior",
            "power": 0.95,
            "multiplier": np.asarray(
                [0.95, cap(ratio[1] ** 0.18, 1.0, 2.5), 1.0, cap(ratio[3] ** 0.12, 1.0, 2.2), 1.0],
                dtype=np.float64,
            ),
            "rationale": "mild same-dataset prevalence correction",
        },
    ]


def raw_dual(b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    return np.concatenate([b0, b1], axis=1).astype(np.float32)


def split_by_record(y: np.ndarray, pid: np.ndarray, seed: int, train_frac: float = 0.67) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    records = np.unique(pid)
    needed_classes = [c for c in range(4) if np.sum(y == c) >= 20]
    for _ in range(2000):
        shuffled = records.copy()
        rng.shuffle(shuffled)
        n_train = max(1, int(round(len(shuffled) * train_frac)))
        train_records = set(shuffled[:n_train])
        train_mask = np.asarray([p in train_records for p in pid])
        test_mask = ~train_mask
        ok = True
        for c in needed_classes:
            ok &= np.any(y[train_mask] == c) and np.any(y[test_mask] == c)
        if ok:
            return train_mask, test_mask
    raise RuntimeError("Could not build a record-level split with all needed classes.")


def cap_augmented_training(
    x: np.ndarray, y: np.ndarray, b0: np.ndarray, b1: np.ndarray, pid: np.ndarray, seed: int, n_cap: int = 50000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep = []
    for cls in range(config.N_CLASSES):
        idx = np.flatnonzero(y == cls)
        if cls == 0 and len(idx) > n_cap:
            idx = rng.choice(idx, size=n_cap, replace=False)
        keep.append(idx)
    idx = np.concatenate(keep)
    rng.shuffle(idx)
    return x[idx], y[idx], b0[idx], b1[idx], pid[idx]


def train_val_split(y: np.ndarray, seed: int, val_fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
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


def make_scenario(data: np.lib.npyio.NpzFile, dataset: str, seed: int) -> dict:
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
            "train_mask_full": None,
            "test_mask_full": None,
            "precap_train_size": int(len(data["mit_tr_y"])),
        }

    prefix = "inc" if dataset == "INCART" else "sv"
    y = data[f"{prefix}_y"].astype(np.int64)
    pid = data[f"{prefix}_pid"]
    train_mask, test_mask = split_by_record(y, pid, seed)
    x_tr = data[f"{prefix}_x"][train_mask].astype(np.float32)
    y_tr = y[train_mask]
    pid_tr = pid[train_mask]
    b0_tr = data[f"{prefix}_b0"][train_mask].astype(np.float32)
    b1_tr = data[f"{prefix}_b1"][train_mask].astype(np.float32)
    precap_train_size = int(len(y_tr))
    x_tr, y_tr, b0_tr, b1_tr, pid_tr = cap_augmented_training(x_tr, y_tr, b0_tr, b1_tr, pid_tr, seed=seed)
    full_name = "INCART" if dataset == "INCART" else "SVDB"
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
        "train_mask_full": train_mask,
        "test_mask_full": test_mask,
        "precap_train_size": precap_train_size,
    }


def add_proto_features_local(
    x_tr_base: np.ndarray,
    y_tr: np.ndarray,
    x_te_base: np.ndarray,
    b0_tr: np.ndarray,
    b1_tr: np.ndarray,
    b0_te: np.ndarray,
    b1_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    proto0 = []
    proto1 = []
    for cls in range(config.N_CLASSES):
        mask = y_tr == cls
        proto0.append(b0_tr[mask].mean(axis=0) if mask.any() else b0_tr.mean(axis=0))
        proto1.append(b1_tr[mask].mean(axis=0) if mask.any() else b1_tr.mean(axis=0))
    proto0 = np.asarray(proto0, dtype=np.float32)
    proto1 = np.asarray(proto1, dtype=np.float32)

    def calc(b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
        b0n = b0 / (np.linalg.norm(b0, axis=1, keepdims=True) + 1e-8)
        b1n = b1 / (np.linalg.norm(b1, axis=1, keepdims=True) + 1e-8)
        p0n = proto0 / (np.linalg.norm(proto0, axis=1, keepdims=True) + 1e-8)
        p1n = proto1 / (np.linalg.norm(proto1, axis=1, keepdims=True) + 1e-8)
        corr0 = b0n @ p0n.T
        corr1 = b1n @ p1n.T
        d0 = ((b0[:, None, :] - proto0[None, :, :]) ** 2).mean(axis=2)
        d1 = ((b1[:, None, :] - proto1[None, :, :]) ** 2).mean(axis=2)
        margins = np.stack(
            [
                corr0[:, 1] - corr0[:, 0],
                corr0[:, 3] - corr0[:, 0],
                corr0[:, 3] - corr0[:, 2],
                corr1[:, 1] - corr1[:, 0],
                corr1[:, 3] - corr1[:, 0],
                corr1[:, 3] - corr1[:, 2],
                d0[:, 0] - d0[:, 1],
                d0[:, 0] - d0[:, 3],
                d0[:, 2] - d0[:, 3],
                d1[:, 0] - d1[:, 1],
                d1[:, 0] - d1[:, 3],
                d1[:, 2] - d1[:, 3],
            ],
            axis=1,
        )
        return np.concatenate([corr0, corr1, np.log1p(d0), np.log1p(d1), margins], axis=1).astype(np.float32)

    return (
        np.concatenate([x_tr_base, calc(b0_tr, b1_tr)], axis=1).astype(np.float32),
        np.concatenate([x_te_base, calc(b0_te, b1_te)], axis=1).astype(np.float32),
    )


def class_weight_value(mode: str):
    if mode == "default_cost":
        return {0: 1.0, 1: 3.0, 2: 1.5, 3: 12.0, 4: 1.0}
    if mode == "no_cost":
        return None
    if mode == "balanced":
        return "balanced"
    if mode == "moderate_tail":
        return {0: 1.0, 1: 2.0, 2: 1.25, 3: 6.0, 4: 1.0}
    if mode == "strong_tail":
        return {0: 0.95, 1: 4.0, 2: 1.5, 3: 16.0, 4: 1.0}
    raise ValueError(f"Unknown class-cost mode: {mode}")


def align_proba(classes, proba: np.ndarray) -> np.ndarray:
    out = np.zeros((proba.shape[0], config.N_CLASSES), dtype=np.float64)
    for i, cls in enumerate(classes):
        out[:, int(cls)] = proba[:, i]
    missing = out.sum(axis=1) <= 0
    if np.any(missing):
        out[missing] = 1.0 / config.N_CLASSES
    return normalize_prob(out)


def align_scores_to_proba(classes, scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    out = np.full((scores.shape[0], config.N_CLASSES), -1e6, dtype=np.float64)
    for i, cls in enumerate(classes):
        out[:, int(cls)] = scores[:, i]
    out = out - np.max(out, axis=1, keepdims=True)
    exp = np.exp(out)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)


def model_classes(model):
    if hasattr(model, "classes_"):
        return model.classes_
    if hasattr(model, "named_steps"):
        for step in reversed(model.named_steps.values()):
            if hasattr(step, "classes_"):
                return step.classes_
    raise AttributeError("Fitted model does not expose classes_.")


def sklearn_proba(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return align_proba(model_classes(model), model.predict_proba(x))
    if hasattr(model, "decision_function"):
        return align_scores_to_proba(model_classes(model), model.decision_function(x))
    pred = np.asarray(model.predict(x)).astype(np.int64)
    out = np.full((len(pred), config.N_CLASSES), 1e-6, dtype=np.float64)
    out[np.arange(len(pred)), pred] = 1.0
    return normalize_prob(out)


def fit_extra_trees_backbone(scenario: dict, train_idx: np.ndarray, val_idx: np.ndarray, seed: int) -> dict:
    spec = next(s for s in ml_specs(seed) if s["model"] == "ExtraTrees_raw")
    x_all = raw_dual(scenario["b0_tr"], scenario["b1_tr"])
    x_test = raw_dual(scenario["b0_te"], scenario["b1_te"])
    model = copy.deepcopy(spec["estimator"])
    t0 = time.time()
    model.fit(x_all[train_idx], scenario["y_tr"][train_idx])
    train_time = time.time() - t0
    t1 = time.time()
    p_val = sklearn_proba(model, x_all[val_idx])
    p_test = sklearn_proba(model, x_test)
    predict_time = time.time() - t1
    return {
        "method": "ExtraTrees_raw",
        "family": "raw_machine_learning",
        "proba_val": p_val,
        "proba_test": p_test,
        "train_time": train_time,
        "predict_time": predict_time,
        "epochs_run": 0,
        "best_val_macro_f1_4": np.nan,
        "batch_size": 0,
        "model_config": spec["config"],
    }


def deep_input_arrays(scenario: dict, input_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if input_mode == "single":
        return scenario["b0_tr"][:, None, :].astype(np.float32), scenario["b0_te"][:, None, :].astype(np.float32)
    if input_mode == "dual":
        return (
            np.stack([scenario["b0_tr"], scenario["b1_tr"]], axis=1).astype(np.float32),
            np.stack([scenario["b0_te"], scenario["b1_te"]], axis=1).astype(np.float32),
        )
    raise ValueError(f"Unknown input_mode: {input_mode}")


@torch.no_grad()
def predict_proba_deep(model, x: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    pin_memory = os.environ.get("CEMR_PIN_MEMORY", "0") == "1"
    loader = torch.utils.data.DataLoader(
        BeatDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    model.eval()
    targets = []
    probas = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probas.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(targets), np.concatenate(probas).astype(np.float64)


def fit_deep_backbone(spec, scenario: dict, train_idx: np.ndarray, val_idx: np.ndarray, seed: int, device: torch.device, smoke: bool = False) -> dict:
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

    batch_size = int(os.environ.get("CEMR_BATCH_SIZE_OVERRIDE", spec.batch_size))
    epochs = 1 if smoke else spec.epochs
    patience = 1 if smoke else spec.patience
    while True:
        try:
            train_loader = make_loader(x_train, y_train, batch_size, train=True, sampler_mode=spec.sampler_mode, augment=spec.augment)
            model = spec.make_model().to(device)
            loss_fn = make_loss(spec, y_train, device)
            optimizer = make_optimizer(spec, model)
            use_amp = os.environ.get("CEMR_USE_AMP", "0") == "1"
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and use_amp)
            break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device.type == "cuda" and batch_size > 16:
                torch.cuda.empty_cache()
                batch_size = max(16, batch_size // 2)
                print(f"[{scenario['dataset']} {spec.name} seed={seed}] OOM during setup; retry batch_size={batch_size}", flush=True)
                continue
            raise

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

        _, p_val_epoch = predict_proba_deep(model, x_val, y_val, batch_size, device)
        val_score = metrics_row(y_val, p_val_epoch.argmax(axis=1))["macro_f1_4"]
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"[supp {scenario['dataset']} {spec.name} seed={seed}] epoch={epoch:03d} "
            f"loss={total_loss / max(1, len(y_train)):.4f} val_MF1(4)={val_score:.4f} best={best_val:.4f}",
            flush=True,
        )
        if bad_epochs >= patience:
            break

    train_time = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    t1 = time.time()
    _, p_val = predict_proba_deep(model, x_val, y_val, batch_size, device)
    _, p_test = predict_proba_deep(model, x_test, scenario["y_te"], batch_size, device)
    predict_time = time.time() - t1
    return {
        "method": spec.name,
        "family": "raw_deep_or_time_series",
        "proba_val": p_val,
        "proba_test": p_test,
        "train_time": train_time,
        "predict_time": predict_time,
        "epochs_run": best_epoch,
        "best_val_macro_f1_4": best_val,
        "batch_size": batch_size,
        "model_config": f"{spec.params} resample={used_resampling or spec.resample_mode or ''}".strip(),
    }


def fit_backbone(method: str, scenario: dict, train_idx: np.ndarray, val_idx: np.ndarray, seed: int, device: torch.device, smoke: bool = False) -> dict:
    if method == "ExtraTrees_raw":
        return fit_extra_trees_backbone(scenario, train_idx, val_idx, seed)
    spec = next((s for s in model_specs() if s.name == method), None)
    if spec is None:
        raise ValueError(f"Unknown representative method: {method}")
    return fit_deep_backbone(spec, scenario, train_idx, val_idx, seed, device, smoke=smoke)


def build_evidence_features(scenario: dict, train_idx: np.ndarray, target: str) -> tuple[np.ndarray, np.ndarray]:
    if target == "val":
        target_idx = train_val_split(scenario["y_tr"], 0)[1]
        raise RuntimeError("Internal error: target='val' requires explicit target indices.")
    raise RuntimeError("Use build_evidence_features_for_target or build_evidence_features_for_test.")


def build_evidence_features_for_target(scenario: dict, fit_idx: np.ndarray, target_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return add_proto_features_local(
        scenario["x_tr_base"][fit_idx],
        scenario["y_tr"][fit_idx],
        scenario["x_tr_base"][target_idx],
        scenario["b0_tr"][fit_idx],
        scenario["b1_tr"][fit_idx],
        scenario["b0_tr"][target_idx],
        scenario["b1_tr"][target_idx],
    )


def build_evidence_features_for_test(scenario: dict, fit_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return add_proto_features_local(
        scenario["x_tr_base"][fit_idx],
        scenario["y_tr"][fit_idx],
        scenario["x_te_base"],
        scenario["b0_tr"][fit_idx],
        scenario["b1_tr"][fit_idx],
        scenario["b0_te"],
        scenario["b1_te"],
    )


def fit_evidence_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_target: np.ndarray, seed: int, cost_mode: str) -> np.ndarray:
    model = ExtraTreesClassifier(
        n_estimators=700,
        max_features=0.35,
        min_samples_leaf=2,
        class_weight=class_weight_value(cost_mode),
        bootstrap=False,
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(x_fit, y_fit)
    return align_proba(model.classes_, model.predict_proba(x_target))


def evidence_cache_key(dataset: str, seed: int, cost_mode: str) -> Path:
    return EVIDENCE_CACHE / f"{dataset}_{int(seed)}_{cost_mode}_evidence.npz"


def load_or_build_evidence(scenario: dict, train_idx: np.ndarray, val_idx: np.ndarray, seed: int, cost_mode: str, force: bool = False) -> dict:
    EVIDENCE_CACHE.mkdir(parents=True, exist_ok=True)
    path = evidence_cache_key(scenario["dataset"], seed, cost_mode)
    if path.exists() and not force:
        cached = np.load(path, allow_pickle=True)
        return {
            "p_evidence_val": cached["p_evidence_val"].astype(np.float64),
            "p_evidence_test": cached["p_evidence_test"].astype(np.float64),
            "train_time": float(cached["train_time"]),
            "predict_time": float(cached["predict_time"]),
            "path": str(path),
        }
    y_fit = scenario["y_tr"][train_idx]
    t0 = time.time()
    x_fit, x_val = build_evidence_features_for_target(scenario, train_idx, val_idx)
    p_val = fit_evidence_predict(x_fit, y_fit, x_val, seed, cost_mode)
    x_fit_full, x_test = build_evidence_features_for_test(scenario, train_idx)
    p_test = fit_evidence_predict(x_fit_full, y_fit, x_test, seed, cost_mode)
    elapsed = time.time() - t0
    np.savez_compressed(path, p_evidence_val=p_val, p_evidence_test=p_test, train_time=np.asarray(elapsed), predict_time=np.asarray(0.0))
    return {"p_evidence_val": p_val, "p_evidence_test": p_test, "train_time": elapsed, "predict_time": 0.0, "path": str(path)}


def select_adapter(dataset: str, p_backbone_val: np.ndarray, p_evidence_val: np.ndarray, y_val: np.ndarray, y_fit: np.ndarray) -> tuple[dict, list[dict]]:
    candidates = []
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
        clinical = next(d for d in decoder_candidates(y_fit) if d["decoder"] == "clinical_default")
        selected = dict(selected)
        selected["decoder"] = clinical
        selected["domain_shift_policy"] = "MIT-BIH uses clinical_default decoder with method-specific validation alpha"
    return selected, candidates


def one_hot(y: np.ndarray) -> np.ndarray:
    out = np.zeros((len(y), config.N_CLASSES), dtype=np.float64)
    out[np.arange(len(y)), np.asarray(y, dtype=np.int64)] = 1.0
    return out


def multiclass_brier(y: np.ndarray, proba: np.ndarray) -> float:
    return float(np.mean(np.sum((normalize_prob(proba) - one_hot(y)) ** 2, axis=1)))


def nll(y: np.ndarray, proba: np.ndarray) -> float:
    p = normalize_prob(proba)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), np.asarray(y, dtype=np.int64)], 1e-12, 1.0))))


def ece(y: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    p = normalize_prob(proba)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == np.asarray(y, dtype=np.int64)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    ece_value = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        if n:
            acc = float(correct[mask].mean())
            avg_conf = float(conf[mask].mean())
            gap = abs(acc - avg_conf)
            ece_value += (n / total) * gap
        else:
            acc = avg_conf = gap = float("nan")
        bins.append({"bin": i + 1, "lo": float(lo), "hi": float(hi), "n": n, "accuracy": acc, "confidence": avg_conf, "abs_gap": gap})
    return float(ece_value), bins


def calibration_rows(dataset: str, method: str, seed: int, y_test: np.ndarray, raw_p: np.ndarray, cemr_p: np.ndarray) -> tuple[dict, list[dict]]:
    raw_pred = raw_p.argmax(axis=1).astype(np.int64)
    cemr_pred = cemr_p.argmax(axis=1).astype(np.int64)
    raw_metrics = metrics_row(y_test, raw_pred)
    cemr_metrics = metrics_row(y_test, cemr_pred)
    raw_ece, raw_bins = ece(y_test, raw_p)
    cemr_ece, cemr_bins = ece(y_test, cemr_p)
    row = {
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "raw_brier": multiclass_brier(y_test, raw_p),
        "cemr_brier": multiclass_brier(y_test, cemr_p),
        "delta_brier": multiclass_brier(y_test, cemr_p) - multiclass_brier(y_test, raw_p),
        "raw_ece": raw_ece,
        "cemr_ece": cemr_ece,
        "delta_ece": cemr_ece - raw_ece,
        "raw_nll": nll(y_test, raw_p),
        "cemr_nll": nll(y_test, cemr_p),
        "delta_nll": nll(y_test, cemr_p) - nll(y_test, raw_p),
        "raw_accuracy": raw_metrics["accuracy"],
        "cemr_accuracy": cemr_metrics["accuracy"],
        "delta_accuracy": cemr_metrics["accuracy"] - raw_metrics["accuracy"],
        "raw_macro_f1_4": raw_metrics["macro_f1_4"],
        "cemr_macro_f1_4": cemr_metrics["macro_f1_4"],
        "delta_macro_f1_4": cemr_metrics["macro_f1_4"] - raw_metrics["macro_f1_4"],
        "raw_F1_S": raw_metrics["F1_S"],
        "cemr_F1_S": cemr_metrics["F1_S"],
        "raw_F1_F": raw_metrics["F1_F"],
        "cemr_F1_F": cemr_metrics["F1_F"],
    }
    bins = []
    for source, source_bins in [("raw", raw_bins), ("cemr", cemr_bins)]:
        for b in source_bins:
            bins.append({"dataset": dataset, "method": method, "seed": int(seed), "source": source, **b})
    return row, bins


def proba_cache_key(dataset: str, method: str, seed: int) -> Path:
    safe_method = method.replace("/", "_").replace(" ", "_")
    return PROBA_CACHE / f"{dataset}_{safe_method}_{int(seed)}.npz"


def save_probability_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_probability_cache(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def completed_calibration_keys() -> set[tuple[str, str, int]]:
    if not CAL_DETAIL.exists():
        return set()
    df = pd.read_csv(CAL_DETAIL)
    return set(zip(df["dataset"].astype(str), df["method"].astype(str), df["seed"].astype(int)))


def append_csv(path: Path, rows: list[dict] | dict) -> None:
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def run_calibration(args, data, device: torch.device) -> None:
    done = completed_calibration_keys()
    conf = json.loads(CONFUSION.read_text(encoding="utf-8")) if CONFUSION.exists() else {}
    for dataset in args.datasets:
        for seed in args.seeds:
            scenario = make_scenario(data, dataset, int(seed))
            train_idx, val_idx = train_val_split(scenario["y_tr"], int(seed))
            y_fit = scenario["y_tr"][train_idx]
            y_val = scenario["y_tr"][val_idx]
            evidence = load_or_build_evidence(scenario, train_idx, val_idx, int(seed), "default_cost", force=args.force_evidence)
            for method in args.methods:
                key = (dataset, method, int(seed))
                if key in done and not args.force:
                    print(f"Skip completed calibration {dataset} {method} seed={seed}", flush=True)
                    continue
                print(f"Running calibration {dataset} {method} seed={seed}", flush=True)
                p_path = proba_cache_key(dataset, method, int(seed))
                if p_path.exists() and not args.force:
                    cached = load_probability_cache(p_path)
                    backbone = {
                        "method": method,
                        "family": str(cached["family"]),
                        "proba_val": cached["p_backbone_val"],
                        "proba_test": cached["p_backbone_test"],
                        "train_time": float(cached["backbone_train_time"]),
                        "predict_time": float(cached["backbone_predict_time"]),
                        "epochs_run": int(cached["epochs_run"]),
                        "best_val_macro_f1_4": float(cached["best_val_macro_f1_4"]),
                        "batch_size": int(cached["batch_size"]),
                        "model_config": str(cached["model_config"]),
                    }
                else:
                    backbone = fit_backbone(method, scenario, train_idx, val_idx, int(seed), device, smoke=args.smoke)
                selected, candidates = select_adapter(dataset, backbone["proba_val"], evidence["p_evidence_val"], y_val, y_fit)
                p_fused_test = fuse_backbone_evidence(backbone["proba_test"], evidence["p_evidence_test"], selected["alpha"])
                p_cemr_test = decoder_scores(p_fused_test, selected["decoder"])
                row, bin_rows = calibration_rows(dataset, method, int(seed), scenario["y_te"], backbone["proba_test"], p_cemr_test)
                row.update(
                    {
                        "family": backbone["family"],
                        "protocol": scenario["protocol"],
                        "train_size": int(len(train_idx)),
                        "val_size": int(len(val_idx)),
                        "test_size": int(len(scenario["y_te"])),
                        "train_counts": counts_text(y_fit),
                        "val_counts": counts_text(y_val),
                        "test_counts": counts_text(scenario["y_te"]),
                        "selected_alpha": float(selected["alpha"]),
                        "selected_decoder": selected["decoder"]["decoder"],
                        "selected_power": float(selected["decoder"]["power"]),
                        "selected_multiplier": json.dumps([float(x) for x in selected["decoder"]["multiplier"]], ensure_ascii=False),
                        "domain_shift_policy": selected.get("domain_shift_policy", ""),
                        "model_config": backbone["model_config"],
                        "batch_size": int(backbone["batch_size"]),
                        "epochs_run": int(backbone["epochs_run"]),
                        "best_val_macro_f1_4": float(backbone["best_val_macro_f1_4"]),
                        "backbone_train_time": float(backbone["train_time"]),
                        "backbone_predict_time": float(backbone["predict_time"]),
                        "evidence_cache": evidence["path"],
                        "proba_cache": str(p_path),
                    }
                )
                if not args.smoke:
                    save_probability_cache(
                        p_path,
                        {
                            "dataset": np.asarray(dataset),
                            "method": np.asarray(method),
                            "family": np.asarray(backbone["family"]),
                            "seed": np.asarray(int(seed)),
                            "p_backbone_val": backbone["proba_val"].astype(np.float32),
                            "p_backbone_test": backbone["proba_test"].astype(np.float32),
                            "p_cemr_test": p_cemr_test.astype(np.float32),
                            "y_val": y_val.astype(np.int64),
                            "y_test": scenario["y_te"].astype(np.int64),
                            "selected_alpha": np.asarray(float(selected["alpha"])),
                            "selected_decoder": np.asarray(selected["decoder"]["decoder"]),
                            "selected_multiplier": np.asarray(selected["decoder"]["multiplier"], dtype=np.float32),
                            "backbone_train_time": np.asarray(float(backbone["train_time"])),
                            "backbone_predict_time": np.asarray(float(backbone["predict_time"])),
                            "epochs_run": np.asarray(int(backbone["epochs_run"])),
                            "best_val_macro_f1_4": np.asarray(float(backbone["best_val_macro_f1_4"])),
                            "batch_size": np.asarray(int(backbone["batch_size"])),
                            "model_config": np.asarray(backbone["model_config"]),
                        },
                    )
                    append_csv(CAL_DETAIL, row)
                    append_csv(CAL_BINS, bin_rows)
                    conf[f"{dataset}|{method}|{seed}|raw"] = confusion_matrix(
                        scenario["y_te"], backbone["proba_test"].argmax(axis=1), labels=list(range(config.N_CLASSES))
                    ).astype(int).tolist()
                    conf[f"{dataset}|{method}|{seed}|CEMR-BioAdaptive"] = confusion_matrix(
                        scenario["y_te"], p_cemr_test.argmax(axis=1), labels=list(range(config.N_CLASSES))
                    ).astype(int).tolist()
                    CONFUSION.write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")
                    done.add(key)
                print(
                    f"[cal {dataset} {method} seed={seed}] raw_MF1={row['raw_macro_f1_4']*100:.2f} "
                    f"CEMR_MF1={row['cemr_macro_f1_4']*100:.2f} "
                    f"Brier {row['raw_brier']:.4f}->{row['cemr_brier']:.4f} "
                    f"ECE {row['raw_ece']:.4f}->{row['cemr_ece']:.4f}",
                    flush=True,
                )
                if args.smoke:
                    return
    summarize_calibration()


def summarize_calibration() -> pd.DataFrame:
    if not CAL_DETAIL.exists():
        return pd.DataFrame()
    df = pd.read_csv(CAL_DETAIL)
    metrics = [
        "raw_brier",
        "cemr_brier",
        "delta_brier",
        "raw_ece",
        "cemr_ece",
        "delta_ece",
        "raw_nll",
        "cemr_nll",
        "delta_nll",
        "raw_accuracy",
        "cemr_accuracy",
        "delta_accuracy",
        "raw_macro_f1_4",
        "cemr_macro_f1_4",
        "delta_macro_f1_4",
        "raw_F1_S",
        "cemr_F1_S",
        "raw_F1_F",
        "cemr_F1_F",
    ]
    rows = []
    for (dataset, method), g in df.groupby(["dataset", "method"], sort=True):
        row = {"dataset": dataset, "method": method, "n_seeds": int(g["seed"].nunique()), "seeds": " ".join(map(str, sorted(g["seed"].unique())))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=0))
        rows.append(row)
    for method, g in df.groupby("method", sort=True):
        row = {"dataset": "Overall", "method": method, "n_seeds": int(len(g)), "seeds": "all"}
        for metric in metrics:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=0))
        rows.append(row)
    overall = {"dataset": "Overall", "method": "All representatives", "n_seeds": int(len(df)), "seeds": "all"}
    for metric in metrics:
        overall[f"{metric}_mean"] = float(df[metric].mean())
        overall[f"{metric}_std"] = float(df[metric].std(ddof=0))
    rows.append(overall)
    out = pd.DataFrame(rows)
    out.to_csv(CAL_SUMMARY, index=False, encoding="utf-8")
    return out


def run_split_audit(args, data) -> None:
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            scenario = make_scenario(data, dataset, int(seed))
            train_idx, val_idx = train_val_split(scenario["y_tr"], int(seed))
            fit_records = set(map(str, scenario["pid_tr"][train_idx]))
            val_records = set(map(str, scenario["pid_tr"][val_idx]))
            test_records = set(map(str, scenario["pid_te"]))
            train_records = set(map(str, scenario["pid_tr"]))
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "protocol": scenario["protocol"],
                    "train_record_policy": scenario["train_record_policy"],
                    "test_record_policy": scenario["test_record_policy"],
                    "precap_train_size": int(scenario["precap_train_size"]),
                    "fit_size": int(len(train_idx)),
                    "val_size": int(len(val_idx)),
                    "test_size": int(len(scenario["y_te"])),
                    "train_pool_records": int(len(train_records)),
                    "fit_records": int(len(fit_records)),
                    "val_records": int(len(val_records)),
                    "test_records": int(len(test_records)),
                    "train_test_record_overlap": int(len(train_records & test_records)),
                    "fit_test_record_overlap": int(len(fit_records & test_records)),
                    "val_test_record_overlap": int(len(val_records & test_records)),
                    "fit_val_record_overlap": int(len(fit_records & val_records)),
                    "fit_counts": counts_text(scenario["y_tr"][train_idx]),
                    "val_counts": counts_text(scenario["y_tr"][val_idx]),
                    "test_counts": counts_text(scenario["y_te"]),
                    "train_records_list": ";".join(sorted(train_records)),
                    "test_records_list": ";".join(sorted(test_records)),
                    "audit_interpretation": "No test-record leakage" if not (train_records & test_records) else "Record overlap detected",
                }
            )
    pd.DataFrame(rows).to_csv(SPLIT_AUDIT, index=False, encoding="utf-8")
    print(f"Wrote {SPLIT_AUDIT}", flush=True)


def completed_sensitivity_keys() -> set[tuple[str, str, int, str]]:
    if not SENS_DETAIL.exists():
        return set()
    df = pd.read_csv(SENS_DETAIL)
    return set(zip(df["dataset"].astype(str), df["method"].astype(str), df["seed"].astype(int), df["class_cost_profile"].astype(str)))


def run_sensitivity(args, data, device: torch.device) -> None:
    profiles = ["no_cost", "balanced", "moderate_tail", "default_cost", "strong_tail"]
    done = completed_sensitivity_keys()
    for dataset in args.datasets:
        for seed in args.seeds:
            scenario = make_scenario(data, dataset, int(seed))
            train_idx, val_idx = train_val_split(scenario["y_tr"], int(seed))
            y_fit = scenario["y_tr"][train_idx]
            y_val = scenario["y_tr"][val_idx]
            for method in args.methods:
                p_path = proba_cache_key(dataset, method, int(seed))
                if p_path.exists():
                    cached = load_probability_cache(p_path)
                    p_backbone_val = cached["p_backbone_val"].astype(np.float64)
                    p_backbone_test = cached["p_backbone_test"].astype(np.float64)
                else:
                    print(f"Sensitivity needs probability cache; building {dataset} {method} seed={seed}", flush=True)
                    backbone = fit_backbone(method, scenario, train_idx, val_idx, int(seed), device, smoke=args.smoke)
                    p_backbone_val = backbone["proba_val"]
                    p_backbone_test = backbone["proba_test"]
                    save_probability_cache(
                        p_path,
                        {
                            "dataset": np.asarray(dataset),
                            "method": np.asarray(method),
                            "family": np.asarray(backbone["family"]),
                            "seed": np.asarray(int(seed)),
                            "p_backbone_val": p_backbone_val.astype(np.float32),
                            "p_backbone_test": p_backbone_test.astype(np.float32),
                            "y_val": y_val.astype(np.int64),
                            "y_test": scenario["y_te"].astype(np.int64),
                            "backbone_train_time": np.asarray(float(backbone["train_time"])),
                            "backbone_predict_time": np.asarray(float(backbone["predict_time"])),
                            "epochs_run": np.asarray(int(backbone["epochs_run"])),
                            "best_val_macro_f1_4": np.asarray(float(backbone["best_val_macro_f1_4"])),
                            "batch_size": np.asarray(int(backbone["batch_size"])),
                            "model_config": np.asarray(backbone["model_config"]),
                        },
                    )
                for profile in profiles:
                    key = (dataset, method, int(seed), profile)
                    if key in done and not args.force:
                        print(f"Skip completed sensitivity {dataset} {method} seed={seed} profile={profile}", flush=True)
                        continue
                    evidence = load_or_build_evidence(scenario, train_idx, val_idx, int(seed), profile, force=args.force_evidence)
                    selected, _ = select_adapter(dataset, p_backbone_val, evidence["p_evidence_val"], y_val, y_fit)
                    p_fused = fuse_backbone_evidence(p_backbone_test, evidence["p_evidence_test"], selected["alpha"])
                    p_cemr = decoder_scores(p_fused, selected["decoder"])
                    pred = p_cemr.argmax(axis=1).astype(np.int64)
                    raw_pred = p_backbone_test.argmax(axis=1).astype(np.int64)
                    metrics = metrics_row(scenario["y_te"], pred)
                    raw_metrics = metrics_row(scenario["y_te"], raw_pred)
                    row = {
                        "dataset": dataset,
                        "method": method,
                        "seed": int(seed),
                        "class_cost_profile": profile,
                        "selected_alpha": float(selected["alpha"]),
                        "selected_decoder": selected["decoder"]["decoder"],
                        "selected_multiplier": json.dumps([float(x) for x in selected["decoder"]["multiplier"]], ensure_ascii=False),
                        "raw_macro_f1_4": raw_metrics["macro_f1_4"],
                        "macro_f1_4": metrics["macro_f1_4"],
                        "delta_macro_f1_4": metrics["macro_f1_4"] - raw_metrics["macro_f1_4"],
                        "accuracy": metrics["accuracy"],
                        "raw_accuracy": raw_metrics["accuracy"],
                        "F1_S": metrics["F1_S"],
                        "F1_V": metrics["F1_V"],
                        "F1_F": metrics["F1_F"],
                        "raw_F1_S": raw_metrics["F1_S"],
                        "raw_F1_F": raw_metrics["F1_F"],
                        "evidence_cache": evidence["path"],
                    }
                    if not args.smoke:
                        append_csv(SENS_DETAIL, row)
                        done.add(key)
                    print(
                        f"[sens {dataset} {method} seed={seed} {profile}] MF1={metrics['macro_f1_4']*100:.2f} "
                        f"delta={row['delta_macro_f1_4']*100:.2f}",
                        flush=True,
                    )
                    if args.smoke:
                        return
    summarize_sensitivity()


def summarize_sensitivity() -> pd.DataFrame:
    if not SENS_DETAIL.exists():
        return pd.DataFrame()
    df = pd.read_csv(SENS_DETAIL)
    rows = []
    for (dataset, method, profile), g in df.groupby(["dataset", "method", "class_cost_profile"], sort=True):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "class_cost_profile": profile,
                "n_runs": int(len(g)),
                "macro_f1_4_mean": float(g["macro_f1_4"].mean()),
                "macro_f1_4_std": float(g["macro_f1_4"].std(ddof=0)),
                "delta_macro_f1_4_mean": float(g["delta_macro_f1_4"].mean()),
                "accuracy_mean": float(g["accuracy"].mean()),
                "F1_S_mean": float(g["F1_S"].mean()),
                "F1_F_mean": float(g["F1_F"].mean()),
            }
        )
    for profile, g in df.groupby("class_cost_profile", sort=True):
        rows.append(
            {
                "dataset": "Overall",
                "method": "All representatives",
                "class_cost_profile": profile,
                "n_runs": int(len(g)),
                "macro_f1_4_mean": float(g["macro_f1_4"].mean()),
                "macro_f1_4_std": float(g["macro_f1_4"].std(ddof=0)),
                "delta_macro_f1_4_mean": float(g["delta_macro_f1_4"].mean()),
                "accuracy_mean": float(g["accuracy"].mean()),
                "F1_S_mean": float(g["F1_S"].mean()),
                "F1_F_mean": float(g["F1_F"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(SENS_SUMMARY, index=False, encoding="utf-8")
    return out


def write_report() -> None:
    cal = summarize_calibration()
    sens = summarize_sensitivity()
    lines = [
        "# Representative Supplementary Audits for CEMR-ECG",
        "",
        "Scope: one representative method per model family: ExtraTrees_raw (machine learning), CAT-Net (ECG deep model), and TimeMixer (modern time-series model), evaluated on MIT-BIH, INCART and SVDB with seeds 303, 1303, 2303, 3303 and 4303.",
        "",
        "These analyses are supplementary diagnostics. They do not replace the main 20-method, three-dataset, five-seed framework result.",
        "",
    ]
    if not cal.empty:
        overall = cal[(cal["dataset"] == "Overall") & (cal["method"] == "All representatives")]
        if not overall.empty:
            r = overall.iloc[0]
            lines.extend(
                [
                    "## Calibration audit",
                    "",
                    f"- Representative runs: {int(r['n_seeds'])}.",
                    f"- Mean M-F1(4) changed from {r['raw_macro_f1_4_mean']*100:.2f}% to {r['cemr_macro_f1_4_mean']*100:.2f}%.",
                    f"- Mean Brier score changed from {r['raw_brier_mean']:.4f} to {r['cemr_brier_mean']:.4f}; delta={r['delta_brier_mean']:.4f}.",
                    f"- Mean ECE changed from {r['raw_ece_mean']:.4f} to {r['cemr_ece_mean']:.4f}; delta={r['delta_ece_mean']:.4f}.",
                    "",
                ]
            )
        lines.extend(["### Calibration summary table", "", cal.to_markdown(index=False), ""])
    if SPLIT_AUDIT.exists():
        split = pd.read_csv(SPLIT_AUDIT)
        max_test_overlap = int(split[["train_test_record_overlap", "fit_test_record_overlap", "val_test_record_overlap"]].max().max())
        lines.extend(
            [
                "## Split audit",
                "",
                f"- Maximum observed train/fit/validation to test record overlap: {max_test_overlap}.",
                "- Internal fit/validation record overlap is reported separately and is not test leakage; it reflects the class-stratified validation fold used only for adapter selection.",
                "",
            ]
        )
    if not sens.empty:
        overall = sens[sens["dataset"] == "Overall"].copy()
        lines.extend(["## Class-cost sensitivity", "", overall.to_markdown(index=False), ""])
    lines.extend(
        [
            "## Boundaries",
            "",
            "- Calibration metrics are reported on representative backbones because sample-level probabilities were not retained for all 20 main-result methods.",
            "- The full performance claim remains based on the complete 20-method result table.",
            "- SVDB F remains a limited-support endpoint and is not reinterpreted as recovered.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Representative supplementary calibration, split and sensitivity audits.")
    parser.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS)
    parser.add_argument("--methods", nargs="*", default=REPRESENTATIVE_METHODS, choices=REPRESENTATIVE_METHODS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--phase", choices=["all", "calibration", "split", "sensitivity", "summary"], default="all")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-evidence", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    EVIDENCE_CACHE.mkdir(parents=True, exist_ok=True)
    PROBA_CACHE.mkdir(parents=True, exist_ok=True)
    if args.force:
        for p in [CAL_DETAIL, CAL_SUMMARY, CAL_BINS, SENS_DETAIL, SENS_SUMMARY, REPORT, CONFUSION]:
            if p.exists():
                p.unlink()

    data = np.load(cache_path(), allow_pickle=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Representative supplementary audit device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if args.phase in {"all", "split"}:
        run_split_audit(args, data)
    if args.phase in {"all", "calibration"}:
        run_calibration(args, data, device)
    if args.phase in {"all", "sensitivity"}:
        run_sensitivity(args, data, device)
    if args.phase == "summary":
        summarize_calibration()
        summarize_sensitivity()
    write_report()


if __name__ == "__main__":
    main()
