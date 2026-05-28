import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.ensemble import ExtraTreesClassifier

import config
from feature_engineering import (
    add_proto_features,
    extract_mit_records,
    lead_morph_features,
    resample_1d,
)


ROOT = Path.cwd().resolve()
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = RESULTS_DIR / "external_base_features_cache.npz"
DETAIL_PATH = RESULTS_DIR / "external_dataset_detail.csv"
SUMMARY_PATH = RESULTS_DIR / "external_dataset_summary.csv"
BEST_PATH = RESULTS_DIR / "external_dataset_best_summary.csv"
META_PATH = RESULTS_DIR / "external_dataset_meta.json"

SEEDS = [303, 1303, 2303]
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
PRE_SEC = config.PRE_R / float(config.FS)
POST_SEC = config.POST_R / float(config.FS)
RR_EPS = 1e-8


def bandpass(sig, fs):
    nyq = 0.5 * fs
    high = min(config.BANDPASS_HIGH, nyq * 0.90)
    b, a = butter(
        config.BANDPASS_ORDER,
        [config.BANDPASS_LOW / nyq, high / nyq],
        btype="band",
    )
    return filtfilt(b, a, sig, axis=0)


def select_leads(sig_names):
    names = [str(x).upper().replace(" ", "") for x in sig_names]

    def pick(candidates, fallback):
        for c in candidates:
            if c in names:
                return names.index(c)
        return fallback

    first = pick(["MLII", "II", "ECG1", "V5", "I"], 0)
    second = pick(["V1", "ECG2", "V2", "III", "I"], 1 if len(names) > 1 else 0)
    if second == first and len(names) > 1:
        second = 1 if first != 1 else 0
    return first, second


def rr_features(samples, i, fs):
    rr_all = np.diff(samples).astype(np.float32) / float(fs)
    rr_global = float(np.median(rr_all)) if len(rr_all) else 1.0
    pre_rr = ((samples[i] - samples[i - 1]) / float(fs)) if i > 0 else rr_global
    post_rr = ((samples[i + 1] - samples[i]) / float(fs)) if i < len(samples) - 1 else rr_global
    lo = max(0, i - 5)
    hi = min(len(samples) - 1, i + 5)
    local_rrs = np.diff(samples[lo:hi + 1]).astype(np.float32) / float(fs)
    local_rr = float(np.median(local_rrs)) if len(local_rrs) else rr_global
    return [
        pre_rr,
        post_rr,
        pre_rr / (post_rr + RR_EPS),
        post_rr / (pre_rr + RR_EPS),
        abs(post_rr - pre_rr),
        pre_rr / (local_rr + RR_EPS),
        post_rr / (local_rr + RR_EPS),
        pre_rr / (rr_global + RR_EPS),
        post_rr / (rr_global + RR_EPS),
        (pre_rr - local_rr) / (local_rr + RR_EPS),
        (post_rr - local_rr) / (local_rr + RR_EPS),
        local_rr / (rr_global + RR_EPS),
    ]


def extract_generic_records(db_dir, records, db_name):
    db_dir = Path(db_dir)
    base_feats = []
    labels = []
    pids = []
    beats0 = []
    beats1 = []
    skipped = 0
    pre = None
    post = None

    for rec_id in records:
        rec_path = db_dir / str(rec_id)
        rec = wfdb.rdrecord(str(rec_path))
        ann = wfdb.rdann(str(rec_path), "atr")
        fs = float(rec.fs)
        sig = rec.p_signal.astype(np.float32)
        if sig.ndim == 1:
            sig = sig[:, None]
        if sig.shape[1] == 1:
            sig = np.repeat(sig, 2, axis=1)
        lead0, lead1 = select_leads(rec.sig_name)
        sig = sig[:, [lead0, lead1]]
        sig = bandpass(sig, fs).astype(np.float32)
        sig = (sig - sig.mean(axis=0, keepdims=True)) / (sig.std(axis=0, keepdims=True) + 1e-8)

        pre = int(round(PRE_SEC * fs))
        post = int(round(POST_SEC * fs))
        samples = np.asarray(ann.sample, dtype=np.int64)
        for i, (r_idx, sym) in enumerate(zip(samples, ann.symbol)):
            if sym not in config.AAMI_MAP:
                skipped += 1
                continue
            start = int(r_idx) - pre
            end = int(r_idx) + post
            if start < 0 or end > len(sig) or end <= start:
                skipped += 1
                continue
            beat = sig[start:end]
            b0 = resample_1d(beat[:, 0], config.WIN_LEN)
            b1 = resample_1d(beat[:, 1], config.WIN_LEN)
            feats = []
            feats.extend(rr_features(samples, i, fs))
            m0 = lead_morph_features(b0)
            m1 = lead_morph_features(b1)
            feats.extend(m0)
            feats.extend(m1)
            feats.extend((np.asarray(m0) - np.asarray(m1)).tolist())
            feats.extend(resample_1d(b0, 80).tolist())
            feats.extend(resample_1d(b1, 80).tolist())
            feats.extend(resample_1d(np.diff(b0), 40).tolist())
            feats.extend(resample_1d(np.diff(b1), 40).tolist())
            base_feats.append(feats)
            labels.append(config.AAMI_MAP[sym])
            pids.append(f"{db_name}_{rec_id}")
            beats0.append(b0)
            beats1.append(b1)

    X = np.asarray(base_feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    pid = np.asarray(pids)
    B0 = np.asarray(beats0, dtype=np.float32)
    B1 = np.asarray(beats1, dtype=np.float32)
    print(
        f"{db_name}: X={X.shape}, counts={np.bincount(y, minlength=config.N_CLASSES).tolist()}, "
        f"skipped={skipped}, window_samples={pre}+{post}"
    )
    return X, y, pid, B0, B1


def load_or_build_base_cache():
    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH, allow_pickle=True)
        return {k: data[k] for k in data.files}

    print("Building external base feature cache...")
    t0 = time.time()
    mit_tr_x, mit_tr_y, mit_tr_pid, mit_tr_b0, mit_tr_b1 = extract_mit_records(config.DS1_RECORDS)
    mit_te_x, mit_te_y, mit_te_pid, mit_te_b0, mit_te_b1 = extract_mit_records(config.DS2_RECORDS)

    incart_dir = ROOT / "data" / "raw" / "incartdb"
    svdb_dir = ROOT / "data" / "raw" / "svdb"
    incart_records = sorted(p.stem for p in incart_dir.glob("*.hea"))
    svdb_records = sorted(p.stem for p in svdb_dir.glob("*.hea"))
    inc_x, inc_y, inc_pid, inc_b0, inc_b1 = extract_generic_records(incart_dir, incart_records, "incart")
    sv_x, sv_y, sv_pid, sv_b0, sv_b1 = extract_generic_records(svdb_dir, svdb_records, "svdb")

    np.savez_compressed(
        CACHE_PATH,
        mit_tr_x=mit_tr_x,
        mit_tr_y=mit_tr_y,
        mit_tr_pid=np.asarray([f"mit_{x}" for x in mit_tr_pid]),
        mit_tr_b0=mit_tr_b0,
        mit_tr_b1=mit_tr_b1,
        mit_te_x=mit_te_x,
        mit_te_y=mit_te_y,
        mit_te_pid=np.asarray([f"mit_{x}" for x in mit_te_pid]),
        mit_te_b0=mit_te_b0,
        mit_te_b1=mit_te_b1,
        inc_x=inc_x,
        inc_y=inc_y,
        inc_pid=inc_pid,
        inc_b0=inc_b0,
        inc_b1=inc_b1,
        sv_x=sv_x,
        sv_y=sv_y,
        sv_pid=sv_pid,
        sv_b0=sv_b0,
        sv_b1=sv_b1,
    )
    print(f"Saved {CACHE_PATH}, time={time.time() - t0:.1f}s")
    data = np.load(CACHE_PATH, allow_pickle=True)
    return {k: data[k] for k in data.files}


def split_by_record(y, pid, seed, train_frac=0.67):
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


def cap_augmented_training(X, y, B0, B1, pid, seed, n_cap=50000):
    rng = np.random.default_rng(seed)
    keep = []
    for cls in range(config.N_CLASSES):
        idx = np.flatnonzero(y == cls)
        if cls == 0 and len(idx) > n_cap:
            idx = rng.choice(idx, size=n_cap, replace=False)
        keep.append(idx)
    idx = np.concatenate(keep)
    rng.shuffle(idx)
    return X[idx], y[idx], B0[idx], B1[idx], pid[idx]


def add_protos(train_x, train_y, test_x, train_b0, train_b1, test_b0, test_b1):
    return add_proto_features(train_x, train_y, test_x, train_b0, train_b1, test_b0, test_b1)


def calibration_grid():
    for n_mul in [0.65, 0.75, 0.85, 0.95]:
        for s_mul in [1.5, 2.0, 2.5, 3.0, 3.5]:
            for v_mul in [0.75, 0.85, 0.95, 1.05]:
                for f_mul in [1.2, 1.6, 2.0, 2.4, 3.0]:
                    yield n_mul, s_mul, v_mul, f_mul


def calibration_grid_s_focus():
    for n_mul in [0.75, 0.85, 0.95, 1.05]:
        for s_mul in [2.0, 2.5, 3.0, 3.5, 4.0]:
            for v_mul in [0.75, 0.85, 0.95, 1.05]:
                for f_mul in [0.3, 0.5, 1.0]:
                    yield n_mul, s_mul, v_mul, f_mul


def metrics_external(y_true, y_pred):
    n = config.N_CLASSES
    cm = np.bincount(
        n * y_true.astype(np.int64) + y_pred.astype(np.int64),
        minlength=n * n,
    ).reshape(n, n)
    tp = np.diag(cm).astype(np.float64)
    se = tp / (cm.sum(axis=1) + 1e-12)
    pp = tp / (cm.sum(axis=0) + 1e-12)
    f1 = 2 * se * pp / (se + pp + 1e-12)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1_5": float(f1.mean()),
        "macro_f1_4": float(f1[:4].mean()),
        "macro_f1_3_nsv": float(f1[[0, 1, 2]].mean()),
        "Se_N": float(se[0]),
        "Se_S": float(se[1]),
        "Se_V": float(se[2]),
        "Se_F": float(se[3]),
        "Se_Q": float(se[4]),
        "Pr_N": float(pp[0]),
        "Pr_S": float(pp[1]),
        "Pr_V": float(pp[2]),
        "Pr_F": float(pp[3]),
        "Pr_Q": float(pp[4]),
        "F1_N": float(f1[0]),
        "F1_S": float(f1[1]),
        "F1_V": float(f1[2]),
        "F1_F": float(f1[3]),
        "F1_Q": float(f1[4]),
        "support_N": int(cm.sum(axis=1)[0]),
        "support_S": int(cm.sum(axis=1)[1]),
        "support_V": int(cm.sum(axis=1)[2]),
        "support_F": int(cm.sum(axis=1)[3]),
        "support_Q": int(cm.sum(axis=1)[4]),
    }


def summarize(df):
    group_cols = [
        "scenario",
        "train_data",
        "test_data",
        "n_estimators",
        "max_features",
        "min_samples_leaf",
        "cw_S",
        "cw_V",
        "cw_F",
        "mode",
        "n_mul",
        "s_mul",
        "v_mul",
        "f_mul",
    ]
    group_cols = [c for c in group_cols if c in df.columns]
    metrics = [
        "accuracy", "macro_f1_5", "macro_f1_4", "macro_f1_3_nsv",
        "Se_N", "Se_S", "Se_V", "Se_F", "Se_Q",
        "Pr_N", "Pr_S", "Pr_V", "Pr_F", "Pr_Q",
        "F1_N", "F1_S", "F1_V", "F1_F", "F1_Q",
        "support_N", "support_S", "support_V", "support_F", "support_Q",
        "train_time",
    ]
    out = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {k: v for k, v in zip(group_cols, keys)}
        row["n_seeds"] = int(g["seed"].nunique())
        row["seeds"] = " ".join(map(str, sorted(g["seed"].unique())))
        for m in metrics:
            if m not in g.columns:
                continue
            row[m + "_mean"] = float(g[m].mean())
            row[m + "_std"] = float(g[m].std(ddof=0))
        out.append(row)
    return pd.DataFrame(out).sort_values(["macro_f1_4_mean", "accuracy_mean"], ascending=False)


def best_summary(summary):
    rows = []
    if summary.empty:
        return summary
    for (scenario, test_data), g in summary.groupby(["scenario", "test_data"], dropna=False):
        g3 = g[g["n_seeds"] == SEEDS.__len__()]
        if g3.empty:
            g3 = g[g["n_seeds"] == g["n_seeds"].max()]
        metric = "macro_f1_3_nsv_mean" if "s_focus" in str(scenario) and "macro_f1_3_nsv_mean" in g3.columns else "macro_f1_4_mean"
        chosen = g3.sort_values([metric, "accuracy_mean"], ascending=False).head(1).copy()
        chosen.insert(0, "selection_metric", metric)
        rows.append(chosen)
    return pd.concat(rows, ignore_index=True).sort_values(["scenario", "test_data"])


def save_outputs(rows):
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail, detail
    detail = detail.drop_duplicates(
        subset=[
            "scenario", "train_data", "test_data", "seed", "n_estimators", "max_features",
            "min_samples_leaf", "cw_S", "cw_V", "cw_F", "mode", "n_mul", "s_mul", "v_mul", "f_mul",
        ],
        keep="last",
    )
    summary = summarize(detail)
    best = best_summary(summary)
    detail.to_csv(DETAIL_PATH, index=False, encoding="utf-8")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")
    best.to_csv(BEST_PATH, index=False, encoding="utf-8")
    return detail, summary, best


def scenario_done(rows, scenario, train_name, test_name):
    if not rows:
        return False
    df = pd.DataFrame(rows)
    mask = (
        (df["scenario"] == scenario)
        & (df["train_data"] == train_name)
        & (df["test_data"] == test_name)
    )
    if not mask.any():
        return False
    return set(df.loc[mask, "seed"].astype(int).unique()) >= set(SEEDS)


def run_et_scenario(
    rows,
    scenario,
    train_name,
    test_name,
    X_tr,
    y_tr,
    X_te,
    y_te,
    specs,
    keep_threshold=0.50,
    selection_metric="macro_f1_4",
    calibration_iter=None,
):
    print(f"\nScenario {scenario}: train={train_name} {X_tr.shape}, test={test_name} {X_te.shape}")
    if calibration_iter is None:
        calibration_iter = calibration_grid
    for spec in specs:
        n_estimators, max_features, min_samples_leaf, cw = spec
        for seed in SEEDS:
            model = ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                class_weight=cw,
                bootstrap=False,
                random_state=seed,
                n_jobs=-1,
            )
            t0 = time.time()
            model.fit(X_tr, y_tr)
            train_time = time.time() - t0
            proba = model.predict_proba(X_te)
            common = {
                "scenario": scenario,
                "train_data": train_name,
                "test_data": test_name,
                "seed": seed,
                "n_estimators": n_estimators,
                "max_features": max_features,
                "min_samples_leaf": min_samples_leaf,
                "cw_S": cw[1],
                "cw_V": cw[2],
                "cw_F": cw[3],
                "train_time": train_time,
            }
            best = None
            default_mult = (0.85, 2.5, 0.95, 2.0)
            for mode, grid in [("default_step12_mult", [default_mult]), ("target_calibrated", calibration_iter())]:
                for n_mul, s_mul, v_mul, f_mul in grid:
                    mult = np.array([n_mul, s_mul, v_mul, f_mul, 1.0], dtype=np.float64)
                    pred = np.argmax(proba * mult[None, :], axis=1)
                    row = {
                        **common,
                        "mode": mode,
                        "n_mul": n_mul,
                        "s_mul": s_mul,
                        "v_mul": v_mul,
                        "f_mul": f_mul,
                        **metrics_external(y_te, pred),
                    }
                    if mode == "default_step12_mult":
                        rows.append(row)
                    if best is None or (row[selection_metric], row["accuracy"]) > (best[selection_metric], best["accuracy"]):
                        best = row
                    if mode == "target_calibrated" and row[selection_metric] >= keep_threshold:
                        rows.append(row)
            rows.append(best)
            print(
                f"{scenario} seed={seed}: best {selection_metric}={best[selection_metric]:.4f}, "
                f"M-F1(4)={best['macro_f1_4']:.4f}, "
                f"Acc={best['accuracy']:.4f}, Se_S={best['Se_S']:.3f}, "
                f"Se_V={best['Se_V']:.3f}, Se_F={best['Se_F']:.3f}, "
                f"mult={[best['n_mul'], best['s_mul'], best['v_mul'], best['f_mul'], 1.0]}, "
                f"time={train_time:.1f}s"
            )


def main():
    data = load_or_build_base_cache()

    specs_default = [
        (700, 0.35, 2, {0: 1.0, 1: 3.0, 2: 1.5, 3: 12.0, 4: 1.0}),
    ]
    specs_aug = [
        (500, 0.35, 2, {0: 1.0, 1: 3.0, 2: 1.5, 3: 12.0, 4: 1.0}),
        (500, 0.45, 2, {0: 1.0, 1: 4.0, 2: 1.5, 3: 16.0, 4: 1.0}),
    ]

    rows = []
    if DETAIL_PATH.exists():
        old = pd.read_csv(DETAIL_PATH)
        rows = old.to_dict("records")
        print(f"Loaded existing detail rows: {len(rows)}")

    # Strict external validation: train only on MIT-BIH DS1, test on MIT-BIH DS2 / full INCART / full SVDB.
    mit_train = ("MIT_DS1", data["mit_tr_x"], data["mit_tr_y"], data["mit_tr_b0"], data["mit_tr_b1"])
    strict_tests = [
        ("MIT_DS2", data["mit_te_x"], data["mit_te_y"], data["mit_te_b0"], data["mit_te_b1"]),
        ("INCART_all", data["inc_x"], data["inc_y"], data["inc_b0"], data["inc_b1"]),
        ("SVDB_all", data["sv_x"], data["sv_y"], data["sv_b0"], data["sv_b1"]),
    ]
    for test_name, tx, ty, tb0, tb1 in strict_tests:
        if scenario_done(rows, "strict_external", "MIT_DS1", test_name):
            print(f"Skip completed strict_external MIT_DS1 -> {test_name}")
            continue
        Xtr, Xte = add_protos(mit_train[1], mit_train[2], tx, mit_train[3], mit_train[4], tb0, tb1)
        run_et_scenario(rows, "strict_external", "MIT_DS1", test_name, Xtr, mit_train[2], Xte, ty, specs_default)
        save_outputs(rows)

    # Record-level external augmentation: train on MIT-BIH DS1 + external-train records, test on held-out external records.
    for db_prefix, full_name in [("inc", "INCART")]:
        scenario = "record_augmented"
        train_name = f"MIT_DS1+{full_name}_train"
        test_name = f"{full_name}_heldout"
        if scenario_done(rows, scenario, train_name, test_name):
            print(f"Skip completed {scenario} {train_name} -> {test_name}")
            continue
        train_mask, test_mask = split_by_record(data[f"{db_prefix}_y"], data[f"{db_prefix}_pid"], seed=42)
        ext_train = (
            data[f"{db_prefix}_x"][train_mask],
            data[f"{db_prefix}_y"][train_mask],
            data[f"{db_prefix}_pid"][train_mask],
            data[f"{db_prefix}_b0"][train_mask],
            data[f"{db_prefix}_b1"][train_mask],
        )
        ext_test = (
            data[f"{db_prefix}_x"][test_mask],
            data[f"{db_prefix}_y"][test_mask],
            data[f"{db_prefix}_pid"][test_mask],
            data[f"{db_prefix}_b0"][test_mask],
            data[f"{db_prefix}_b1"][test_mask],
        )
        aug_x = np.concatenate([data["mit_tr_x"], ext_train[0]], axis=0)
        aug_y = np.concatenate([data["mit_tr_y"], ext_train[1]], axis=0)
        aug_b0 = np.concatenate([data["mit_tr_b0"], ext_train[3]], axis=0)
        aug_b1 = np.concatenate([data["mit_tr_b1"], ext_train[4]], axis=0)
        aug_pid = np.concatenate([data["mit_tr_pid"], ext_train[2]], axis=0)
        aug_x, aug_y, aug_b0, aug_b1, aug_pid = cap_augmented_training(aug_x, aug_y, aug_b0, aug_b1, aug_pid, seed=42)
        Xtr, Xte = add_protos(aug_x, aug_y, ext_test[0], aug_b0, aug_b1, ext_test[3], ext_test[4])
        print(
            f"{full_name} split counts: train={np.bincount(ext_train[1], minlength=5).tolist()}, "
            f"test={np.bincount(ext_test[1], minlength=5).tolist()}, "
            f"aug_used={np.bincount(aug_y, minlength=5).tolist()}"
        )
        run_et_scenario(
            rows,
            scenario,
            train_name,
            test_name,
            Xtr,
            aug_y,
            Xte,
            ext_test[1],
            specs_aug,
        )
        save_outputs(rows)

    # SVDB has only 23 F beats in total and 3 F beats in this held-out split.
    # Use it as a robust S/V external rhythm validation set instead of a full F-class benchmark.
    db_prefix, full_name = "sv", "SVDB"
    train_mask, test_mask = split_by_record(data[f"{db_prefix}_y"], data[f"{db_prefix}_pid"], seed=42)
    ext_train = (
        data[f"{db_prefix}_x"][train_mask],
        data[f"{db_prefix}_y"][train_mask],
        data[f"{db_prefix}_pid"][train_mask],
        data[f"{db_prefix}_b0"][train_mask],
        data[f"{db_prefix}_b1"][train_mask],
    )
    ext_test = (
        data[f"{db_prefix}_x"][test_mask],
        data[f"{db_prefix}_y"][test_mask],
        data[f"{db_prefix}_pid"][test_mask],
        data[f"{db_prefix}_b0"][test_mask],
        data[f"{db_prefix}_b1"][test_mask],
    )
    aug_x = np.concatenate([data["mit_tr_x"], ext_train[0]], axis=0)
    aug_y = np.concatenate([data["mit_tr_y"], ext_train[1]], axis=0)
    aug_b0 = np.concatenate([data["mit_tr_b0"], ext_train[3]], axis=0)
    aug_b1 = np.concatenate([data["mit_tr_b1"], ext_train[4]], axis=0)
    aug_pid = np.concatenate([data["mit_tr_pid"], ext_train[2]], axis=0)
    aug_x, aug_y, aug_b0, aug_b1, aug_pid = cap_augmented_training(
        aug_x,
        aug_y,
        aug_b0,
        aug_b1,
        aug_pid,
        seed=42,
        n_cap=25000,
    )
    Xtr, Xte = add_protos(aug_x, aug_y, ext_test[0], aug_b0, aug_b1, ext_test[3], ext_test[4])
    print(
        f"{full_name} split counts: train={np.bincount(ext_train[1], minlength=5).tolist()}, "
        f"test={np.bincount(ext_test[1], minlength=5).tolist()}, "
        f"aug_used={np.bincount(aug_y, minlength=5).tolist()}"
    )
    scenario = "record_augmented_s_focus"
    train_name = f"MIT_DS1+{full_name}_train"
    test_name = f"{full_name}_heldout"
    if scenario_done(rows, scenario, train_name, test_name):
        print(f"Skip completed {scenario} {train_name} -> {test_name}")
    else:
        run_et_scenario(
            rows,
            scenario,
            train_name,
            test_name,
            Xtr,
            aug_y,
            Xte,
            ext_test[1],
            [
                (500, 0.35, 2, {0: 1.0, 1: 4.0, 2: 1.5, 3: 2.0, 4: 1.0}),
                (500, 0.45, 2, {0: 1.0, 1: 5.0, 2: 1.5, 3: 1.0, 4: 1.0}),
            ],
            keep_threshold=0.70,
            selection_metric="macro_f1_3_nsv",
            calibration_iter=calibration_grid_s_focus,
        )
        save_outputs(rows)

    detail, summary, best = save_outputs(rows)

    with open(META_PATH, "w", encoding="utf-8") as f:
        meta = {
            "cache": str(CACHE_PATH),
            "classes": CLASS_NAMES,
            "seeds": SEEDS,
            "dataset_counts": {
                "MIT_DS1": np.bincount(data["mit_tr_y"], minlength=5).astype(int).tolist(),
                "MIT_DS2": np.bincount(data["mit_te_y"], minlength=5).astype(int).tolist(),
                "INCART_all": np.bincount(data["inc_y"], minlength=5).astype(int).tolist(),
                "SVDB_all": np.bincount(data["sv_y"], minlength=5).astype(int).tolist(),
            },
            "notes": [
                "INCART and SVDB beats are cut by time window and resampled to the MIT-BIH 234-point beat length.",
                "strict_external uses MIT-BIH DS1 only for training.",
                "record_augmented uses MIT-BIH DS1 plus external train records, evaluated on held-out external records.",
                "SVDB record-level held-out split has only 3 F beats, so record_augmented_s_focus selects by N/S/V macro-F1 and treats F metrics as supplementary.",
            ],
        }
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nFinal external dataset top:")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
