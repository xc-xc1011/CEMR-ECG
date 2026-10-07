import os
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

import config


N_CLASS = 0
S_CLASS = 1
V_CLASS = 2
F_CLASS = 3
Q_CLASS = 4
RR_EPS = 1e-8


def bandpass(sig, fs=None):
    if fs is None:
        fs = config.FS
    nyq = 0.5 * float(fs)
    high = min(config.BANDPASS_HIGH, nyq * 0.90)
    b, a = butter(
        config.BANDPASS_ORDER,
        [config.BANDPASS_LOW / nyq, high / nyq],
        btype="band",
    )
    return filtfilt(b, a, sig, axis=0)


def resample_1d(x, n_points=80):
    old = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    new = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    return np.interp(new, old, x).astype(np.float32)


def lead_morph_features(x):
    t = len(x)
    c = config.PRE_R
    left = x[:c]
    right = x[c:]
    qrs = x[max(0, c - 45):min(t, c + 55)]
    st = x[min(t, c + 25):min(t, c + 100)]
    dx = np.diff(x)
    absx = np.abs(x)
    qrs_abs = np.abs(qrs)
    peak = int(np.argmax(x))
    trough = int(np.argmin(x))
    r_amp = float(x[peak])
    min_amp = float(x[trough])
    left_energy = float(np.mean(left * left))
    right_energy = float(np.mean(right * right))
    qrs_energy = float(np.mean(qrs * qrs))
    total_energy = float(np.mean(x * x))
    max_qrs = float(np.max(qrs_abs) + 1e-8)
    return [
        r_amp,
        min_amp,
        r_amp - min_amp,
        peak - c,
        trough - c,
        float(np.mean(x)),
        float(np.std(x)),
        float(np.mean(absx)),
        float(np.mean(qrs)),
        float(np.std(qrs)),
        qrs_energy,
        total_energy,
        left_energy,
        right_energy,
        left_energy / (right_energy + 1e-8),
        float(np.max(dx)),
        float(np.min(dx)),
        float(np.max(dx) - np.min(dx)),
        float(np.mean(dx * dx)),
        float(np.mean(np.diff(np.signbit(x)))),
        float(np.mean(qrs_abs > 0.3 * max_qrs)),
        float(np.mean(qrs_abs > 0.5 * max_qrs)),
        float(np.mean(st)) if len(st) else 0.0,
        float(np.std(st)) if len(st) else 0.0,
        float(np.sum(qrs)),
        float(np.sum(np.abs(qrs))),
    ]


def rr_features(samples, i, fs=None):
    if fs is None:
        fs = config.FS
    fs = float(fs)
    rr_all = np.diff(samples).astype(np.float32) / fs
    rr_global = float(np.median(rr_all)) if len(rr_all) else 1.0
    pre_rr = ((samples[i] - samples[i - 1]) / fs) if i > 0 else rr_global
    post_rr = ((samples[i + 1] - samples[i]) / fs) if i < len(samples) - 1 else rr_global
    lo = max(0, i - 5)
    hi = min(len(samples) - 1, i + 5)
    local_rrs = np.diff(samples[lo:hi + 1]).astype(np.float32) / fs
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


def read_mit_dual_record(record_id):
    path = os.path.join(config.RAW_DIR, str(record_id))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    sig = rec.p_signal.astype(np.float32)
    if sig.ndim == 1:
        sig = sig[:, None]
    if sig.shape[1] == 1:
        sig = np.repeat(sig, 2, axis=1)
    sig = sig[:, :2]
    sig = bandpass(sig, config.FS).astype(np.float32)
    sig = (sig - sig.mean(axis=0, keepdims=True)) / (sig.std(axis=0, keepdims=True) + 1e-8)
    return sig, ann


def build_beat_feature(b0, b1, samples, i, fs=None):
    m0 = lead_morph_features(b0)
    m1 = lead_morph_features(b1)
    feats = []
    feats.extend(rr_features(samples, i, fs))
    feats.extend(m0)
    feats.extend(m1)
    feats.extend((np.asarray(m0) - np.asarray(m1)).tolist())
    feats.extend(resample_1d(b0, 80).tolist())
    feats.extend(resample_1d(b1, 80).tolist())
    feats.extend(resample_1d(np.diff(b0), 40).tolist())
    feats.extend(resample_1d(np.diff(b1), 40).tolist())
    return feats


def extract_mit_records(record_ids):
    beats0 = []
    beats1 = []
    features = []
    labels = []
    pids = []
    for rec_id in record_ids:
        sig, ann = read_mit_dual_record(rec_id)
        samples = np.asarray(ann.sample, dtype=np.int64)
        for i, (r_idx, sym) in enumerate(zip(samples, ann.symbol)):
            if sym not in config.AAMI_MAP:
                continue
            start = int(r_idx) - config.PRE_R
            end = int(r_idx) + config.POST_R
            if start < 0 or end > len(sig):
                continue
            beat = sig[start:end]
            if beat.shape[0] != config.WIN_LEN:
                continue
            b0 = beat[:, 0].astype(np.float32)
            b1 = beat[:, 1].astype(np.float32)
            labels.append(config.AAMI_MAP[sym])
            pids.append(rec_id)
            beats0.append(b0)
            beats1.append(b1)
            features.append(build_beat_feature(b0, b1, samples, i, config.FS))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(pids),
        np.asarray(beats0, dtype=np.float32),
        np.asarray(beats1, dtype=np.float32),
    )


def squared_distance_to_protos(B, proto):
    """Mean squared distance from each row of B to each prototype.

    Equivalent to ``((B[:, None, :] - proto[None, :, :]) ** 2).mean(axis=2)``
    but computed through the expansion
    ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b`` so that no
    ``(n_beats, n_protos, n_samples)`` temporary is ever materialised. The
    broadcast form needs several hundred megabytes per call on the full
    MIT-BIH training pool and exhausted memory on this machine.
    """
    B = np.asarray(B, dtype=np.float32)
    proto = np.asarray(proto, dtype=np.float32)
    b_sq = np.einsum("ij,ij->i", B, B)[:, None]
    p_sq = np.einsum("ij,ij->i", proto, proto)[None, :]
    dist = b_sq + p_sq - 2.0 * (B @ proto.T)
    # The expansion can produce small negative values through cancellation.
    return np.maximum(dist, 0.0, out=dist) / float(B.shape[1])


def add_proto_features(X_tr_base, y_tr, X_te_base, B0_tr, B1_tr, B0_te, B1_te):
    proto0 = []
    proto1 = []
    for cls in range(config.N_CLASSES):
        mask = y_tr == cls
        proto0.append(B0_tr[mask].mean(axis=0) if mask.any() else B0_tr.mean(axis=0))
        proto1.append(B1_tr[mask].mean(axis=0) if mask.any() else B1_tr.mean(axis=0))
    proto0 = np.asarray(proto0, dtype=np.float32)
    proto1 = np.asarray(proto1, dtype=np.float32)

    def calc(B0, B1):
        B0n = B0 / (np.linalg.norm(B0, axis=1, keepdims=True) + 1e-8)
        B1n = B1 / (np.linalg.norm(B1, axis=1, keepdims=True) + 1e-8)
        P0n = proto0 / (np.linalg.norm(proto0, axis=1, keepdims=True) + 1e-8)
        P1n = proto1 / (np.linalg.norm(proto1, axis=1, keepdims=True) + 1e-8)
        corr0 = B0n @ P0n.T
        corr1 = B1n @ P1n.T
        d0 = squared_distance_to_protos(B0, proto0)
        d1 = squared_distance_to_protos(B1, proto1)
        margins = np.stack([
            corr0[:, S_CLASS] - corr0[:, N_CLASS],
            corr0[:, F_CLASS] - corr0[:, N_CLASS],
            corr0[:, F_CLASS] - corr0[:, V_CLASS],
            corr1[:, S_CLASS] - corr1[:, N_CLASS],
            corr1[:, F_CLASS] - corr1[:, N_CLASS],
            corr1[:, F_CLASS] - corr1[:, V_CLASS],
            d0[:, N_CLASS] - d0[:, S_CLASS],
            d0[:, N_CLASS] - d0[:, F_CLASS],
            d0[:, V_CLASS] - d0[:, F_CLASS],
            d1[:, N_CLASS] - d1[:, S_CLASS],
            d1[:, N_CLASS] - d1[:, F_CLASS],
            d1[:, V_CLASS] - d1[:, F_CLASS],
        ], axis=1)
        return np.concatenate([corr0, corr1, np.log1p(d0), np.log1p(d1), margins], axis=1).astype(np.float32)

    return (
        np.concatenate([X_tr_base, calc(B0_tr, B1_tr)], axis=1).astype(np.float32),
        np.concatenate([X_te_base, calc(B0_te, B1_te)], axis=1).astype(np.float32),
    )


def load_or_build_mit_features(cache_path=None):
    if cache_path is None:
        cache_path = Path(config.RESULTS_DIR) / "mit_strong_features_cache.npz"
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["X_tr"], data["y_tr"], data["X_te"], data["y_te"]

    X_tr_base, y_tr, _pid_tr, B0_tr, B1_tr = extract_mit_records(config.DS1_RECORDS)
    X_te_base, y_te, _pid_te, B0_te, B1_te = extract_mit_records(config.DS2_RECORDS)
    X_tr, X_te = add_proto_features(X_tr_base, y_tr, X_te_base, B0_tr, B1_tr, B0_te, B1_te)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)
    return X_tr, y_tr, X_te, y_te
