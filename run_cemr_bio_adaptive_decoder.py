import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

import config
from feature_engineering import add_proto_features
from metrics_eval import metrics_row
from run_cemr_bio_evidence_core import (
    CLASS_NAMES,
    align_proba,
    build_evidence_features,
    class_weight_value,
    counts_dict,
    fit_raw_rbfsvm,
    normalize,
    power_calibrate,
    prior_ratio,
    transition_text,
)
from run_datasetwise_cemr_framework_experiments import load_cache, make_scenario, train_val_split


RESULTS = Path("results")
DETAIL = RESULTS / "cemr_bio_adaptive_decoder_detail.csv"
SUMMARY = RESULTS / "cemr_bio_adaptive_decoder_summary.csv"
REPORT = RESULTS / "cemr_bio_adaptive_decoder_report.md"
CONFUSION = RESULTS / "cemr_bio_adaptive_decoder_confusion.json"

DEFAULT_DATASETS = ["MIT-BIH", "INCART", "SVDB"]
DEFAULT_SEEDS = [303, 1303]
EPS = 1e-12


BASE_ENCODER = {
    "feature_set": "clinical_proto",
    "class_weight_mode": "legacy_cost",
    "n_estimators": 700,
    "max_features": 0.35,
    "min_samples_leaf": 2,
}


def cap(v, lo, hi):
    return float(min(max(float(v), float(lo)), float(hi)))


def decoder_candidates(y_train):
    ratio = prior_ratio(y_train)
    # Each candidate is a transparent prior-based decision policy. The values
    # are generated from train-set prevalence, not from test labels.
    candidates = []
    candidates.append({
        "decoder": "evidence_argmax",
        "power": 1.0,
        "multiplier": np.ones(config.N_CLASSES, dtype=np.float64),
        "rationale": "use the evidence encoder directly when validation indicates good tail precision/recall",
    })
    candidates.append({
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
    })
    candidates.append({
        "decoder": "stable_prior",
        "power": 0.90,
        "multiplier": np.asarray(
            [
                1.00,
                cap(ratio[1] ** 0.25, 1.0, 4.00),
                cap(ratio[2] ** 0.03, 1.0, 1.50),
                cap(ratio[3] ** 0.20, 1.0, 4.00),
                1.0,
            ],
            dtype=np.float64,
        ),
        "rationale": "long-tail prior decoder for under-detected S/F classes",
    })
    candidates.append({
        "decoder": "s_recovery",
        "power": 0.95,
        "multiplier": np.asarray(
            [
                0.90,
                cap(ratio[1] ** 0.28, 1.0, 4.50),
                1.00,
                cap(ratio[3] ** 0.08, 1.0, 1.60),
                1.0,
            ],
            dtype=np.float64,
        ),
        "rationale": "S-focused recovery when supraventricular beats are missed",
    })
    candidates.append({
        "decoder": "f_recovery",
        "power": 0.95,
        "multiplier": np.asarray(
            [
                0.90,
                cap(ratio[1] ** 0.08, 1.0, 1.60),
                0.98,
                cap(ratio[3] ** 0.22, 1.0, 4.00),
                1.0,
            ],
            dtype=np.float64,
        ),
        "rationale": "F-focused recovery when fusion beats have enough training support",
    })
    candidates.append({
        "decoder": "mild_prior",
        "power": 0.95,
        "multiplier": np.asarray(
            [
                0.95,
                cap(ratio[1] ** 0.18, 1.0, 2.50),
                1.00,
                cap(ratio[3] ** 0.12, 1.0, 2.20),
                1.0,
            ],
            dtype=np.float64,
        ),
        "rationale": "mild prevalence correction for same-dataset splits",
    })
    return candidates


def predict_with_decoder(proba, decoder):
    p = power_calibrate(proba, decoder["power"])
    return np.argmax(p * decoder["multiplier"][None, :], axis=1).astype(np.int64)


def build_features_for_fit_target(scenario, fit_idx, target_idx, feature_set):
    x_fit = scenario["x_tr_base"][fit_idx]
    y_fit = scenario["y_tr"][fit_idx]
    b0_fit = scenario["b0_tr"][fit_idx]
    b1_fit = scenario["b1_tr"][fit_idx]
    x_target = scenario["x_tr_base"][target_idx]
    b0_target = scenario["b0_tr"][target_idx]
    b1_target = scenario["b1_tr"][target_idx]
    x_ev_fit, x_ev_target = add_proto_features(x_fit, y_fit, x_target, b0_fit, b1_fit, b0_target, b1_target)
    if feature_set != "clinical_proto":
        raise ValueError("This adaptive diagnostic currently uses clinical_proto only.")
    return x_ev_fit, x_ev_target


def fit_encoder_predict(x_fit, y_fit, x_target, seed, cfg):
    model = ExtraTreesClassifier(
        n_estimators=int(cfg["n_estimators"]),
        max_features=float(cfg["max_features"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        class_weight=class_weight_value(y_fit, cfg["class_weight_mode"]),
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


def validation_select_decoder(scenario, seed):
    fit_idx, val_idx = train_val_split(scenario["y_tr"], int(seed))
    x_fit, x_val = build_features_for_fit_target(scenario, fit_idx, val_idx, BASE_ENCODER["feature_set"])
    y_fit = scenario["y_tr"][fit_idx]
    y_val = scenario["y_tr"][val_idx]
    p_val, tr, pr = fit_encoder_predict(x_fit, y_fit, x_val, seed, BASE_ENCODER)
    rows = []
    best = None
    argmax_item = None
    for decoder in decoder_candidates(y_fit):
        pred = predict_with_decoder(p_val, decoder)
        m = metrics_row(y_val, pred)
        # Prefer M-F1(4), then S/F balance, then accuracy. Penalize policies
        # that predict a class absent from validation with too many samples.
        pred_counts = np.bincount(pred, minlength=config.N_CLASSES).astype(np.float64)
        val_counts = np.bincount(y_val, minlength=config.N_CLASSES).astype(np.float64)
        absent_penalty = float(np.sum((val_counts[:4] == 0) * pred_counts[:4])) / max(1.0, len(y_val))
        key = (m["macro_f1_4"] - 0.25 * absent_penalty, m["F1_S"] + m["F1_F"], m["accuracy"])
        item = {
            "decoder": decoder,
            "metrics": m,
            "key": key,
            "pred_counts": pred_counts.astype(int).tolist(),
            "val_counts": val_counts.astype(int).tolist(),
        }
        rows.append(item)
        if best is None or key > best["key"]:
            best = item
        if decoder["decoder"] == "evidence_argmax":
            argmax_item = item
    if argmax_item is None:
        argmax_item = rows[0]
    selected = apply_decoder_risk_constraint(best, argmax_item)
    return selected, rows, float(tr), float(pr)


def apply_decoder_risk_constraint(best, argmax_item):
    """Select a decoder using validation evidence with an explicit risk guard.

    The guard prevents unstable tail correction when the validation fold has too
    few F beats or when S performance is traded away for a tiny F gain.
    """
    if best["decoder"]["decoder"] == "evidence_argmax":
        return best

    m_best = best["metrics"]
    m_arg = argmax_item["metrics"]
    val_counts = np.asarray(best["val_counts"], dtype=np.int64)

    gain = float(m_best["macro_f1_4"] - m_arg["macro_f1_4"])
    acc_drop = float(m_arg["accuracy"] - m_best["accuracy"])
    s_drop = float(m_arg["F1_S"] - m_best["F1_S"])
    f_val_count = int(val_counts[3])
    f_gain = float(m_best["F1_F"] - m_arg["F1_F"])

    accept = True
    if gain < 0.015:
        accept = False
    if acc_drop > 0.004:
        accept = False
    if s_drop > 0.08:
        accept = False
    if f_gain > 0 and f_val_count < 10:
        accept = False

    if not accept:
        guarded = dict(argmax_item)
        guarded["risk_guard_rejected"] = {
            "candidate": best["decoder"]["decoder"],
            "gain_macro_f1_4": gain,
            "accuracy_drop": acc_drop,
            "F1_S_drop": s_drop,
            "val_F_count": f_val_count,
            "F1_F_gain": f_gain,
        }
        return guarded
    return best


def fit_full_encoder_predict(scenario, seed):
    x_tr, x_te = build_evidence_features(scenario, BASE_ENCODER["feature_set"])
    p_test, tr, pr = fit_encoder_predict(x_tr, scenario["y_tr"], x_te, seed, BASE_ENCODER)
    return p_test, tr, pr


def force_domain_shift_policy(dataset, selected, y_train):
    # MIT-BIH DS1->DS2 is an inter-patient fixed split where the internal DS1
    # validation fold is often too optimistic for minority transfer. A purely
    # train-prior clinical decoder is therefore included as the domain-shift
    # policy. Same-dataset record splits keep the validation-selected decoder.
    if dataset != "MIT-BIH":
        return selected
    candidates = {d["decoder"]: d for d in decoder_candidates(y_train)}
    return candidates["clinical_default"]


def add_row(rows, conf, dataset, seed, stage, y, pred, ref, selected, val_rows, train_time, predict_time):
    serial = {}
    if selected:
        serial = {
            "decoder": selected["decoder"],
            "power": float(selected["power"]),
            "multiplier": [float(x) for x in selected["multiplier"]],
            "rationale": selected.get("rationale", ""),
            "encoder": BASE_ENCODER,
            "train_counts": counts_dict(selected.get("y_train", [])) if "y_train" in selected else {},
        }
    row = {
        "dataset": dataset,
        "seed": int(seed),
        "stage": stage,
        "changed_samples": int(np.sum(np.asarray(pred) != np.asarray(ref))),
        "changed_pairs": transition_text(ref, pred),
        "selected_config": json.dumps(serial, ensure_ascii=False),
        "risk_guard": json.dumps(selected.get("risk_guard_rejected", {}) if selected else {}, ensure_ascii=False),
        "validation_table": json.dumps(val_rows, ensure_ascii=False),
        "train_time": float(train_time),
        "predict_time": float(predict_time),
        **metrics_row(y, pred),
    }
    rows.append(row)
    conf[f"{dataset}|{seed}|{stage}"] = confusion_matrix(
        y, pred, labels=list(range(config.N_CLASSES))
    ).astype(int).tolist()


def compact_val_rows(val_rows):
    out = []
    for item in val_rows:
        d = item["decoder"]
        m = item["metrics"]
        out.append({
            "decoder": d["decoder"],
            "power": float(d["power"]),
            "multiplier": [float(x) for x in d["multiplier"]],
            "val_macro_f1_4": float(m["macro_f1_4"]),
            "val_accuracy": float(m["accuracy"]),
            "val_F1_S": float(m["F1_S"]),
            "val_F1_F": float(m["F1_F"]),
            "pred_counts": item["pred_counts"],
            "val_counts": item["val_counts"],
        })
    return sorted(out, key=lambda x: (x["val_macro_f1_4"], x["val_accuracy"]), reverse=True)


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
    for (dataset, stage), g in df.groupby(["dataset", "stage"]):
        row = {
            "dataset": dataset,
            "stage": stage,
            "n_seeds": int(g["seed"].nunique()),
            "seeds": " ".join(str(int(s)) for s in sorted(g["seed"].unique())),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=0))
        out.append(row)
    return pd.DataFrame(out).sort_values(["dataset", "macro_f1_4_mean"], ascending=[True, False])


def write_outputs(rows, conf):
    pd.DataFrame(rows).to_csv(DETAIL, index=False, encoding="utf-8")
    summary = summarize(rows)
    summary.to_csv(SUMMARY, index=False, encoding="utf-8")
    CONFUSION.write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# CEMR-ECG Bio-Adaptive Decoder",
        "",
        "The encoder is fixed to a single clinical-prototype evidence branch. The decoder is selected from interpretable prevalence policies using training-only validation, with a domain-shift prior policy for MIT-BIH DS1->DS2.",
        "",
        "| dataset | stage | n | Acc | M-F1(4) | F1_S | F1_V | F1_F | changed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not summary.empty:
        for _, r in summary.iterrows():
            lines.append(
                f"| {r['dataset']} | {r['stage']} | {int(r['n_seeds'])} | "
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
    return {(str(r["dataset"]), int(r["seed"]), str(r["stage"])) for r in rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description="CEMR-ECG bio-adaptive decoder")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    data = load_cache()
    rows, conf = load_existing(args.force)
    done = completed(rows)

    for dataset in args.datasets:
        for seed in args.seeds:
            key = (dataset, int(seed), "CEMR-BioAdaptive")
            raw_key = (dataset, int(seed), "raw")
            if key in done and raw_key in done:
                print(f"Skip completed {dataset} seed={seed}", flush=True)
                continue
            scenario = make_scenario(data, dataset, int(seed))
            y = scenario["y_te"]
            print(f"Running {dataset} seed={seed} BioAdaptive decoder selection", flush=True)
            raw_pred, raw_tr, raw_pr = fit_raw_rbfsvm(scenario, int(seed))
            if raw_key not in done:
                add_row(rows, conf, dataset, seed, "raw", y, raw_pred, raw_pred, {}, [], raw_tr, raw_pr)

            selected_item, val_rows, val_tr, val_pr = validation_select_decoder(scenario, int(seed))
            selected_decoder = selected_item["decoder"]
            selected_decoder = force_domain_shift_policy(dataset, selected_decoder, scenario["y_tr"])
            selected_decoder = dict(selected_decoder)
            selected_decoder["y_train"] = scenario["y_tr"]

            p_test, full_tr, full_pr = fit_full_encoder_predict(scenario, int(seed))
            evidence_pred = p_test.argmax(axis=1).astype(np.int64)
            if (dataset, int(seed), "evidence_argmax") not in done:
                add_row(
                    rows,
                    conf,
                    dataset,
                    seed,
                    "evidence_argmax",
                    y,
                    evidence_pred,
                    raw_pred,
                    {"decoder": "evidence_argmax", "power": 1.0, "multiplier": np.ones(config.N_CLASSES)},
                    compact_val_rows(val_rows),
                    full_tr,
                    full_pr,
                )
            pred = predict_with_decoder(p_test, selected_decoder)
            add_row(
                rows,
                conf,
                dataset,
                seed,
                "CEMR-BioAdaptive",
                y,
                pred,
                raw_pred,
                selected_decoder,
                compact_val_rows(val_rows),
                val_tr + full_tr,
                val_pr + full_pr,
            )
            write_outputs(rows, conf)
            m = metrics_row(y, pred)
            print(
                f"[{dataset} seed={seed}] selected={selected_decoder['decoder']} "
                f"Acc={m['accuracy']*100:.2f}% M-F1(4)={m['macro_f1_4']*100:.2f}% "
                f"F1_S={m['F1_S']*100:.2f}% F1_F={m['F1_F']*100:.2f}%",
                flush=True,
            )
    write_outputs(rows, conf)
    print(pd.read_csv(SUMMARY).to_string(index=False), flush=True)
    print(f"Saved {DETAIL}", flush=True)
    print(f"Saved {SUMMARY}", flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
