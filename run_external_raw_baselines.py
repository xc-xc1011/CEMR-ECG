import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

import config
from metrics_eval import metrics_row


warnings.filterwarnings("ignore")

RESULTS = Path("results")
CACHE = RESULTS / "external_base_features_cache.npz"
DETAIL = RESULTS / "external_raw_baselines_detail.csv"
SUMMARY = RESULTS / "external_raw_baselines_summary.csv"
REPORT = RESULTS / "external_raw_baselines_report.md"
CONFUSION = RESULTS / "external_raw_baselines_confusion.json"

SEEDS = [303, 1303, 2303, 3303, 4303]
CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def raw_dual(data, prefix):
    return np.concatenate(
        [data[f"{prefix}_b0"].astype(np.float32), data[f"{prefix}_b1"].astype(np.float32)],
        axis=1,
    ).astype(np.float32)


def load_datasets():
    d = np.load(CACHE, allow_pickle=True)
    train_x = raw_dual(d, "mit_tr")
    train_y = d["mit_tr_y"].astype(np.int64)
    tests = {
        "INCART_all": (raw_dual(d, "inc"), d["inc_y"].astype(np.int64)),
        "SVDB_all": (raw_dual(d, "sv"), d["sv_y"].astype(np.int64)),
    }
    return train_x, train_y, tests


def specs(seed):
    out = [
        {
            "model": "LogReg_raw",
            "estimator": make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed),
            ),
            "config": "StandardScaler + LogisticRegression(C=1.0, solver=lbfgs, max_iter=1000), no class/sample weights.",
        },
        {
            "model": "LinearSVM_raw",
            "estimator": make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=5000, random_state=seed)),
            "config": "StandardScaler + LinearSVC(C=1.0, max_iter=5000), no class/sample weights.",
        },
        {
            "model": "RbfSVM_raw",
            "estimator": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed)),
            "config": "StandardScaler + SVC(C=1.0, kernel=rbf, gamma=scale), no class/sample weights.",
        },
        {
            "model": "KNN_raw_k7",
            "estimator": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="uniform", n_jobs=-1)),
            "config": "StandardScaler + KNN(k=7, weights=uniform), no class/sample weights.",
        },
        {
            "model": "RandomForest_raw",
            "estimator": RandomForestClassifier(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=-1,
            ),
            "config": "RandomForest(n_estimators=400, max_features=sqrt, min_samples_leaf=1), no class/sample weights.",
        },
        {
            "model": "ExtraTrees_raw",
            "estimator": ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=-1,
            ),
            "config": "ExtraTrees(n_estimators=500, max_features=sqrt, min_samples_leaf=1), no class/sample weights.",
        },
        {
            "model": "HGB_raw",
            "estimator": HistGradientBoostingClassifier(
                max_iter=260,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                early_stopping=True,
                random_state=seed,
            ),
            "config": "HistGradientBoosting(max_iter=260, lr=0.05, max_leaf_nodes=31, l2=0.05), no class/sample weights.",
        },
        {
            "model": "MLP_raw",
            "estimator": make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=512,
                    learning_rate_init=1e-3,
                    max_iter=120,
                    early_stopping=True,
                    n_iter_no_change=12,
                    random_state=seed,
                ),
            ),
            "config": "StandardScaler + MLP(256,128, relu, alpha=1e-4, lr=1e-3, max_iter=120), no class/sample weights.",
        },
    ]
    try:
        from xgboost import XGBClassifier

        out.append(
            {
                "model": "XGBoost_raw",
                "estimator": XGBClassifier(
                    n_estimators=350,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=-1,
                ),
                "config": "XGBoost(n_estimators=350, max_depth=6, lr=0.05, subsample=0.85, colsample=0.85), no class/sample weights.",
            }
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        out.append(
            {
                "model": "LightGBM_raw",
                "estimator": LGBMClassifier(
                    n_estimators=350,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=-1,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="multiclass",
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=-1,
                ),
                "config": "LightGBM(n_estimators=350, lr=0.05, num_leaves=31, subsample=0.85, colsample=0.85), no class/sample weights.",
            }
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier

        out.append(
            {
                "model": "CatBoost_raw",
                "estimator": CatBoostClassifier(
                    iterations=350,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="MultiClass",
                    random_seed=seed,
                    verbose=False,
                    thread_count=-1,
                ),
                "config": "CatBoost(iterations=350, depth=6, lr=0.05, loss=MultiClass), no class/sample weights.",
            }
        )
    except Exception:
        pass
    return out


def completed():
    if not DETAIL.exists():
        return set()
    df = pd.read_csv(DETAIL)
    return set(zip(df["dataset"].astype(str), df["model"].astype(str), df["seed"].astype(int)))


def append(row):
    pd.DataFrame([row]).to_csv(DETAIL, mode="a", index=False, header=not DETAIL.exists(), encoding="utf-8")


def load_conf():
    if CONFUSION.exists():
        return json.loads(CONFUSION.read_text(encoding="utf-8"))
    return {}


def save_conf(conf):
    CONFUSION.write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize():
    if not DETAIL.exists():
        return pd.DataFrame()
    detail = pd.read_csv(DETAIL)
    metrics = [
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
        "train_time",
        "predict_time",
    ]
    rows = []
    for (dataset, model), g in detail.groupby(["dataset", "model"]):
        row = {
            "dataset": dataset,
            "model": model,
            "n_seeds": int(g["seed"].nunique()),
            "seeds": " ".join(str(int(x)) for x in sorted(g["seed"].unique())),
            "feature_set": str(g["feature_set"].iloc[0]),
            "model_config": str(g["model_config"].iloc[0]),
            "train_size": int(g["train_size"].iloc[0]),
            "test_size": int(g["test_size"].iloc[0]),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=0))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["dataset", "macro_f1_4_mean", "accuracy_mean"], ascending=[True, False, False])
    out.to_csv(SUMMARY, index=False, encoding="utf-8")
    return out


def write_report():
    df = summarize()
    if df.empty:
        return
    cols = [
        "dataset",
        "model",
        "n_seeds",
        "accuracy_mean",
        "macro_f1_5_mean",
        "macro_f1_4_mean",
        "macro_f1_3_nsv_mean",
        "Se_N_mean",
        "Se_S_mean",
        "Se_V_mean",
        "Se_F_mean",
        "Se_Q_mean",
        "Pr_S_mean",
        "Pr_F_mean",
        "F1_N_mean",
        "F1_S_mean",
        "F1_V_mean",
        "F1_F_mean",
        "F1_Q_mean",
        "train_time_mean",
        "predict_time_mean",
    ]
    lines = [
        "# External Raw Baselines",
        "",
        "Training data: MIT-BIH DS1 raw dual-lead heartbeat waveform.",
        "",
        "Test data: INCART_all and SVDB_all.",
        "",
        "No CEMR-ECG hand-crafted RR/prototype/calibration modules are used in these baselines.",
        "",
    ]
    for dataset, g in df.groupby("dataset"):
        lines += [f"## {dataset}", "", g[cols].to_markdown(index=False), ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="INCART_all,SVDB_all")
    parser.add_argument("--methods", default="")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = parser.parse_args()

    wanted_datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}
    wanted_methods = {x.strip() for x in args.methods.split(",") if x.strip()}
    run_seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    x_train, y_train, tests = load_datasets()
    done = completed()
    conf = load_conf()
    print("train MIT_DS1", x_train.shape, np.bincount(y_train, minlength=5).tolist())
    for ds, (_x, y) in tests.items():
        if ds in wanted_datasets:
            print("test", ds, _x.shape, np.bincount(y, minlength=5).tolist())

    for seed in run_seeds:
        for spec in specs(seed):
            name = spec["model"]
            if wanted_methods and name not in wanted_methods:
                continue
            print(f"\n[fit] {name} seed={seed}")
            t0 = time.time()
            model = spec["estimator"]
            model.fit(x_train, y_train)
            train_time = time.time() - t0
            for dataset, (x_test, y_test) in tests.items():
                if dataset not in wanted_datasets:
                    continue
                key = (dataset, name, seed)
                if key in done:
                    print(f"skip {dataset} {name} seed={seed}")
                    continue
                t1 = time.time()
                pred = np.asarray(model.predict(x_test)).ravel().astype(np.int64)
                pred_time = time.time() - t1
                met = metrics_row(y_test, pred)
                row = {
                    "dataset": dataset,
                    "model": name,
                    "seed": seed,
                    "feature_set": "raw_dual_wave_flattened",
                    "model_config": spec["config"],
                    "train_data": "MIT-BIH_DS1",
                    "train_size": int(len(y_train)),
                    "test_size": int(len(y_test)),
                    "train_time": float(train_time),
                    "predict_time": float(pred_time),
                    **met,
                }
                append(row)
                done.add(key)
                conf[f"{dataset}|{name}|seed{seed}"] = confusion_matrix(
                    y_test, pred, labels=list(range(config.N_CLASSES))
                ).astype(int).tolist()
                save_conf(conf)
                print(
                    f"{dataset} {name} seed={seed}: Acc={met['accuracy']:.4f}, "
                    f"M-F1(4)={met['macro_f1_4']:.4f}, M-F1(3)={met['macro_f1_3_nsv']:.4f}, "
                    f"F1_S={met['F1_S']:.4f}, F1_F={met['F1_F']:.4f}, pred={pred_time:.1f}s"
                )
            summary = summarize()
            write_report()
            if not summary.empty:
                print("current top")
                print(
                    summary.groupby("dataset")
                    .head(5)[["dataset", "model", "n_seeds", "accuracy_mean", "macro_f1_4_mean", "macro_f1_3_nsv_mean", "F1_S_mean", "F1_F_mean"]]
                    .to_string(index=False)
                )

    summary = summarize()
    write_report()
    print("\nFinal external raw baseline summary")
    print(
        summary[["dataset", "model", "n_seeds", "accuracy_mean", "macro_f1_4_mean", "macro_f1_3_nsv_mean", "F1_S_mean", "F1_F_mean"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
