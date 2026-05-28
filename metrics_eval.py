import numpy as np

import config


CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def classification_metrics(y_true, y_pred, n_classes=None):
    if n_classes is None:
        n_classes = config.N_CLASSES
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.bincount(
        n_classes * y_true + y_pred,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    se = tp / (cm.sum(axis=1) + 1e-12)
    pp = tp / (cm.sum(axis=0) + 1e-12)
    f1 = 2 * se * pp / (se + pp + 1e-12)
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1_5": float(f1.mean()),
        "macro_f1_4": float(f1[:4].mean()),
        "macro_f1_3_nsv": float(f1[:3].mean()),
        "confusion_matrix": cm,
    }
    for i, name in enumerate(CLASS_NAMES[:n_classes]):
        out[f"Se_{name}"] = float(se[i])
        out[f"Pr_{name}"] = float(pp[i])
        out[f"F1_{name}"] = float(f1[i])
    return out


def metrics_row(y_true, y_pred):
    out = classification_metrics(y_true, y_pred)
    out.pop("confusion_matrix", None)
    return out
