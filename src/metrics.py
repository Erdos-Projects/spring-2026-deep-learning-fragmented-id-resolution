import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["pr_auc"] = float("nan")

    return metrics


def select_threshold(y_true, y_prob, mode="f1", precision_target=None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    sweep_df = pd.DataFrame(rows)

    if mode == "fixed":
        best_row = sweep_df.iloc[(sweep_df["threshold"] - 0.5).abs().argmin()]
        return float(best_row["threshold"]), best_row.to_dict(), sweep_df

    if mode == "precision_target":
        if precision_target is None:
            raise ValueError("precision_target mode requires precision_target.")
        candidates = sweep_df[sweep_df["precision"] >= precision_target]
        if candidates.empty:
            best_row = sweep_df.sort_values(["precision", "recall"], ascending=[False, False]).iloc[0]
        else:
            best_row = candidates.sort_values(["recall", "f1"], ascending=[False, False]).iloc[0]
        return float(best_row["threshold"]), best_row.to_dict(), sweep_df

    if mode == "f1":
        best_row = sweep_df.sort_values(["f1", "precision", "recall"], ascending=[False, False, False]).iloc[0]
        return float(best_row["threshold"]), best_row.to_dict(), sweep_df

    raise ValueError(f"Unknown threshold selection mode '{mode}'.")
