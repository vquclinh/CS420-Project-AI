"""Evaluation metrics. Scores are on a 0–100 scale to match the published tables."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .config import LABELS

LABEL_IDS = list(range(len(LABELS)))


def score_predictions(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, zero_division=0
    )
    result = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "macro_f1": f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0) * 100,
        "weighted_f1": f1_score(y_true, y_pred, labels=LABEL_IDS, average="weighted", zero_division=0) * 100,
    }
    for index, label in enumerate(LABELS):
        result[f"f1_{label}"] = f1[index] * 100
        result[f"precision_{label}"] = precision[index] * 100
        result[f"recall_{label}"] = recall[index] * 100
        result[f"support_{label}"] = int(support[index])
    return result


def compute_metrics(eval_prediction) -> dict:
    """Metric function for the Trainer."""
    logits, labels = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    return score_predictions(labels, np.argmax(logits, axis=-1))


def confusion(y_true, y_pred) -> list:
    return confusion_matrix(y_true, y_pred, labels=LABEL_IDS).tolist()


def majority_baseline(y_true) -> dict:
    """Scores of always predicting CLEAN."""
    y_true = np.asarray(list(y_true))
    return score_predictions(y_true, np.zeros_like(y_true))


def bootstrap_interval(y_true, y_pred, n_resamples: int = 2000, seed: int = 0) -> dict:
    """95% confidence intervals of Accuracy and macro F1, via bootstrap on the test set."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    size = len(y_true)
    accuracy, macro = [], []
    for _ in range(n_resamples):
        index = rng.integers(0, size, size)
        accuracy.append((y_true[index] == y_pred[index]).mean() * 100)
        macro.append(f1_score(y_true[index], y_pred[index], labels=LABEL_IDS,
                              average="macro", zero_division=0) * 100)
    low, high = 2.5, 97.5
    return {
        "accuracy_ci95": [float(np.percentile(accuracy, low)), float(np.percentile(accuracy, high))],
        "macro_f1_ci95": [float(np.percentile(macro, low)), float(np.percentile(macro, high))],
    }


def paired_bootstrap_difference(y_true, pred_a, pred_b, n_resamples: int = 2000, seed: int = 0) -> dict:
    """95% confidence interval of the macro F1 difference (A minus B) on the same test samples."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    size = len(y_true)
    differences = []
    for _ in range(n_resamples):
        index = rng.integers(0, size, size)
        score_a = f1_score(y_true[index], pred_a[index], labels=LABEL_IDS, average="macro", zero_division=0)
        score_b = f1_score(y_true[index], pred_b[index], labels=LABEL_IDS, average="macro", zero_division=0)
        differences.append((score_a - score_b) * 100)
    observed = (f1_score(y_true, pred_a, labels=LABEL_IDS, average="macro", zero_division=0)
                - f1_score(y_true, pred_b, labels=LABEL_IDS, average="macro", zero_division=0)) * 100
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": float(observed), "ci95": [float(low), float(high)],
            "significant": bool(low > 0 or high < 0)}
