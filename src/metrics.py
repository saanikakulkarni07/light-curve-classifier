"""Evaluation metrics for the PLAsTiCC classification task."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, log_loss,
)
from config import DATA_CONFIG, VIS_CONFIG


def weighted_log_loss(y_true, y_pred_proba, class_weights=None):
    """Compute PLAsTiCC-style weighted multi-class log-loss.

    Parameters
    ----------
    y_true : np.ndarray (n_samples,)
        Integer class labels (encoded 0..n_classes-1).
    y_pred_proba : np.ndarray (n_samples, n_classes)
        Predicted probabilities.
    class_weights : dict or None
        {encoded_class_index: weight}. If None, uniform weights.

    Returns
    -------
    float
        Weighted log-loss (lower is better).
    """
    n_classes = y_pred_proba.shape[1]
    # Clip predictions to avoid log(0)
    eps = 1e-15
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)

    if class_weights is None:
        return log_loss(y_true, y_pred_proba)

    # Weighted version
    total_loss = 0.0
    total_weight = 0.0
    for cls in range(n_classes):
        mask = y_true == cls
        if not mask.any():
            continue
        w = class_weights.get(cls, 1.0)
        cls_loss = -np.mean(np.log(y_pred_proba[mask, cls]))
        total_loss += w * cls_loss
        total_weight += w

    return total_loss / total_weight if total_weight > 0 else 0.0


def per_class_metrics(y_true, y_pred, class_names=None):
    """Compute precision, recall, F1 per class.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    class_names : dict, optional
        {encoded_index: name}

    Returns
    -------
    pd.DataFrame
        Columns: class, precision, recall, f1, support
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for cls_str, metrics in report.items():
        if cls_str in ("accuracy", "macro avg", "weighted avg"):
            continue
        cls_idx = int(cls_str)
        name = class_names.get(cls_idx, str(cls_idx)) if class_names else str(cls_idx)
        rows.append({
            "class": name,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1-score"],
            "support": int(metrics["support"]),
        })
    return pd.DataFrame(rows)


def macro_f1(y_true, y_pred):
    """Compute macro-averaged F1 score."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, ax=None):
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    class_names : list of str
    normalize : bool
        If True, show percentages rather than counts.
    ax : matplotlib.axes.Axes, optional
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = ".2f"
    else:
        fmt = "d"

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10), dpi=VIS_CONFIG["dpi"])

    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, square=True, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    return ax


def classification_summary(y_true, y_pred, y_pred_proba, class_names=None):
    """Print a full classification report and return all metrics.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    y_pred_proba : np.ndarray (n_samples, n_classes)
    class_names : dict, optional

    Returns
    -------
    dict
        'log_loss', 'macro_f1', 'accuracy', 'per_class' (DataFrame)
    """
    ll = weighted_log_loss(y_true, y_pred_proba)
    mf1 = macro_f1(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    per_class = per_class_metrics(y_true, y_pred, class_names)

    print(f"Weighted Log-Loss: {ll:.4f}")
    print(f"Macro F1:          {mf1:.4f}")
    print(f"Accuracy:          {acc:.4f}")
    print()
    print(per_class.to_string(index=False))

    return {
        "log_loss": ll,
        "macro_f1": mf1,
        "accuracy": acc,
        "per_class": per_class,
    }
