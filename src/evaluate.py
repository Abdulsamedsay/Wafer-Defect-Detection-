"""
src/evaluate.py

Shared evaluation utilities used by baseline, CNN, and dashboard.
All evaluation goes through these functions so metrics are consistent
across every experiment logged in wiki/experiment-log.md.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_loader import DEFECT_CLASSES, LABEL_MAP_INV


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = DEFECT_CLASSES,
) -> dict:
    """
    Compute all evaluation metrics for one experiment.

    Returns a dict with:
      accuracy, macro_f1, weighted_f1,
      macro_precision, macro_recall,
      per_class: {class_name: {precision, recall, f1, support}}
    """
    metrics = {
        "accuracy":         accuracy_score(y_true, y_pred),
        "macro_f1":         f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "weighted_f1":      f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision":  precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "macro_recall":     recall_score(y_true, y_pred, average="macro",       zero_division=0),
    }

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class"] = {
        cls: {
            "precision": report[cls]["precision"],
            "recall":    report[cls]["recall"],
            "f1":        report[cls]["f1-score"],
            "support":   int(report[cls]["support"]),
        }
        for cls in class_names
    }
    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Results") -> None:
    """Print a formatted metrics summary."""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision:   {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:      {metrics['macro_recall']:.4f}")
    print(f"\n  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*55}")
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10,}")
    print(f"{'='*55}\n")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = DEFECT_CLASSES,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    normalize: bool = True,
) -> None:
    """
    Plot and optionally save a confusion matrix.

    Parameters
    ----------
    normalize : if True, shows row-normalized percentages (better for imbalanced data)
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = None

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_display,
        annot=True, fmt=fmt,
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=ax,
        vmin=0, vmax=vmax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


# ── Comparison table ──────────────────────────────────────────────────────────

def compare_experiments(experiments: dict[str, dict]) -> None:
    """
    Print a side-by-side comparison table for multiple experiments.

    Parameters
    ----------
    experiments : {experiment_name: metrics_dict}
    """
    print(f"\n{'='*70}")
    print("  Model Comparison")
    print(f"{'='*70}")
    header = f"  {'Model':<22} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}"
    print(header)
    print(f"  {'-'*66}")
    for name, m in experiments.items():
        print(f"  {name:<22} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['weighted_f1']:>12.4f}")
    print(f"{'='*70}\n")
