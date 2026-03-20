"""
evaluate.py
-----------
Evaluation utilities: metrics, confusion matrix, ROC curve.

All plots are saved to ``results/`` as high-resolution PNGs.
A ``metrics.json`` summary is also written there.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless – no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
TARGET       = "stroke"

# ── plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGSIZE = (7, 5)
DPI     = 150


# ── public API ─────────────────────────────────────────────────────────────────

def evaluate(
    predictor: TabularPredictor,
    test_df: pd.DataFrame,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """
    Run full evaluation on *test_df* and save all artefacts.

    Steps
    -----
    1. Generate class-probability predictions.
    2. Compute classification metrics (ROC-AUC, F1, accuracy, AP).
    3. Save ``metrics.json``.
    4. Plot and save confusion matrix PNG.
    5. Plot and save ROC curve PNG.

    Parameters
    ----------
    predictor : TabularPredictor
        Fitted AutoGluon predictor.
    test_df : pd.DataFrame
        Test split (must include the target column).
    results_dir : Path
        Directory where artefacts are written.

    Returns
    -------
    dict
        Metrics dictionary (also written to ``results/metrics.json``).
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    y_true   = test_df[TARGET].values
    y_proba  = predictor.predict_proba(test_df)

    # AutoGluon returns a DataFrame with one column per class; grab P(stroke=1)
    if isinstance(y_proba, pd.DataFrame):
        y_score = y_proba[1].values
    else:
        y_score = np.array(y_proba)

    # Default threshold = 0.5 for hard predictions
    y_pred = (y_score >= 0.5).astype(int)

    metrics = _compute_metrics(y_true, y_pred, y_score)
    _save_metrics(metrics, results_dir)
    _plot_confusion_matrix(y_true, y_pred, results_dir)
    _plot_roc_curve(y_true, y_score, metrics["roc_auc"], results_dir)

    logger.info("Evaluation complete. Metrics: %s", metrics)
    return metrics


# ── private helpers ────────────────────────────────────────────────────────────

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """Return a flat dictionary of scalar evaluation metrics."""
    return {
        "roc_auc":           round(float(roc_auc_score(y_true, y_score)), 4),
        "average_precision":  round(float(average_precision_score(y_true, y_score)), 4),
        "f1_macro":          round(float(f1_score(y_true, y_pred, average="macro",  zero_division=0)), 4),
        "f1_stroke":         round(float(f1_score(y_true, y_pred, pos_label=1,      zero_division=0)), 4),
        "accuracy":          round(float(accuracy_score(y_true, y_pred)), 4),
        "positive_rate_true": round(float(y_true.mean()), 4),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["No Stroke", "Stroke"],
            zero_division=0,
        ),
    }


def _save_metrics(metrics: dict, results_dir: Path) -> None:
    """Write metrics (excluding long string fields) to ``metrics.json``."""
    serialisable = {k: v for k, v in metrics.items() if not isinstance(v, str)}
    out = results_dir / "metrics.json"
    with open(out, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Metrics saved → %s", out)

    # Also print the classification report to console
    print("\n── Classification Report ──────────────────────────────────────")
    print(metrics["classification_report"])
    print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"  Average Precision  : {metrics['average_precision']:.4f}")
    print(f"  F1 (stroke class)  : {metrics['f1_stroke']:.4f}")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print("───────────────────────────────────────────────────────────────\n")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    results_dir: Path,
) -> None:
    """Save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = np.array(
        [[f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
         for row_v, row_p in zip(cm, cm_pct)]
    )
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues",
        xticklabels=["No Stroke", "Stroke"],
        yticklabels=["No Stroke", "Stroke"],
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label", labelpad=10)
    ax.set_title("Confusion Matrix", fontweight="bold", pad=12)
    fig.tight_layout()

    out = results_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", out)


def _plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    auc: float,
    results_dir: Path,
) -> None:
    """Save an annotated ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", labelpad=10)
    ax.set_ylabel("True Positive Rate", labelpad=10)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    out = results_dir / "roc_curve.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    logger.info("ROC curve saved → %s", out)
