"""Métricas de la clase premium y figuras de evaluación."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_with_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)


def compute_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Métricas de la clase premium (1) con el umbral de decisión dado."""
    y_pred = predict_with_threshold(proba, threshold)
    return {
        "f1_premium": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_premium": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_premium": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def report_dict(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, object]:
    y_pred = predict_with_threshold(proba, threshold)
    return classification_report(
        y_true,
        y_pred,
        target_names=["no_premium", "premium"],
        output_dict=True,
        zero_division=0,
    )


def make_figures(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, Figure]:
    """Matriz de confusión, curva ROC y curva precisión-recall."""
    y_pred = predict_with_threshold(proba, threshold)
    figures: dict[str, Figure] = {}

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["no premium", "premium"], ax=ax
    )
    ax.set_title(f"Matriz de confusión (umbral {threshold:g})")
    figures["confusion_matrix.png"] = fig

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax)
    ax.set_title("Curva ROC")
    figures["roc_curve.png"] = fig

    fig, ax = plt.subplots(figsize=(4.5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax)
    ax.set_title("Curva precisión-recall")
    figures["pr_curve.png"] = fig

    return figures
