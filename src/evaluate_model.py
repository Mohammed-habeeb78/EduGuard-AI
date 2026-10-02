"""
evaluate_model.py
EduGuard AI – Student Academic Risk Prediction System

Provides evaluation utilities used both during training
and by the Streamlit dashboard for displaying charts.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
)


def build_comparison_dataframe(comparison_table: list[dict]) -> pd.DataFrame:
    """Convert the raw comparison list from the saved bundle into a DataFrame.

    Args:
        comparison_table: List of dicts produced during training,
                          each with keys Model, Accuracy, Precision,
                          Recall, F1-Score, ROC-AUC.

    Returns:
        Styled DataFrame ready for st.dataframe().
    """
    df = pd.DataFrame(comparison_table)
    # Round all numeric columns for display
    numeric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    return df


def confusion_matrix_to_df(cm: list[list[int]]) -> pd.DataFrame:
    """Wrap a raw confusion matrix list into a labelled DataFrame.

    Args:
        cm: 2×2 list [[TN, FP], [FN, TP]]

    Returns:
        DataFrame with row/column labels for display.
    """
    labels = ["Lower Risk (Actual)", "High Risk (Actual)"]
    cols   = ["Predicted: Lower Risk", "Predicted: High Risk"]
    return pd.DataFrame(np.array(cm), index=labels, columns=cols)


def get_roc_data(pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict | None:
    """Compute ROC curve data for Plotly.

    Args:
        pipeline: Fitted sklearn Pipeline with a *predict_proba* step.
        X_test:   Test features.
        y_test:   True labels.

    Returns:
        Dict with keys fpr, tpr, auc, or None if proba not available.
    """
    clf = pipeline.named_steps.get("clf")
    if clf is None or not hasattr(clf, "predict_proba"):
        return None

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc)}


def get_pr_data(pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict | None:
    """Compute Precision-Recall curve data for Plotly.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test:   Test features.
        y_test:   True labels.

    Returns:
        Dict with keys precision, recall, or None if proba not available.
    """
    clf = pipeline.named_steps.get("clf")
    if clf is None or not hasattr(clf, "predict_proba"):
        return None

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    return {
        "precision": precision.tolist(),
        "recall":    recall.tolist(),
    }


def summarise_metrics(metrics: dict) -> pd.DataFrame:
    """Return a one-row DataFrame of key scalar metrics for display.

    Args:
        metrics: Dict with keys accuracy, precision, recall,
                 f1_score, roc_auc (may be None).

    Returns:
        Single-row DataFrame.
    """
    row = {
        "Accuracy":  round(metrics.get("accuracy",  0), 4),
        "Precision": round(metrics.get("precision", 0), 4),
        "Recall":    round(metrics.get("recall",    0), 4),
        "F1-Score":  round(metrics.get("f1_score",  0), 4),
        "ROC-AUC":   round(metrics["roc_auc"], 4)
                     if metrics.get("roc_auc") is not None else "N/A",
    }
    return pd.DataFrame([row])
