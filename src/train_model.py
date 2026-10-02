"""
train_model.py
EduGuard AI – Student Academic Risk Prediction System

Trains multiple classifiers, selects the best by F1-score,
and persists the winning pipeline + metadata to disk.

Run from the project root:
    python src/train_model.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import preprocess

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "academic_risk_model.pkl")
META_PATH    = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "student-mat.csv")

# ─── Model Definitions ─────────────────────────────────────────────────────────

def build_pipelines() -> dict:
    """Return a dictionary of named sklearn Pipelines, one per algorithm."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


# ─── Evaluation Helper ──────────────────────────────────────────────────────────

def evaluate_pipeline(pipeline, X_test, y_test, model_name: str) -> dict:
    """Evaluate a fitted pipeline and return a metrics dictionary."""
    y_pred  = pipeline.predict(X_test)
    y_proba = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline.named_steps["clf"], "predict_proba")
        else None
    )

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    rpt  = classification_report(y_test, y_pred,
                                  target_names=["Lower Risk", "High Risk"])

    auc = None
    if y_proba is not None:
        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            pass

    print(f"\n── {model_name} ─────────────────────────────")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    if auc is not None:
        print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{rpt}")

    return {
        "model_name":  model_name,
        "accuracy":    float(acc),
        "precision":   float(prec),
        "recall":      float(rec),
        "f1_score":    float(f1),
        "roc_auc":     auc,
        "confusion_matrix": cm,
        "classification_report": rpt,
    }


# ─── Feature Importance ─────────────────────────────────────────────────────────

def get_feature_importance(pipeline, feature_names: list) -> list[dict]:
    """Extract feature importances from the classifier step if available.

    Returns a list of dicts sorted by importance descending.
    """
    clf = pipeline.named_steps["clf"]
    importances = None

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])

    if importances is None:
        return []

    paired = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [{"feature": f, "importance": round(v, 6)} for f, v in paired]


# ─── Main Training Routine ─────────────────────────────────────────────────────

def train(data_path: str = DATA_PATH) -> None:
    """Complete training workflow: preprocess → train → evaluate → save."""

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════╗")
    print("║       EduGuard AI – Model Training           ║")
    print("╚══════════════════════════════════════════════╝\n")

    data = preprocess(path=data_path)
    X_train       = data["X_train"]
    X_test        = data["X_test"]
    y_train       = data["y_train"]
    y_test        = data["y_test"]
    feature_names = data["feature_names"]
    encoders      = data["encoders"]

    # ── 2. Train all pipelines ────────────────────────────────────────────────
    pipelines = build_pipelines()
    results   = {}
    fitted    = {}

    print("\n── Training Models ───────────────────────────────────────────────────")
    for name, pipeline in pipelines.items():
        print(f"  Fitting: {name} …", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        print("done")
        fitted[name] = pipeline

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    print("\n── Evaluation Results ────────────────────────────────────────────────")
    for name, pipeline in fitted.items():
        metrics = evaluate_pipeline(pipeline, X_test, y_test, name)
        results[name] = metrics

    # ── 4. Select best model by F1-score ─────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_pipeline = fitted[best_name]
    best_metrics  = results[best_name]

    print(f"\n[INFO] Best model selected: {best_name}")
    print(f"  F1-Score : {best_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC  : {best_metrics.get('roc_auc', 'N/A')}")

    # ── 5. Feature importance for the best model ──────────────────────────────
    feature_importance = get_feature_importance(best_pipeline, feature_names)
    if feature_importance:
        print("\n── Top-10 Feature Importances ────────────────────────────────────────")
        for row in feature_importance[:10]:
            print(f"  {row['feature']:<20} {row['importance']:.6f}")

    # ── 6. Build comparison table ─────────────────────────────────────────────
    comparison_rows = []
    for name, m in results.items():
        comparison_rows.append({
            "Model":     name,
            "Accuracy":  round(m["accuracy"],  4),
            "Precision": round(m["precision"], 4),
            "Recall":    round(m["recall"],    4),
            "F1-Score":  round(m["f1_score"],  4),
            "ROC-AUC":   round(m["roc_auc"], 4) if m["roc_auc"] else None,
        })

    # ── 7. Persist model ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    save_bundle = {
        "pipeline":           best_pipeline,
        "feature_names":      feature_names,
        "encoders":           encoders,
        "best_model_name":    best_name,
        "all_results":        results,
        "comparison_table":   comparison_rows,
        "feature_importance": feature_importance,
    }
    joblib.dump(save_bundle, MODEL_PATH)
    print(f"\n[INFO] Model bundle saved to: {MODEL_PATH}")

    # ── 8. Persist metadata (JSON, human-readable) ────────────────────────────
    meta = {
        "best_model":         best_name,
        "feature_names":      feature_names,
        "best_metrics":       {
            k: v for k, v in best_metrics.items()
            if k != "classification_report"   # skip long string
        },
        "comparison_table":   comparison_rows,
        "feature_importance": feature_importance,
        "n_train":            int(len(X_train)),
        "n_test":             int(len(X_test)),
        "n_features":         int(len(feature_names)),
    }
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[INFO] Metadata saved to: {META_PATH}")

    print("\n╔══════════════════════════════════════════════╗")
    print("║  Training complete – ready to run app.py    ║")
    print("╚══════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    train()
