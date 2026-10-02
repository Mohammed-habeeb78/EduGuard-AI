"""
prediction.py
EduGuard AI – Student Academic Risk Prediction System

Provides a prediction interface used by the Streamlit app.
Accepts a dictionary of raw student attributes, applies the
same preprocessing used during training, and returns the
predicted risk label, probability, and contributing factors.
"""

import os
import numpy as np
import pandas as pd
import joblib

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "academic_risk_model.pkl")

# ─── Label Mappings ─────────────────────────────────────────────────────────────
RISK_LABELS = {0: "Lower Academic Risk", 1: "High Academic Risk"}
RISK_EMOJI  = {0: "🟢", 1: "🔴"}


def load_model_bundle(path: str = MODEL_PATH) -> dict:
    """Load the saved model bundle from disk.

    Args:
        path: Path to the .pkl file written by train_model.py.

    Returns:
        Dictionary containing pipeline, feature_names, encoders,
        best_model_name, all_results, comparison_table, feature_importance.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'.\n"
            "Please run: python src/train_model.py"
        )
    return joblib.load(path)


def _encode_input(raw_input: dict, feature_names: list, encoders: dict) -> np.ndarray:
    """Transform raw user input into a feature vector.

    This mirrors the encoding performed in data_preprocessing.preprocess()
    so predictions are consistent with training.

    Args:
        raw_input:     Dictionary mapping feature name → raw value.
        feature_names: Ordered list of features the model expects.
        encoders:      Dict of {column_name: LabelEncoder} from training.

    Returns:
        1×N NumPy array ready for pipeline.predict().
    """
    row = {}
    for feature in feature_names:
        value = raw_input.get(feature)

        if value is None:
            raise ValueError(f"Missing input for feature: '{feature}'")

        # Apply the same label encoder used at training time
        if feature in encoders:
            le = encoders[feature]
            str_val = str(value)
            if str_val in le.classes_:
                row[feature] = le.transform([str_val])[0]
            else:
                # Unseen category → use the most frequent class (index 0)
                row[feature] = 0
        else:
            row[feature] = float(value)

    return np.array([row[f] for f in feature_names]).reshape(1, -1)


def predict_risk(raw_input: dict, bundle: dict) -> dict:
    """Predict academic risk for a single student.

    Args:
        raw_input: Dictionary of {feature_name: raw_value} as collected
                   from the Streamlit form.
        bundle:    Model bundle loaded by load_model_bundle().

    Returns:
        Dictionary with keys:
            label         – "Lower Academic Risk" or "High Academic Risk"
            emoji         – 🟢 or 🔴
            risk_class    – 0 or 1
            probability   – float probability of High Risk (class 1)
            top_factors   – list of {feature, importance} for the top 5 features
    """
    pipeline      = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    encoders      = bundle["encoders"]
    feat_imp      = bundle.get("feature_importance", [])

    # Build encoded feature vector
    X = _encode_input(raw_input, feature_names, encoders)

    # Prediction
    risk_class = int(pipeline.predict(X)[0])
    label = RISK_LABELS[risk_class]
    emoji = RISK_EMOJI[risk_class]

    # Probability (confidence)
    probability = None
    clf = pipeline.named_steps.get("clf")
    if clf is not None and hasattr(clf, "predict_proba"):
        proba = pipeline.predict_proba(X)[0]
        probability = float(proba[1])   # probability of High Risk

    # Top contributing factors from overall feature importance
    top_factors = feat_imp[:5] if feat_imp else []

    return {
        "label":       label,
        "emoji":       emoji,
        "risk_class":  risk_class,
        "probability": probability,
        "top_factors": top_factors,
    }


def format_probability(probability: float | None) -> str:
    """Return a human-readable confidence string."""
    if probability is None:
        return "N/A"
    pct = probability * 100
    return f"{pct:.1f}%"
