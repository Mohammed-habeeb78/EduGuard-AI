"""
data_preprocessing.py
EduGuard AI – Student Academic Risk Prediction System

Handles dataset loading, cleaning, feature engineering,
and preparation of training/testing splits.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─── Constants ────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")

# Columns that must be dropped to prevent target leakage.
# G3 is the final grade used to derive the target; G1/G2 are period grades
# that are highly correlated with G3 and would constitute leakage in a
# real early-warning setting (they are collected at the same time as G3).
LEAKAGE_COLUMNS = ["G1", "G2", "G3"]

# Categorical columns present in the raw dataset
CATEGORICAL_COLUMNS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

# Numerical columns (ordinal or continuous)
NUMERICAL_COLUMNS = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences",
]


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the UCI Student Performance dataset from a CSV file.

    The dataset uses a semicolon (;) as the delimiter.

    Args:
        path: Absolute or relative path to student-mat.csv.

    Returns:
        Raw DataFrame as loaded from disk.

    Raises:
        FileNotFoundError: If the CSV file does not exist at *path*.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please download student-mat.csv from:\n"
            "  https://archive.ics.uci.edu/dataset/320/student+performance\n"
            "and place it in the EduGuard-AI/data/ directory."
        )

    df = pd.read_csv(path, sep=";")
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def display_dataset_info(df: pd.DataFrame) -> None:
    """Print a concise summary of the raw dataset."""
    print("\n── Dataset Shape ─────────────────────────────")
    print(f"  Rows: {df.shape[0]}   Columns: {df.shape[1]}")

    print("\n── Column Names & Data Types ─────────────────")
    print(df.dtypes.to_string())

    print("\n── Missing Values ────────────────────────────")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.sum() > 0 else "  No missing values found.")

    print("\n── Duplicate Records ─────────────────────────")
    dupes = df.duplicated().sum()
    print(f"  Duplicate rows: {dupes}")

    print("\n── Categorical Columns ───────────────────────")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"  {cat_cols}")

    print("\n── Numerical Columns ─────────────────────────")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  {num_cols}")

    print("\n── Target Distribution (G3 raw) ──────────────")
    if "G3" in df.columns:
        print(df["G3"].describe().to_string())


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary *academic_risk* target column derived from G3.

    Encoding:
        1 (High Risk)   – final grade G3 < 10
        0 (Lower Risk)  – final grade G3 >= 10

    G3 is retained in the DataFrame at this stage so callers can inspect it;
    it must be removed from the feature matrix before model training.

    Args:
        df: DataFrame containing a numeric *G3* column.

    Returns:
        DataFrame with a new *academic_risk* column appended.
    """
    if "G3" not in df.columns:
        raise ValueError("Column 'G3' is required to create the target variable.")

    df = df.copy()
    df["academic_risk"] = (df["G3"] < 10).astype(int)

    high_risk = df["academic_risk"].sum()
    low_risk  = len(df) - high_risk
    print(f"\n[INFO] Target created → High Risk: {high_risk}  Lower Risk: {low_risk}")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode all categorical (object-dtype) columns.

    Returns:
        Tuple of (encoded DataFrame, mapping of column → LabelEncoder).
    """
    df = df.copy()
    encoders: dict = {}

    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def preprocess(
    path: str = DATA_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """Full preprocessing pipeline.

    Steps
    -----
    1. Load dataset.
    2. Drop duplicate rows.
    3. Create binary *academic_risk* target from G3.
    4. Drop leakage columns (G1, G2, G3).
    5. Label-encode categorical columns.
    6. Stratified train/test split.

    Args:
        path:         Path to student-mat.csv.
        test_size:    Fraction of data held out for testing.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with keys:
            X_train, X_test, y_train, y_test,
            feature_names, encoders, df_full, df_processed
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    df_raw = load_dataset(path)
    display_dataset_info(df_raw)

    # ── 2. Drop duplicates ────────────────────────────────────────────────────
    before = len(df_raw)
    df = df_raw.drop_duplicates().copy()
    if len(df) < before:
        print(f"[INFO] Dropped {before - len(df)} duplicate row(s).")

    # ── 3. Create target ─────────────────────────────────────────────────────
    df = create_target(df)

    # Keep a copy of the full processed DataFrame (with G3) for EDA
    df_full = df.copy()

    # ── 4. Remove leakage columns ─────────────────────────────────────────────
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"[INFO] Dropped leakage columns: {cols_to_drop}")

    # ── 5. Encode categoricals ────────────────────────────────────────────────
    target_col = "academic_risk"
    feature_cols = [c for c in df.columns if c != target_col]

    df_features = df[feature_cols].copy()
    df_features, encoders = encode_categoricals(df_features)

    df_processed = df_features.copy()
    df_processed[target_col] = df[target_col].values

    X = df_features.values
    y = df[target_col].values

    # ── 6. Stratified split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"[INFO] Train size: {len(X_train)}   Test size: {len(X_test)}")
    print(f"[INFO] Features used for modelling: {len(feature_cols)}")

    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "feature_names": feature_cols,
        "encoders":      encoders,
        "df_full":       df_full,
        "df_processed":  df_processed,
    }
