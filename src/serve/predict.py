# src/serve/predict.py
import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Path config: adjust if your models are stored elsewhere
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Filenames (update if your filenames differ)
SCALER_FNAME = os.path.join(MODELS_DIR, "scaler.joblib")
# Use xgboost model if that's what you trained; keep the same filename you have in models/
# e.g. "xgboost_model.joblib" or "logistic_baseline.joblib"
MODEL_FNAME = os.path.join(MODELS_DIR, "logistic_baseline.joblib")
FEATURE_COLS_FNAME = os.path.join(MODELS_DIR, "feature_cols.csv")

# Load artifacts
if not os.path.exists(SCALER_FNAME):
    raise FileNotFoundError(f"Scaler not found at {SCALER_FNAME}")
if not os.path.exists(MODEL_FNAME):
    raise FileNotFoundError(f"Model not found at {MODEL_FNAME}")
if not os.path.exists(FEATURE_COLS_FNAME):
    raise FileNotFoundError(
        f"feature_cols.csv not found at {FEATURE_COLS_FNAME}"
    )

scaler = joblib.load(SCALER_FNAME)
model = joblib.load(MODEL_FNAME)
feature_cols = pd.read_csv(FEATURE_COLS_FNAME, header=None).iloc[:, 0].tolist()


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has all expected columns (V1..V28, Time, Amount_log or Amount) and order."""
    # If Amount present but model expects Amount_log, create it
    if "Amount" in df.columns and "Amount_log" not in df.columns:
        df["Amount_log"] = np.log1p(df["Amount"])
        df = df.drop(columns=["Amount"])
    # If Time present and must be scaled we leave it; scaler handling below.
    # Ensure all feature cols exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df[feature_cols]
    return df


def _scale_df(df: pd.DataFrame) -> np.ndarray:
    """
    Safely scale appropriate columns using the loaded scaler.
    Two common cases:
     - scaler was trained only on ['Time', 'Amount_log'] => scaler.n_features_in_ == 2
     - scaler was trained on entire feature vector => scaler.n_features_in_ == len(feature_cols)
    """
    try:
        n_in = getattr(scaler, "n_features_in_", None)
    except Exception:
        n_in = None

    if n_in is None:
        # fallback: try to transform entire df
        return scaler.transform(df.values)
    if n_in == df.shape[1]:
        return scaler.transform(df.values)
    # If scaler expects fewer features, try selecting Time and Amount_log columns if present
    if n_in == 2:
        subset = []
        if "Time" in df.columns and "Amount_log" in df.columns:
            subset = df[["Time", "Amount_log"]].values
            return scaler.transform(subset)
        # if missing, build zeros
        subset = np.zeros((df.shape[0], 2))
        return scaler.transform(subset)
    # last resort: try to transform full vector (may raise)
    return scaler.transform(df.values)


def preprocess(data: List[Dict[str, Any]]) -> (np.ndarray, pd.DataFrame):
    """
    Input: list of dicts (each dict is one transaction)
    Returns: scaled numpy array X and dataframe (for logging)
    """
    df = pd.DataFrame(data)
    df = _ensure_columns(df)
    # Scaling
    X = _scale_df(df.copy())
    return X, df


def predict_proba_single(data: Dict[str, Any]) -> float:
    X, _ = preprocess([data])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[:, 1][0])
    else:
        # some models (tree booster) support predict returning probabilities differently
        prob = float(model.predict(X)[0])
    return prob


def predict_proba_batch(data: List[Dict[str, Any]]) -> List[float]:
    X, _ = preprocess(data)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1].tolist()
    else:
        probs = model.predict(X).tolist()
    return [float(p) for p in probs]
