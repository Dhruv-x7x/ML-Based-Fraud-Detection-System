# src/serve/predict.py
import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load artifacts
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

# Load feature cols
_fc = pd.read_csv(os.path.join(MODELS_DIR, "feature_cols.csv"))
feature_cols = _fc.iloc[:, 0].tolist()

# Load all models
MODELS = {
    "logistic": joblib.load(os.path.join(MODELS_DIR, "logistic.joblib")),
    "lightgbm": joblib.load(os.path.join(MODELS_DIR, "lightgbm.joblib")),
    "xgboost": joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib")),
}

def preprocess(data: list[dict]) -> np.ndarray:
    df = pd.DataFrame(data)
    # Scale Time and Amount using the scaler (trained on Time, Amount)
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    # Ensure all feature columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df = df[feature_cols]
    return df.values

def predict_proba(data: list[dict], model_name: str = "lightgbm") -> list[float]:
    X = preprocess(data)
    model = MODELS[model_name]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].tolist()
    return model.predict(X).tolist()

