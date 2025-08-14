import os
import joblib
import numpy as np
import pandas as pd
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")

# Load artifacts
scaler: StandardScaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
model: LogisticRegression = joblib.load(os.path.join(MODELS_DIR, "logistic_baseline.joblib"))
feature_cols: List[str] = pd.read_csv(os.path.join(MODELS_DIR, "feature_cols.csv")).iloc[:, 0].tolist()

def preprocess_input(data: dict) -> np.ndarray:
    """
    Preprocess raw JSON into model-ready features.
    Expected keys: 'Time', 'Amount', 'V1'...'V28' (all numeric)
    """
    df = pd.DataFrame([data])
    # Log-transform amount
    df["Amount_log"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Amount"])
    # Scale time & amount_log
    df[["Time", "Amount_log"]] = scaler.transform(df[["Time", "Amount_log"]])
    # Ensure column order
    df = df[feature_cols]
    return df

def predict_proba(data: dict) -> float:
    X = preprocess_input(data)
    proba = model.predict_proba(X)[:, 1][0]
    return proba
