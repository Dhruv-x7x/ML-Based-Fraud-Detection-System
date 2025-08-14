from fastapi import FastAPI
from pydantic import BaseModel
from .predict import predict_proba

app = FastAPI(title="Fraud Detection API", version="1.0")

class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

@app.post("/predict")
def predict(txn: Transaction):
    proba = predict_proba(txn.dict())
    return {
        "fraud_probability": proba,
        "prediction": int(proba >= 0.5)
    }
