# src/serve/api.py
import time
import json
import os
import sqlite3
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .predict import predict_proba_single, predict_proba_batch

# Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRED_DB = os.environ.get("PRED_DB", os.path.join(BASE_DIR, "predictions.db"))

# Prometheus metrics
PRED_COUNTER = Counter("fd_predictions_total", "Total number of predictions", ["outcome"])
LATENCY_HIST = Histogram("fd_inference_latency_seconds", "Inference latency seconds")

app = FastAPI(title="Fraud Detection API", version="2.0")

# DB init
def init_db():
    conn = sqlite3.connect(PRED_DB)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            input_json TEXT,
            pred INTEGER,
            prob REAL,
            latency REAL
        )
    """
    )
    conn.commit()
    conn.close()

init_db()

# Pydantic models
class Transaction(BaseModel):
    # we accept dynamic content; validate keys present at runtime
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

class BatchRequest(BaseModel):
    transactions: List[Transaction]

# Endpoints
@app.post("/predict")
def predict(txn: Transaction):
    t0 = time.time()
    try:
        prob = predict_proba_single(txn.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    latency = time.time() - t0
    pred = 1 if prob >= 0.5 else 0

    # metrics
    PRED_COUNTER.labels(outcome="fraud" if pred == 1 else "legit").inc()
    LATENCY_HIST.observe(latency)

    # persist
    conn = sqlite3.connect(PRED_DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (ts, input_json, pred, prob, latency) VALUES (?, ?, ?, ?, ?)",
        (time.time(), json.dumps(txn.dict()), pred, float(prob), float(latency)),
    )
    conn.commit()
    conn.close()

    return {"fraud_probability": prob, "prediction": pred, "latency": latency}


@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    t0 = time.time()
    try:
        list_dicts = [tx.dict() for tx in req.transactions]
        probs = predict_proba_batch(list_dicts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    latency = time.time() - t0
    preds = [1 if p >= 0.5 else 0 for p in probs]

    # metrics
    for p in preds:
        PRED_COUNTER.labels(outcome="fraud" if p == 1 else "legit").inc()
    LATENCY_HIST.observe(latency)

    # persist
    conn = sqlite3.connect(PRED_DB)
    c = conn.cursor()
    ts = time.time()
    for inp, pr, pd in zip(list_dicts, probs, preds):
        c.execute(
            "INSERT INTO predictions (ts, input_json, pred, prob, latency) VALUES (?, ?, ?, ?, ?)",
            (ts, json.dumps(inp), int(pd), float(pr), float(latency)),
        )
    conn.commit()
    conn.close()

    return {"results": [{"pred": int(p), "prob": float(pr)} for p, pr in zip(preds, probs)], "latency": latency}


@app.get("/recent")
def recent(limit: int = 100):
    conn = sqlite3.connect(PRED_DB)
    c = conn.cursor()
    rows = c.execute(
        "SELECT ts, input_json, pred, prob, latency FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({"ts": r[0], "input": json.loads(r[1]), "pred": r[2], "prob": r[3], "latency": r[4]})
    return {"recent": out}


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
