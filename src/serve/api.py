# src/serve/api.py
import time
import json
import os
import sqlite3
from collections import deque
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .predict import predict_proba

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRED_DB = os.environ.get("PRED_DB", os.path.join(BASE_DIR, "predictions.db"))

# Prometheus metrics
PRED_COUNTER = Counter("fd_predictions_total", "Total predictions", ["model", "outcome"])
LATENCY_HIST = Histogram("fd_inference_latency_seconds", "Inference latency", ["model"])

# Rolling window for metrics (last 1000 predictions)
ROLLING_WINDOW = 1000
rolling_metrics = deque(maxlen=ROLLING_WINDOW)

app = FastAPI(title="Fraud Detection API", version="2.0")

# DB init
def init_db():
    conn = sqlite3.connect(PRED_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, model TEXT, input_json TEXT, pred INTEGER, prob REAL, latency REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float

class BatchRequest(BaseModel):
    transactions: list[Transaction]
    model: str = "lightgbm"

@app.post("/predict")
def predict_single(txn: Transaction, model: str = "lightgbm"):
    t0 = time.time()
    prob = predict_proba([txn.model_dump()], model)[0]
    latency = time.time() - t0
    pred = 1 if prob >= 0.5 else 0
    
    PRED_COUNTER.labels(model=model, outcome="fraud" if pred else "legit").inc()
    LATENCY_HIST.labels(model=model).observe(latency)
    rolling_metrics.append({"pred": pred, "prob": prob, "latency": latency, "model": model})
    
    conn = sqlite3.connect(PRED_DB)
    conn.execute("INSERT INTO predictions (ts, model, input_json, pred, prob, latency) VALUES (?,?,?,?,?,?)",
                 (time.time(), model, json.dumps(txn.model_dump()), pred, prob, latency))
    conn.commit()
    conn.close()
    
    return {"prediction": pred, "probability": prob, "latency": latency, "model": model}

@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    t0 = time.time()
    data = [tx.model_dump() for tx in req.transactions]
    probs = predict_proba(data, req.model)
    latency = time.time() - t0
    preds = [1 if p >= 0.5 else 0 for p in probs]
    
    for p in preds:
        PRED_COUNTER.labels(model=req.model, outcome="fraud" if p else "legit").inc()
    LATENCY_HIST.labels(model=req.model).observe(latency)
    
    conn = sqlite3.connect(PRED_DB)
    ts = time.time()
    for inp, pr, pd in zip(data, probs, preds):
        conn.execute("INSERT INTO predictions (ts, model, input_json, pred, prob, latency) VALUES (?,?,?,?,?,?)",
                     (ts, req.model, json.dumps(inp), pd, pr, latency / len(data)))
        rolling_metrics.append({"pred": pd, "prob": pr, "latency": latency / len(data), "model": req.model})
    conn.commit()
    conn.close()
    
    return {"results": [{"pred": p, "prob": pr} for p, pr in zip(preds, probs)], "latency": latency}

@app.get("/recent")
def recent(limit: int = 100):
    conn = sqlite3.connect(PRED_DB)
    rows = conn.execute("SELECT ts, model, pred, prob, latency FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return {"recent": [{"ts": r[0], "model": r[1], "pred": r[2], "prob": r[3], "latency": r[4]} for r in rows]}

@app.get("/stats")
def stats():
    if not rolling_metrics:
        return {"count": 0}
    latencies = [m["latency"] for m in rolling_metrics]
    preds = [m["pred"] for m in rolling_metrics]
    probs = [m["prob"] for m in rolling_metrics]
    latencies_sorted = sorted(latencies)
    n = len(latencies)
    return {
        "count": n,
        "fraud_count": sum(preds),
        "fraud_rate": sum(preds) / n,
        "avg_prob": sum(probs) / n,
        "latency_avg": sum(latencies) / n,
        "latency_p50": latencies_sorted[n // 2],
        "latency_p95": latencies_sorted[int(n * 0.95)],
        "latency_p99": latencies_sorted[int(n * 0.99)] if n >= 100 else latencies_sorted[-1],
    }

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

