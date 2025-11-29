# src/dashboard/app.py
import streamlit as st
import requests
import pandas as pd
import time
import os
import subprocess
import signal

API_URL = os.environ.get("API_URL", "http://localhost:8000")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Real-time Fraud Detection")

# Session state
if "producer_proc" not in st.session_state:
    st.session_state.producer_proc = None
if "consumer_proc" not in st.session_state:
    st.session_state.consumer_proc = None

# Sidebar controls
st.sidebar.header("Controls")
model = st.sidebar.selectbox("Model", ["lightgbm", "logistic", "xgboost"])
rate = st.sidebar.slider("Transactions/sec", 1, 50, 10)
max_rows = st.sidebar.number_input("Max transactions", 100, 10000, 1000)
refresh_rate = st.sidebar.slider("Refresh rate (s)", 1, 5, 2)

col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("▶ Start")
stop_btn = col2.button("⏹ Stop")

if start_btn:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    st.session_state.producer_proc = subprocess.Popen([
        "python", "-m", "src.streaming.producer_kafka",
        "--kafka", KAFKA_BOOTSTRAP, "--rate", str(rate), "--max_rows", str(max_rows)
    ], cwd=base_dir)
    time.sleep(1)
    st.session_state.consumer_proc = subprocess.Popen([
        "python", "-m", "src.streaming.consumer_kafka",
        "--kafka", KAFKA_BOOTSTRAP, "--api", API_URL, "--model", model
    ], cwd=base_dir)
    st.sidebar.success("Streaming started!")

if stop_btn:
    for proc in [st.session_state.producer_proc, st.session_state.consumer_proc]:
        if proc:
            proc.terminate()
    st.session_state.producer_proc = None
    st.session_state.consumer_proc = None
    st.sidebar.info("Streaming stopped")

# Main dashboard
st.header("📊 Live Metrics")
metrics_cols = st.columns(4)
stats_placeholder = st.empty()
chart_placeholder = st.empty()

st.header("📜 Recent Predictions")
table_placeholder = st.empty()

# Auto-refresh loop
def fetch_stats():
    try:
        return requests.get(f"{API_URL}/stats", timeout=3).json()
    except:
        return {}

def fetch_recent(limit=50):
    try:
        return requests.get(f"{API_URL}/recent?limit={limit}", timeout=3).json().get("recent", [])
    except:
        return []

while True:
    stats = fetch_stats()
    if stats.get("count", 0) > 0:
        metrics_cols[0].metric("Total Predictions", stats["count"])
        metrics_cols[1].metric("Fraud Count", stats["fraud_count"])
        metrics_cols[2].metric("Fraud Rate", f"{stats['fraud_rate']*100:.2f}%")
        metrics_cols[3].metric("Avg Latency", f"{stats['latency_avg']*1000:.1f}ms")
        
        stats_placeholder.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | P50 Latency | {stats['latency_p50']*1000:.2f} ms |
        | P95 Latency | {stats['latency_p95']*1000:.2f} ms |
        | P99 Latency | {stats['latency_p99']*1000:.2f} ms |
        | Avg Probability | {stats['avg_prob']:.4f} |
        """)
    
    recent = fetch_recent(100)
    if recent:
        df = pd.DataFrame(recent)
        df["ts"] = pd.to_datetime(df["ts"], unit="s").dt.strftime("%H:%M:%S")
        df["result"] = df["pred"].map({0: "✅ Legit", 1: "🚨 FRAUD"})
        table_placeholder.dataframe(df[["ts", "model", "result", "prob", "latency"]], use_container_width=True)
        chart_placeholder.line_chart(df["prob"].astype(float).values[::-1])
    
    time.sleep(refresh_rate)

