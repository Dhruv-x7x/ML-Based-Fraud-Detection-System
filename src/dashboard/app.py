# src/dashboard/app.py
import streamlit as st
import requests
import pandas as pd
import time
import os

# Default API URL used by dashboard. When running locally without Docker use http://localhost:8000
API_URL = os.environ.get("ST_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Real-time Fraud Detection")

tabs = st.tabs(["Manual Predict", "Live Dashboard"])

# ---------- Manual Predict tab ----------
with tabs[0]:
    st.header("Manual transaction prediction")
    st.markdown(
        "Fill the form and click **Predict**. The V1..V28 fields are collapsed to keep the UI tidy."
    )
    with st.form("txn_form"):
        col1, col2 = st.columns(2)
        Time = col1.number_input(
            "Time (seconds since first transaction)", min_value=0.0, value=0.0
        )
        Amount = col2.number_input("Amount", min_value=0.0, value=0.0)
        with st.expander("Advanced: V1..V28 (collapsed)"):
            V_fields = {}
            # Keep labels matching your model: V1..V28
            for i in range(1, 29):
                key = f"V{i}"
                V_fields[key] = st.number_input(key, value=0.0, key=key)
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {"Time": float(Time), "Amount": float(Amount)}
        payload.update({k: float(v) for k, v in V_fields.items()})
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if r.status_code == 200:
                j = r.json()
                st.metric(
                    "Fraud Probability", f"{j['fraud_probability']*100:.2f}%"
                )
                st.metric(
                    "Prediction",
                    "FRAUD" if j["prediction"] == 1 else "Not Fraud",
                )
                st.json(j)
            else:
                st.error(f"API error: {r.status_code} - {r.text}")
        except Exception as e:
            st.error(f"Request error: {e}")

# ---------- Live Dashboard tab ----------
with tabs[1]:
    st.header("Live recent predictions")
    st.markdown(
        "This panel polls the API `/recent` endpoint and shows recent predictions. Run the consumer/producer to populate."
    )
    col1, col2 = st.columns([3, 1])
    with col2:
        REFRESH = st.number_input(
            "Refresh every (s)", min_value=1, max_value=10, value=3
        )
        MAX_ROWS = st.number_input(
            "Rows to show", min_value=10, max_value=500, value=100
        )
        st.write("API URL:", API_URL)

    holder = st.empty()
    stat_cols = st.columns(3)
    chart_slot = st.empty()

    def fetch_recent(limit: int = 100):
        try:
            r = requests.get(f"{API_URL}/recent?limit={limit}", timeout=5)
            if r.status_code == 200:
                return r.json().get("recent", [])
            else:
                return []
        except Exception:
            return []

    # non-blocking loop using button to start/stop live mode
    if "running" not in st.session_state:
        st.session_state.running = False

    start_stop = col1.button(
        "Start Live Poll" if not st.session_state.running else "Stop Live Poll"
    )
    if start_stop:
        st.session_state.running = not st.session_state.running

    while st.session_state.running:
        rows = fetch_recent(MAX_ROWS)
        if rows:
            # Normalize to dataframe
            df = pd.DataFrame(
                [
                    {
                        "ts": time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])
                        ),
                        "pred": r["pred"],
                        "prob": r["prob"],
                    }
                    for r in rows
                ]
            )
            holder.dataframe(df, use_container_width=True)
            total = len(df)
            frauds = int(df["pred"].sum())
            fraud_rate = frauds / total if total > 0 else 0.0
            avg_prob = df["prob"].mean() if total > 0 else 0.0
            stat_cols[0].metric("Recent total", total)
            stat_cols[1].metric("Recent frauds", frauds)
            stat_cols[2].metric("Fraud rate", f"{fraud_rate:.4f}")
            chart_slot.line_chart(df["prob"].astype(float))
        else:
            holder.write(
                "No recent predictions yet (start consumer/producer or run local fallback)."
            )
        time.sleep(REFRESH)
        # allow manual stop
        if not st.session_state.running:
            break

    if not st.session_state.running:
        st.info("Live poll stopped. Click 'Start Live Poll' to begin.")
