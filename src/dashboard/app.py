import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000/predict"  # Change if running in Docker with a network

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Real-time Fraud Detection")

st.markdown("Enter transaction details to get a fraud probability prediction:")

# Input fields
with st.form("txn_form"):
    col1, col2 = st.columns(2)
    Time = col1.number_input("Time (seconds since first transaction)", min_value=0.0)
    Amount = col2.number_input("Amount", min_value=0.0)
    V_fields = {}
    for i in range(1, 29):
        V_fields[f"V{i}"] = st.number_input(f"V{i}", value=0.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {"Time": Time, "Amount": Amount, **V_fields}
    res = requests.post(API_URL, json=payload)
    if res.status_code == 200:
        out = res.json()
        st.metric("Fraud Probability", f"{out['fraud_probability']*100:.2f}%")
        st.metric("Prediction", "Fraud" if out["prediction"] else "Not Fraud")
    else:
        st.error(f"API Error: {res.status_code}")
