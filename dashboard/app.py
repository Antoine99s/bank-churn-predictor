import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Bank Churn Predictor", layout="centered")
st.title("Bank Churn Predictor")

with st.form("predict_form"):
    credit_score = st.number_input("CreditScore", min_value=300.0, max_value=900.0, value=650.0)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=18.0, max_value=100.0, value=40.0)
    tenure = st.number_input("Tenure", min_value=0.0, max_value=20.0, value=5.0)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    num_products = st.number_input("NumOfProducts", min_value=1.0, max_value=4.0, value=1.0)
    has_card = st.selectbox("HasCrCard", [0, 1], index=1)
    is_active = st.selectbox("IsActiveMember", [0, 1], index=1)
    estimated_salary = st.number_input("EstimatedSalary", min_value=0.0, value=100000.0)

    submitted = st.form_submit_button("Predict churn")

if submitted:
    payload = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": estimated_salary,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        st.success(f"Prediction: {data['label']}")
        st.metric("Churn probability", f"{data['probability']:.2%}")
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
