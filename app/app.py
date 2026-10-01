"""Streamlit dashboard for interactive fraud predictions."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import predict_transaction


st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🔎", layout="centered")
st.title("Credit Card Fraud Detection")
st.caption("Score one transaction with the trained model and inspect the decision signals.")

model_options = sorted(Path("models").glob("*.joblib"))
if not model_options:
	st.warning("No trained model found. Train a model first with: python src/train.py <csv-path>")
	st.stop()

model_path = st.selectbox("Model", model_options, format_func=lambda path: path.stem)
uploaded_file = st.file_uploader("Upload one transaction as CSV", type=["csv"])

if uploaded_file:
	transaction_frame = pd.read_csv(uploaded_file)
	if transaction_frame.empty:
		st.error("The uploaded CSV has no rows.")
		st.stop()
	transaction = transaction_frame.iloc[0].to_dict()
	st.dataframe(transaction_frame.head(1), use_container_width=True)
else:
	amount = st.number_input("Transaction amount", min_value=0.0, value=100.0, step=1.0)
	merchant_category = st.text_input("Merchant category", value="general")
	transaction_time = st.text_input("Transaction time", value="2026-09-26 14:30:00")
	transaction = {
		"amount": amount,
		"merchant_category": merchant_category,
		"transaction_time": transaction_time,
	}

if st.button("Analyze transaction", type="primary"):
	try:
		result = predict_transaction(transaction, model_path)
		probability = result["fraud_probability"]
		st.metric("Fraud probability", f"{probability:.1%}")
		if result["prediction"] == "FRAUD":
			st.error("HIGH RISK: FRAUD")
		else:
			st.success("LOWER RISK: LEGITIMATE")
		if result["risk_factors"]:
			st.subheader("Risk factors")
			for factor in result["risk_factors"]:
				st.write(f"- {factor}")
	except Exception as error:
		st.error(f"Could not score this transaction: {error}")
