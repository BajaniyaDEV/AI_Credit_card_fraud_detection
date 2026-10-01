"""Make a fraud prediction for one transaction."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

try:
	from .features import engineer_features
except ImportError:
	from features import engineer_features


def predict_transaction(
	transaction: dict,
	model_path: str | Path = "models/random_forest__class_weight.joblib",
) -> dict:
	"""Return fraud probability, label, and a small set of data-driven risk factors."""
	model = joblib.load(model_path)
	frame = engineer_features(pd.DataFrame([transaction]))
	probability = float(model.predict_proba(frame)[:, 1][0])
	risk_factors = []
	amount = next((frame[column].iloc[0] for column in frame if "amount" in str(column).lower()), None)
	if amount is not None and pd.notna(amount):
		risk_factors.append("Transaction amount was included in the model decision")
	if any("hour" in str(column).lower() for column in frame):
		risk_factors.append("Transaction time was included in the model decision")
	return {
		"prediction": "FRAUD" if probability >= 0.5 else "LEGITIMATE",
		"fraud_probability": probability,
		"risk_factors": risk_factors,
	}
