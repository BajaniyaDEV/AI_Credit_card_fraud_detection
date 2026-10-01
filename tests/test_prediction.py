from __future__ import annotations

import joblib

from src.features import engineer_features
from src.predict import predict_transaction
from src.preprocessing import build_preprocessor, build_sampling_pipeline
from sklearn.linear_model import LogisticRegression


def test_prediction_output_has_expected_contract(small_dataset, tmp_path) -> None:
	features = engineer_features(small_dataset.drop(columns=["Class"]))
	pipeline = build_sampling_pipeline(
		LogisticRegression(max_iter=500, class_weight="balanced"),
		build_preprocessor(features),
		"class_weight",
	)
	pipeline.fit(features, small_dataset["Class"])
	model_path = tmp_path / "model.joblib"
	joblib.dump(pipeline, model_path)

	result = predict_transaction(
		small_dataset.drop(columns=["Class"]).iloc[0].to_dict(), model_path
	)

	assert result["prediction"] in {"FRAUD", "LEGITIMATE"}
	assert 0.0 <= result["fraud_probability"] <= 1.0
	assert isinstance(result["risk_factors"], list)
