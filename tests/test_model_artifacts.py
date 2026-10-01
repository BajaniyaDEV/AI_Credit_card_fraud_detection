from __future__ import annotations

import joblib

from src.features import engineer_features
from src.preprocessing import build_preprocessor, build_sampling_pipeline
from sklearn.linear_model import LogisticRegression


def test_saved_model_artifact_can_be_loaded_and_used(small_dataset, tmp_path) -> None:
	features = engineer_features(small_dataset.drop(columns=["Class"]))
	pipeline = build_sampling_pipeline(
		LogisticRegression(max_iter=500, class_weight="balanced"),
		build_preprocessor(features),
		"class_weight",
	)
	pipeline.fit(features, small_dataset["Class"])
	artifact_path = tmp_path / "model.joblib"
	joblib.dump(pipeline, artifact_path)

	loaded_pipeline = joblib.load(artifact_path)
	probabilities = loaded_pipeline.predict_proba(features)

	assert probabilities.shape == (len(features), 2)
