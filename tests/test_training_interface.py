from __future__ import annotations

import pytest

from src.train import build_models


def test_build_models_exposes_required_training_models() -> None:
	models = build_models("class_weight", random_state=7)

	assert {"logistic_regression", "decision_tree", "random_forest"}.issubset(models)
	assert all(model.random_state == 7 for model in models.values() if hasattr(model, "random_state"))


def test_build_models_rejects_unknown_strategy() -> None:
	with pytest.raises(ValueError, match="strategy must be"):
		build_models("unknown")
