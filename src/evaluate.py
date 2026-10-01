"""Evaluate saved fraud models with imbalance-aware metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
	from .data_loader import discover_csv, load_dataset
	from .modeling import evaluate_estimator, prepare_model_splits
except ImportError:
	from data_loader import discover_csv, load_dataset
	from modeling import evaluate_estimator, prepare_model_splits


def evaluate_model(
	model_path: str | Path,
	data: pd.DataFrame,
	target_column: str | None = None,
	random_state: int = 42,
) -> dict[str, Any]:
	"""Evaluate one saved pipeline on the deterministic final test partition."""
	prepared = prepare_model_splits(
		data, target_column=target_column, random_state=random_state
	)
	model = joblib.load(model_path)
	return evaluate_estimator(model, prepared["x_test"], prepared["y_test"])


def evaluate_saved_models(
	data_path: str | Path | None = None,
	models_dir: str | Path = "models",
	target_column: str | None = None,
) -> list[dict[str, Any]]:
	"""Evaluate every saved model artifact except preprocessing metadata."""
	source = Path(data_path) if data_path else discover_csv()
	data = load_dataset(source)
	metadata_path = Path(models_dir) / "metadata.json"
	random_state = 42
	if metadata_path.exists():
		metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
		random_state = int(metadata.get("random_state", random_state))
	prepared = prepare_model_splits(
		data, target_column=target_column, random_state=random_state
	)
	rows = []
	for model_path in sorted(Path(models_dir).glob("*.joblib")):
		if model_path.name == "preprocessing.joblib":
			continue
		model = joblib.load(model_path)
		metrics = evaluate_estimator(model, prepared["x_test"], prepared["y_test"])
		model_name, _, strategy = model_path.stem.partition("__")
		rows.append({"Model": model_name, "ImbalanceStrategy": strategy, **metrics})
	return rows


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data", help="Path to the input CSV; defaults to data/raw/creditcard.csv")
	parser.add_argument("--models", default="models")
	parser.add_argument("--target")
	args = parser.parse_args()
	print(json.dumps(evaluate_saved_models(args.data, args.models, args.target), indent=2))


if __name__ == "__main__":
	main()
