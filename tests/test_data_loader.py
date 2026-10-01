from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import discover_csv, infer_target_column, load_dataset


def test_load_dataset_and_target_detection(tmp_path) -> None:
	csv_path = tmp_path / "creditcard.csv"
	pd.DataFrame({"Amount": [10.0, 20.0], "Class": [0, 1]}).to_csv(csv_path, index=False)

	loaded = load_dataset(csv_path)

	assert loaded.shape == (2, 2)
	assert infer_target_column(loaded) == "Class"
	assert discover_csv(tmp_path) == csv_path


def test_invalid_dataset_inputs_raise_actionable_errors(tmp_path) -> None:
	with pytest.raises(FileNotFoundError, match="Dataset not found"):
		load_dataset(tmp_path / "missing.csv")

	non_csv_path = tmp_path / "dataset.txt"
	non_csv_path.write_text("not a csv", encoding="utf-8")
	with pytest.raises(ValueError, match="Only CSV"):
		load_dataset(non_csv_path)

	with pytest.raises(ValueError, match="target column"):
		infer_target_column(pd.DataFrame({"amount": [1.0]}))
