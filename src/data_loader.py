"""Reusable dataset discovery, loading, and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


TARGET_CANDIDATES = ("is_fraud", "fraud", "class", "label", "target")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def discover_csv(data_dir: str | Path | None = None) -> Path:
	"""Return the project dataset, or fail with an actionable message."""
	directory = PROJECT_ROOT / "data" / "raw" if data_dir is None else Path(data_dir)
	if not directory.exists():
		raise FileNotFoundError(f"Dataset directory not found: {directory}")
	csv_files = sorted(directory.rglob("*.csv"))
	if not csv_files:
		raise FileNotFoundError(
		f"No CSV dataset found in {directory}. Place the dataset there or pass --data."
	)
	preferred = [path for path in csv_files if path.name.lower() == "creditcard.csv"]
	if preferred:
		return preferred[0]
	if len(csv_files) > 1:
		raise ValueError(
		f"Found multiple CSV datasets in {directory}. Pass one file explicitly with --data."
	)
	return csv_files[0]


def load_dataset(path: str | Path) -> pd.DataFrame:
	"""Load a CSV and reject unsupported or empty inputs."""
	source = Path(path)
	if not source.exists():
		raise FileNotFoundError(f"Dataset not found: {source}")
	if source.suffix.lower() != ".csv":
		raise ValueError("Only CSV datasets are supported currently.")
	data = pd.read_csv(source)
	if data.empty:
		raise ValueError(f"Dataset is empty: {source}")
	return data


def infer_target_column(data: pd.DataFrame, target_column: str | None = None) -> str:
	"""Resolve the fraud label from an explicit name or a common exact name."""
	if target_column:
		if target_column not in data.columns:
			raise ValueError(
				f"Target column '{target_column}' is missing. Available columns: {list(data.columns)}"
			)
		return target_column
	normalized = {str(column).strip().lower(): column for column in data.columns}
	for candidate in TARGET_CANDIDATES:
		if candidate in normalized:
			return normalized[candidate]
	raise ValueError(
		"Expected fraud target column was not found. Pass --target with the exact column name. "
		f"Available columns: {list(data.columns)}"
	)


def encode_target(target: pd.Series) -> pd.Series:
	"""Encode a binary fraud target with zero as non-fraud and one as fraud."""
	if target.isna().any():
		raise ValueError("Target column contains missing values; resolve them before encoding.")
	if pd.api.types.is_numeric_dtype(target):
		values = set(target.unique())
		if values.issubset({0, 1}):
			return target.astype(int)
	positive_values = {"1", "true", "yes", "fraud", "fraudulent", "positive"}
	normalized_target = target.map(lambda value: str(value).strip().lower())
	encoded = normalized_target.isin(positive_values).astype(int)
	if encoded.sum() == 0 or encoded.sum() == len(encoded):
		raise ValueError("Target must contain both fraud and non-fraud classes.")
	return encoded


def summarize_dataset(
	data: pd.DataFrame, source: str | Path, target_column: str | None = None
) -> dict[str, Any]:
	"""Return a JSON-serializable structural and target-quality report."""
	target_name = infer_target_column(data, target_column)
	encoded_target = encode_target(data[target_name])
	return {
		"filename": Path(source).name,
		"path": str(source),
		"rows": int(len(data)),
		"columns": int(data.shape[1]),
		"column_names": [str(column) for column in data.columns],
		"data_types": {str(column): str(dtype) for column, dtype in data.dtypes.items()},
		"missing_values": {str(column): int(count) for column, count in data.isna().sum().items()},
		"duplicate_rows": int(data.duplicated().sum()),
		"target_column": target_name,
		"target_distribution": {
			"non_fraud": int((encoded_target == 0).sum()),
			"fraud": int((encoded_target == 1).sum()),
			"fraud_rate": float(encoded_target.mean()),
		},
	}