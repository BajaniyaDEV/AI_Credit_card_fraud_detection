"""Leakage-safe preprocessing and split utilities for fraud detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

try:
	from .data_loader import encode_target as resolve_target_encoding
	from .data_loader import infer_target_column as resolve_target_column
	from .data_loader import load_dataset
except ImportError:
	from data_loader import encode_target as resolve_target_encoding
	from data_loader import infer_target_column as resolve_target_column
	from data_loader import load_dataset


def load_data(path: str | Path) -> pd.DataFrame:
	"""Load a CSV dataset and fail with a useful message for unsupported files."""
	return load_dataset(path)


def infer_target_column(data: pd.DataFrame, target_column: str | None = None) -> str:
	"""Find the fraud label column using an explicit name or common conventions."""
	return resolve_target_column(data, target_column)


def encode_target(target: pd.Series) -> pd.Series:
	"""Convert common binary fraud labels to integers where zero means legitimate."""
	return resolve_target_encoding(target)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
	"""Build a preprocessing transformer for mixed numeric and categorical data."""
	numeric_columns = features.select_dtypes(include="number").columns.tolist()
	categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)
	return ColumnTransformer(
		transformers=[
			("numeric", numeric_pipeline, numeric_columns),
			("categorical", categorical_pipeline, categorical_columns),
		],
		remainder="drop",
	)


def recommend_imbalance_strategy(target: pd.Series) -> dict[str, Any]:
	"""Recommend a conservative baseline; resampling remains an experiment."""
	counts = target.value_counts().sort_index()
	minority = int(counts.min())
	majority = int(counts.max())
	ratio = majority / minority if minority else float("inf")
	return {
		"counts": {str(label): int(count) for label, count in counts.items()},
		"imbalance_ratio": ratio,
		"recommendation": "class_weight",
		"reason": "Weighted learning preserves all observations; compare samplers only inside training folds.",
	}


def build_sampling_pipeline(
	estimator: Any,
	preprocessor: ColumnTransformer,
	strategy: str = "class_weight",
	random_state: int = 42,
) -> ImbPipeline:
	"""Build an optional training pipeline; samplers run after train-only preprocessing."""
	steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
	if strategy == "random_under_sampler":
		steps.append(("sampler", RandomUnderSampler(random_state=random_state)))
	elif strategy == "smote":
		steps.append(("sampler", SMOTE(random_state=random_state)))
	elif strategy != "class_weight":
		raise ValueError("strategy must be class_weight, random_under_sampler, or smote")
	steps.append(("model", estimator))
	return ImbPipeline(steps=steps)


def prepare_data(
	data: pd.DataFrame,
	target_column: str | None = None,
	test_size: float = 0.2,
	random_state: int = 42,
	validation_size: float = 0.2,
	remove_duplicates: bool = True,
) -> dict[str, Any]:
	"""Remove exact duplicates and create reproducible stratified raw-data splits.

	Transformers are built from the training partition but are not fitted here. This
	keeps split creation separate from model fitting and prevents holdout statistics
	from entering preprocessing.
	"""
	target_name = infer_target_column(data, target_column)
	clean_data = data.dropna(subset=[target_name]).copy()
	duplicate_count = int(clean_data.duplicated().sum())
	if remove_duplicates:
		clean_data = clean_data.drop_duplicates().reset_index(drop=True)
	target = encode_target(clean_data.pop(target_name))
	features = clean_data

	holdout_size = validation_size + test_size
	x_train, x_holdout, y_train, y_holdout = train_test_split(
		features,
		target,
		test_size=holdout_size,
		random_state=random_state,
		stratify=target,
	)
	x_validation, x_test, y_validation, y_test = train_test_split(
		x_holdout,
		y_holdout,
		test_size=test_size / holdout_size,
		random_state=random_state,
		stratify=y_holdout,
	)
	return {
		"target_column": target_name,
		"feature_columns": features.columns.tolist(),
		"preprocessor": build_preprocessor(x_train),
		"duplicate_rows_removed": duplicate_count if remove_duplicates else 0,
		"x_train": x_train,
		"x_validation": x_validation,
		"x_test": x_test,
		"y_train": y_train,
		"y_validation": y_validation,
		"y_test": y_test,
		"imbalance": recommend_imbalance_strategy(y_train),
		"split_distributions": {
			"train": _class_distribution(y_train),
			"validation": _class_distribution(y_validation),
			"test": _class_distribution(y_test),
		},
	}


def _class_distribution(target: pd.Series) -> dict[str, Any]:
	"""Return counts and fraud rate for one split."""
	counts = target.value_counts().reindex([0, 1], fill_value=0)
	return {
		"rows": int(len(target)),
		"non_fraud": int(counts[0]),
		"fraud": int(counts[1]),
		"fraud_rate": float(target.mean()),
	}


def save_preprocessing_artifact(prepared: dict[str, Any], output_path: str | Path) -> None:
	"""Persist the train-fitted preprocessor and schema metadata for inference."""
	artifact = {
		"preprocessor": prepared["preprocessor"].fit(prepared["x_train"]),
		"target_column": prepared["target_column"],
		"feature_columns": prepared["feature_columns"],
		"imbalance": prepared["imbalance"],
	}
	destination = Path(output_path)
	destination.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(artifact, destination)
