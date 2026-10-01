"""Feature engineering helpers for transaction-level fraud data."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _find_column(data: pd.DataFrame, names: tuple[str, ...]) -> str | None:
	normalized = {str(column).lower().replace(" ", "_"): column for column in data.columns}
	for name in names:
		if name in normalized:
			return normalized[name]
	return None


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
	"""Add safe, schema-adaptive transaction features without removing source columns."""
	features = data.copy()
	amount_column = _find_column(features, ("amount", "transaction_amount", "amt"))
	customer_column = _find_column(
		features, ("customer_id", "customer", "card_id", "cc_num", "account_id")
	)
	time_column = _find_column(
		features, ("transaction_time", "timestamp", "datetime", "date", "time")
	)

	if amount_column:
		amount = pd.to_numeric(features[amount_column], errors="coerce")
		features["amount_log"] = np.log1p(amount.clip(lower=0))
		if customer_column:
			grouped = features.groupby(customer_column, dropna=False)[amount_column]
			features["customer_average_amount"] = grouped.transform("mean")
			features["amount_deviation"] = amount - features["customer_average_amount"]
			features["customer_transaction_frequency"] = grouped.transform("count")

	if time_column:
		raw_time = features[time_column]
		parsed_time = pd.to_datetime(raw_time, errors="coerce")
		if parsed_time.notna().any():
			features["transaction_hour"] = parsed_time.dt.hour
			features["transaction_day_of_week"] = parsed_time.dt.dayofweek
			if customer_column:
				ordered = features.assign(_parsed_time=parsed_time).sort_values(
					[customer_column, "_parsed_time"]
				)
				intervals = ordered.groupby(customer_column)["_parsed_time"].diff().dt.total_seconds()
				features["previous_transaction_interval"] = intervals.reindex(features.index)
		else:
			numeric_time = pd.to_numeric(raw_time, errors="coerce")
			if numeric_time.notna().any():
				features["transaction_hour"] = (numeric_time // 3600) % 24

	object_columns = [
		column
		for column in features.columns
		if pd.api.types.is_object_dtype(features[column])
		or pd.api.types.is_string_dtype(features[column])
	]
	for column in object_columns:
		if re.search(r"category|merchant|location|type", str(column), re.IGNORECASE):
			features[column] = features[column].fillna("unknown").astype(str)
	return features
