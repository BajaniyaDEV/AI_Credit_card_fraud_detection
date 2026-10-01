from __future__ import annotations

import pandas as pd

from src.features import engineer_features


def test_engineer_features_preserves_rows_and_adds_safe_features() -> None:
	data = pd.DataFrame(
		{
			"Amount": [10.0, None],
			"customer_id": ["a", "a"],
			"transaction_time": ["2026-01-01 01:00:00", "2026-01-01 02:00:00"],
			"merchant_category": ["retail", None],
		}
	)

	transformed = engineer_features(data)

	assert transformed.shape[0] == data.shape[0]
	assert {"amount_log", "customer_average_amount", "transaction_hour"}.issubset(
		transformed.columns
	)
	assert transformed["merchant_category"].tolist() == ["retail", "unknown"]
