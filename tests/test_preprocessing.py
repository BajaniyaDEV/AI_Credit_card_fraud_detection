from __future__ import annotations

import pandas as pd

from src.preprocessing import build_preprocessor, prepare_data


def test_prepare_data_removes_duplicates_and_preserves_stratification(small_dataset) -> None:
	duplicated = small_dataset.iloc[[0]].copy()
	prepared = prepare_data(pd.concat([small_dataset, duplicated], ignore_index=True), random_state=42)

	assert prepared["duplicate_rows_removed"] == 1
	assert prepared["split_distributions"]["train"]["fraud"] > 0
	assert prepared["split_distributions"]["validation"]["fraud"] > 0
	assert prepared["split_distributions"]["test"]["fraud"] > 0

	transformed = build_preprocessor(prepared["x_train"]).fit_transform(prepared["x_train"])
	assert transformed.shape[0] == len(prepared["x_train"])
