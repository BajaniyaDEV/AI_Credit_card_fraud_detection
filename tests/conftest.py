from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def small_dataset() -> pd.DataFrame:
	rows = []
	for index in range(60):
		rows.append(
			{
				"Time": float(index * 60),
				"Amount": float(index + 1),
				"category": "online" if index % 2 else "retail",
				"Class": 1 if index % 10 == 0 else 0,
			}
		)
	return pd.DataFrame(rows)
