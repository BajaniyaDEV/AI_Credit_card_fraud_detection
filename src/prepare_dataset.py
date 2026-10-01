"""Prepare a fraud dataset for later model training without fitting a model."""

from __future__ import annotations

import argparse
import json

try:
	from .data_loader import discover_csv, load_dataset
	from .preprocessing import prepare_data, save_preprocessing_artifact
except ImportError:
	from data_loader import discover_csv, load_dataset
	from preprocessing import prepare_data, save_preprocessing_artifact


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data", help="Path to one CSV; defaults to the only CSV in data/raw")
	parser.add_argument("--target", help="Exact fraud target column name")
	parser.add_argument("--artifact", default="models/preprocessing.joblib")
	args = parser.parse_args()
	source = args.data or discover_csv()
	prepared = prepare_data(load_dataset(source), target_column=args.target)
	save_preprocessing_artifact(prepared, args.artifact)
	result = {
		"source": str(source),
		"target_column": prepared["target_column"],
		"feature_columns": prepared["feature_columns"],
		"split_rows": {
			"train": len(prepared["x_train"]),
			"validation": len(prepared["x_validation"]),
			"test": len(prepared["x_test"]),
		},
		"duplicate_rows_removed": prepared["duplicate_rows_removed"],
		"imbalance": prepared["imbalance"],
		"artifact": args.artifact,
	}
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()