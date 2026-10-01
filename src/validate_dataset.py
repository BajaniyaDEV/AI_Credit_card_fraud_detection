"""Validate and summarize the fraud dataset without training a model."""

from __future__ import annotations

import argparse
import json

try:
	from .data_loader import discover_csv, load_dataset, summarize_dataset
except ImportError:
	from data_loader import discover_csv, load_dataset, summarize_dataset


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data", help="Path to one CSV; defaults to the only CSV in data/raw")
	parser.add_argument("--target", help="Exact fraud target column name")
	args = parser.parse_args()
	source = args.data or discover_csv()
	data = load_dataset(source)
	print(json.dumps(summarize_dataset(data, source, args.target), indent=2))


if __name__ == "__main__":
	main()