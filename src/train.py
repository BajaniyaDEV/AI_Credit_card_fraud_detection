"""Train and compare leakage-safe fraud detection experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

try:
	from .data_loader import discover_csv, load_dataset
	from .modeling import (
		RESULT_COLUMNS,
		evaluate_estimator,
		prepare_model_splits,
		save_confusion_matrix,
		save_curve_plots,
		save_metric_comparison,
	)
	from .preprocessing import build_preprocessor, build_sampling_pipeline
except ImportError:
	from data_loader import discover_csv, load_dataset
	from modeling import (
		RESULT_COLUMNS,
		evaluate_estimator,
		prepare_model_splits,
		save_confusion_matrix,
		save_curve_plots,
		save_metric_comparison,
	)
	from preprocessing import build_preprocessor, build_sampling_pipeline


STRATEGIES = ("class_weight", "random_under_sampler", "smote")


def build_models(
	strategy: str = "class_weight",
	random_state: int = 42,
	training_target: pd.Series | None = None,
) -> dict[str, Any]:
	"""Return the required estimators configured for one imbalance strategy."""
	if strategy not in STRATEGIES:
		raise ValueError(f"strategy must be one of: {', '.join(STRATEGIES)}")
	weighted = strategy == "class_weight"
	class_weight = "balanced" if weighted else None
	models: dict[str, Any] = {
		"logistic_regression": LogisticRegression(
			max_iter=1000, class_weight=class_weight, random_state=random_state
		),
		"decision_tree": DecisionTreeClassifier(
			max_depth=12, class_weight=class_weight, random_state=random_state
		),
		"random_forest": RandomForestClassifier(
			n_estimators=100,
			class_weight="balanced_subsample" if weighted else None,
			n_jobs=-1,
			random_state=random_state,
		),
	}
	try:
		from xgboost import XGBClassifier

		positive_weight = 1.0
		if weighted and training_target is not None:
			counts = training_target.value_counts()
			positive_weight = float(counts.get(0, 0) / max(counts.get(1, 1), 1))
		models["xgboost"] = XGBClassifier(
			n_estimators=100,
			max_depth=6,
			learning_rate=0.08,
			eval_metric="logloss",
			tree_method="hist",
			scale_pos_weight=positive_weight,
			n_jobs=-1,
			random_state=random_state,
		)
	except ImportError:
		pass
	return models


def train_models(
	data_path: str | Path | None = None,
	target_column: str | None = None,
	output_dir: str | Path = "models",
	reports_dir: str | Path = "reports",
	strategies: tuple[str, ...] = STRATEGIES,
	random_state: int = 42,
) -> dict[str, Any]:
	"""Train all selected model/imbalance experiments and save test reports."""
	if any(strategy not in STRATEGIES for strategy in strategies):
		raise ValueError(f"strategies must be selected from: {', '.join(STRATEGIES)}")
	source = Path(data_path) if data_path else discover_csv()
	data = load_dataset(source)
	prepared = prepare_model_splits(data, target_column=target_column, random_state=random_state)
	models_path = Path(output_dir)
	reports_path = Path(reports_dir)
	figures_path = reports_path / "figures"
	models_path.mkdir(parents=True, exist_ok=True)
	figures_path.mkdir(parents=True, exist_ok=True)

	preprocessing_artifact = {
		"preprocessor": build_preprocessor(prepared["x_train"]).fit(prepared["x_train"]),
		"feature_columns": prepared["feature_columns"],
		"target_column": prepared["target_column"],
		"duplicate_rows_removed": prepared["duplicate_rows_removed"],
		"random_state": random_state,
	}
	joblib.dump(preprocessing_artifact, models_path / "preprocessing.joblib")

	results: list[dict[str, Any]] = []
	validation_results: list[dict[str, Any]] = []
	curves: list[dict[str, Any]] = []
	for strategy in strategies:
		strategy_models = build_models(strategy, random_state, prepared["y_train"])
		if strategy == "smote":
			strategy_models = {"logistic_regression": strategy_models["logistic_regression"]}
		for model_name, estimator in strategy_models.items():
			pipeline = build_sampling_pipeline(
				estimator,
				build_preprocessor(prepared["x_train"]),
				strategy,
				random_state,
			)
			pipeline.fit(prepared["x_train"], prepared["y_train"])
			experiment_name = f"{model_name}__{strategy}"
			model_path = models_path / f"{experiment_name}.joblib"
			joblib.dump(pipeline, model_path)
			validation_metrics = evaluate_estimator(
				pipeline, prepared["x_validation"], prepared["y_validation"]
			)
			test_metrics = evaluate_estimator(pipeline, prepared["x_test"], prepared["y_test"])
			results.append({"Model": model_name, "ImbalanceStrategy": strategy, **test_metrics})
			validation_results.append(
				{"Model": model_name, "ImbalanceStrategy": strategy, **validation_metrics}
			)
			save_confusion_matrix(
				test_metrics,
				f"{experiment_name} confusion matrix",
				figures_path / f"{experiment_name}_confusion_matrix.png",
			)
			probabilities = pipeline.predict_proba(prepared["x_test"])[:, 1]
			curves.append(
				{
					"label": experiment_name,
					"y_true": prepared["y_test"],
					"probabilities": probabilities,
					**test_metrics,
				}
			)

	results_frame = pd.DataFrame(results, columns=RESULT_COLUMNS)
	results_frame.to_csv(reports_path / "model_comparison.csv", index=False)
	pd.DataFrame(validation_results).to_csv(reports_path / "validation_comparison.csv", index=False)
	save_curve_plots(curves, figures_path)
	save_metric_comparison(results_frame, figures_path / "model_metric_comparison.png")

	configuration = {
		"source": str(source),
		"target_column": prepared["target_column"],
		"random_state": random_state,
		"duplicate_rows_removed": prepared["duplicate_rows_removed"],
		"split_distributions": prepared["split_distributions"],
		"strategies": list(strategies),
		"strategy_model_scope": {
			"class_weight": "all configured models",
			"random_under_sampler": "all configured models",
			"smote": ["logistic_regression"],
		},
		"models": sorted({row["Model"] for row in results}),
		"test_results": results,
		"validation_results": validation_results,
	}
	(reports_path / "experiment_config.json").write_text(
		json.dumps(configuration, indent=2), encoding="utf-8"
	)
	(models_path / "metadata.json").write_text(
		json.dumps(
			{
				"target_column": prepared["target_column"],
				"feature_columns": prepared["feature_columns"],
				"models": [f"{row['Model']}__{row['ImbalanceStrategy']}" for row in results],
				"random_state": random_state,
			},
			indent=2,
		),
		encoding="utf-8",
	)
	return configuration


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data", help="Path to the input CSV; defaults to data/raw/creditcard.csv")
	parser.add_argument("--target", help="Fraud label column name")
	parser.add_argument("--output-dir", default="models")
	parser.add_argument("--reports-dir", default="reports")
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument(
		"--strategies",
		nargs="+",
		choices=STRATEGIES,
		default=list(STRATEGIES),
		help="Imbalance strategies to compare",
	)
	args = parser.parse_args()
	configuration = train_models(
		args.data,
		args.target,
		args.output_dir,
		args.reports_dir,
		tuple(args.strategies),
		args.random_state,
	)
	print(json.dumps(configuration, indent=2))


if __name__ == "__main__":
	main()
