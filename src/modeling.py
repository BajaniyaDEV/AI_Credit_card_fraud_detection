"""Shared model evaluation and reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
	precision_recall_curve,
)

try:
	from .features import engineer_features
	from .preprocessing import prepare_data
except ImportError:
	from features import engineer_features
	from preprocessing import prepare_data


RESULT_COLUMNS = [
	"Model",
	"ImbalanceStrategy",
	"Accuracy",
	"Precision",
	"Recall",
	"F1",
	"ROC-AUC",
	"PR-AUC",
	"True Negatives",
	"False Positives",
	"False Negatives",
	"True Positives",
]


def prepare_model_splits(
	data: pd.DataFrame,
	target_column: str | None = None,
	random_state: int = 42,
) -> dict[str, Any]:
	"""Prepare raw splits and engineer each split independently."""
	prepared = prepare_data(data, target_column=target_column, random_state=random_state)
	train_features = engineer_features(prepared["x_train"])
	validation_features = engineer_features(prepared["x_validation"])
	test_features = engineer_features(prepared["x_test"])
	for split_name, features in (
		("x_train", train_features),
		("x_validation", validation_features),
		("x_test", test_features),
	):
		prepared[split_name] = features.reindex(columns=train_features.columns)
	prepared["feature_columns"] = train_features.columns.tolist()
	return prepared


def evaluate_predictions(
	y_true: pd.Series,
	predictions: Any,
	probabilities: Any,
) -> dict[str, Any]:
	"""Return fraud-focused metrics and confusion-matrix counts."""
	matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
	true_negative, false_positive, false_negative, true_positive = matrix.ravel()
	return {
		"Accuracy": float(accuracy_score(y_true, predictions)),
		"Precision": float(precision_score(y_true, predictions, zero_division=0)),
		"Recall": float(recall_score(y_true, predictions, zero_division=0)),
		"F1": float(f1_score(y_true, predictions, zero_division=0)),
		"ROC-AUC": float(roc_auc_score(y_true, probabilities)),
		"PR-AUC": float(average_precision_score(y_true, probabilities)),
		"True Negatives": int(true_negative),
		"False Positives": int(false_positive),
		"False Negatives": int(false_negative),
		"True Positives": int(true_positive),
	}


def evaluate_estimator(estimator: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
	"""Evaluate a fitted estimator without changing its decision threshold."""
	predictions = estimator.predict(x)
	probabilities = estimator.predict_proba(x)[:, 1]
	return evaluate_predictions(y, predictions, probabilities)


def save_confusion_matrix(metrics: dict[str, Any], title: str, output_path: str | Path) -> None:
	"""Save a labeled confusion-matrix plot for one experiment."""
	matrix = [
		[metrics["True Negatives"], metrics["False Positives"]],
		[metrics["False Negatives"], metrics["True Positives"]],
	]
	figure, axis = plt.subplots(figsize=(5, 4))
	sns.heatmap(
		matrix,
		annot=True,
		fmt="d",
		cmap="Blues",
		cbar=False,
		xticklabels=["Legitimate", "Fraud"],
		yticklabels=["Legitimate", "Fraud"],
		ax=axis,
	)
	axis.set_xlabel("Predicted label")
	axis.set_ylabel("True label")
	axis.set_title(title)
	figure.tight_layout()
	figure.savefig(output_path, dpi=160)
	plt.close(figure)


def save_curve_plots(
	curves: Iterable[dict[str, Any]],
	output_dir: str | Path,
) -> None:
	"""Save ROC and precision-recall curves for final test predictions."""
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	roc_figure, roc_axis = plt.subplots(figsize=(8, 6))
	pr_figure, pr_axis = plt.subplots(figsize=(8, 6))
	for curve in curves:
		label = curve["label"]
		false_positive_rate, true_positive_rate, _ = roc_curve(curve["y_true"], curve["probabilities"])
		precision, recall, _ = precision_recall_curve(curve["y_true"], curve["probabilities"])
		roc_axis.plot(false_positive_rate, true_positive_rate, label=f"{label} (AUC={curve['ROC-AUC']:.3f})")
		pr_axis.plot(recall, precision, label=f"{label} (AP={curve['PR-AUC']:.3f})")
	roc_axis.plot([0, 1], [0, 1], "k--", linewidth=1)
	roc_axis.set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
	pr_axis.set(title="Precision-recall curves", xlabel="Recall", ylabel="Precision")
	for axis in (roc_axis, pr_axis):
		axis.legend(fontsize="small")
		axis.grid(alpha=0.25)
	roc_figure.tight_layout()
	pr_figure.tight_layout()
	roc_figure.savefig(output_path / "roc_curves.png", dpi=160)
	pr_figure.savefig(output_path / "precision_recall_curves.png", dpi=160)
	plt.close(roc_figure)
	plt.close(pr_figure)


def save_metric_comparison(results: pd.DataFrame, output_path: str | Path) -> None:
	"""Save a compact comparison plot focused on fraud detection metrics."""
	long_results = results.melt(
		id_vars=["Model", "ImbalanceStrategy"],
		value_vars=["Precision", "Recall", "F1", "PR-AUC"],
		var_name="Metric",
		value_name="Score",
	)
	long_results["Experiment"] = (
		long_results["Model"] + " / " + long_results["ImbalanceStrategy"]
	)
	figure, axis = plt.subplots(figsize=(max(10, len(results) * 0.65), 6))
	sns.barplot(data=long_results, x="Experiment", y="Score", hue="Metric", ax=axis)
	axis.set_title("Fraud detection metric comparison on the final test set")
	axis.set_xlabel("")
	axis.set_ylabel("Score")
	axis.tick_params(axis="x", rotation=70)
	axis.set_ylim(0, 1.05)
	axis.legend(title="Metric")
	figure.tight_layout()
	figure.savefig(output_path, dpi=160)
	plt.close(figure)
