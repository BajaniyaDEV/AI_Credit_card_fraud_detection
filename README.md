# AI-Powered Credit Card Fraud Detection System

Academic capstone project for detecting fraudulent credit-card transactions with machine
learning. The system will compare interpretable and ensemble models while prioritizing
precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrices over accuracy alone.

## Project Status

Phase 5: dataset ingestion, exploratory analysis, leakage-safe preprocessing, imbalance
experiments, model comparison, evaluation reports, and model artifacts are implemented.

Remaining work includes threshold tuning, SHAP explainability, REST API integration, and
dashboard development.

## Structure

```text
data/raw/          Source datasets (not committed)
data/processed/    Generated datasets (not committed)
notebooks/         Reproducible analysis and experiment notebooks
src/               Reusable preprocessing, feature, training, and inference code
models/            Persisted model artifacts (not committed)
reports/           Generated metrics and figures (not committed)
app/               Application entry points
tests/             Automated tests
```

## Setup

Create a Python 3.11+ virtual environment, activate it, and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python --version
python -m pip --version
python -c "import numpy, pandas, sklearn, imblearn, xgboost, shap, fastapi, streamlit; print('Environment OK')"
```

## Dataset validation and analysis

Place exactly one CSV in `data/raw/`, or pass a CSV path explicitly. The loader identifies
the target only from an exact `is_fraud`, `fraud`, `class`, `label`, or `target` column name;
otherwise it fails and asks for `--target` rather than guessing.

Validate and print the filename, shape, schema, missing values, duplicates, target, and class
distribution without training a model:

```powershell
python -m src.validate_dataset
python -m src.validate_dataset --data data/raw/your_file.csv --target your_label
```

Run the EDA notebook at `notebooks/06_eda.ipynb`. It discovers the actual numeric and
amount-like columns at runtime and does not train models.

Prepare train/validation/test splits and save a train-fitted preprocessing artifact:

```powershell
python -m src.prepare_dataset --data data/raw/your_file.csv --target your_label
```

The preprocessing removes exact duplicates before splitting, drops rows with missing target
values, imputes and encodes features inside a scikit-learn pipeline, and fits the artifact on
training data only. The default imbalance recommendation is `class_weight`, because it keeps
all observations and avoids synthetic examples before the feature schema and minority class
behavior have been studied. `RandomUnderSampler` and `SMOTE` are available through
`build_sampling_pipeline` for controlled training-fold experiments later; neither is applied
by the preparation command.

## Model training and comparison

The included `creditcard.csv` dataset contains 284,807 transactions and 31 columns. The
target is `Class`: 284,315 legitimate transactions and 492 fraudulent transactions. Exact
duplicate rows (1,081 in this dataset) are removed before splitting because they can otherwise
inflate evaluation results when copies cross a split boundary.

Run the complete comparison from the project root:

```powershell
.\.venv\Scripts\python.exe -m src.train
```

Training uses a fixed random state of 42 and stratified train/validation/test partitions.
Numeric values are median-imputed and scaled; categorical values are mode-imputed and
one-hot encoded. Each model has its own train-fitted preprocessing pipeline. The comparison
tests `class_weight`, `random_under_sampler`, and `smote`; samplers run only inside the
training pipeline, so validation and final test distributions remain unchanged. SMOTE is
applied to the Logistic Regression baseline as a controlled comparison because synthetic
training sets make tree ensembles substantially more resource-intensive.

The models are Logistic Regression, Decision Tree, Random Forest, and XGBoost when XGBoost
is installed. Results are compared using accuracy, fraud precision, fraud recall, F1, ROC-AUC,
PR-AUC, and confusion-matrix counts. Accuracy is not used as the sole selection criterion.

Outputs are written to:

- `reports/model_comparison.csv`: final test comparison table.
- `reports/validation_comparison.csv`: validation metrics for experiment review.
- `reports/figures/`: confusion matrices, ROC curves, precision-recall curves, and metric comparison.
- `reports/experiment_config.json`: source, split distributions, random state, and experiment results.
- `models/*__*.joblib`: fitted model pipelines containing preprocessing and the estimator.
- `models/preprocessing.joblib`: fitted preprocessing artifact and feature metadata.

Evaluate saved artifacts again with:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```
