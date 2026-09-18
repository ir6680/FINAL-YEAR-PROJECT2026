# Research Project

## Project Overview

This clean package contains the research materials for a spending-aware DeFi assistant. The pipeline converts personal-finance records into engineered financial metrics, creates cluster-derived proxy risk labels, trains multiple machine-learning models to approximate those labels from deployable user-input features, and compares model performance before selecting a final Multilayer Perceptron model.

## Research Objective

Evaluate whether a leakage-aware supervised model can learn a financial risk segmentation derived from surplus, expense ratio, and debt pressure, then support DeFi recommendation logic with documented validation results.

## Directory Structure

```text
clean_research_project/
  README.md
  requirements.txt
  notebooks/main_research_notebook.ipynb
  research_summary/
  src/
  configs/
  data/
  results/
  images/
  models/
  scripts/
  docs/source_docs/
```

## Dataset

The included dataset is `data/raw/personal_finance_tracker_dataset.csv` with 3000 rows and 28 columns. Derived research tables are in `data/processed/` and `results/tables/`.

## Methodology

The notebook loads the dataset, recomputes financial metrics, validates leakage constraints, generates KMeans risk labels, trains four candidate classifiers, evaluates them on a production-aligned holdout split and cross-validation, then saves metrics, figures, and the final bundle.

## Models

Compared models:

- Logistic Regression
- Decision Tree
- Multilayer Perceptron
- Random Forest

Selected final model: Multilayer Perceptron.

## Experimental Setup

Use `notebooks/main_research_notebook.ipynb` for the full workflow. To regenerate confusion matrices:

```bash
python scripts/generate_model_confusion_matrices.py
```

## Evaluation Metrics

Reported metrics include holdout accuracy, holdout macro F1, ROC-AUC OVR macro, PR-AUC macro, 5-fold CV accuracy/F1, and confusion matrices.

## Results

Main result files:

- `results/model_comparisons/model_comparison.csv`
- `results/metrics/cv_validation.csv`
- `results/metrics/finance_risk_metrics_final.json`
- `results/metrics/confusion_matrix_generation_report.json`

## Figures

Figures are organized under `images/`. Confusion matrices for all compared models are in `images/confusion_matrices/` and were generated from actual holdout predictions.

## How to Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/main_research_notebook.ipynb
```

If Jupyter is not installed, install a notebook interface separately or run the Python scripts directly.

## Dependencies

The minimal Python dependencies are listed in `requirements.txt`.

## Important Notes

- Risk labels are cluster-derived proxy labels, not externally verified default or investment-loss outcomes.
- The dataset is included because it is small in this repository. Confirm sharing permission before external distribution.
- Archived notebooks, backup datasets, cache files, full Streamlit UI code, and unrelated app assets were intentionally excluded.
