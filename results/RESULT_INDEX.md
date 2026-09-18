# Result Index

## Holdout Metrics

Source: `model_comparisons/model_comparison.csv`

| Model | Metric | Value | Related Figure |
|---|---|---:|---|
| Multilayer Perceptron | Test Accuracy | 0.910000 | `images/results/performance_comparison.png` |
| Multilayer Perceptron | Test Macro F1 | 0.880765 | `images/results/performance_comparison.png` |
| Random Forest | Test Accuracy | 0.895000 | `images/results/performance_comparison.png` |
| Random Forest | Test Macro F1 | 0.868082 | `images/results/performance_comparison.png` |
| Logistic Regression | Test Accuracy | 0.891667 | `images/results/performance_comparison.png` |
| Logistic Regression | Test Macro F1 | 0.863716 | `images/results/performance_comparison.png` |
| Decision Tree | Test Accuracy | 0.795000 | `images/results/performance_comparison.png` |
| Decision Tree | Test Macro F1 | 0.764769 | `images/results/performance_comparison.png` |

## Confusion Matrices

Source: `metrics/confusion_matrix_generation_report.json`

| Model | Matrix Total | Related Figure |
|---|---:|---|
| Logistic Regression | 600 | `images/confusion_matrices/confusion_matrix_logistic_regression.png` |
| Decision Tree | 600 | `images/confusion_matrices/confusion_matrix_decision_tree.png` |
| Multilayer Perceptron | 600 | `images/confusion_matrices/confusion_matrix_multilayer_perceptron.png` |
| Random Forest | 600 | `images/confusion_matrices/confusion_matrix_random_forest.png` |

## Cross-Validation Metrics

Source: `metrics/cv_validation.csv`

| Model | CV Accuracy Mean | CV Macro F1 Mean |
|---|---:|---:|
| Multilayer Perceptron | 0.912333 | 0.866533 |
| Logistic Regression | 0.895000 | 0.849303 |
| Random Forest | 0.898667 | 0.846013 |
| Decision Tree | 0.827000 | 0.788719 |

## Final Bundle

Source: `models/finance_risk_bundle_final.joblib`

Selected final model: Multilayer Perceptron.
