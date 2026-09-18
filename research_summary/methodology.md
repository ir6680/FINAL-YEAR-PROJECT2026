# Methodology

## Data

The notebook loads `data/raw/personal_finance_tracker_dataset.csv`. The file contains 3000 records and 28 columns.

## Preprocessing

The notebook recomputes `monthly_expense_total` from `rent_or_mortgage`, `essential_spending`, and `discretionary_spending`. It then derives `surplus`, `expense_ratio`, and `debt_pressure`. Infinite or invalid ratio values are replaced with `999`.

## Feature Engineering

Clustering features:

- `surplus`
- `expense_ratio`
- `debt_pressure`

Classifier features:

- `monthly_income`
- `credit_score`
- `rent_or_mortgage`
- `essential_spending`
- `discretionary_spending`

Target-like and clustering-derived columns are blocked from the classifier to reduce leakage.

## Label Construction

KMeans is fit with `n_clusters=3`, `n_init=50`, and `random_state=42`. Clusters are ranked by severity using expense ratio, debt pressure, and a surplus component. The ordered clusters are mapped to:

- `0`: Low Risk
- `1`: Medium Risk
- `2`: High Risk

## Training

The split is an 80/20 holdout split using `train_test_split`, `random_state=42`, shuffling enabled, and stratification by monthly-income quantile bins. Each classifier is built as a scikit-learn pipeline with median imputation and standard scaling before the estimator.

## Validation

The project reports 5-fold cross-validation using production-style folds. Each fold refits KMeans on training data, predicts clusters for test data, maps risk labels, trains the supervised classifier, and evaluates fold predictions.

## Testing

Holdout testing uses 600 test samples. The same test labels and class order are used for all compared models.

## Evaluation

Evaluation metrics include accuracy, macro F1, ROC-AUC OVR macro, PR-AUC macro, weighted precision/recall/F1/accuracy for comparison plots, micro-average ROC/PR curves, selected-model classwise curves, uncertainty-accuracy behavior, and confusion matrices.
