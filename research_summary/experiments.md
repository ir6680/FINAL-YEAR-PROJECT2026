# Experiments

## Models Evaluated

- Logistic Regression
- Decision Tree
- Multilayer Perceptron
- Random Forest

## Hyperparameters

Logistic Regression: `max_iter=3000`, `class_weight="balanced"`, `random_state=42`.

Decision Tree: `max_depth=6`, `min_samples_leaf=10`, `class_weight="balanced"`, `random_state=42`.

Multilayer Perceptron: hidden layers `(64, 32)`, activation `relu`, solver `adam`, `alpha=0.001`, `learning_rate_init=0.001`, `max_iter=1000`, `early_stopping=True`, `validation_fraction=0.15`, `n_iter_no_change=20`, `random_state=42`.

Random Forest: `n_estimators=300`, `min_samples_leaf=1`, `class_weight="balanced_subsample"`, `random_state=42`, `n_jobs=-1`.

## Baselines

Logistic Regression and Decision Tree serve as simpler baselines. Random Forest serves as an ensemble comparison. The MLP is the selected final model in the project design.

## Evaluation Metrics

The notebook reports train/test accuracy, train/test macro F1, ROC-AUC OVR macro, PR-AUC macro, 5-fold CV accuracy and macro F1, curve AUC summaries, bootstrap confidence intervals for the selected model, and McNemar-style disagreement checks.

## Comparison Methodology

All candidate models are trained and evaluated on the same production-aligned holdout split and the same KMeans-derived risk labels. Class order is fixed as Low Risk, Medium Risk, High Risk.
