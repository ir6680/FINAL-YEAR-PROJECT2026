# Research Overview

## Research Title

Spending-Aware DeFi Assistant Using Leakage-Aware Machine Learning Risk Classification

## Research Problem

Retail DeFi recommendation systems can become unsuitable when they ignore a user's basic financial health. This project studies a pipeline that estimates user risk bands from personal-finance behavior and uses those bands to support safer portfolio allocation logic.

## Research Objective

Compare supervised models that learn cluster-derived financial risk labels using deployable input features only, then select a final model for the assistant workflow.

## Research Gap

The project does not use externally validated credit-default or investment-loss labels. Instead, it derives an interpretable financial risk proxy from spending, surplus, and debt pressure while preventing clustering-derived variables from entering the supervised classifier.

## Proposed Approach

The workflow computes corrected financial metrics, fits KMeans with three clusters on `surplus`, `expense_ratio`, and `debt_pressure`, maps clusters to risk bands, trains candidate classifiers on allowed user-input features, and evaluates them on a production-aligned holdout split.

## Main Contributions

- Leakage-aware separation between label-generation features and classifier input features.
- Cluster-derived proxy labels mapped to Low, Medium, and High Risk.
- Comparison of Logistic Regression, Decision Tree, Multilayer Perceptron, and Random Forest.
- Publication-quality figures, including confusion matrices for all compared models.
- Traceable metrics and saved final model bundle for reproducibility.

## Complete Research Pipeline

Dataset -> financial feature correction -> leakage validation -> KMeans risk labeling -> 80/20 holdout split -> candidate model training -> holdout and CV evaluation -> model comparison -> final MLP bundle -> result figures.

## Main Findings

The Multilayer Perceptron achieved the highest holdout accuracy and macro F1 in the saved model comparison table: accuracy `0.910000` and macro F1 `0.880765`.
