# Results Summary

## Holdout Model Comparison

| Model | Train Accuracy | Test Accuracy | Train Macro F1 | Test Macro F1 | ROC-AUC OVR Macro | PR-AUC Macro |
|---|---:|---:|---:|---:|---:|---:|
| Multilayer Perceptron | 0.919583 | 0.910000 | 0.888000 | 0.880765 | 0.985903 | 0.951365 |
| Random Forest | 1.000000 | 0.895000 | 1.000000 | 0.868082 | 0.976438 | 0.923256 |
| Logistic Regression | 0.905833 | 0.891667 | 0.879961 | 0.863716 | 0.986053 | 0.951977 |
| Decision Tree | 0.863750 | 0.795000 | 0.845036 | 0.764769 | 0.927602 | 0.835853 |

## Cross-Validation Summary

| Model | CV Accuracy Mean | CV Accuracy Std | CV Macro F1 Mean | CV Macro F1 Std |
|---|---:|---:|---:|---:|
| Multilayer Perceptron | 0.912333 | 0.009226 | 0.866533 | 0.017966 |
| Logistic Regression | 0.895000 | 0.012953 | 0.849303 | 0.024844 |
| Random Forest | 0.898667 | 0.008654 | 0.846013 | 0.021082 |
| Decision Tree | 0.827000 | 0.009854 | 0.788719 | 0.019648 |

## Best-Performing Model

The Multilayer Perceptron is best by holdout accuracy and holdout macro F1 in the saved results. It is also the selected final deployment model.

## Important Observations

- Random Forest reaches `1.000000` train accuracy but lower holdout accuracy than the MLP, indicating stronger overfitting risk.
- Logistic Regression has the highest ROC-AUC OVR macro and PR-AUC macro by a small margin, while MLP has the best classification accuracy and macro F1.
- Decision Tree is the weakest evaluated candidate on the holdout split.

## Limitations

The risk classes are proxy labels generated from clustering, not external real-world labels. The results should be described as learning a financial-behavior risk segmentation rather than predicting verified default, fraud, or investment-loss outcomes.
