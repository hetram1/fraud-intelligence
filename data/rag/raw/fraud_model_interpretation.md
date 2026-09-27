# Fraud Model Interpretation Guide

Document type: Synthetic machine-learning guidance
Version: 1.0

## Model Output

The fraud model produces a probability-like score representing estimated fraud risk for a claim.

The score should be treated as a ranking signal for investigation.

## Thresholds

A classification threshold determines whether a claim is flagged by the application.

Changing the threshold changes the trade-off between precision and recall.

Threshold selection should be based on validation data and documented business requirements.

## Evaluation Metrics

PR-AUC is useful for evaluating fraud detection under class imbalance.

ROC-AUC measures ranking performance across classification thresholds.

Precision measures the fraction of flagged claims that are actually positive in the evaluation data.

Recall measures the fraction of positive claims that are successfully flagged.

## Limitations

Evaluation metrics describe performance on the available evaluation population.

They do not guarantee performance on future claims or different populations.

Model predictions should therefore be reviewed alongside other evidence.

## Explainability

Feature-level explanations can identify inputs that contributed to an individual prediction.

Explanations describe model behavior and should not be presented as independent proof of fraudulent intent.
