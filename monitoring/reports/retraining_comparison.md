# Retraining Comparison Report

## Purpose

This report documents how restaurants4u compares the current production model with a candidate retrained model after drift detection or scheduled retraining.

The goal is to prevent automatic promotion of a weaker model.

## Current Production Model

The current production model is the Random Forest restaurant classifier from the Phase 1 MLOps pipeline.

The model predicts whether a restaurant is likely to be highly rated based on structured metadata such as category, location, and contact information availability.

## Candidate Retrained Model

The candidate retrained model is produced when the training pipeline is rerun after new data is available or drift is detected.

## Comparison Metrics

| Metric | Current Model | Candidate Model | Promotion Rule |
|---|---:|---:|---|
| Accuracy | 0.8508 | To be measured after retraining | Candidate should remain stable or improve |
| Precision | 0.8605 | To be measured after retraining | Candidate should remain stable or improve |
| Recall | 0.9816 | To be measured after retraining | Candidate should not significantly reduce recall |
| F1 Score | To be measured | To be measured | Candidate should improve or remain stable |
| ROC-AUC | To be measured | To be measured | Candidate should improve or remain stable |
| Balanced Accuracy | To be measured | To be measured | Candidate should improve or remain stable |

## Promotion Rule

The candidate model should only be promoted if it improves or maintains F1 score, balanced accuracy, and recall. If the candidate model performs worse than the current model, the current model remains in production.

## Retraining Trigger

Retraining should be triggered when one or more of the following conditions occur:

- EvidentlyAI detects significant data drift.
- The positive recommendation rate changes by more than 15% from baseline.
- Model performance drops below the accepted threshold.
- New restaurant data differs significantly from the original training distribution.
- Production prediction distribution becomes unstable.

## Current Phase 2 Status

For the Phase 2 submission, the retraining workflow is implemented and the model comparison process is documented. Final candidate metrics should be updated after the full DVC-backed training pipeline is rerun by the DVC/deployment owner.