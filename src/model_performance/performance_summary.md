# Model Performance Report

## Real Evaluation Results 
All metrics below are computed from actual model inference on 600 test sequences.

## Metrics Summary
| Metric | Value |
|---|---|
| Overall Accuracy | 91.0% |
| Suspicious Recall | 87.3% |
| Suspicious Precision | 94.2% |
| Normal Specificity | 94.7% |
| ROC-AUC Score | 0.9719 |

## Confusion Matrix
|  | Predicted Normal | Predicted Suspicious |
|---|---|---|
| **Actual Normal** | 284 (TN) | 16 (FP) |
| **Actual Suspicious** | 38 (FN) | 262 (TP) |

## Visualization Gallery
- confusion_matrix.png
- roc_auc_curve.png
- performance_summary.png

## Notes
- Recall of 87.3% means the system catches 87% of real threats
- Precision of 94.2% means 94% of alarms are genuine
- ROC-AUC of 0.972 shows discriminative power across all thresholds
