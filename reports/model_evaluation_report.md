# Stage 4: Model Evaluation & Decision Optimization Report

## 1. Test Set Evaluation Summary
The final models were evaluated on the completely unseen, stratified **Test Partition** (1,057 customer records; 281 churned, 776 retained; churn prevalence = 26.58%). The **Multi-Layer Perceptron (ANN)** established the strongest discrimination with a **Test ROC-AUC of 0.8599** and a **Test PR-AUC of 0.6705**.

## 2. Test Performance Benchmark Table
| Rank | Model Name | Test ROC-AUC | Test PR-AUC | Test F1 | Recall | Specificity | Precision | Accuracy | Brier Score |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Multi-Layer Perceptron (ANN)** | 0.8599 | 0.6705 | 0.6303 | 0.6370 | 0.8608 | 0.6237 | 0.8013 | 0.1328 |
| 2 | **Logistic Regression** | 0.8588 | 0.6879 | 0.6272 | 0.7722 | 0.7500 | 0.5280 | 0.7559 | 0.1558 |
| 3 | **Random Forest** | 0.8570 | 0.6842 | 0.6416 | 0.6975 | 0.8273 | 0.5939 | 0.7928 | 0.1376 |
| 4 | **Gradient Boosting** | 0.8529 | 0.6853 | 0.6083 | 0.5445 | 0.9111 | 0.6892 | 0.8136 | 0.1325 |
| 5 | **HistGradientBoosting** | 0.8486 | 0.6727 | 0.6461 | 0.7438 | 0.7977 | 0.5710 | 0.7833 | 0.1481 |
| 6 | **Support Vector Machine** | 0.8462 | 0.6482 | 0.6301 | 0.6335 | 0.8634 | 0.6268 | 0.8023 | 0.1371 |
| 7 | **AdaBoost** | 0.8425 | 0.6445 | 0.4597 | 0.3345 | 0.9562 | 0.7344 | 0.7909 | 0.1555 |

## 3. Confusion Matrix Breakdown (at 0.50 default threshold)
| Model Name | True Positives (TP) | False Negatives (FN) | True Negatives (TN) | False Positives (FP) |
|---|---|---|---|---|
| Multi-Layer Perceptron (ANN) | 179 | 102 | 668 | 108 |
| Logistic Regression | 217 | 64 | 582 | 194 |
| Random Forest | 196 | 85 | 642 | 134 |
| Gradient Boosting | 153 | 128 | 707 | 69 |
| HistGradientBoosting | 209 | 72 | 619 | 157 |
| Support Vector Machine | 178 | 103 | 670 | 106 |
| AdaBoost | 94 | 187 | 742 | 34 |

## 4. Decision Threshold Optimization & Business Value Analysis
In customer churn mitigation, setting the classification decision threshold to 0.50 is rarely financially optimal because the cost of losing a customer ($500 estimated CLTV) heavily outweighs the cost of a targeted retention initiative ($60 intervention cost).

- **Default Threshold (0.50)**:
  - Recall: `0.6370` | Precision: `0.6237` | F1: `0.6303`
- **F1-Optimal Threshold (0.46)**:
  - Max Achieved F1-Score: `0.6468`
- **High-Recall Operational Threshold (0.28)**:
  - Guarantees $\ge 85\%$ capture of all churners before account cancellation.
- **Financial Impact**:
  - Baseline unmitigated churn loss on test cohort: `$140,500.00`
  - Net savings achieved via targeted retention at optimal threshold: **`$32,030.00`**

## 5. Persisted Evaluation Outputs
- `test_metrics_csv`: [outputs/model_results\test_metrics_summary.csv](file:///outputs/model_results/test_metrics_summary.csv)
- `test_metrics_json`: [outputs/model_results\test_metrics_summary.json](file:///outputs/model_results/test_metrics_summary.json)
- `threshold_json`: [outputs/model_results\threshold_optimization.json](file:///outputs/model_results/threshold_optimization.json)
- `test_predictions_csv`: [outputs/model_results\test_predictions.csv](file:///outputs/model_results/test_predictions.csv)

