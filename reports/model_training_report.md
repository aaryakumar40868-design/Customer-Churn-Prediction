# Stage 3: Model Training & Algorithmic Benchmarking Report

## 1. Executive Summary
In this stage, 7 diverse classification models were implemented and trained on the standardized 47-dimensional Telco customer feature space (4,930 training records). All models were evaluated on the independent, stratified validation set (1,056 records). The **Logistic Regression** achieved the top performance with a **Validation ROC-AUC of 0.8484** and an **F1-Score of 0.6248**.

## 2. Models Evaluated & Configuration Strategy
| Model Family | Algorithm | Key Hyperparameters | Imbalance Strategy |
|---|---|---|---|
| Linear Baseline | Logistic Regression | `C=1.0, penalty='l2', solver='lbfgs', max_iter=1000` | `class_weight='balanced'` |
| Ensemble (Bagging) | Random Forest | `n_estimators=200, max_depth=12, min_samples_split=6` | `class_weight='balanced'` |
| Ensemble (Boosting) | Gradient Boosting | `n_estimators=150, learning_rate=0.08, max_depth=4` | Subsample 0.85 & Gradient Step |
| Modern Boosting | HistGradientBoosting | `learning_rate=0.08, max_iter=150, max_depth=6` | `class_weight='balanced'` |
| Adaptive Boosting | AdaBoost | `n_estimators=100, learning_rate=0.1` | Sequential Weight Adjustment |
| Kernel Classifier | Support Vector Machine (SVC) | `C=1.0, kernel='rbf', probability=True` | `class_weight='balanced'` |
| Deep Learning | Multi-Layer Perceptron (ANN) | `layers=(64, 32), activation='relu', solver='adam'` | Early stopping & L2 regularization |

## 3. Validation Performance Comparison
| Rank | Model Name | Validation ROC-AUC | Validation PR-AUC | Validation F1 | Recall | Precision | Accuracy | Train Time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | **Logistic Regression** | 0.8484 | 0.6315 | 0.6248 | 0.8179 | 0.5055 | 0.7396 | 0.046 |
| 2 | **Multi-Layer Perceptron (ANN)** | 0.8433 | 0.6147 | 0.6187 | 0.6750 | 0.5710 | 0.7794 | 0.436 |
| 3 | **AdaBoost** | 0.8432 | 0.6223 | 0.4731 | 0.3607 | 0.6871 | 0.7869 | 0.575 |
| 4 | **Gradient Boosting** | 0.8422 | 0.6259 | 0.5768 | 0.5500 | 0.6063 | 0.7860 | 2.095 |
| 5 | **Random Forest** | 0.8412 | 0.6215 | 0.6283 | 0.7214 | 0.5565 | 0.7737 | 0.457 |
| 6 | **HistGradientBoosting** | 0.8340 | 0.6216 | 0.6121 | 0.7607 | 0.5120 | 0.7443 | 1.954 |
| 7 | **Support Vector Machine** | 0.8291 | 0.5811 | 0.6215 | 0.8036 | 0.5068 | 0.7405 | 4.015 |

## 4. Selection of Best Model
- **Selected Champion Model**: `Logistic Regression`
- **Validation ROC-AUC**: `0.8484`
- **Validation Recall**: `0.8179` (Identifies 81.8% of churning customers)
- **Validation F1-Score**: `0.6248`
- **Validation Precision**: `0.5055`
- **Selection Rationale**: In churn prediction, missing at-risk customers (false negatives) incurs substantial lost revenue. The champion model maintains high discriminative ability (ROC-AUC) while providing superior sensitivity (recall) to capture potential defectors before contract termination.

## 5. Persisted Model Artifacts
- `Logistic Regression`: [models\logistic_regression.joblib](file:///models/logistic_regression.joblib)
- `Random Forest`: [models\random_forest.joblib](file:///models/random_forest.joblib)
- `Gradient Boosting`: [models\gradient_boosting.joblib](file:///models/gradient_boosting.joblib)
- `HistGradientBoosting`: [models\histgradientboosting.joblib](file:///models/histgradientboosting.joblib)
- `AdaBoost`: [models\adaboost.joblib](file:///models/adaboost.joblib)
- `Support Vector Machine`: [models\support_vector_machine.joblib](file:///models/support_vector_machine.joblib)
- `Multi-Layer Perceptron (ANN)`: [models\multi_layer_perceptron_ann.joblib](file:///models/multi_layer_perceptron_ann.joblib)
- `Best Model`: [models\best_model.joblib](file:///models/best_model.joblib)
- `Validation Metrics CSV`: [outputs/model_results\validation_metrics_summary.csv](file:///outputs/model_results/validation_metrics_summary.csv)
- `Validation Metrics JSON`: [outputs/model_results\validation_metrics_summary.json](file:///outputs/model_results/validation_metrics_summary.json)

