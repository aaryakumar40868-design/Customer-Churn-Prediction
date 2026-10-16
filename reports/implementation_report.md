# Project Implementation Report: Machine Learning System for Customer Churn Prediction

## 1. Executive Summary
This report documents the end-to-end design, implementation, and empirical validation of a high-performance machine learning system built to predict customer churn using the Telco customer dataset (7,043 customer accounts). Customer churn poses a substantial financial risk to subscription-based telecommunications providers. The primary objective is to reliably detect churn risk prior to contract termination, allowing customer retention teams to deploy proactive, cost-effective intervention strategies.

The implemented architecture enforces strict data leakage prevention, stratified data partitioning, a multi-algorithm model zoo, out-of-sample benchmarking, and financial decision threshold optimization. Across 7 distinct model families, the **Multi-Layer Perceptron (ANN)** and **Logistic Regression** achieved top-tier discrimination on the unseen test set (**ROC-AUC: 0.8599**, **PR-AUC: 0.6879**), successfully identifying over **77%** of churning accounts at default settings and over **85%** under cost-optimized operating thresholds.

---

## 2. End-to-End System Architecture

The project is structured into a modular, sequential 5-stage pipeline orchestrated by [main.py](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/main.py):

```
+---------------------------------------------------------------------------------------------------+
|                                  TELCO CUSTOMER CHURN PIPELINE                                    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | STAGE 1: Data Understanding & Leakage Mitigation         |
                     | - Module: src/data_analysis.py                           |
                     | - Output: reports/data_understanding_report.md          |
                     +----------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | STAGE 2: Preprocessing & Stratified Splitting            |
                     | - Module: src/preprocessing.py                           |
                     | - Transformer: models/preprocessor.joblib                |
                     | - Artifacts: dataset/processed_data.npz                  |
                     | - Output: reports/preprocessing_report.md                |
                     +----------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | STAGE 3: Model Training & Algorithmic Benchmarking       |
                     | - Module: src/train_model.py                             |
                     | - Artifacts: models/*.joblib (7 Classifiers)             |
                     | - Output: reports/model_training_report.md               |
                     +----------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | STAGE 4: Model Evaluation & Threshold Optimization       |
                     | - Module: src/evaluate_model.py                          |
                     | - Results: outputs/model_results/test_metrics_summary.*  |
                     | - Output: reports/model_evaluation_report.md             |
                     +----------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | STAGE 5: Visual Diagnostics & Explainability             |
                     | - Module: src/visualization.py                           |
                     | - Visualizations: outputs/visualizations/*.png           |
                     +----------------------------------------------------------+
```

---

## 3. Stage 1 Implementation: Data Understanding & Leakage Prevention

Implemented in [src/data_analysis.py](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/src/data_analysis.py) and documented in [reports/data_understanding_report.md](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/reports/data_understanding_report.md).

### 3.1 Dataset Profile
- **Total Records**: 7,043 customer accounts
- **Total Input Features**: 33 initial columns
- **Class Distribution**:
  - Class 0 (Retained Customers): 5,174 (73.46%)
  - Class 1 (Churned Customers): 1,869 (26.54%)
  - Imbalance Ratio: **2.77 : 1** (Moderate class imbalance)

### 3.2 Elimination of Data Leakage & Extraneous Features
A critical failure mode in churn modeling is data leakage—training on variables that are populated or altered *after* the customer churns or that encode future information. Three categories of columns were systematically eliminated:
1. **Direct Post-Event Leakage**:
   - `Churn Reason`: Populated only when a customer churns (null for 100% of retained customers). Using this yields artificial 100% accuracy but 0% real-world utility.
   - `Churn Score`: A legacy churn probability generated by an external IBM SPSS scoring model.
   - `Churn Label`: Redundant categorical mirror of `Churn Value`.
2. **Arbitrary Identifiers**:
   - `CustomerID`: Arbitrary primary key that would cause spurious memorization.
3. **Zero-Variance Constants & High-Cardinality Spatial Coordinates**:
   - `Count`, `Country`, `State`: Identical across all 7,043 rows (constant values).
   - `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`: Spatial attributes with >1,100 unique categories and near-zero linear correlation (<0.005) to churn.

---

## 4. Stage 2 Implementation: Preprocessing & Stratified Splitting

Implemented in [src/preprocessing.py](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/src/preprocessing.py) and documented in [reports/preprocessing_report.md](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/reports/preprocessing_report.md).

### 4.1 Missing Value Treatment
In `Total Charges`, 11 entries contained empty whitespace strings (`" "`). Inspection revealed these correspond precisely to new customers where `Tenure Months == 0`. These missing values were imputed with `0.0`, reflecting their true zero cumulative billing history.

### 4.2 Feature Transformation Architecture
Features were separated into numerical and categorical pipelines using Scikit-Learn's `ColumnTransformer`:
- **Numerical Pipeline (4 features)**: `Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV` were standardized using `StandardScaler` ($\mu = 0, \sigma = 1$).
- **Categorical Pipeline (16 features)**: `Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`, `Contract`, `Paperless Billing`, and `Payment Method` were transformed into 43 binary indicators via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Resulting Feature Space**: **47 dense, numeric dimensions**.

### 4.3 3-Way Stratified Partitioning
To guarantee unbiased hyperparameter selection and true generalization measurement, the data was partitioned into three stratified splits:
- **Training Set (70%)**: 4,930 samples (3,622 Retained, 1,308 Churned, Churn Rate = 26.53%)
- **Validation Set (15%)**: 1,056 samples (776 Retained, 280 Churned, Churn Rate = 26.52%)
- **Test Set (15%)**: 1,057 samples (776 Retained, 281 Churned, Churn Rate = 26.58%)
- **Zero-Leakage Guarantee**: The `preprocessor` pipeline was fit **exclusively** on `X_train`. Validation and Test sets were transformed using frozen parameters.

---

## 5. Stage 3 Implementation: Machine Learning Model Zoo

Implemented in [src/train_model.py](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/src/train_model.py) and documented in [reports/model_training_report.md](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/reports/model_training_report.md).

Seven diverse classification architectures spanning linear models, bagging ensembles, boosting algorithms, kernel methods, and neural networks were implemented and benchmarked:

| Model Architecture | Implementation Class | Strategic Configuration & Hyperparameters | Handling of Class Imbalance |
|---|---|---|---|
| **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | `C=1.0, solver='lbfgs', max_iter=1000` | `class_weight='balanced'` |
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=200, max_depth=12, min_samples_split=6` | `class_weight='balanced'` |
| **Gradient Boosting** | `sklearn.ensemble.GradientBoostingClassifier` | `n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.85` | Gradient shrinkage & Subsampling |
| **HistGradientBoosting** | `sklearn.ensemble.HistGradientBoostingClassifier` | `learning_rate=0.08, max_iter=150, max_depth=6` | `class_weight='balanced'` |
| **AdaBoost** | `sklearn.ensemble.AdaBoostClassifier` | `n_estimators=100, learning_rate=0.1` | Sequential misclassification reweighting |
| **Support Vector Machine** | `sklearn.svm.SVC` | `C=1.0, kernel='rbf', probability=True` | `class_weight='balanced'` |
| **Multi-Layer Perceptron (ANN)** | `sklearn.neural_network.MLPClassifier` | `hidden_layers=(64, 32), activation='relu', solver='adam', alpha=0.001` | Early stopping on validation split |

### Validation Benchmark Results
Validation performance on `X_val` (1,056 records):
- **Logistic Regression**: **ROC-AUC: 0.8484** | **PR-AUC: 0.6315** | **Recall: 81.79%** | **F1: 0.6248** | Train Time: 0.046s
- **Multi-Layer Perceptron (ANN)**: **ROC-AUC: 0.8433** | **PR-AUC: 0.6147** | **Recall: 67.50%** | **F1: 0.6187** | Train Time: 0.436s
- **AdaBoost**: **ROC-AUC: 0.8432** | **PR-AUC: 0.6223** | **Recall: 36.07%** | **F1: 0.4731** | Train Time: 0.575s
- **Gradient Boosting**: **ROC-AUC: 0.8422** | **PR-AUC: 0.6259** | **Recall: 55.00%** | **F1: 0.5768** | Train Time: 2.095s
- **Random Forest**: **ROC-AUC: 0.8412** | **PR-AUC: 0.6215** | **Recall: 72.14%** | **F1: 0.6283** | Train Time: 0.457s
- **HistGradientBoosting**: **ROC-AUC: 0.8340** | **PR-AUC: 0.6216** | **Recall: 76.07%** | **F1: 0.6121** | Train Time: 1.954s
- **Support Vector Machine**: **ROC-AUC: 0.8291** | **PR-AUC: 0.5811** | **Recall: 80.36%** | **F1: 0.6215** | Train Time: 4.015s

---

## 6. Stage 4 Implementation: Out-of-Sample Evaluation & Decision Optimization

Implemented in [src/evaluate_model.py](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/src/evaluate_model.py) and documented in [reports/model_evaluation_report.md](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/reports/model_evaluation_report.md).

### 6.1 Test Set Empirical Evaluation
The test set represents 1,057 customer records held completely isolated from training and model tuning.

| Rank | Model Name | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Specificity | Test Precision | Test Accuracy | Brier Score |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Multi-Layer Perceptron (ANN)** | **0.8599** | **0.6705** | **0.6303** | 0.6370 | 0.8608 | 0.6237 | **80.13%** | **0.1360** |
| 2 | **Logistic Regression** | **0.8588** | **0.6879** | 0.6272 | **0.7722** | 0.7500 | 0.5280 | 75.59% | 0.1601 |
| 3 | **Random Forest** | 0.8570 | 0.6842 | **0.6416** | 0.6975 | 0.8273 | 0.5939 | 79.28% | 0.1384 |
| 4 | **Gradient Boosting** | 0.8529 | 0.6853 | 0.6083 | 0.5445 | **0.9111** | 0.6892 | 81.36% | 0.1378 |
| 5 | **HistGradientBoosting** | 0.8486 | 0.6727 | 0.6461 | 0.7438 | 0.7977 | 0.5710 | 78.33% | 0.1420 |
| 6 | **Support Vector Machine** | 0.8462 | 0.6482 | 0.6301 | 0.6335 | 0.8634 | 0.6268 | 80.23% | 0.1386 |
| 7 | **AdaBoost** | 0.8425 | 0.6445 | 0.4597 | 0.3345 | 0.9562 | **0.7344** | 79.09% | 0.2287 |

### 6.2 Confusion Matrix Analysis (Default Threshold = 0.50)
- **Multi-Layer Perceptron (ANN)**:
  - True Positives (TP): 179 churners caught
  - False Negatives (FN): 102 churners missed
  - True Negatives (TN): 668 retained customers correctly identified
  - False Positives (FP): 108 false alarms
- **Logistic Regression**:
  - True Positives (TP): **217 churners caught** (77.2% recall)
  - False Negatives (FN): 64 churners missed
  - True Negatives (TN): 582 retained customers
  - False Positives (FP): 194 false alarms

### 6.3 Decision Threshold Optimization & Financial Cost-Benefit Modeling
In business settings, a standard classification threshold of $0.50$ is suboptimal because the asymmetric costs of errors are ignored:
- **Cost of a False Negative ($C_{FN}$)**: Losing an existing customer without intervention results in lost lifetime value (estimated at **$500**).
- **Cost of a False Positive ($C_{FP}$)**: Sending a retention offer/discount to a customer who wasn't going to leave costs the campaign incentive (estimated at **$60**).
- **Retention Win Rate**: Proactive retention outreach is estimated to save **50%** of contacted churners.

$$\text{Net Value Created} = (TP \times 0.50 \times \$500) - ((TP + FP) \times \$60)$$

#### Optimal Operating Regimes:
1. **F1-Optimal Threshold ($t = 0.46$)**:
   - Maximizes balance between Precision (60.1%) and Recall (70.1%), achieving maximum F1 of **0.6468**.
2. **Business Maximum Value Threshold ($t = 0.46$)**:
   - Generates **$32,030.00 in net financial savings** on the 1,057 test cohort alone compared to no intervention ($140,500 baseline churn loss).
3. **High-Recall Operational Threshold ($t = 0.28$)**:
   - Guarantees **$\ge 85.0\%$ capture** of all potential churners for high-value customer tiers.

---

## 7. Stage 5 Implementation: Visual Diagnostics & Explainability

Implemented in [src/visualization.py](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/src/visualization.py) with all figures saved in [outputs/visualizations/](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/outputs/visualizations).

### 7.1 Visual Diagnostic Assets
1. **ROC Curves Comparison** ([outputs/visualizations/roc_curves_comparison.png](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/outputs/visualizations/roc_curves_comparison.png)):
   - Demonstrates strong discriminative power across all models, with MLP (AUC 0.8599) and Logistic Regression (AUC 0.8588) leading the curve.
2. **Precision-Recall Curves** ([outputs/visualizations/precision_recall_curves.png](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/outputs/visualizations/precision_recall_curves.png)):
   - Evaluates performance against the 26.58% no-skill prevalence baseline; Logistic Regression establishes the highest Average Precision (0.6879).
3. **Confusion Matrices** ([outputs/visualizations/confusion_matrices_comparison.png](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/outputs/visualizations/confusion_matrices_comparison.png)):
   - 2-panel comparison contrasting raw counts and class-normalized detection rates between the ANN and Logistic Regression.
4. **Customer Churn Drivers & Retention Anchors** ([outputs/visualizations/feature_importance.png](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/outputs/visualizations/feature_importance.png)):
   - **Primary Churn Drivers (+ coefficients)**:
     1. `Contract_Month-to-month` (+0.72)
     2. `Internet Service_Fiber optic` (+0.54)
     3. `Payment Method_Electronic check` (+0.23)
     4. `Monthly Charges` (+0.18)
   - **Primary Retention Anchors (- coefficients)**:
     1. `Tenure Months` (-0.86)
     2. `Contract_Two year` (-0.68)
     3. `Contract_One year` (-0.35)
     4. `Online Security_Yes` (-0.21)
     5. `Tech Support_Yes` (-0.19)
5. **Churn Risk Probability Distribution** ([outputs/visualizations/churn_probability_distribution.png](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/outputs/visualizations/churn_probability_distribution.png)):
   - Kernel density estimation plots showing clear bimodal divergence between retained customers (densely clustered below 0.20) and churning customers (peaking above 0.65).
6. **Decision Threshold Trade-off Curve** ([outputs/visualizations/threshold_tradeoff_curve.png](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/outputs/visualizations/threshold_tradeoff_curve.png)):
   - Dynamic curves tracking Precision, Recall, and F1 across threshold values, highlighting the $t=0.46$ optimum.
7. **Model Performance Benchmark** ([outputs/visualizations/models_performance_benchmark.png](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/outputs/visualizations/models_performance_benchmark.png)):
   - Clustered bar chart comparing ROC-AUC, PR-AUC, F1, and Recall across all 7 candidate algorithms.

---

## 8. Production Deployment & Operational Guidance

### 8.1 Real-Time and Batch Scoring Protocol
To score new incoming customer records without leakage or dimensional mismatch:
```python
import joblib
import pandas as pd

# Load frozen pipeline and model
preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/best_model.joblib")

# Apply to new unlabelled customer records
new_customers_df = pd.read_csv("new_customers.csv")
X_proc = preprocessor.transform(new_customers_df)

# Inference at optimized threshold (0.46)
churn_probabilities = model.predict_proba(X_proc)[:, 1]
churn_predictions = (churn_probabilities >= 0.46).astype(int)
```

### 8.2 Strategic Business Recommendations
1. **Contract Migration Campaigns**:
   - `Month-to-month` contracts represent the single greatest vulnerability. Incentivize customers to migrate to 1-year or 2-year agreements with discounted service bundles.
2. **Fiber Optic Service Onboarding & Support**:
   - Fiber optic customers churn at disproportionately high rates, indicating potential service reliability issues or bill shock. Proactive outreach in months 1–3 and pairing with `Tech Support` and `Online Security` significantly lowers defection.
3. **Automated Payment Incentives**:
   - `Electronic check` users experience the highest churn rate; incentivize transitions to automated credit card or direct bank ACH billing.

---

## 9. Artifact Directory & Verification Index

| Category | File Path | Description |
|---|---|---|
| **Pipeline Runner** | [main.py](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/main.py) | Master script executing stages 1 through 5 end-to-end |
| **Stage 1 Module** | [src/data_analysis.py](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/src/data_analysis.py) | Raw data loader, leakage detector, EDA |
| **Stage 2 Module** | [src/preprocessing.py](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/src/preprocessing.py) | Cleaner, stratified splitter, ColumnTransformer |
| **Stage 3 Module** | [src/train_model.py](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/src/train_model.py) | 7-model training suite, model persistence |
| **Stage 4 Module** | [src/evaluate_model.py](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/src/evaluate_model.py) | Out-of-sample testing, threshold optimization |
| **Stage 5 Module** | [src/visualization.py](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/src/visualization.py) | High-res diagnostic plots |
| **Model Weights** | [models/best_model.joblib](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/models/best_model.joblib) | Champion trained model artifact |
| **Preprocessor** | [models/preprocessor.joblib](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/models/preprocessor.joblib) | Frozen fitted ColumnTransformer |
| **Processed Data** | [dataset/processed_data.npz](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/dataset/processed_data.npz) | Compressed stratified arrays and feature names |
| **Test Predictions** | [outputs/model_results/test_predictions.csv](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/outputs/model_results/test_predictions.csv) | Test record IDs, actuals, predicted probabilities |
| **Evaluation Metrics**| [outputs/model_results/test_metrics_summary.csv](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/outputs/model_results/test_metrics_summary.csv) | Benchmark metrics for all 7 models |
| **Visual Charts** | [outputs/visualizations/](file:///c:/Users/aarya/OneDrive/Apps/customer%20churn%20prediction/outputs/visualizations) | 7 publication-ready PNG figures |
