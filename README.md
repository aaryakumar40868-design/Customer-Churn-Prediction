# Telco Customer Churn Prediction System

An end-to-end, production-grade Machine Learning pipeline for predicting customer churn in subscription telecommunications services, featuring strict data leakage prevention, stratified 3-way dataset splitting, a benchmarked 7-model zoo, financial decision threshold optimization, and comprehensive visual diagnostics.

---

## 🚀 Quickstart & Pipeline Execution

### 1. Environment Setup
Install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execute Entire Pipeline End-to-End
Run all 5 stages in sequence (~20 seconds total execution time):
```bash
python main.py
```

### 3. Or Execute Individual Stages Sequentially:
```bash
# Stage 1: Dataset Inspection & Data Leakage Analysis
python src/data_analysis.py

# Stage 2: Data Preprocessing, Imputation & Stratified Splitting
python src/preprocessing.py

# Stage 3: Train & Benchmark 7 Machine Learning Models
python src/train_model.py

# Stage 4: Out-of-Sample Evaluation & Decision Threshold Tuning
python src/evaluate_model.py

# Stage 5: Visual Diagnostics & Explainability Plotting
python src/visualization.py
```

---

## 📊 Pipeline Architecture & Reports

The project is structured into 5 cohesive stages, each generating dedicated reports and artifacts:

| Stage | Script | Output Report | Core Responsibility |
|---|---|---|---|
| **Stage 1: Data Understanding** | [`src/data_analysis.py`](src/data_analysis.py) | [`reports/data_understanding_report.md`](reports/data_understanding_report.md) | Schema inspection, null checks, post-event data leakage removal (`Churn Reason`, `Churn Score`, `Churn Label`). |
| **Stage 2: Preprocessing & Splitting** | [`src/preprocessing.py`](src/preprocessing.py) | [`reports/preprocessing_report.md`](reports/preprocessing_report.md) | Stratified 70/15/15 split, numerical standardization (`StandardScaler`), one-hot encoding (`OneHotEncoder`). Transformers fitted strictly on `X_train`. |
| **Stage 3: Model Training** | [`src/train_model.py`](src/train_model.py) | [`reports/model_training_report.md`](reports/model_training_report.md) | Trains 7 diverse models with balanced class weights, compares validation performance, and persists `.joblib` model artifacts. |
| **Stage 4: Evaluation & Optimization** | [`src/evaluate_model.py`](src/evaluate_model.py) | [`reports/model_evaluation_report.md`](reports/model_evaluation_report.md) | Out-of-sample testing on 1,057 unseen test records, confusion matrix decomposition, and cost-benefit threshold optimization ($t^* = 0.46$). |
| **Stage 5: Visual Diagnostics** | [`src/visualization.py`](src/visualization.py) | [`outputs/visualizations/`](outputs/visualizations/) | Generates high-res ROC curves, PR curves, Confusion Matrices, Feature Importance, Probability Distributions, and Benchmark charts. |
| **Master Summary** | — | [`reports/implementation_report.md`](reports/implementation_report.md) | Comprehensive project implementation report synthesized end-to-end. |

---

## 🏆 Test Set Benchmarks (Held-Out 1,057 Customers)

| Rank | Model Architecture | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Specificity | Test Accuracy |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Multi-Layer Perceptron (ANN)** | **0.8599** | **0.6705** | **0.6303** | 0.6370 | 0.8608 | **80.13%** |
| 2 | **Logistic Regression** | **0.8588** | **0.6879** | 0.6272 | **0.7722** | 0.7500 | 75.59% |
| 3 | **Random Forest** | 0.8570 | 0.6842 | **0.6416** | 0.6975 | 0.8273 | 79.28% |
| 4 | **Gradient Boosting** | 0.8529 | 0.6853 | 0.6083 | 0.5445 | **0.9111** | 81.36% |
| 5 | **HistGradientBoosting** | 0.8486 | 0.6727 | 0.6461 | 0.7438 | 0.7977 | 78.33% |
| 6 | **Support Vector Machine (SVC)** | 0.8462 | 0.6482 | 0.6301 | 0.6335 | 0.8634 | 80.23% |
| 7 | **AdaBoost** | 0.8425 | 0.6445 | 0.4597 | 0.3345 | 0.9562 | 79.09% |

---

## 💡 Key Business Insights & Risk Factors

From model explainability and coefficient analysis:
- **Major Churn Catalysts**:
  1. `Month-to-month contracts` (+0.72 log-odds): Customers without long-term commitments churn at nearly 4x the rate of contracted customers.
  2. `Fiber optic internet` (+0.54 log-odds): Indicates potential service friction, bill shock, or uncompetitive pricing relative to competitor alternatives.
  3. `Electronic check payments` (+0.23 log-odds): Unautomated payment methods lead to higher friction and lapse rates.
- **Strongest Retention Anchors**:
  1. `Tenure Months` (-0.86 log-odds): Longevity is the single strongest buffer against churn.
  2. `Two-year & One-year contracts` (-0.68 and -0.35 log-odds): Multi-year agreements lock in customer loyalty.
  3. `Tech Support & Online Security add-ons`: Customers subscribed to technical assistance and security services have significantly lower defection rates.

---

## 📁 Repository Layout

```
customer churn prediction/
├── dataset/
│   ├── Telco_customer_churn.csv.xlsx   # Original raw dataset
│   ├── processed_data.npz              # Scaled & one-hot encoded arrays (train/val/test)
│   ├── train_clean.csv                 # 70% training split
│   ├── val_clean.csv                   # 15% validation split
│   └── test_clean.csv                  # 15% test split
├── models/
│   ├── preprocessor.joblib             # Fitted Scikit-Learn ColumnTransformer
│   ├── best_model.joblib               # Top-performing champion model
│   └── *.joblib                        # Individual serialized models
├── outputs/
│   ├── model_results/                  # Test & validation metrics (CSV/JSON), predictions
│   └── visualizations/                 # 7 high-resolution diagnostic charts (.png)
├── reports/
│   ├── data_understanding_report.md    # Stage 1 inspection & leakage report
│   ├── preprocessing_report.md         # Stage 2 preprocessing & splitting report
│   ├── model_training_report.md        # Stage 3 model zoo training report
│   ├── model_evaluation_report.md      # Stage 4 out-of-sample testing report
│   └── implementation_report.md        # Master comprehensive implementation report
├── src/
│   ├── data_analysis.py                # Stage 1 EDA & leakage audit
│   ├── preprocessing.py                # Stage 2 data preprocessing pipeline
│   ├── train_model.py                  # Stage 3 model training pipeline
│   ├── evaluate_model.py               # Stage 4 model evaluation pipeline
│   └── visualization.py                # Stage 5 diagnostics generator
├── main.py                             # Master pipeline orchestrator
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```
