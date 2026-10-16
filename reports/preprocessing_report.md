# Stage 2: Data Preprocessing & Stratified Splitting Report

## 1. Preprocessing Summary
- **Raw Dataset Source**: `dataset/Telco_customer_churn.csv.xlsx` (Preserved intact)
- **Missing Value Imputation**: Handled 11 blank entries in `Total Charges` by imputing `0.0` (tenure=0 new joiners).
- **Leakage Columns Removed**: `Churn Reason`, `Churn Score`, `Churn Label`
- **Non-Predictive / Constant Columns Removed**: `CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`

## 2. Dataset Splitting & Stratification Verification
| Partition | Rows | % of Total | Retained (0) | Churned (1) | Churn Rate (%) |
|---|---|---|---|---|---|
| Training Set (70%) | 4,930 | 70.0% | 3,622 | 1,308 | 26.53% |
| Validation Set (15%) | 1,056 | 15.0% | 776 | 280 | 26.52% |
| Testing Set (15%) | 1,057 | 15.0% | 776 | 281 | 26.58% |
| **Full Dataset** | 7,043 | 100.0% | 5,174 | 1,869 | 26.54% |

## 3. Transformation & Feature Encoding Details
- **Numerical Scaling**: `StandardScaler` applied to `Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV`.
- **Categorical Encoding**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to 16 categorical features.
- **Total Resulting Features**: 47 features.
- **Leakage Prevention**: Strictly called `fit` on `X_train` only. Validation and Test sets were evaluated using frozen transformer weights.

## 4. Persisted Artifacts
- `preprocessor`: [models\preprocessor.joblib](file:///models/preprocessor.joblib) (6.6 KB)
- `train_csv`: [dataset\train_clean.csv](file:///dataset/train_clean.csv) (640.1 KB)
- `val_csv`: [dataset\val_clean.csv](file:///dataset/val_clean.csv) (136.2 KB)
- `test_csv`: [dataset\test_clean.csv](file:///dataset/test_clean.csv) (137.8 KB)
- `processed_npz`: [dataset\processed_data.npz](file:///dataset/processed_data.npz) (268.7 KB)
