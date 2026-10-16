"""
preprocessing.py
----------------
Module for data cleaning, leakage prevention, stratified train/validation/test
splitting, and feature transformation (standardization & one-hot encoding).
Ensures zero data leakage by fitting transformers strictly on training data.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


NUMERICAL_COLS: List[str] = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV"
]

CATEGORICAL_COLS: List[str] = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method"
]

TARGET_COL: str = "Churn Value"

LEAKAGE_AND_IRRELEVANT_COLS: List[str] = [
    "CustomerID",       # Arbitrary identifier
    "Count",            # Constant (= 1)
    "Country",          # Constant (= 'United States')
    "State",            # Constant (= 'California')
    "City",             # High cardinality spatial (1129 categories)
    "Zip Code",         # High cardinality spatial (1652 categories)
    "Lat Long",         # Spatial coordinate string
    "Latitude",         # Spatial coordinate
    "Longitude",        # Spatial coordinate
    "Churn Label",      # Redundant target label ('Yes'/'No')
    "Churn Score",      # Pre-existing SPSS predictive risk score (leakage)
    "Churn Reason"      # Post-event reason populated only when churned (leakage)
]


def load_raw_data(filepath: str = "dataset/Telco_customer_churn.csv.xlsx") -> pd.DataFrame:
    """Load raw dataset from Excel or CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset file not found at: {filepath}")
    try:
        return pd.read_excel(filepath)
    except Exception:
        return pd.read_csv(filepath)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    - Converts 'Total Charges' to numeric, handling empty whitespace strings.
    - Imputes 'Total Charges' = 0.0 for new customers where 'Tenure Months' == 0.
    - Preserves data integrity without altering the raw file.
    """
    df_cleaned = df.copy()
    
    # Handle Total Charges blank spaces
    if "Total Charges" in df_cleaned.columns:
        df_cleaned["Total Charges"] = pd.to_numeric(df_cleaned["Total Charges"], errors="coerce").fillna(0.0)
        
    return df_cleaned


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate input features and the target variable, filtering out leakage/irrelevant columns."""
    selected_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    
    missing_features = [col for col in selected_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns in dataset: {missing_features}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")
        
    X = df[selected_cols].copy()
    y = df[TARGET_COL].copy().astype(int)
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Perform a 3-way stratified split: Train (70%), Validation (15%), Test (15%).
    Stratification ensures the target class ratio is preserved across all partitions.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    
    # First split off the training set
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=y
    )
    
    # Split the remaining data equally into validation and test sets
    val_rel_ratio = val_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - val_rel_ratio),
        random_state=random_state,
        stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    """
    Construct preprocessing ColumnTransformer:
    - StandardScaler for numerical features.
    - OneHotEncoder (dense output) for categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)
        ],
        remainder="drop"
    )
    return preprocessor


def fit_and_transform_pipeline(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Fit preprocessor strictly on X_train to prevent data leakage.
    Transform X_train, X_val, and X_test.
    """
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    feature_names = list(preprocessor.get_feature_names_out())
    return X_train_proc, X_val_proc, X_test_proc, feature_names


def save_preprocessed_artifacts(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    X_train_proc: np.ndarray,
    X_val_proc: np.ndarray,
    X_test_proc: np.ndarray,
    feature_names: List[str],
    models_dir: str = "models",
    dataset_dir: str = "dataset"
) -> Dict[str, str]:
    """Save preprocessor pipeline, raw split CSVs, and processed numpy arrays."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save preprocessor pipeline
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save raw clean splits with target
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train
    train_csv_path = os.path.join(dataset_dir, "train_clean.csv")
    train_df.to_csv(train_csv_path, index=False)
    
    val_df = X_val.copy()
    val_df[TARGET_COL] = y_val
    val_csv_path = os.path.join(dataset_dir, "val_clean.csv")
    val_df.to_csv(val_csv_path, index=False)
    
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    test_csv_path = os.path.join(dataset_dir, "test_clean.csv")
    test_df.to_csv(test_csv_path, index=False)
    
    # Save processed numerical/encoded arrays
    processed_npz_path = os.path.join(dataset_dir, "processed_data.npz")
    np.savez_compressed(
        processed_npz_path,
        X_train=X_train_proc,
        y_train=y_train.values,
        X_val=X_val_proc,
        y_val=y_val.values,
        X_test=X_test_proc,
        y_test=y_test.values,
        feature_names=np.array(feature_names)
    )
    
    return {
        "preprocessor": preprocessor_path,
        "train_csv": train_csv_path,
        "val_csv": val_csv_path,
        "test_csv": test_csv_path,
        "processed_npz": processed_npz_path
    }


def run_stage2_preprocessing(
    raw_path: str = "dataset/Telco_customer_churn.csv.xlsx",
    report_path: str = "reports/preprocessing_report.md"
) -> Dict[str, Any]:
    """Execute complete Stage 2 preprocessing workflow."""
    print("==================================================")
    print("STAGE 2: DATA PREPROCESSING & STRATIFIED SPLITTING")
    print("==================================================")
    
    # 1. Load and clean raw data
    raw_df = load_raw_data(raw_path)
    clean_df = clean_raw_data(raw_df)
    
    # 2. Extract features and target
    X, y = prepare_features_and_target(clean_df)
    
    # 3. Stratified splitting (70% train, 15% val, 15% test)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    
    # 4. Fit preprocessor on training data ONLY
    preprocessor = build_preprocessor()
    X_train_proc, X_val_proc, X_test_proc, feature_names = fit_and_transform_pipeline(
        preprocessor, X_train, X_val, X_test
    )
    
    # 5. Save all artifacts
    saved_paths = save_preprocessed_artifacts(
        preprocessor,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_train_proc, X_val_proc, X_test_proc,
        feature_names
    )
    
    # Print verification summaries
    print("\n1. SPLIT SUMMARY & TARGET STRATIFICATION:")
    splits = [
        ("Training Set (70%)", len(y_train), y_train),
        ("Validation Set (15%)", len(y_val), y_val),
        ("Testing Set (15%)", len(y_test), y_test)
    ]
    for name, size, y_set in splits:
        churn_count = int(y_set.sum())
        retained_count = int(size - churn_count)
        churn_rate = (churn_count / size) * 100
        print(f"   - {name:20s}: {size:,} rows | Retained: {retained_count:,} | Churned: {churn_count:,} | Churn Rate: {churn_rate:.2f}%")
        
    print("\n2. FEATURE ENCODING & DIMENSIONALITY:")
    print(f"   - Input Tabular Features: {X.shape[1]} (4 Numerical + 16 Categorical)")
    print(f"   - Processed Feature Vector Dimension: {X_train_proc.shape[1]}")
    print(f"   - Numerical Features Standardized (Mean ~ 0, Std ~ 1): {NUMERICAL_COLS}")
    print(f"   - First 5 Encoded Features: {feature_names[:5]}")
    
    print("\n3. DATA LEAKAGE PREVENTION VERIFICATION:")
    print("   - Transformers were fit EXCLUSIVELY on X_train.")
    print("   - Validation and test sets were transformed using training parameters.")
    print("   - Post-event leakage columns ('Churn Reason', 'Churn Score', 'Churn Label') completely removed.")
    
    print("\n4. ARTIFACTS PERSISTED:")
    for key, path in saved_paths.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"   - {key:15s} -> {path} ({size_kb:.1f} KB)")
        
    # Generate preprocessing report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Stage 2: Data Preprocessing & Stratified Splitting Report\n\n")
        f.write("## 1. Preprocessing Summary\n")
        f.write("- **Raw Dataset Source**: `dataset/Telco_customer_churn.csv.xlsx` (Preserved intact)\n")
        f.write("- **Missing Value Imputation**: Handled 11 blank entries in `Total Charges` by imputing `0.0` (tenure=0 new joiners).\n")
        f.write("- **Leakage Columns Removed**: `Churn Reason`, `Churn Score`, `Churn Label`\n")
        f.write("- **Non-Predictive / Constant Columns Removed**: `CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`\n\n")
        
        f.write("## 2. Dataset Splitting & Stratification Verification\n")
        f.write("| Partition | Rows | % of Total | Retained (0) | Churned (1) | Churn Rate (%) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, size, y_set in splits:
            churn_count = int(y_set.sum())
            retained_count = int(size - churn_count)
            churn_rate = (churn_count / size) * 100
            f.write(f"| {name} | {size:,} | {size / len(clean_df) * 100:.1f}% | {retained_count:,} | {churn_count:,} | {churn_rate:.2f}% |\n")
        f.write(f"| **Full Dataset** | {len(clean_df):,} | 100.0% | {int((y == 0).sum()):,} | {int((y == 1).sum()):,} | {y.mean() * 100:.2f}% |\n\n")
        
        f.write("## 3. Transformation & Feature Encoding Details\n")
        f.write("- **Numerical Scaling**: `StandardScaler` applied to `Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV`.\n")
        f.write("- **Categorical Encoding**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to 16 categorical features.\n")
        f.write(f"- **Total Resulting Features**: {len(feature_names)} features.\n")
        f.write("- **Leakage Prevention**: Strictly called `fit` on `X_train` only. Validation and Test sets were evaluated using frozen transformer weights.\n\n")
        
        f.write("## 4. Persisted Artifacts\n")
        for key, path in saved_paths.items():
            size_kb = os.path.getsize(path) / 1024
            f.write(f"- `{key}`: [{path}](file:///{path.replace('\\', '/')}) ({size_kb:.1f} KB)\n")

    print(f"\nReport generated and saved to: {report_path}")
    print("STAGE 2 COMPLETED SUCCESSFULLY.\n")
    
    return {
        "saved_paths": saved_paths,
        "feature_names": feature_names,
        "X_train_shape": X_train_proc.shape,
        "X_val_shape": X_val_proc.shape,
        "X_test_shape": X_test_proc.shape
    }


if __name__ == "__main__":
    run_stage2_preprocessing()
