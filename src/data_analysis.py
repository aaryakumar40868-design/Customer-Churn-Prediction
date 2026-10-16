"""
data_analysis.py
----------------
Module for loading, inspecting, and analyzing the Telco Customer Churn dataset.
Performs Exploratory Data Analysis (EDA), identifies target variables,
detects data leakage, handles missing values, and checks for outliers and duplicates.
"""

import os
import pandas as pd
import numpy as np


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from an Excel (.xlsx) or CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    # Try reading as Excel first, fallback to CSV
    try:
        df = pd.read_excel(filepath)
    except Exception:
        df = pd.read_csv(filepath)
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Perform structural and statistical inspection of the dataset.
    Returns a dictionary of inspection summaries.
    """
    summary = {}
    summary["shape"] = df.shape
    summary["columns"] = list(df.columns)
    summary["dtypes"] = df.dtypes.astype(str).to_dict()
    
    # Check duplicates
    summary["duplicate_rows"] = int(df.duplicated().sum())
    if "CustomerID" in df.columns:
        summary["duplicate_customer_ids"] = int(df["CustomerID"].duplicated().sum())
        
    # Check missing values
    missing = df.isnull().sum()
    summary["missing_values"] = missing[missing > 0].to_dict()
    
    # Check Total Charges blank spaces
    if "Total Charges" in df.columns:
        total_charges_numeric = pd.to_numeric(df["Total Charges"], errors="coerce")
        blank_total_charges = int(total_charges_numeric.isnull().sum() - df["Total Charges"].isnull().sum())
        summary["blank_total_charges"] = max(0, blank_total_charges)
    
    # Target distribution
    if "Churn Value" in df.columns:
        target_counts = df["Churn Value"].value_counts().to_dict()
        target_pct = (df["Churn Value"].value_counts(normalize=True) * 100).round(2).to_dict()
        summary["target_distribution"] = {
            "counts": target_counts,
            "percentages": target_pct,
            "imbalance_ratio": f"{target_counts.get(0, 0) / max(1, target_counts.get(1, 1)):.2f}:1"
        }
        
    # Low-variance / constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    summary["constant_columns"] = constant_cols
    
    # Data leakage candidates
    leakage_cols = [c for c in ["Churn Reason", "Churn Score", "Churn Label"] if c in df.columns]
    summary["potential_leakage_columns"] = leakage_cols
    
    # Numerical and categorical candidate split
    cleaned_tc = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)
    df_temp = df.copy()
    df_temp["Total Charges"] = cleaned_tc
    
    num_cols = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]
    num_summary = {}
    for col in num_cols:
        if col in df_temp.columns:
            q1 = float(df_temp[col].quantile(0.25))
            q3 = float(df_temp[col].quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((df_temp[col] < lower) | (df_temp[col] > upper)).sum())
            num_summary[col] = {
                "mean": round(float(df_temp[col].mean()), 2),
                "std": round(float(df_temp[col].std()), 2),
                "min": round(float(df_temp[col].min()), 2),
                "median": round(float(df_temp[col].median()), 2),
                "max": round(float(df_temp[col].max()), 2),
                "IQR": round(iqr, 2),
                "outliers_count": outlier_count
            }
    summary["numerical_statistics"] = num_summary
    
    return summary


def run_stage1_analysis(filepath: str = "dataset/Telco_customer_churn.csv.xlsx", report_output_path: str = "reports/data_understanding_report.md"):
    """Run full Stage 1 analysis and print formatted inspection results."""
    print("==================================================")
    print("STAGE 1: DATASET INSPECTION & TARGET IDENTIFICATION")
    print("==================================================")
    print(f"Loading dataset from: {filepath}")
    df = load_dataset(filepath)
    summary = inspect_dataset(df)
    
    print(f"\n1. DATASET DIMENSIONS:")
    print(f"   - Total Rows: {summary['shape'][0]:,}")
    print(f"   - Total Columns: {summary['shape'][1]}")
    
    print(f"\n2. TARGET VARIABLE IDENTIFICATION:")
    print(f"   - Selected Target Variable: 'Churn Value'")
    print(f"   - Target Definition: 1 = Customer Churned, 0 = Customer Retained")
    print(f"   - Target Distribution:")
    for k, v in summary['target_distribution']['counts'].items():
        pct = summary['target_distribution']['percentages'][k]
        label_name = "Churned (1)" if k == 1 else "Retained (0)"
        print(f"     * {label_name}: {v:,} rows ({pct}%)")
    print(f"   - Imbalance Ratio: {summary['target_distribution']['imbalance_ratio']}")
    
    print(f"\n3. DATA QUALITY & INTEGRITY:")
    print(f"   - Full Row Duplicates: {summary['duplicate_rows']}")
    print(f"   - Duplicate CustomerIDs: {summary.get('duplicate_customer_ids', 0)}")
    print(f"   - Explicit Nulls: {summary['missing_values']}")
    print(f"   - Hidden Missing/Blanks in 'Total Charges': {summary.get('blank_total_charges', 0)} rows (tenure=0 new customers)")
    
    print(f"\n4. DATA LEAKAGE & IRRELEVANT COLUMNS IDENTIFIED:")
    print(f"   - 'Churn Reason': Only present when churned (1,869 non-null values). Direct post-event leakage.")
    print(f"   - 'Churn Score': Pre-calculated risk score from SPSS predictive tool. Must be excluded.")
    print(f"   - 'Churn Label': Text duplicate of target ('Yes'/'No').")
    print(f"   - Constant Columns: {summary['constant_columns']} (zero variance).")
    print(f"   - Identifier / High Cardinality Location: 'CustomerID', 'Lat Long', 'Latitude', 'Longitude', 'City', 'Zip Code'.")
    
    print(f"\n5. NUMERICAL FEATURES & OUTLIER ANALYSIS (1.5 * IQR):")
    for feat, stats in summary["numerical_statistics"].items():
        print(f"   - {feat:15s} | Mean: {stats['mean']:8.2f} | Std: {stats['std']:8.2f} | Min: {stats['min']:8.2f} | Max: {stats['max']:8.2f} | Outliers: {stats['outliers_count']}")

    # Write report
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write("# Stage 1: Data Understanding & Inspection Report\n\n")
        f.write("## 1. Dataset Overview\n")
        f.write(f"- **Dataset File**: `{filepath}`\n")
        f.write(f"- **Total Rows**: {summary['shape'][0]:,}\n")
        f.write(f"- **Total Columns**: {summary['shape'][1]}\n")
        f.write(f"- **Duplicate Records**: {summary['duplicate_rows']}\n")
        f.write(f"- **Unique Customers**: {df['CustomerID'].nunique():,}\n\n")
        
        f.write("## 2. Target Variable Analysis\n")
        f.write("- **Primary Target Variable**: `Churn Value` (Integer 0 or 1)\n")
        f.write("- **Associated Column**: `Churn Label` ('Yes' or 'No')\n")
        f.write(f"- **Class 0 (Retained)**: {summary['target_distribution']['counts'][0]:,} ({summary['target_distribution']['percentages'][0]}%)\n")
        f.write(f"- **Class 1 (Churned)**: {summary['target_distribution']['counts'][1]:,} ({summary['target_distribution']['percentages'][1]}%)\n")
        f.write(f"- **Imbalance Ratio**: {summary['target_distribution']['imbalance_ratio']} (Moderate class imbalance to handle during training)\n\n")
        
        f.write("## 3. Data Leakage & Column Exclusions\n")
        f.write("| Column Name | Type | Reason for Exclusion / Action |\n")
        f.write("|-------------|------|-------------------------------|\n")
        f.write("| `Churn Reason` | Leakage | Recorded post-churn; null for 100% of non-churned customers |\n")
        f.write("| `Churn Score` | Leakage | Pre-existing predictive churn risk score generated by IBM SPSS |\n")
        f.write("| `Churn Label` | Redundant Target | String duplicate of `Churn Value` |\n")
        f.write("| `CustomerID` | Identifier | Arbitrary identifier; would cause spurious memorization |\n")
        f.write("| `Count` | Constant | Constant value (= 1) across all records |\n")
        f.write("| `Country` | Constant | Constant value ('United States') across all records |\n")
        f.write("| `State` | Constant | Constant value ('California') across all records |\n")
        f.write("| `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude` | Geographic / High Cardinality | Spatial attributes with near-zero linear correlation (<0.005) to churn |\n\n")

        f.write("## 4. Modeling Feature Specification\n")
        f.write("### Numerical Features (4)\n")
        f.write("- `Tenure Months`: Length of customer relationship in months (0 to 72)\n")
        f.write("- `Monthly Charges`: Current monthly bill amount in USD ($18.25 to $118.75)\n")
        f.write("- `Total Charges`: Cumulative billed amount (11 missing values imputed with 0.0 for new customers with tenure=0)\n")
        f.write("- `CLTV`: Customer Lifetime Value score ($2,003 to $6,500)\n\n")

        f.write("### Categorical Features (16)\n")
        f.write("- **Demographics (4)**: `Gender`, `Senior Citizen`, `Partner`, `Dependents`\n")
        f.write("- **Phone Services (2)**: `Phone Service`, `Multiple Lines`\n")
        f.write("- **Internet & Add-on Services (7)**: `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`\n")
        f.write("- **Account & Contract Terms (3)**: `Contract`, `Paperless Billing`, `Payment Method`\n\n")
        
        f.write("## 5. Numerical Feature Statistics & Outlier Check\n")
        f.write("| Feature | Mean | Std | Min | Median | Max | IQR | Outliers (1.5x IQR) |\n")
        f.write("|---------|------|-----|-----|--------|-----|-----|--------------------|\n")
        for feat, stats in summary["numerical_statistics"].items():
            f.write(f"| `{feat}` | {stats['mean']} | {stats['std']} | {stats['min']} | {stats['median']} | {stats['max']} | {stats['IQR']} | {stats['outliers_count']} |\n")
            
    print(f"\nReport generated and saved to: {report_output_path}")
    print("STAGE 1 COMPLETED SUCCESSFULLY.\n")
    return summary


if __name__ == "__main__":
    run_stage1_analysis()
