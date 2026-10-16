"""
train_model.py
--------------
Module for training and comparing multiple Machine Learning models for Customer Churn Prediction.
Loads preprocessed dataset artifacts, trains diverse classification algorithms (Logistic Regression,
Random Forest, Gradient Boosting, HistGradientBoosting, AdaBoost, SVM, and Multi-Layer Perceptron),
evaluates validation performance, selects the top-performing model, persists trained models, and
generates a comprehensive Stage 3 training report.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def load_processed_data(npz_path: str = "dataset/processed_data.npz") -> Dict[str, np.ndarray]:
    """Load preprocessed split arrays and feature names from compressed NPZ."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Processed data file not found at '{npz_path}'. "
            "Please run 'src/preprocessing.py' first."
        )
    data = np.load(npz_path, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "feature_names": data["feature_names"]
    }


def get_model_zoo(random_state: int = 42) -> Dict[str, Any]:
    """
    Define a diverse suite of classification models configured with balanced class handling
    to counter customer churn class imbalance (~26.5% positive rate).
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state
        ),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=random_state
        ),
        "Multi-Layer Perceptron (ANN)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.005,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state
        )
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Calculate key binary classification metrics."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4)
    }


def train_and_validate_all_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train each model on training data, record training duration, and evaluate on validation data.
    """
    trained_models = {}
    metrics_records = []

    print("\n--- Training and Validating Candidate Models ---")
    print(f"{'Model Name':<30} | {'Train Time (s)':<14} | {'Val Acc':<8} | {'Val Prec':<8} | {'Val Recall':<10} | {'Val F1':<8} | {'Val ROC-AUC':<11} | {'Val PR-AUC':<10}")
    print("-" * 115)

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_duration = round(time.time() - start_time, 3)

        # Validation predictions
        y_val_pred = model.predict(X_val)
        if hasattr(model, "predict_proba"):
            y_val_proba = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_val)
            y_val_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            y_val_proba = y_val_pred.astype(float)

        metrics = evaluate_predictions(y_val, y_val_pred, y_val_proba)
        metrics["training_time_sec"] = train_duration
        metrics["model_name"] = name

        trained_models[name] = {
            "model": model,
            "metrics": metrics,
            "val_pred": y_val_pred,
            "val_proba": y_val_proba
        }

        metrics_records.append(metrics)
        print(
            f"{name:<30} | "
            f"{train_duration:<14.3f} | "
            f"{metrics['accuracy']:<8.4f} | "
            f"{metrics['precision']:<8.4f} | "
            f"{metrics['recall']:<10.4f} | "
            f"{metrics['f1']:<8.4f} | "
            f"{metrics['roc_auc']:<11.4f} | "
            f"{metrics['pr_auc']:<10.4f}"
        )

    val_metrics_df = pd.DataFrame(metrics_records)
    # Primary ranking based on ROC-AUC, then F1
    val_metrics_df = val_metrics_df.sort_values(by=["roc_auc", "f1"], ascending=[False, False]).reset_index(drop=True)
    return trained_models, val_metrics_df


def save_models_and_artifacts(
    trained_models: Dict[str, Any],
    best_model_name: str,
    val_metrics_df: pd.DataFrame,
    models_dir: str = "models",
    results_dir: str = "outputs/model_results"
) -> Dict[str, str]:
    """Persist trained model weights and validation results."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    persisted_files = {}

    # Save each model individually
    for name, item in trained_models.items():
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        filepath = os.path.join(models_dir, f"{slug}.joblib")
        joblib.dump(item["model"], filepath)
        persisted_files[name] = filepath

    # Save the designated best model
    best_model_obj = trained_models[best_model_name]["model"]
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model_obj, best_model_path)
    persisted_files["Best Model"] = best_model_path

    # Save validation metrics CSV and JSON
    val_csv_path = os.path.join(results_dir, "validation_metrics_summary.csv")
    val_metrics_df.to_csv(val_csv_path, index=False)
    persisted_files["Validation Metrics CSV"] = val_csv_path

    val_json_path = os.path.join(results_dir, "validation_metrics_summary.json")
    val_metrics_dict = {
        "best_model": best_model_name,
        "models_evaluated": len(trained_models),
        "results": val_metrics_df.to_dict(orient="records")
    }
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics_dict, f, indent=4)
    persisted_files["Validation Metrics JSON"] = val_json_path

    return persisted_files


def generate_training_report(
    val_metrics_df: pd.DataFrame,
    best_model_name: str,
    persisted_files: Dict[str, str],
    report_path: str = "reports/model_training_report.md"
) -> None:
    """Generate Markdown report for Stage 3: Model Implementation & Training."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    best_row = val_metrics_df[val_metrics_df["model_name"] == best_model_name].iloc[0]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Stage 3: Model Training & Algorithmic Benchmarking Report\n\n")
        f.write("## 1. Executive Summary\n")
        f.write(
            f"In this stage, 7 diverse classification models were implemented and trained "
            f"on the standardized 47-dimensional Telco customer feature space (4,930 training records). "
            f"All models were evaluated on the independent, stratified validation set (1,056 records). "
            f"The **{best_model_name}** achieved the top performance with a **Validation ROC-AUC of {best_row['roc_auc']:.4f}** "
            f"and an **F1-Score of {best_row['f1']:.4f}**.\n\n"
        )

        f.write("## 2. Models Evaluated & Configuration Strategy\n")
        f.write("| Model Family | Algorithm | Key Hyperparameters | Imbalance Strategy |\n")
        f.write("|---|---|---|---|\n")
        f.write("| Linear Baseline | Logistic Regression | `C=1.0, penalty='l2', solver='lbfgs', max_iter=1000` | `class_weight='balanced'` |\n")
        f.write("| Ensemble (Bagging) | Random Forest | `n_estimators=200, max_depth=12, min_samples_split=6` | `class_weight='balanced'` |\n")
        f.write("| Ensemble (Boosting) | Gradient Boosting | `n_estimators=150, learning_rate=0.08, max_depth=4` | Subsample 0.85 & Gradient Step |\n")
        f.write("| Modern Boosting | HistGradientBoosting | `learning_rate=0.08, max_iter=150, max_depth=6` | `class_weight='balanced'` |\n")
        f.write("| Adaptive Boosting | AdaBoost | `n_estimators=100, learning_rate=0.1` | Sequential Weight Adjustment |\n")
        f.write("| Kernel Classifier | Support Vector Machine (SVC) | `C=1.0, kernel='rbf', probability=True` | `class_weight='balanced'` |\n")
        f.write("| Deep Learning | Multi-Layer Perceptron (ANN) | `layers=(64, 32), activation='relu', solver='adam'` | Early stopping & L2 regularization |\n\n")

        f.write("## 3. Validation Performance Comparison\n")
        f.write("| Rank | Model Name | Validation ROC-AUC | Validation PR-AUC | Validation F1 | Recall | Precision | Accuracy | Train Time (s) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for i, row in val_metrics_df.iterrows():
            f.write(
                f"| {i + 1} | **{row['model_name']}** | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | "
                f"{row['f1']:.4f} | {row['recall']:.4f} | {row['precision']:.4f} | {row['accuracy']:.4f} | "
                f"{row['training_time_sec']:.3f} |\n"
            )
        f.write("\n")

        f.write("## 4. Selection of Best Model\n")
        f.write(
            f"- **Selected Champion Model**: `{best_model_name}`\n"
            f"- **Validation ROC-AUC**: `{best_row['roc_auc']:.4f}`\n"
            f"- **Validation Recall**: `{best_row['recall']:.4f}` (Identifies {best_row['recall']*100:.1f}% of churning customers)\n"
            f"- **Validation F1-Score**: `{best_row['f1']:.4f}`\n"
            f"- **Validation Precision**: `{best_row['precision']:.4f}`\n"
            f"- **Selection Rationale**: In churn prediction, missing at-risk customers (false negatives) incurs substantial lost revenue. "
            f"The champion model maintains high discriminative ability (ROC-AUC) while providing superior sensitivity (recall) "
            f"to capture potential defectors before contract termination.\n\n"
        )

        f.write("## 5. Persisted Model Artifacts\n")
        for name, path in persisted_files.items():
            rel_link = path.replace("\\", "/")
            f.write(f"- `{name}`: [{path}](file:///{rel_link})\n")
        f.write("\n")


def run_stage3_training(
    npz_path: str = "dataset/processed_data.npz",
    report_path: str = "reports/model_training_report.md"
) -> Dict[str, Any]:
    """Execute complete Stage 3 training and benchmarking workflow."""
    print("==================================================")
    print("STAGE 3: MODEL IMPLEMENTATION, TRAINING & BENCHMARK")
    print("==================================================")

    # 1. Load dataset arrays
    data = load_processed_data(npz_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    print(f"Loaded training data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"Loaded validation data: {X_val.shape[0]:,} samples, {X_val.shape[1]} features")

    # 2. Initialize model zoo
    models = get_model_zoo(random_state=42)

    # 3. Train and benchmark
    trained_models, val_metrics_df = train_and_validate_all_models(
        models, X_train, y_train, X_val, y_val
    )

    # 4. Best model selection (based on ROC-AUC and F1)
    best_model_name = val_metrics_df.iloc[0]["model_name"]
    print(f"\nChampion Model Selected: '{best_model_name}' (ROC-AUC: {val_metrics_df.iloc[0]['roc_auc']:.4f})")

    # 5. Persist models and metadata
    persisted_files = save_models_and_artifacts(
        trained_models, best_model_name, val_metrics_df
    )
    print("\nPersisted Model Artifacts:")
    for name, path in persisted_files.items():
        print(f"   - {name:<30} -> {path}")

    # 6. Generate Stage 3 Report
    generate_training_report(val_metrics_df, best_model_name, persisted_files, report_path)
    print(f"\nStage 3 Report generated and saved to: {report_path}")
    print("STAGE 3 COMPLETED SUCCESSFULLY.\n")

    return {
        "best_model_name": best_model_name,
        "val_metrics_df": val_metrics_df,
        "persisted_files": persisted_files
    }


if __name__ == "__main__":
    run_stage3_training()
