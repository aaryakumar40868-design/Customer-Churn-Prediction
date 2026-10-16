"""
visualization.py
----------------
Module for generating publication-quality visual diagnostics, performance curves,
feature importance rankings, and business impact charts for the Customer Churn project.
Persists all plots to outputs/visualizations/ with consistent styling and high resolution.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Apply consistent aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 15,
    "figure.titleweight": "bold"
})


def load_data_and_models(
    npz_path: str = "dataset/processed_data.npz",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """Load test data arrays, feature names, and trained model artifacts."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Processed dataset not found at '{npz_path}'.")
    data = np.load(npz_path, allow_pickle=True)
    
    model_mapping = {
        "Multi-Layer Perceptron (ANN)": "multi_layer_perceptron_ann.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
        "HistGradientBoosting": "histgradientboosting.joblib",
        "Support Vector Machine": "support_vector_machine.joblib",
        "AdaBoost": "adaboost.joblib"
    }
    
    models = {}
    for name, filename in model_mapping.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            
    return {
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "feature_names": data["feature_names"],
        "models": models
    }


def compute_test_probabilities(models: Dict[str, Any], X_test: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract class 1 positive churn probabilities for all models."""
    probas = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probas[name] = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            probas[name] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            probas[name] = model.predict(X_test).astype(float)
    return probas


def plot_roc_curves(
    y_test: np.ndarray,
    probas: Dict[str, np.ndarray],
    output_dir: str = "outputs/visualizations"
) -> str:
    """Plot multi-model ROC Curves with AUC benchmark legend."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    palette = sns.color_palette("tab10", n_colors=len(probas))
    
    for idx, (name, y_prob) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_val = auc(fpr, tpr)
        lw = 2.2 if idx < 2 else 1.5
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.4f})", color=palette[idx], linewidth=lw)
        
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)", linewidth=1.2, alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves - Test Cohort")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "roc_curves_comparison.png")
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_pr_curves(
    y_test: np.ndarray,
    probas: Dict[str, np.ndarray],
    output_dir: str = "outputs/visualizations"
) -> str:
    """Plot multi-model Precision-Recall curves with Average Precision legend."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    baseline = np.mean(y_test)
    palette = sns.color_palette("tab10", n_colors=len(probas))
    
    for idx, (name, y_prob) in enumerate(probas.items()):
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        lw = 2.2 if idx < 2 else 1.5
        ax.plot(rec, prec, label=f"{name} (AP = {ap:.4f})", color=palette[idx], linewidth=lw)
        
    ax.axhline(baseline, color="navy", linestyle="--", linewidth=1.2, label=f"No-Skill Baseline ({baseline*100:.1f}%)", alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision-Recall (PR) Curves - Test Cohort")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "precision_recall_curves.png")
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_confusion_matrices(
    y_test: np.ndarray,
    probas: Dict[str, np.ndarray],
    top_models: List[str],
    output_dir: str = "outputs/visualizations"
) -> str:
    """Plot dual confusion matrices (counts & normalized) for champion models."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(len(top_models), 2, figsize=(11, 4.8 * len(top_models)))
    
    if len(top_models) == 1:
        axes = np.array([axes])
        
    for row_idx, model_name in enumerate(top_models):
        y_prob = probas[model_name]
        y_pred = (y_prob >= 0.50).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[row_idx, 0],
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"]
        )
        axes[row_idx, 0].set_title(f"{model_name}\nRaw Counts (Default Threshold = 0.50)")
        axes[row_idx, 0].set_ylabel("True Ground Truth")
        axes[row_idx, 0].set_xlabel("Predicted Label")
        
        # Normalized percentages
        sns.heatmap(
            cm_norm, annot=True, fmt=".1%", cmap="Greens", cbar=False, ax=axes[row_idx, 1],
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"]
        )
        axes[row_idx, 1].set_title(f"{model_name}\nNormalized by Class True Rate")
        axes[row_idx, 1].set_ylabel("True Ground Truth")
        axes[row_idx, 1].set_xlabel("Predicted Label")
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrices_comparison.png")
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_feature_importance(
    models: Dict[str, Any],
    feature_names: np.ndarray,
    output_dir: str = "outputs/visualizations"
) -> str:
    """Plot positive and negative predictive feature importance drivers."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use Logistic Regression coefficients for exact directionality (log-odds impact)
    lr_model = models.get("Logistic Regression")
    if lr_model is None or not hasattr(lr_model, "coef_"):
        # Fallback to Random Forest feature importances
        rf_model = models.get("Random Forest")
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(np.array(feature_names)[indices][::-1], importances[indices][::-1], color="#2b5c8f")
        ax.set_xlabel("Gini Importance Score")
        ax.set_title("Top 20 Predictive Features - Random Forest")
        plt.tight_layout()
        save_path = os.path.join(output_dir, "feature_importance.png")
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    # Clean feature names for presentation
    clean_names = [
        f.replace("num__", "").replace("cat__", "")
        for f in feature_names
    ]
    coefs = lr_model.coef_[0]
    df_coefs = pd.DataFrame({"Feature": clean_names, "Coefficient": coefs})
    
    # Top 10 Churn Drivers (positive coef) and Top 10 Retention Drivers (negative coef)
    top_churn = df_coefs.sort_values(by="Coefficient", ascending=False).head(10)
    top_retention = df_coefs.sort_values(by="Coefficient", ascending=True).head(10)
    df_plot = pd.concat([top_retention, top_churn]).sort_values(by="Coefficient", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 7.5))
    colors = ["#2e7d32" if c < 0 else "#c62828" for c in df_plot["Coefficient"]]
    ax.barh(df_plot["Feature"], df_plot["Coefficient"], color=colors, edgecolor="black", linewidth=0.5)
    
    ax.axvline(0, color="black", linestyle="-", linewidth=1.0)
    ax.set_xlabel("Model Coefficient (Log-Odds Impact on Churn)")
    ax.set_title("Customer Churn Drivers vs. Retention Anchors (Logistic Model)")
    
    # Custom legend
    import matplotlib.patches as mpatches
    retention_patch = mpatches.Patch(color="#2e7d32", label="Retention Drivers (Reduces Churn Risk)")
    churn_patch = mpatches.Patch(color="#c62828", label="Churn Drivers (Increases Churn Risk)")
    ax.legend(handles=[retention_patch, churn_patch], loc="lower right", frameon=True)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_churn_probability_distribution(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    champion_name: str,
    output_dir: str = "outputs/visualizations"
) -> str:
    """Plot predicted probability density distribution comparing retained vs churned customers."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    sns.kdeplot(
        y_proba[y_test == 0],
        fill=True, color="#1976d2", label="Retained Customers (Class 0)",
        alpha=0.4, linewidth=2, ax=ax
    )
    sns.kdeplot(
        y_proba[y_test == 1],
        fill=True, color="#d32f2f", label="Churned Customers (Class 1)",
        alpha=0.4, linewidth=2, ax=ax
    )
    
    ax.axvline(0.50, color="gray", linestyle="--", linewidth=1.5, label="Default Cutoff (0.50)")
    ax.axvline(0.35, color="purple", linestyle=":", linewidth=2, label="High-Recall Cutoff (0.35)")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel("Predicted Probability of Churn")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Predicted Churn Risk Distribution by Ground Truth ({champion_name})")
    ax.legend(loc="upper center", frameon=True)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "churn_probability_distribution.png")
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_threshold_tradeoff_curves(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    output_dir: str = "outputs/visualizations"
) -> str:
    """Plot Precision, Recall, and F1 across decision thresholds."""
    os.makedirs(output_dir, exist_ok=True)
    thresholds = np.linspace(0.05, 0.95, 91)
    
    precs, recs, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(thresholds, precs, color="#1565c0", linewidth=2.0, label="Precision")
    ax.plot(thresholds, recs, color="#c62828", linewidth=2.0, label="Recall (Sensitivity)")
    ax.plot(thresholds, f1s, color="#2e7d32", linewidth=2.5, linestyle="-", label="F1-Score")
    
    best_idx = int(np.argmax(f1s))
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    
    ax.axvline(best_t, color="#2e7d32", linestyle="--", alpha=0.8, label=f"Max F1 Threshold ({best_t:.2f})")
    ax.scatter([best_t], [best_f1], color="#2e7d32", s=70, zorder=5)
    
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Decision Classification Threshold")
    ax.set_ylabel("Metric Score")
    ax.set_title("Metric Trade-offs across Decision Thresholds")
    ax.legend(loc="center right", frameon=True)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "threshold_tradeoff_curve.png")
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_model_benchmarks_barchart(
    metrics_csv_path: str = "outputs/model_results/test_metrics_summary.csv",
    output_dir: str = "outputs/visualizations"
) -> str:
    """Plot multi-metric benchmark comparison across all models."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(metrics_csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.18
    
    ax.bar(x - 1.5 * width, df["roc_auc"], width, label="ROC-AUC", color="#1f77b4")
    ax.bar(x - 0.5 * width, df["pr_auc"], width, label="PR-AUC", color="#ff7f0e")
    ax.bar(x + 0.5 * width, df["f1"], width, label="F1-Score", color="#2ca02c")
    ax.bar(x + 1.5 * width, df["recall"], width, label="Recall", color="#d62728")
    
    ax.set_ylabel("Score")
    ax.set_title("Comprehensive Out-of-Sample Model Benchmarks (Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=25, ha="right")
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "models_performance_benchmark.png")
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def run_stage5_visualizations(
    npz_path: str = "dataset/processed_data.npz",
    models_dir: str = "models",
    output_dir: str = "outputs/visualizations"
) -> Dict[str, str]:
    """Execute complete Stage 5 visualization workflow and save diagnostic charts."""
    print("==================================================")
    print("STAGE 5: VISUAL DIAGNOSTICS & EXPLAINABILITY PLOTS")
    print("==================================================")
    
    # 1. Load data and models
    context = load_data_and_models(npz_path, models_dir)
    X_test = context["X_test"]
    y_test = context["y_test"]
    feature_names = context["feature_names"]
    models = context["models"]
    
    # 2. Extract positive probabilities
    probas = compute_test_probabilities(models, X_test)
    
    # 3. Generate individual charts
    saved_charts = {}
    
    print("Generating ROC Curve comparisons...")
    saved_charts["ROC Curves"] = plot_roc_curves(y_test, probas, output_dir)
    
    print("Generating Precision-Recall Curve comparisons...")
    saved_charts["Precision-Recall Curves"] = plot_pr_curves(y_test, probas, output_dir)
    
    print("Generating Confusion Matrices...")
    top_models = ["Multi-Layer Perceptron (ANN)", "Logistic Regression"]
    saved_charts["Confusion Matrices"] = plot_confusion_matrices(y_test, probas, top_models, output_dir)
    
    print("Generating Feature Importance / Coefficients...")
    saved_charts["Feature Importance"] = plot_feature_importance(models, feature_names, output_dir)
    
    print("Generating Churn Risk Probability Distribution...")
    saved_charts["Probability Distribution"] = plot_churn_probability_distribution(
        y_test, probas["Multi-Layer Perceptron (ANN)"], "Multi-Layer Perceptron (ANN)", output_dir
    )
    
    print("Generating Decision Threshold Trade-off curves...")
    saved_charts["Threshold Trade-offs"] = plot_threshold_tradeoff_curves(
        y_test, probas["Multi-Layer Perceptron (ANN)"], output_dir
    )
    
    print("Generating Multi-Model Benchmark Barchart...")
    saved_charts["Model Benchmarks"] = plot_model_benchmarks_barchart(output_dir=output_dir)
    
    print("\nVisual Artifacts Persisted Successfully:")
    for label, path in saved_charts.items():
        print(f"   - {label:<25} -> {path}")
        
    print("STAGE 5 COMPLETED SUCCESSFULLY.\n")
    return saved_charts


if __name__ == "__main__":
    run_stage5_visualizations()
