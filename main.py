"""
main.py
-------
Master pipeline runner for the Customer Churn Prediction project.
Executes all 5 stages sequentially:
  - Stage 1: Data Understanding & Inspection (data_analysis.py)
  - Stage 2: Preprocessing, Leakage Prevention & Stratified Splitting (preprocessing.py)
  - Stage 3: Model Zoo Training & Validation Benchmarking (train_model.py)
  - Stage 4: Out-of-Sample Testing & Decision Threshold Optimization (evaluate_model.py)
  - Stage 5: Visual Diagnostics & Explainability Plotting (visualization.py)
"""

import sys
import time
from src.data_analysis import run_stage1_analysis
from src.preprocessing import run_stage2_preprocessing
from src.train_model import run_stage3_training
from src.evaluate_model import run_stage4_evaluation
from src.visualization import run_stage5_visualizations


def run_full_pipeline():
    start_time = time.time()
    print("=" * 65)
    print("CUSTOMER CHURN PREDICTION: END-TO-END PIPELINE EXECUTION")
    print("=" * 65)

    try:
        # Stage 1
        print("\n>>> EXECUTING STAGE 1: DATA UNDERSTANDING & INSPECTION...")
        run_stage1_analysis()

        # Stage 2
        print("\n>>> EXECUTING STAGE 2: PREPROCESSING & STRATIFIED SPLITTING...")
        run_stage2_preprocessing()

        # Stage 3
        print("\n>>> EXECUTING STAGE 3: MODEL TRAINING & BENCHMARKING...")
        run_stage3_training()

        # Stage 4
        print("\n>>> EXECUTING STAGE 4: MODEL EVALUATION & THRESHOLD OPTIMIZATION...")
        run_stage4_evaluation()

        # Stage 5
        print("\n>>> EXECUTING STAGE 5: VISUAL DIAGNOSTICS & EXPLAINABILITY...")
        run_stage5_visualizations()

        total_time = time.time() - start_time
        print("=" * 65)
        print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
        print("All reports generated in 'reports/' directory.")
        print("All visual figures generated in 'outputs/visualizations/' directory.")
        print("All metrics exported in 'outputs/model_results/' directory.")
        print("All models persisted in 'models/' directory.")
        print("=" * 65)

    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {str(e)}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    run_full_pipeline()
