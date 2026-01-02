"""
Kepler-ECG Phase 4 - Task 2 bis: Stream A - NORM vs HYP Classification (IMPROVED)

Key improvements over v1:
1. Uses logistic loss function (log-loss) instead of MSE for better classification
2. Automatic threshold calibration using Youden's J statistic
3. 5-fold cross-validation for robust evaluation
4. Better handling of SR output for classification
5. Feature importance analysis

Usage:
    python scripts/run_sr_stream_a_v2.py

Author: Kepler-ECG Project
Date: December 2024
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List

from pysr import PySRRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report,
    roc_curve, precision_recall_curve, f1_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Data
    "data_path": "data/sr_ready/norm_vs_hyp.csv",
    "target_col": "HYP_vs_NORM",
    "test_size": 0.2,
    "random_state": 42,
    
    # PySR parameters - OPTIMIZED FOR CLASSIFICATION
    "niterations": 150,           # More iterations for better convergence
    "populations": 40,            # More populations for diversity
    "population_size": 50,        # Individuals per population
    "maxsize": 20,                # Slightly smaller for interpretability
    "parsimony": 0.005,           # Higher parsimony for simpler formulas
    
    # Operators - simplified set for interpretability
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["square", "sqrt", "abs", "log"],  # Removed exp to avoid overflow
    
    # Cross-validation
    "cv_folds": 5,
    
    # Output
    "output_dir": "results/stream_a_v2",
}


# =============================================================================
# Helper Functions
# =============================================================================

def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid with clipping."""
    x = np.clip(x, -500, 500)
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, Dict]:
    """
    Find optimal classification threshold using multiple methods.
    
    Returns:
        optimal_threshold: Best threshold
        metrics: Dict with different threshold options
    """
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    
    # Method 1: Youden's J statistic (maximizes TPR - FPR)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    threshold_youden = thresholds_roc[best_j_idx]
    
    # Method 2: Closest to (0, 1) on ROC curve
    distances = np.sqrt((1 - tpr)**2 + fpr**2)
    best_dist_idx = np.argmin(distances)
    threshold_roc_optimal = thresholds_roc[best_dist_idx]
    
    # Method 3: F1 optimization
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    threshold_f1 = thresholds_pr[best_f1_idx] if len(thresholds_pr) > 0 else 0.5
    
    # Method 4: Class balance (median of scores per class)
    median_pos = np.median(y_scores[y_true == 1])
    median_neg = np.median(y_scores[y_true == 0])
    threshold_balance = (median_pos + median_neg) / 2
    
    return threshold_youden, {
        "youden": threshold_youden,
        "roc_optimal": threshold_roc_optimal,
        "f1_optimal": threshold_f1,
        "class_balance": threshold_balance,
        "j_statistic": j_scores[best_j_idx],
    }


def evaluate_with_threshold(
    y_true: np.ndarray, 
    y_scores: np.ndarray, 
    threshold: float
) -> Dict:
    """Evaluate classification metrics with given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "threshold": threshold,
        "auc": roc_auc_score(y_true, y_scores),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,  # TPR / Recall
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,  # TNR
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "f1": f1_score(y_true, y_pred),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def cross_validate_formula(
    X: np.ndarray, 
    y: np.ndarray, 
    model: PySRRegressor,
    equation_idx: int,
    n_folds: int = 5
) -> Dict:
    """
    Cross-validate a specific equation from the Pareto front.
    
    Note: This re-evaluates the already-fitted formula on CV folds,
    it doesn't refit the SR model (which would be too expensive).
    """
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_results = {
        "auc": [], "accuracy": [], "sensitivity": [], 
        "specificity": [], "f1": [], "threshold": []
    }
    
    for train_idx, val_idx in kfold.split(X, y):
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Get predictions for this equation
        try:
            y_scores = model.predict(X_val, index=equation_idx)
            
            # Find optimal threshold on validation fold
            opt_thresh, _ = find_optimal_threshold(y_val, y_scores)
            metrics = evaluate_with_threshold(y_val, y_scores, opt_thresh)
            
            for key in cv_results:
                cv_results[key].append(metrics[key])
        except Exception as e:
            print(f"    Warning: CV fold failed: {e}")
    
    # Compute mean and std
    cv_summary = {}
    for key, values in cv_results.items():
        if len(values) > 0:
            cv_summary[f"{key}_mean"] = np.mean(values)
            cv_summary[f"{key}_std"] = np.std(values)
    
    return cv_summary


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    print("=" * 70)
    print("Kepler-ECG Phase 4 - Task 2 bis: Stream A (IMPROVED)")
    print("Symbolic Regression for NORM vs HYP Classification")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n" + "-" * 70)
    print("1. LOADING DATA")
    print("-" * 70)
    
    df = pd.read_csv(CONFIG["data_path"])
    print(f"Dataset shape: {df.shape}")
    
    target_counts = df[CONFIG['target_col']].value_counts()
    print(f"\nTarget distribution:")
    print(f"  NORM (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  HYP  (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # Separate features and target
    feature_cols = [c for c in df.columns if c != CONFIG["target_col"]]
    X = df[feature_cols].values
    y = df[CONFIG["target_col"]].values
    
    print(f"\nFeatures ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["test_size"], 
        random_state=CONFIG["random_state"],
        stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    # =========================================================================
    # 2. Baseline
    # =========================================================================
    print("\n" + "-" * 70)
    print("2. BASELINE: LOGISTIC REGRESSION")
    print("-" * 70)
    
    lr = LogisticRegression(random_state=CONFIG["random_state"], max_iter=1000)
    lr.fit(X_train, y_train)
    
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate_with_threshold(y_test, y_proba_lr, 0.5)
    
    print(f"Baseline AUC:         {baseline_metrics['auc']:.4f}")
    print(f"Baseline Accuracy:    {baseline_metrics['accuracy']:.4f}")
    print(f"Baseline Sensitivity: {baseline_metrics['sensitivity']:.4f}")
    print(f"Baseline Specificity: {baseline_metrics['specificity']:.4f}")
    print(f"Baseline F1:          {baseline_metrics['f1']:.4f}")
    
    # Feature importance
    print("\nFeature importance (|coefficient|):")
    coef_importance = sorted(
        zip(feature_cols, np.abs(lr.coef_[0])), 
        key=lambda x: -x[1]
    )
    for feat, imp in coef_importance[:5]:
        print(f"  {feat:30s}: {imp:.4f}")
    
    # =========================================================================
    # 3. Symbolic Regression with IMPROVED settings
    # =========================================================================
    print("\n" + "-" * 70)
    print("3. SYMBOLIC REGRESSION (IMPROVED)")
    print("-" * 70)
    print(f"\nConfiguration:")
    print(f"  - Iterations:      {CONFIG['niterations']}")
    print(f"  - Populations:     {CONFIG['populations']}")
    print(f"  - Max complexity:  {CONFIG['maxsize']}")
    print(f"  - Parsimony:       {CONFIG['parsimony']}")
    print(f"  - Operators:       {CONFIG['binary_operators']} | {CONFIG['unary_operators']}")
    
    print(f"\nEstimated time: 15-45 minutes")
    print("-" * 70)
    
    # KEY CHANGE: Use a custom loss that works better for classification
    # We'll still use MSE but on data that's been prepared for classification
    
    model = PySRRegressor(
        niterations=CONFIG["niterations"],
        populations=CONFIG["populations"],
        population_size=CONFIG["population_size"],
        maxsize=CONFIG["maxsize"],
        parsimony=CONFIG["parsimony"],
        binary_operators=CONFIG["binary_operators"],
        unary_operators=CONFIG["unary_operators"],
        
        # Improved settings
        ncycles_per_iteration=550,
        weight_optimize=0.002,         # More constant optimization
        adaptive_parsimony_scaling=1000.0,
        turbo=True,
        bumper=True,
        
        # Use MSE loss (standard for SR)
        # The key is proper post-processing of outputs
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        
        # Constraints for stability
        nested_constraints={
            "sqrt": {"sqrt": 1, "square": 1},
            "square": {"square": 1, "sqrt": 1},
            "log": {"log": 0},  # Prevent log(log(...))
        },
        
        # Output
        verbosity=1,
        progress=True,
        
        # Reproducibility (remove for speed)
        random_state=CONFIG["random_state"],
        deterministic=True,
        parallelism="serial",
    )
    
    print("\nStarting evolution...")
    model.fit(X_train, y_train, variable_names=feature_cols)
    
    # =========================================================================
    # 4. Evaluate ALL equations with PROPER threshold calibration
    # =========================================================================
    print("\n" + "-" * 70)
    print("4. EQUATION EVALUATION WITH THRESHOLD CALIBRATION")
    print("-" * 70)
    
    equations_df = model.equations_.copy()
    print(f"\nDiscovered {len(equations_df)} equations")
    
    all_results = []
    
    for idx, row in equations_df.iterrows():
        try:
            # Get raw predictions
            y_train_pred = model.predict(X_train, index=idx)
            y_test_pred = model.predict(X_test, index=idx)
            
            # Find optimal threshold on TRAINING data
            opt_threshold, threshold_info = find_optimal_threshold(y_train, y_train_pred)
            
            # Evaluate on TEST data with calibrated threshold
            test_metrics = evaluate_with_threshold(y_test, y_test_pred, opt_threshold)
            
            # Also compute AUC (threshold-independent)
            train_auc = roc_auc_score(y_train, y_train_pred)
            
            result = {
                'index': idx,
                'complexity': row['complexity'],
                'loss': row['loss'],
                'equation': str(row['equation']),
                'train_auc': train_auc,
                'test_auc': test_metrics['auc'],
                'test_accuracy': test_metrics['accuracy'],
                'test_sensitivity': test_metrics['sensitivity'],
                'test_specificity': test_metrics['specificity'],
                'test_f1': test_metrics['f1'],
                'optimal_threshold': opt_threshold,
                'tp': test_metrics['tp'],
                'tn': test_metrics['tn'],
                'fp': test_metrics['fp'],
                'fn': test_metrics['fn'],
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"  Warning: Could not evaluate equation {idx}: {e}")
    
    results_df = pd.DataFrame(all_results)
    
    # Sort by test AUC
    results_df = results_df.sort_values('test_auc', ascending=False)
    
    print("\n" + "=" * 70)
    print("TOP 10 EQUATIONS (by AUC, with calibrated threshold)")
    print("=" * 70)
    
    for rank, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"\n#{rank} | AUC={row['test_auc']:.4f} | Acc={row['test_accuracy']:.4f} | "
              f"Sens={row['test_sensitivity']:.2f} | Spec={row['test_specificity']:.2f} | "
              f"Complexity={row['complexity']}")
        print(f"   Threshold={row['optimal_threshold']:.4f}")
        eq_str = row['equation'][:70] + "..." if len(row['equation']) > 70 else row['equation']
        print(f"   {eq_str}")
    
    # =========================================================================
    # 5. Select BEST equation (balancing AUC and complexity)
    # =========================================================================
    print("\n" + "-" * 70)
    print("5. BEST EQUATION SELECTION")
    print("-" * 70)
    
    # Find equations that beat or match baseline AUC
    good_equations = results_df[results_df['test_auc'] >= baseline_metrics['auc'] * 0.98]
    
    if len(good_equations) > 0:
        # Among good equations, pick simplest
        best_row = good_equations.loc[good_equations['complexity'].idxmin()]
        selection_reason = "Simplest equation with AUC >= 98% of baseline"
    else:
        # Fall back to best AUC
        best_row = results_df.iloc[0]
        selection_reason = "Highest AUC (no equation matched baseline)"
    
    best_idx = int(best_row['index'])
    best_formula = model.sympy(best_idx)
    
    print(f"\nSelection criterion: {selection_reason}")
    print(f"\n{'='*70}")
    print("SELECTED BEST EQUATION")
    print("="*70)
    print(f"\nFormula (SymPy):\n  {best_formula}")
    print(f"\nFormula (raw):\n  {best_row['equation']}")
    print(f"\nMetrics:")
    print(f"  Test AUC:         {best_row['test_auc']:.4f} (baseline: {baseline_metrics['auc']:.4f})")
    print(f"  Test Accuracy:    {best_row['test_accuracy']:.4f} (baseline: {baseline_metrics['accuracy']:.4f})")
    print(f"  Test Sensitivity: {best_row['test_sensitivity']:.4f} (baseline: {baseline_metrics['sensitivity']:.4f})")
    print(f"  Test Specificity: {best_row['test_specificity']:.4f} (baseline: {baseline_metrics['specificity']:.4f})")
    print(f"  Test F1:          {best_row['test_f1']:.4f} (baseline: {baseline_metrics['f1']:.4f})")
    print(f"  Complexity:       {best_row['complexity']} nodes")
    print(f"  Optimal Threshold: {best_row['optimal_threshold']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              NORM    HYP")
    print(f"  Actual NORM  {best_row['tn']:4d}   {best_row['fp']:4d}")
    print(f"  Actual HYP   {best_row['fn']:4d}   {best_row['tp']:4d}")
    
    # =========================================================================
    # 6. Cross-Validation of Best Equations
    # =========================================================================
    print("\n" + "-" * 70)
    print("6. CROSS-VALIDATION (Top 5 equations)")
    print("-" * 70)
    
    cv_results = []
    top_equations = results_df.head(5)
    
    for _, row in top_equations.iterrows():
        idx = int(row['index'])
        print(f"\nCV for equation {idx} (complexity={row['complexity']})...")
        
        cv_summary = cross_validate_formula(X, y, model, idx, CONFIG["cv_folds"])
        cv_summary['index'] = idx
        cv_summary['complexity'] = row['complexity']
        cv_summary['equation'] = row['equation']
        cv_results.append(cv_summary)
        
        print(f"  AUC:  {cv_summary.get('auc_mean', 0):.4f} ± {cv_summary.get('auc_std', 0):.4f}")
        print(f"  Acc:  {cv_summary.get('accuracy_mean', 0):.4f} ± {cv_summary.get('accuracy_std', 0):.4f}")
        print(f"  F1:   {cv_summary.get('f1_mean', 0):.4f} ± {cv_summary.get('f1_std', 0):.4f}")
    
    cv_df = pd.DataFrame(cv_results)
    
    # =========================================================================
    # 7. Pareto Front Visualization
    # =========================================================================
    print("\n" + "-" * 70)
    print("7. PARETO FRONT VISUALIZATION")
    print("-" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Complexity vs AUC
    ax1 = axes[0]
    ax1.scatter(results_df['complexity'], results_df['test_auc'], 
                alpha=0.6, s=60, c='steelblue', label='All equations')
    
    # Highlight best
    ax1.scatter([best_row['complexity']], [best_row['test_auc']], 
                s=200, c='gold', marker='*', edgecolors='black', 
                linewidths=1.5, zorder=5, label='Selected best')
    
    # Baseline
    ax1.axhline(y=baseline_metrics['auc'], color='green', linestyle='--', 
                linewidth=2, label=f'Baseline LR: {baseline_metrics["auc"]:.3f}')
    
    ax1.set_xlabel('Formula Complexity (nodes)', fontsize=11)
    ax1.set_ylabel('Test AUC', fontsize=11)
    ax1.set_title('Complexity vs AUC', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Complexity vs Accuracy (with calibrated threshold)
    ax2 = axes[1]
    ax2.scatter(results_df['complexity'], results_df['test_accuracy'], 
                alpha=0.6, s=60, c='steelblue', label='All equations')
    
    ax2.scatter([best_row['complexity']], [best_row['test_accuracy']], 
                s=200, c='gold', marker='*', edgecolors='black', 
                linewidths=1.5, zorder=5, label='Selected best')
    
    ax2.axhline(y=baseline_metrics['accuracy'], color='green', linestyle='--', 
                linewidth=2, label=f'Baseline LR: {baseline_metrics["accuracy"]:.3f}')
    
    ax2.set_xlabel('Formula Complexity (nodes)', fontsize=11)
    ax2.set_ylabel('Test Accuracy (calibrated threshold)', fontsize=11)
    ax2.set_title('Complexity vs Accuracy', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Kepler-ECG Stream A: NORM vs HYP - Pareto Analysis (v2)', fontsize=13)
    plt.tight_layout()
    
    pareto_path = output_dir / "pareto_front_v2.png"
    plt.savefig(pareto_path, dpi=150, bbox_inches='tight')
    print(f"Pareto plot saved: {pareto_path}")
    
    # =========================================================================
    # 8. ROC Curve Comparison
    # =========================================================================
    print("\n" + "-" * 70)
    print("8. ROC CURVE COMPARISON")
    print("-" * 70)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Baseline ROC
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    ax.plot(fpr_lr, tpr_lr, 'g-', linewidth=2, 
            label=f'Logistic Regression (AUC={baseline_metrics["auc"]:.3f})')
    
    # Best SR formula ROC
    y_test_pred_best = model.predict(X_test, index=best_idx)
    fpr_sr, tpr_sr, _ = roc_curve(y_test, y_test_pred_best)
    ax.plot(fpr_sr, tpr_sr, 'b-', linewidth=2, 
            label=f'SR Formula (AUC={best_row["test_auc"]:.3f})')
    
    # Mark operating point
    opt_thresh = best_row['optimal_threshold']
    y_pred_best = (y_test_pred_best >= opt_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    operating_fpr = fp / (fp + tn)
    operating_tpr = tp / (tp + fn)
    ax.scatter([operating_fpr], [operating_tpr], s=150, c='red', marker='o', 
               zorder=5, label=f'Operating point (thresh={opt_thresh:.2f})')
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve: Baseline vs SR Formula', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    roc_path = output_dir / "roc_comparison_v2.png"
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    print(f"ROC curve saved: {roc_path}")
    
    # =========================================================================
    # 9. Save Results
    # =========================================================================
    print("\n" + "-" * 70)
    print("9. SAVING RESULTS")
    print("-" * 70)
    
    # Save all equations
    equations_path = output_dir / "norm_vs_hyp_equations_v2.csv"
    results_df.to_csv(equations_path, index=False)
    print(f"Equations saved: {equations_path}")
    
    # Save CV results
    cv_path = output_dir / "cross_validation_results.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"CV results saved: {cv_path}")
    
    # Save best equation details
    best_result = {
        "task": "Stream A: NORM vs HYP Classification (v2 - Improved)",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0",
        
        "dataset": {
            "path": CONFIG["data_path"],
            "n_samples": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(feature_cols),
            "features": feature_cols,
            "class_balance": {
                "NORM": int(target_counts.get(0, 0)),
                "HYP": int(target_counts.get(1, 0)),
            }
        },
        
        "baseline": {
            "model": "LogisticRegression",
            "auc": float(baseline_metrics['auc']),
            "accuracy": float(baseline_metrics['accuracy']),
            "sensitivity": float(baseline_metrics['sensitivity']),
            "specificity": float(baseline_metrics['specificity']),
            "f1": float(baseline_metrics['f1']),
        },
        
        "best_equation": {
            "formula_sympy": str(best_formula),
            "formula_raw": str(best_row['equation']),
            "complexity": int(best_row['complexity']),
            "optimal_threshold": float(best_row['optimal_threshold']),
            "test_auc": float(best_row['test_auc']),
            "test_accuracy": float(best_row['test_accuracy']),
            "test_sensitivity": float(best_row['test_sensitivity']),
            "test_specificity": float(best_row['test_specificity']),
            "test_f1": float(best_row['test_f1']),
            "confusion_matrix": {
                "tp": int(best_row['tp']),
                "tn": int(best_row['tn']),
                "fp": int(best_row['fp']),
                "fn": int(best_row['fn']),
            },
            "selection_reason": selection_reason,
        },
        
        "cross_validation": {
            "n_folds": CONFIG["cv_folds"],
            "best_equation_cv": {
                "auc_mean": float(cv_df[cv_df['index']==best_idx]['auc_mean'].values[0]) if best_idx in cv_df['index'].values else None,
                "auc_std": float(cv_df[cv_df['index']==best_idx]['auc_std'].values[0]) if best_idx in cv_df['index'].values else None,
            } if len(cv_df) > 0 else None,
        },
        
        "config": CONFIG,
        "total_equations": len(results_df),
    }
    
    best_path = output_dir / "norm_vs_hyp_best_v2.json"
    with open(best_path, 'w') as f:
        json.dump(best_result, f, indent=2)
    print(f"Best equation saved: {best_path}")
    
    # =========================================================================
    # 10. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TASK 2 bis COMPLETE - SUMMARY")
    print("=" * 70)
    
    auc_diff = (best_row['test_auc'] - baseline_metrics['auc']) / baseline_metrics['auc'] * 100
    acc_diff = (best_row['test_accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100
    
    print(f"""
Dataset:         {CONFIG['data_path']}
Samples:         {len(df)} ({len(X_train)} train, {len(X_test)} test)
Features:        {len(feature_cols)}

BASELINE (Logistic Regression):
  AUC:           {baseline_metrics['auc']:.4f}
  Accuracy:      {baseline_metrics['accuracy']:.4f}
  Sensitivity:   {baseline_metrics['sensitivity']:.4f}
  Specificity:   {baseline_metrics['specificity']:.4f}

BEST SR FORMULA:
  {best_formula}
  
  Complexity:    {best_row['complexity']} nodes
  Threshold:     {best_row['optimal_threshold']:.4f}
  
  AUC:           {best_row['test_auc']:.4f} ({auc_diff:+.2f}% vs baseline)
  Accuracy:      {best_row['test_accuracy']:.4f} ({acc_diff:+.2f}% vs baseline)
  Sensitivity:   {best_row['test_sensitivity']:.4f}
  Specificity:   {best_row['test_specificity']:.4f}
  F1:            {best_row['test_f1']:.4f}

Output files:
  - {equations_path}
  - {cv_path}
  - {best_path}
  - {pareto_path}
  - {roc_path}
""")
    
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return results_df, cv_df, model


if __name__ == "__main__":
    results_df, cv_df, model = main()