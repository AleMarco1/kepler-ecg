"""
Kepler-ECG Phase 4 - Task 3: Stream B - Cardiac Age Regression

Discovers interpretable formulas to predict biological cardiac age from ECG features.

Key concept: If predicted_age > actual_age, the heart is "older" than expected,
potentially indicating compromised cardiovascular health.

Usage:
    python scripts/run_sr_stream_b.py

Output:
    - results/stream_b/cardiac_age_equations.csv
    - results/stream_b/cardiac_age_best.json
    - results/stream_b/pareto_front_age.png
    - results/stream_b/age_prediction_scatter.png

Author: Kepler-ECG Project
Date: December 2024
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List

from pysr import PySRRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    explained_variance_score
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Data
    "data_path": "data/sr_ready/age_regression_norm.csv",
    "target_col": "age",
    "test_size": 0.2,
    "random_state": 42,
    
    # PySR parameters - OPTIMIZED FOR REGRESSION
    "niterations": 150,           # Good balance of exploration
    "populations": 40,            # Diverse search
    "population_size": 50,        
    "maxsize": 25,                # Allow moderately complex formulas
    "parsimony": 0.003,           # Balance complexity vs accuracy
    
    # Operators for regression (more mathematical operations)
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["square", "sqrt", "abs", "log", "exp"],
    
    # Cross-validation
    "cv_folds": 5,
    
    # Output
    "output_dir": "results/stream_b",
}


# =============================================================================
# Helper Functions
# =============================================================================

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute regression metrics."""
    # Clip predictions to reasonable range
    y_pred = np.clip(y_pred, 0, 120)  # Age between 0 and 120
    
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mse": mean_squared_error(y_true, y_pred),
        "explained_variance": explained_variance_score(y_true, y_pred),
        "mean_error": np.mean(y_pred - y_true),  # Bias
        "std_error": np.std(y_pred - y_true),    # Precision
    }


def cross_validate_formula(
    X: np.ndarray, 
    y: np.ndarray, 
    model: PySRRegressor,
    equation_idx: int,
    n_folds: int = 5
) -> Dict:
    """Cross-validate a specific equation."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_results = {"r2": [], "mae": [], "rmse": []}
    
    for train_idx, val_idx in kfold.split(X):
        X_val, y_val = X[val_idx], y[val_idx]
        
        try:
            y_pred = model.predict(X_val, index=equation_idx)
            y_pred = np.clip(y_pred, 0, 120)
            
            cv_results["r2"].append(r2_score(y_val, y_pred))
            cv_results["mae"].append(mean_absolute_error(y_val, y_pred))
            cv_results["rmse"].append(np.sqrt(mean_squared_error(y_val, y_pred)))
        except Exception as e:
            print(f"    Warning: CV fold failed: {e}")
    
    return {
        f"{k}_mean": np.mean(v) for k, v in cv_results.items() if len(v) > 0
    } | {
        f"{k}_std": np.std(v) for k, v in cv_results.items() if len(v) > 0
    }


def analyze_cardiac_age_delta(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    output_dir: Path
) -> Dict:
    """
    Analyze the cardiac age delta (predicted - actual).
    
    Positive delta = heart appears older than chronological age
    Negative delta = heart appears younger
    """
    delta = y_pred - y_true
    
    analysis = {
        "mean_delta": float(np.mean(delta)),
        "std_delta": float(np.std(delta)),
        "median_delta": float(np.median(delta)),
        "pct_older": float(np.mean(delta > 0) * 100),  # % with older cardiac age
        "pct_younger": float(np.mean(delta < 0) * 100),
        "delta_percentiles": {
            "p10": float(np.percentile(delta, 10)),
            "p25": float(np.percentile(delta, 25)),
            "p50": float(np.percentile(delta, 50)),
            "p75": float(np.percentile(delta, 75)),
            "p90": float(np.percentile(delta, 90)),
        }
    }
    
    # Plot delta distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of delta
    ax1 = axes[0]
    ax1.hist(delta, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax1.axvline(x=np.mean(delta), color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(delta):.1f} years')
    ax1.set_xlabel('Cardiac Age Delta (Predicted - Actual)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Cardiac Age Delta', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Delta vs actual age
    ax2 = axes[1]
    ax2.scatter(y_true, delta, alpha=0.3, s=10, c='steelblue')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # Add trend line
    z = np.polyfit(y_true, delta, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    ax2.plot(x_line, p(x_line), 'g-', linewidth=2, label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')
    
    ax2.set_xlabel('Actual Age (years)', fontsize=11)
    ax2.set_ylabel('Cardiac Age Delta (years)', fontsize=11)
    ax2.set_title('Cardiac Age Delta vs Actual Age', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    delta_path = output_dir / "cardiac_age_delta.png"
    plt.savefig(delta_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    analysis["delta_plot"] = str(delta_path)
    analysis["age_trend_slope"] = float(z[0])  # How delta changes with age
    
    return analysis


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    print("=" * 70)
    print("Kepler-ECG Phase 4 - Task 3: Stream B")
    print("Symbolic Regression for Cardiac Age Prediction")
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
    
    # Separate features and target
    feature_cols = [c for c in df.columns if c != CONFIG["target_col"]]
    X = df[feature_cols].values
    y = df[CONFIG["target_col"]].values
    
    print(f"\nTarget (age) statistics:")
    print(f"  Min:    {y.min():.1f} years")
    print(f"  Max:    {y.max():.1f} years")
    print(f"  Mean:   {y.mean():.1f} years")
    print(f"  Median: {np.median(y):.1f} years")
    print(f"  Std:    {y.std():.1f} years")
    
    print(f"\nFeatures ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["test_size"], 
        random_state=CONFIG["random_state"]
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    # =========================================================================
    # 2. Baseline Models
    # =========================================================================
    print("\n" + "-" * 70)
    print("2. BASELINE MODELS")
    print("-" * 70)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    baseline_lr = evaluate_regression(y_test, y_pred_lr)
    
    print(f"\nLinear Regression:")
    print(f"  R²:   {baseline_lr['r2']:.4f}")
    print(f"  MAE:  {baseline_lr['mae']:.2f} years")
    print(f"  RMSE: {baseline_lr['rmse']:.2f} years")
    
    # Ridge Regression (for comparison)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    baseline_ridge = evaluate_regression(y_test, y_pred_ridge)
    
    print(f"\nRidge Regression:")
    print(f"  R²:   {baseline_ridge['r2']:.4f}")
    print(f"  MAE:  {baseline_ridge['mae']:.2f} years")
    print(f"  RMSE: {baseline_ridge['rmse']:.2f} years")
    
    # Feature importance from linear model
    print("\nFeature importance (|coefficient|):")
    coef_importance = sorted(
        zip(feature_cols, np.abs(lr.coef_)), 
        key=lambda x: -x[1]
    )
    for feat, imp in coef_importance[:5]:
        print(f"  {feat:35s}: {imp:.4f}")
    
    # Use best baseline for comparison
    baseline_metrics = baseline_lr if baseline_lr['r2'] >= baseline_ridge['r2'] else baseline_ridge
    baseline_name = "Linear Regression" if baseline_lr['r2'] >= baseline_ridge['r2'] else "Ridge Regression"
    
    # =========================================================================
    # 3. Symbolic Regression
    # =========================================================================
    print("\n" + "-" * 70)
    print("3. SYMBOLIC REGRESSION")
    print("-" * 70)
    print(f"\nConfiguration:")
    print(f"  - Iterations:      {CONFIG['niterations']}")
    print(f"  - Populations:     {CONFIG['populations']}")
    print(f"  - Max complexity:  {CONFIG['maxsize']}")
    print(f"  - Parsimony:       {CONFIG['parsimony']}")
    print(f"  - Operators:       {CONFIG['binary_operators']} | {CONFIG['unary_operators']}")
    
    print(f"\nEstimated time: 20-60 minutes (larger dataset)")
    print("-" * 70)
    
    model = PySRRegressor(
        niterations=CONFIG["niterations"],
        populations=CONFIG["populations"],
        population_size=CONFIG["population_size"],
        maxsize=CONFIG["maxsize"],
        parsimony=CONFIG["parsimony"],
        binary_operators=CONFIG["binary_operators"],
        unary_operators=CONFIG["unary_operators"],
        
        # Regression-specific settings
        ncycles_per_iteration=550,
        weight_optimize=0.002,
        adaptive_parsimony_scaling=1000.0,
        turbo=True,
        bumper=True,
        
        # Standard MSE loss for regression
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        
        # Constraints
        nested_constraints={
            "sqrt": {"sqrt": 1, "square": 1},
            "square": {"square": 1, "sqrt": 1},
            "log": {"log": 0, "exp": 1},
            "exp": {"exp": 0, "log": 1},
        },
        
        # Complexity constraints
        constraints={
            "/": (-1, 9),  # Denominator max complexity 9
        },
        
        # Output
        verbosity=1,
        progress=True,
        
        # Reproducibility
        random_state=CONFIG["random_state"],
        deterministic=True,
        parallelism="serial",
    )
    
    print("\nStarting evolution...")
    model.fit(X_train, y_train, variable_names=feature_cols)
    
    # =========================================================================
    # 4. Evaluate All Equations
    # =========================================================================
    print("\n" + "-" * 70)
    print("4. EQUATION EVALUATION")
    print("-" * 70)
    
    equations_df = model.equations_.copy()
    print(f"\nDiscovered {len(equations_df)} equations")
    
    all_results = []
    
    for idx, row in equations_df.iterrows():
        try:
            # Get predictions
            y_train_pred = model.predict(X_train, index=idx)
            y_test_pred = model.predict(X_test, index=idx)
            
            # Clip to reasonable range
            y_train_pred = np.clip(y_train_pred, 0, 120)
            y_test_pred = np.clip(y_test_pred, 0, 120)
            
            # Evaluate
            train_metrics = evaluate_regression(y_train, y_train_pred)
            test_metrics = evaluate_regression(y_test, y_test_pred)
            
            result = {
                'index': idx,
                'complexity': row['complexity'],
                'loss': row['loss'],
                'equation': str(row['equation']),
                'train_r2': train_metrics['r2'],
                'train_mae': train_metrics['mae'],
                'test_r2': test_metrics['r2'],
                'test_mae': test_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
                'test_mean_error': test_metrics['mean_error'],
                'test_std_error': test_metrics['std_error'],
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"  Warning: Could not evaluate equation {idx}: {e}")
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('test_r2', ascending=False)
    
    print("\n" + "=" * 70)
    print("TOP 10 EQUATIONS (by R²)")
    print("=" * 70)
    
    for rank, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"\n#{rank} | R²={row['test_r2']:.4f} | MAE={row['test_mae']:.2f}yr | "
              f"RMSE={row['test_rmse']:.2f}yr | Complexity={row['complexity']}")
        eq_str = row['equation'][:65] + "..." if len(row['equation']) > 65 else row['equation']
        print(f"   {eq_str}")
    
    # =========================================================================
    # 5. Select Best Equation
    # =========================================================================
    print("\n" + "-" * 70)
    print("5. BEST EQUATION SELECTION")
    print("-" * 70)
    
    # Find equations that beat baseline R²
    good_equations = results_df[results_df['test_r2'] >= baseline_metrics['r2'] * 0.95]
    
    if len(good_equations) > 0:
        # Among good equations, prefer simpler ones
        best_row = good_equations.loc[good_equations['complexity'].idxmin()]
        selection_reason = "Simplest equation with R² >= 95% of baseline"
    else:
        # Fall back to best R²
        best_row = results_df.iloc[0]
        selection_reason = "Highest R² (no equation matched baseline threshold)"
    
    best_idx = int(best_row['index'])
    best_formula = model.sympy(best_idx)
    
    print(f"\nSelection criterion: {selection_reason}")
    print(f"\n{'='*70}")
    print("SELECTED BEST EQUATION")
    print("="*70)
    print(f"\nFormula (SymPy):\n  {best_formula}")
    print(f"\nFormula (raw):\n  {best_row['equation']}")
    print(f"\nMetrics:")
    print(f"  Test R²:          {best_row['test_r2']:.4f} (baseline: {baseline_metrics['r2']:.4f})")
    print(f"  Test MAE:         {best_row['test_mae']:.2f} years (baseline: {baseline_metrics['mae']:.2f})")
    print(f"  Test RMSE:        {best_row['test_rmse']:.2f} years (baseline: {baseline_metrics['rmse']:.2f})")
    print(f"  Complexity:       {best_row['complexity']} nodes")
    print(f"  Mean Error:       {best_row['test_mean_error']:.2f} years (bias)")
    print(f"  Std Error:        {best_row['test_std_error']:.2f} years (precision)")
    
    # =========================================================================
    # 6. Cross-Validation
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
        
        print(f"  R²:   {cv_summary.get('r2_mean', 0):.4f} ± {cv_summary.get('r2_std', 0):.4f}")
        print(f"  MAE:  {cv_summary.get('mae_mean', 0):.2f} ± {cv_summary.get('mae_std', 0):.2f} years")
        print(f"  RMSE: {cv_summary.get('rmse_mean', 0):.2f} ± {cv_summary.get('rmse_std', 0):.2f} years")
    
    cv_df = pd.DataFrame(cv_results)
    
    # =========================================================================
    # 7. Cardiac Age Delta Analysis
    # =========================================================================
    print("\n" + "-" * 70)
    print("7. CARDIAC AGE DELTA ANALYSIS")
    print("-" * 70)
    
    y_test_pred_best = model.predict(X_test, index=best_idx)
    y_test_pred_best = np.clip(y_test_pred_best, 0, 120)
    
    delta_analysis = analyze_cardiac_age_delta(y_test, y_test_pred_best, output_dir)
    
    print(f"\nCardiac Age Delta (Predicted - Actual):")
    print(f"  Mean:   {delta_analysis['mean_delta']:+.2f} years")
    print(f"  Std:    {delta_analysis['std_delta']:.2f} years")
    print(f"  Median: {delta_analysis['median_delta']:+.2f} years")
    print(f"\n  Patients with 'older' cardiac age: {delta_analysis['pct_older']:.1f}%")
    print(f"  Patients with 'younger' cardiac age: {delta_analysis['pct_younger']:.1f}%")
    print(f"\n  Age trend slope: {delta_analysis['age_trend_slope']:.4f}")
    print(f"    (negative = formula underestimates for older patients)")
    
    # =========================================================================
    # 8. Visualizations
    # =========================================================================
    print("\n" + "-" * 70)
    print("8. VISUALIZATIONS")
    print("-" * 70)
    
    # 8.1 Pareto Front
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.scatter(results_df['complexity'], results_df['test_r2'], 
                alpha=0.6, s=60, c='steelblue', label='All equations')
    ax1.scatter([best_row['complexity']], [best_row['test_r2']], 
                s=200, c='gold', marker='*', edgecolors='black', 
                linewidths=1.5, zorder=5, label='Selected best')
    ax1.axhline(y=baseline_metrics['r2'], color='green', linestyle='--', 
                linewidth=2, label=f'Baseline: R²={baseline_metrics["r2"]:.3f}')
    ax1.set_xlabel('Formula Complexity (nodes)', fontsize=11)
    ax1.set_ylabel('Test R²', fontsize=11)
    ax1.set_title('Complexity vs R²', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.scatter(results_df['complexity'], results_df['test_mae'], 
                alpha=0.6, s=60, c='steelblue', label='All equations')
    ax2.scatter([best_row['complexity']], [best_row['test_mae']], 
                s=200, c='gold', marker='*', edgecolors='black', 
                linewidths=1.5, zorder=5, label='Selected best')
    ax2.axhline(y=baseline_metrics['mae'], color='green', linestyle='--', 
                linewidth=2, label=f'Baseline: MAE={baseline_metrics["mae"]:.1f}yr')
    ax2.set_xlabel('Formula Complexity (nodes)', fontsize=11)
    ax2.set_ylabel('Test MAE (years)', fontsize=11)
    ax2.set_title('Complexity vs MAE', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Lower MAE is better
    
    plt.suptitle('Kepler-ECG Stream B: Cardiac Age - Pareto Analysis', fontsize=13)
    plt.tight_layout()
    
    pareto_path = output_dir / "pareto_front_age.png"
    plt.savefig(pareto_path, dpi=150, bbox_inches='tight')
    print(f"Pareto plot saved: {pareto_path}")
    plt.close()
    
    # 8.2 Prediction Scatter Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # SR Formula predictions
    ax1 = axes[0]
    ax1.scatter(y_test, y_test_pred_best, alpha=0.3, s=20, c='steelblue')
    
    # Perfect prediction line
    min_age, max_age = y_test.min(), y_test.max()
    ax1.plot([min_age, max_age], [min_age, max_age], 'r--', linewidth=2, label='Perfect prediction')
    
    # Add regression line
    z = np.polyfit(y_test, y_test_pred_best, 1)
    p = np.poly1d(z)
    ax1.plot([min_age, max_age], p([min_age, max_age]), 'g-', linewidth=2, 
             label=f'Fit: y={z[0]:.2f}x+{z[1]:.1f}')
    
    ax1.set_xlabel('Actual Age (years)', fontsize=11)
    ax1.set_ylabel('Predicted Age (years)', fontsize=11)
    ax1.set_title(f'SR Formula: R²={best_row["test_r2"]:.3f}, MAE={best_row["test_mae"]:.1f}yr', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Baseline predictions
    ax2 = axes[1]
    ax2.scatter(y_test, y_pred_lr, alpha=0.3, s=20, c='steelblue')
    ax2.plot([min_age, max_age], [min_age, max_age], 'r--', linewidth=2, label='Perfect prediction')
    
    z2 = np.polyfit(y_test, y_pred_lr, 1)
    p2 = np.poly1d(z2)
    ax2.plot([min_age, max_age], p2([min_age, max_age]), 'g-', linewidth=2,
             label=f'Fit: y={z2[0]:.2f}x+{z2[1]:.1f}')
    
    ax2.set_xlabel('Actual Age (years)', fontsize=11)
    ax2.set_ylabel('Predicted Age (years)', fontsize=11)
    ax2.set_title(f'Linear Regression: R²={baseline_lr["r2"]:.3f}, MAE={baseline_lr["mae"]:.1f}yr', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.suptitle('Kepler-ECG Stream B: Actual vs Predicted Cardiac Age', fontsize=13)
    plt.tight_layout()
    
    scatter_path = output_dir / "age_prediction_scatter.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    print(f"Scatter plot saved: {scatter_path}")
    plt.close()
    
    # =========================================================================
    # 9. Save Results
    # =========================================================================
    print("\n" + "-" * 70)
    print("9. SAVING RESULTS")
    print("-" * 70)
    
    # Save all equations
    equations_path = output_dir / "cardiac_age_equations.csv"
    results_df.to_csv(equations_path, index=False)
    print(f"Equations saved: {equations_path}")
    
    # Save CV results
    cv_path = output_dir / "cardiac_age_cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"CV results saved: {cv_path}")
    
    # Save best equation details
    best_result = {
        "task": "Stream B: Cardiac Age Regression",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
        
        "dataset": {
            "path": CONFIG["data_path"],
            "n_samples": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(feature_cols),
            "features": feature_cols,
            "target_stats": {
                "min": float(y.min()),
                "max": float(y.max()),
                "mean": float(y.mean()),
                "std": float(y.std()),
            }
        },
        
        "baseline": {
            "model": baseline_name,
            "r2": float(baseline_metrics['r2']),
            "mae": float(baseline_metrics['mae']),
            "rmse": float(baseline_metrics['rmse']),
        },
        
        "best_equation": {
            "formula_sympy": str(best_formula),
            "formula_raw": str(best_row['equation']),
            "complexity": int(best_row['complexity']),
            "test_r2": float(best_row['test_r2']),
            "test_mae": float(best_row['test_mae']),
            "test_rmse": float(best_row['test_rmse']),
            "test_mean_error": float(best_row['test_mean_error']),
            "test_std_error": float(best_row['test_std_error']),
            "selection_reason": selection_reason,
        },
        
        "cardiac_age_delta": delta_analysis,
        
        "cross_validation": {
            "n_folds": CONFIG["cv_folds"],
            "best_equation_cv": {
                k: float(v) for k, v in cv_df[cv_df['index']==best_idx].iloc[0].items()
                if k not in ['index', 'equation', 'complexity'] and pd.notna(v)
            } if best_idx in cv_df['index'].values else None,
        },
        
        "config": CONFIG,
        "total_equations": len(results_df),
    }
    
    best_path = output_dir / "cardiac_age_best.json"
    with open(best_path, 'w') as f:
        json.dump(best_result, f, indent=2, default=str)
    print(f"Best equation saved: {best_path}")
    
    # =========================================================================
    # 10. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TASK 3 COMPLETE - SUMMARY")
    print("=" * 70)
    
    r2_diff = (best_row['test_r2'] - baseline_metrics['r2']) / baseline_metrics['r2'] * 100
    mae_diff = (best_row['test_mae'] - baseline_metrics['mae']) / baseline_metrics['mae'] * 100
    
    print(f"""
Dataset:         {CONFIG['data_path']}
Samples:         {len(df)} ({len(X_train)} train, {len(X_test)} test)
Features:        {len(feature_cols)}
Age Range:       {y.min():.0f} - {y.max():.0f} years

BASELINE ({baseline_name}):
  R²:            {baseline_metrics['r2']:.4f}
  MAE:           {baseline_metrics['mae']:.2f} years
  RMSE:          {baseline_metrics['rmse']:.2f} years

BEST SR FORMULA:
  {best_formula}
  
  Complexity:    {best_row['complexity']} nodes
  
  R²:            {best_row['test_r2']:.4f} ({r2_diff:+.2f}% vs baseline)
  MAE:           {best_row['test_mae']:.2f} years ({mae_diff:+.2f}% vs baseline)
  RMSE:          {best_row['test_rmse']:.2f} years

CARDIAC AGE DELTA:
  Mean:          {delta_analysis['mean_delta']:+.2f} years
  Older heart:   {delta_analysis['pct_older']:.1f}% of patients
  Younger heart: {delta_analysis['pct_younger']:.1f}% of patients

Output files:
  - {equations_path}
  - {cv_path}
  - {best_path}
  - {pareto_path}
  - {scatter_path}
  - {delta_analysis['delta_plot']}
""")
    
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return results_df, cv_df, model


if __name__ == "__main__":
    results_df, cv_df, model = main()