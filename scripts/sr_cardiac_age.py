#!/usr/bin/env python3
"""
Kepler-ECG: Symbolic Regression for Cardiac Age Prediction

Discovers interpretable formulas to predict biological cardiac age from ECG features.

Key concept: If predicted_age > actual_age, the heart appears "older" than expected,
potentially indicating compromised cardiovascular health.

Usage:
    # Using dataset with age column
    python scripts/sr_cardiac_age.py --dataset ptb-xl
    
    # Custom input
    python scripts/sr_cardiac_age.py --input path/to/features.csv --age-col age
    
    # Only NORM subjects (healthier baseline)
    python scripts/sr_cardiac_age.py --dataset ptb-xl --norm-only

Author: Kepler-ECG Project
Version: 2.0.0
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # PySR parameters
    "niterations": 150,
    "populations": 40,
    "population_size": 50,
    "maxsize": 25,
    "parsimony": 0.003,
    
    # Operators
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["square", "sqrt", "abs", "log", "exp"],
    
    # Data
    "test_size": 0.2,
    "cv_folds": 5,
    "random_state": 42,
    
    # Age constraints
    "min_age": 18,
    "max_age": 100,
}


# ============================================================================
# Helper Functions
# ============================================================================

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute regression metrics."""
    # Clip to reasonable age range
    y_pred = np.clip(y_pred, 0, 120)
    
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mse": mean_squared_error(y_true, y_pred),
        "mean_error": np.mean(y_pred - y_true),  # Bias
        "std_error": np.std(y_pred - y_true),    # Precision
    }


def analyze_cardiac_age_delta(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    output_dir: Path,
) -> Dict:
    """
    Analyze cardiac age delta (predicted - actual).
    
    Positive delta = heart appears older
    Negative delta = heart appears younger
    """
    delta = y_pred - y_true
    
    analysis = {
        "mean_delta": float(np.mean(delta)),
        "std_delta": float(np.std(delta)),
        "median_delta": float(np.median(delta)),
        "pct_older": float(np.mean(delta > 0) * 100),
        "pct_younger": float(np.mean(delta < 0) * 100),
        "percentiles": {
            "p10": float(np.percentile(delta, 10)),
            "p25": float(np.percentile(delta, 25)),
            "p50": float(np.percentile(delta, 50)),
            "p75": float(np.percentile(delta, 75)),
            "p90": float(np.percentile(delta, 90)),
        }
    }
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(delta, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=np.mean(delta), color='green', linewidth=2,
                label=f'Mean: {np.mean(delta):.1f} years')
    ax1.set_xlabel('Cardiac Age Delta (years)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Cardiac Age Delta')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Delta vs actual age
    ax2 = axes[1]
    ax2.scatter(y_true, delta, alpha=0.3, s=10, c='steelblue')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # Trend line
    z = np.polyfit(y_true, delta, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    ax2.plot(x_line, p(x_line), 'g-', linewidth=2,
             label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')
    
    ax2.set_xlabel('Actual Age (years)')
    ax2.set_ylabel('Cardiac Age Delta (years)')
    ax2.set_title('Cardiac Age Delta vs Actual Age')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "cardiac_age_delta.png", dpi=150)
    plt.close()
    
    analysis["age_trend_slope"] = float(z[0])
    
    return analysis


def load_dataset(
    input_path: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    age_col: str = "age",
    norm_only: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Load dataset with age column."""
    
    if input_path and input_path.exists():
        df = pd.read_csv(input_path)
    elif dataset_name:
        features_path = Path(f"results/{dataset_name}/{dataset_name}_features_extracted.csv")
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        df = pd.read_csv(features_path)
    else:
        raise ValueError("Must provide either --input or --dataset")
    
    # Check age column
    if age_col not in df.columns:
        raise ValueError(f"Age column '{age_col}' not found in dataset")
    
    print(f"Loaded {len(df)} records")
    
    # Filter to NORM only if requested
    if norm_only:
        cat_col = None
        for col in ["diag_primary_category", "primary_superclass"]:
            if col in df.columns:
                cat_col = col
                break
        
        if cat_col:
            df = df[df[cat_col] == "NORM"]
            print(f"Filtered to NORM: {len(df)} records")
    
    # Filter valid age range
    df = df[(df[age_col] >= 18) & (df[age_col] <= 100)]
    print(f"Valid age range: {len(df)} records")
    
    return df, age_col


def get_feature_columns(df: pd.DataFrame, age_col: str, min_valid_ratio: float = 0.5) -> List[str]:
    """Get feature columns excluding metadata, with sufficient valid data."""
    exclude = {
        "ecg_id", "success", "processing_time_ms", "scp_codes",
        "quality_level", "is_usable", "diag_primary_code",
        "diag_primary_category", "primary_superclass",
        "diag_is_normal", "diag_is_multi_label", "diag_n_diagnoses",
        age_col, "sex",
    }
    exclude.update([c for c in df.columns if c.startswith("label_")])
    exclude.update([c for c in df.columns if c.startswith("diag_")])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter to columns with sufficient valid data
    valid_cols = []
    n_rows = len(df)
    for col in numeric_cols:
        if col in exclude:
            continue
        valid_ratio = df[col].notna().sum() / n_rows
        if valid_ratio >= min_valid_ratio:
            valid_cols.append(col)
    
    return valid_cols


# ============================================================================
# Main SR Pipeline
# ============================================================================

def run_sr_cardiac_age(
    df: pd.DataFrame,
    age_col: str,
    output_dir: Path,
    config: Dict,
) -> Dict:
    """Run symbolic regression for cardiac age prediction."""
    
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("❌ PySR not installed. Install with: pip install pysr")
        return {"error": "PySR not installed"}
    
    print("\n" + "="*70)
    print("SYMBOLIC REGRESSION: CARDIAC AGE PREDICTION")
    print("="*70)
    
    # Get features
    feature_cols = get_feature_columns(df, age_col)
    print(f"Features: {len(feature_cols)}")
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[age_col].values
    
    # Drop NaN
    valid_mask = ~X.isna().any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Valid samples: {len(y)}")
    print(f"Age range: {y.min():.0f} - {y.max():.0f} years (mean: {y.mean():.1f})")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=config["test_size"],
        random_state=config["random_state"]
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Baseline: Linear Regression
    print("\n--- Baseline: Linear Regression ---")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    baseline_metrics = evaluate_regression(y_test, y_pred_lr)
    print(f"  R²: {baseline_metrics['r2']:.4f}")
    print(f"  MAE: {baseline_metrics['mae']:.2f} years")
    
    # Ridge for comparison
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    ridge_metrics = evaluate_regression(y_test, y_pred_ridge)
    
    # Use best baseline
    if ridge_metrics["r2"] > baseline_metrics["r2"]:
        baseline_metrics = ridge_metrics
        baseline_name = "Ridge"
        y_pred_baseline = y_pred_ridge
    else:
        baseline_name = "LinearRegression"
        y_pred_baseline = y_pred_lr
    
    print(f"  Best baseline: {baseline_name}")
    
    # PySR
    print("\n--- Symbolic Regression ---")
    print(f"Iterations: {config['niterations']}, Max size: {config['maxsize']}")
    
    model = PySRRegressor(
        niterations=config["niterations"],
        populations=config["populations"],
        population_size=config["population_size"],
        maxsize=config["maxsize"],
        parsimony=config["parsimony"],
        binary_operators=config["binary_operators"],
        unary_operators=config["unary_operators"],
        
        nested_constraints={
            "square": {"square": 0},
            "sqrt": {"sqrt": 0},
            "log": {"log": 0, "exp": 0},
            "exp": {"exp": 0, "log": 0},
        },
        
        weight_optimize=0.01,
        
        # Deterministic mode
        deterministic=True,
        random_state=config["random_state"],
        procs=0,
        multithreading=False,
        
        verbosity=1,
        progress=True,
    )
    
    model.fit(X_train_scaled, y_train, variable_names=feature_cols)
    
    # Evaluate all equations
    print("\n--- Evaluating Pareto Front ---")
    equations = model.equations_
    
    results_list = []
    for idx, row in equations.iterrows():
        try:
            y_pred = model.predict(X_test_scaled, index=idx)
            y_pred = np.clip(y_pred, 0, 120)
            metrics = evaluate_regression(y_test, y_pred)
            
            results_list.append({
                "index": idx,
                "complexity": int(row["complexity"]),
                "equation": str(row["equation"]),
                "loss": float(row["loss"]),
                **{f"test_{k}": v for k, v in metrics.items()},
            })
        except Exception as e:
            print(f"  Warning: Failed equation {idx}: {e}")
    
    results_df = pd.DataFrame(results_list)
    
    # Select best (R² with complexity penalty)
    results_df["score"] = results_df["test_r2"] - 0.005 * results_df["complexity"]
    best_idx = results_df["score"].idxmax()
    best_row = results_df.loc[best_idx]
    
    print(f"\n✅ Best equation (complexity {best_row['complexity']}):")
    print(f"   {best_row['equation']}")
    print(f"   R²: {best_row['test_r2']:.4f}, MAE: {best_row['test_mae']:.2f} years")
    
    # Get predictions for best equation
    y_pred_best = model.predict(X_test_scaled, index=best_idx)
    y_pred_best = np.clip(y_pred_best, 0, 120)
    
    # Analyze cardiac age delta
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_analysis = analyze_cardiac_age_delta(y_test, y_pred_best, output_dir)
    
    # Save equations
    eq_path = output_dir / "sr_cardiac_age_equations.csv"
    results_df.to_csv(eq_path, index=False)
    
    # Save plots
    _save_plots(results_df, best_row, baseline_metrics, y_test,
                y_pred_best, y_pred_baseline, output_dir)
    
    # Summary
    result = {
        "task": "SR Cardiac Age Prediction",
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "n_samples": len(df),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_features": len(feature_cols),
            "age_range": {"min": float(y.min()), "max": float(y.max()), "mean": float(y.mean())},
        },
        "baseline": {
            "model": baseline_name,
            **baseline_metrics,
        },
        "best_equation": {
            "formula": str(best_row["equation"]),
            "complexity": int(best_row["complexity"]),
            "r2": float(best_row["test_r2"]),
            "mae": float(best_row["test_mae"]),
            "rmse": float(best_row["test_rmse"]),
        },
        "cardiac_age_delta": delta_analysis,
        "config": config,
    }
    
    summary_path = output_dir / "sr_cardiac_age_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    return result


def _save_plots(results_df, best_row, baseline_metrics, y_test,
                y_pred_best, y_pred_baseline, output_dir):
    """Save visualization plots."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pareto front
    ax1 = axes[0]
    ax1.scatter(results_df["complexity"], results_df["test_r2"],
                alpha=0.6, s=60, c="steelblue")
    ax1.scatter([best_row["complexity"]], [best_row["test_r2"]],
                s=200, c="gold", marker="*", edgecolors="black", zorder=5)
    ax1.axhline(y=baseline_metrics["r2"], color="green", linestyle="--",
                label=f'Baseline: {baseline_metrics["r2"]:.3f}')
    ax1.set_xlabel("Complexity")
    ax1.set_ylabel("Test R²")
    ax1.set_title("Cardiac Age: Pareto Front")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Prediction scatter
    ax2 = axes[1]
    ax2.scatter(y_test, y_pred_best, alpha=0.3, s=20, c="steelblue")
    
    min_age, max_age = y_test.min(), y_test.max()
    ax2.plot([min_age, max_age], [min_age, max_age], 'r--', linewidth=2,
             label="Perfect prediction")
    
    z = np.polyfit(y_test, y_pred_best, 1)
    p = np.poly1d(z)
    ax2.plot([min_age, max_age], p([min_age, max_age]), 'g-', linewidth=2,
             label=f'Fit: y={z[0]:.2f}x+{z[1]:.1f}')
    
    ax2.set_xlabel("Actual Age (years)")
    ax2.set_ylabel("Predicted Age (years)")
    ax2.set_title(f'SR: R²={best_row["test_r2"]:.3f}, MAE={best_row["test_mae"]:.1f}yr')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "sr_cardiac_age_plots.png", dpi=150)
    plt.close()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kepler-ECG: Symbolic Regression for Cardiac Age",
    )
    
    parser.add_argument("--input", "-i", type=str,
                        help="Path to features CSV")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset name")
    parser.add_argument("--age-col", type=str, default="age",
                        help="Age column name (default: age)")
    parser.add_argument("--norm-only", action="store_true",
                        help="Use only NORM subjects")
    parser.add_argument("--output", "-o", type=str,
                        help="Output directory")
    parser.add_argument("--iterations", type=int, default=150,
                        help="SR iterations (default: 150)")
    parser.add_argument("--maxsize", type=int, default=25,
                        help="Max complexity (default: 25)")
    
    args = parser.parse_args()
    
    if not args.input and not args.dataset:
        parser.error("Must provide either --input or --dataset")
    
    config = DEFAULT_CONFIG.copy()
    config["niterations"] = args.iterations
    config["maxsize"] = args.maxsize
    
    if args.output:
        output_dir = Path(args.output)
    elif args.dataset:
        output_dir = Path(f"results/{args.dataset}/sr_cardiac_age")
    else:
        output_dir = Path("results/sr_cardiac_age")
    
    try:
        df, age_col = load_dataset(
            input_path=Path(args.input) if args.input else None,
            dataset_name=args.dataset,
            age_col=args.age_col,
            norm_only=args.norm_only,
        )
        
        result = run_sr_cardiac_age(df, age_col, output_dir, config)
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
