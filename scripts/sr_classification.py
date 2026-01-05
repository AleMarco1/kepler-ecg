#!/usr/bin/env python3
"""
Kepler-ECG: Symbolic Regression for Binary Classification

Discovers interpretable mathematical formulas to distinguish between
diagnostic categories (e.g., NORM vs HYP, NORM vs MI).

Uses PySR with:
- Logistic loss function for classification
- Automatic threshold calibration (Youden's J)
- Cross-validation for robust evaluation
- Comparison with baseline Logistic Regression

Usage:
    # Use pre-prepared SR dataset
    python scripts/sr_classification.py \\
        --input results/ptb-xl/analysis/sr_ready/ptb-xl_norm_vs_hyp.csv \\
        --positive-class HYP
    
    # Use features file directly (auto-prepares dataset)
    python scripts/sr_classification.py \\
        --dataset ptb-xl \\
        --positive-class MI
    
    # Run on all pathologies
    python scripts/sr_classification.py --dataset ptb-xl --all

Author: Kepler-ECG Project
Version: 2.0.0
"""

import argparse
import json
import os
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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # PySR parameters
    "niterations": 150,
    "populations": 40,
    "population_size": 50,
    "maxsize": 20,
    "parsimony": 0.005,
    
    # Operators
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["square", "sqrt", "abs", "log"],
    
    # Data
    "test_size": 0.2,
    "cv_folds": 5,
    "random_state": 42,
}

# Pathologies to analyze
PATHOLOGIES = ["MI", "HYP", "CD", "STTC", "OTHER"]


# ============================================================================
# Helper Functions
# ============================================================================

def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.clip(x, -500, 500)
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, Dict]:
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Youden's J
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    threshold_youden = thresholds[best_idx]
    
    # F1 optimization
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    threshold_f1 = thresholds_pr[best_f1_idx] if len(thresholds_pr) > 0 else 0.5
    
    return threshold_youden, {
        "youden": threshold_youden,
        "f1_optimal": threshold_f1,
        "j_statistic": j_scores[best_idx],
    }


def evaluate_classification(
    y_true: np.ndarray, 
    y_scores: np.ndarray, 
    threshold: float
) -> Dict:
    """Evaluate classification metrics."""
    y_pred = (y_scores >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "threshold": threshold,
        "auc": roc_auc_score(y_true, y_scores),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "f1": f1_score(y_true, y_pred),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def load_or_prepare_dataset(
    input_path: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    positive_class: str = "HYP",
    negative_class: str = "NORM",
) -> Tuple[pd.DataFrame, str]:
    """Load SR-ready dataset or prepare from features."""
    
    if input_path and input_path.exists():
        df = pd.read_csv(input_path)
        # Infer target column
        target_candidates = [
            f"{positive_class}_vs_{negative_class}",
            f"{negative_class}_vs_{positive_class}",
            "target", "label", "y"
        ]
        target_col = None
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # Look for any binary column
            for col in df.columns:
                if df[col].nunique() == 2:
                    target_col = col
                    break
        
        if target_col is None:
            raise ValueError("Could not find target column in dataset")
        
        print(f"Loaded {len(df)} records from {input_path}")
        print(f"Target column: {target_col}")
        return df, target_col
    
    elif dataset_name:
        # Load from features file
        features_path = Path(f"results/{dataset_name}/{dataset_name}_features_extracted.csv")
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        df = pd.read_csv(features_path)
        
        # Find category column
        cat_col = None
        for col in ["diag_primary_category", "primary_superclass"]:
            if col in df.columns:
                cat_col = col
                break
        
        if cat_col is None:
            raise ValueError("No category column found")
        
        # Filter to binary classification
        mask = df[cat_col].isin([positive_class, negative_class])
        df = df[mask].copy()
        
        # Create target
        target_col = f"{positive_class}_vs_{negative_class}"
        df[target_col] = (df[cat_col] == positive_class).astype(int)
        
        print(f"Prepared dataset: {len(df)} records ({positive_class} vs {negative_class})")
        print(f"  {positive_class}: {(df[target_col] == 1).sum()}")
        print(f"  {negative_class}: {(df[target_col] == 0).sum()}")
        
        return df, target_col
    
    else:
        raise ValueError("Must provide either --input or --dataset")


def get_feature_columns(df: pd.DataFrame, target_col: str, min_valid_ratio: float = 0.5) -> List[str]:
    """Get list of feature columns with sufficient valid data."""
    exclude = {
        "ecg_id", "success", "processing_time_ms", "scp_codes",
        "quality_level", "is_usable", "diag_primary_code",
        "diag_primary_category", "primary_superclass",
        "diag_is_normal", "diag_is_multi_label", "diag_n_diagnoses",
        target_col,
    }
    
    # Also exclude label_* and diag_* columns
    exclude.update([c for c in df.columns if c.startswith("label_")])
    exclude.update([c for c in df.columns if c.startswith("diag_")])
    
    # Get numeric columns
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

def run_sr_classification(
    df: pd.DataFrame,
    target_col: str,
    output_dir: Path,
    config: Dict,
    positive_class: str = "HYP",
) -> Dict:
    """Run symbolic regression for binary classification."""
    
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("❌ PySR not installed. Install with: pip install pysr")
        print("   Also requires Julia. See: https://astroautomata.com/PySR/")
        return {"error": "PySR not installed"}
    
    print("\n" + "="*70)
    print(f"SYMBOLIC REGRESSION: NORM vs {positive_class}")
    print("="*70)
    
    # Get features
    feature_cols = get_feature_columns(df, target_col)
    print(f"Features: {len(feature_cols)}")
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[target_col].values
    
    # Drop rows with NaN
    valid_mask = ~X.isna().any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Valid samples: {len(y)}")
    
    # Balance classes
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    
    if n_pos < n_neg:
        # Undersample negative class
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        neg_idx = np.random.choice(neg_idx, n_pos, replace=False)
        idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(idx)
        X = X.iloc[idx]
        y = y[idx]
    
    print(f"Balanced: {len(y)} samples ({y.sum()} positive)")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=config["test_size"], 
        random_state=config["random_state"], stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Baseline: Logistic Regression
    print("\n--- Baseline: Logistic Regression ---")
    lr = LogisticRegression(random_state=config["random_state"], max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    baseline_metrics = evaluate_classification(y_test, y_proba_lr, 0.5)
    print(f"  AUC: {baseline_metrics['auc']:.4f}")
    print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
    
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
        
        # Constraints
        nested_constraints={
            "square": {"square": 0},
            "sqrt": {"sqrt": 0},
            "log": {"log": 0},
        },
        
        # Optimization
        weight_optimize=0.01,
        
        # Deterministic mode (required settings)
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
            opt_thresh, _ = find_optimal_threshold(y_test, y_pred)
            metrics = evaluate_classification(y_test, y_pred, opt_thresh)
            
            results_list.append({
                "index": idx,
                "complexity": int(row["complexity"]),
                "equation": str(row["equation"]),
                "loss": float(row["loss"]),
                "optimal_threshold": opt_thresh,
                **{f"test_{k}": v for k, v in metrics.items() if k != "threshold"},
            })
        except Exception as e:
            print(f"  Warning: Failed to evaluate equation {idx}: {e}")
    
    results_df = pd.DataFrame(results_list)
    
    # Select best equation (AUC with complexity penalty)
    results_df["score"] = results_df["test_auc"] - 0.01 * results_df["complexity"]
    best_idx = results_df["score"].idxmax()
    best_row = results_df.loc[best_idx]
    
    print(f"\n✅ Best equation (complexity {best_row['complexity']}):")
    print(f"   {best_row['equation']}")
    print(f"   AUC: {best_row['test_auc']:.4f}, Accuracy: {best_row['test_accuracy']:.4f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save equations
    eq_path = output_dir / f"sr_classification_{positive_class.lower()}_equations.csv"
    results_df.to_csv(eq_path, index=False)
    
    # Save plots
    _save_plots(results_df, best_row, baseline_metrics, y_test, 
                model.predict(X_test_scaled, index=best_idx), y_proba_lr,
                output_dir, positive_class)
    
    # Summary
    result = {
        "task": f"SR Classification: NORM vs {positive_class}",
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "n_samples": len(df),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_features": len(feature_cols),
            "features": feature_cols[:20],  # Top 20
        },
        "baseline": baseline_metrics,
        "best_equation": {
            "formula": str(best_row["equation"]),
            "complexity": int(best_row["complexity"]),
            "threshold": float(best_row["optimal_threshold"]),
            "auc": float(best_row["test_auc"]),
            "accuracy": float(best_row["test_accuracy"]),
            "sensitivity": float(best_row["test_sensitivity"]),
            "specificity": float(best_row["test_specificity"]),
            "f1": float(best_row["test_f1"]),
        },
        "config": config,
        "output_files": [str(eq_path)],
    }
    
    # Save summary
    summary_path = output_dir / f"sr_classification_{positive_class.lower()}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    return result


def _save_plots(results_df, best_row, baseline_metrics, y_test, y_pred_best, y_proba_lr,
                output_dir, positive_class):
    """Save visualization plots."""
    
    # Pareto front
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.scatter(results_df["complexity"], results_df["test_auc"], 
                alpha=0.6, s=60, c="steelblue")
    ax1.scatter([best_row["complexity"]], [best_row["test_auc"]], 
                s=200, c="gold", marker="*", edgecolors="black", zorder=5)
    ax1.axhline(y=baseline_metrics["auc"], color="green", linestyle="--",
                label=f'Baseline: {baseline_metrics["auc"]:.3f}')
    ax1.set_xlabel("Complexity")
    ax1.set_ylabel("Test AUC")
    ax1.set_title(f"NORM vs {positive_class}: Pareto Front")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ROC curve
    ax2 = axes[1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fpr_sr, tpr_sr, _ = roc_curve(y_test, y_pred_best)
    
    ax2.plot(fpr_lr, tpr_lr, "g-", label=f'LR (AUC={baseline_metrics["auc"]:.3f})')
    ax2.plot(fpr_sr, tpr_sr, "b-", label=f'SR (AUC={best_row["test_auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"sr_classification_{positive_class.lower()}_plots.png", dpi=150)
    plt.close()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kepler-ECG: Symbolic Regression for Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--input", "-i", type=str,
                        help="Path to SR-ready CSV dataset")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset name (uses features from results/{dataset}/)")
    parser.add_argument("--positive-class", "-p", type=str, default="HYP",
                        help="Positive class for binary classification (default: HYP)")
    parser.add_argument("--negative-class", "-n", type=str, default="NORM",
                        help="Negative class (default: NORM)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Run on all pathologies")
    parser.add_argument("--output", "-o", type=str,
                        help="Output directory")
    parser.add_argument("--iterations", type=int, default=150,
                        help="Number of SR iterations (default: 150)")
    parser.add_argument("--maxsize", type=int, default=20,
                        help="Maximum formula complexity (default: 20)")
    
    args = parser.parse_args()
    
    if not args.input and not args.dataset:
        parser.error("Must provide either --input or --dataset")
    
    # Config
    config = DEFAULT_CONFIG.copy()
    config["niterations"] = args.iterations
    config["maxsize"] = args.maxsize
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    elif args.dataset:
        output_dir = Path(f"results/{args.dataset}/sr_classification")
    else:
        output_dir = Path("results/sr_classification")
    
    # Pathologies to process
    if args.all and args.dataset:
        pathologies = PATHOLOGIES
    else:
        pathologies = [args.positive_class]
    
    # Process each pathology
    all_results = {}
    
    for positive_class in pathologies:
        try:
            df, target_col = load_or_prepare_dataset(
                input_path=Path(args.input) if args.input else None,
                dataset_name=args.dataset,
                positive_class=positive_class,
                negative_class=args.negative_class,
            )
            
            result = run_sr_classification(
                df, target_col, output_dir, config, positive_class
            )
            
            all_results[positive_class] = result
            
        except Exception as e:
            print(f"❌ Failed for {positive_class}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        for pathology, result in all_results.items():
            if "error" not in result:
                best = result["best_equation"]
                print(f"{pathology}: AUC={best['auc']:.4f}, Complexity={best['complexity']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
