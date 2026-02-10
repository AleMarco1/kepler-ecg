#!/usr/bin/env python3
"""
Kepler-ECG: Pooled Symbolic Regression for QTc Discovery

Discovers interpretable QTc correction formulas using PySR on the pooled
dataset combining all 6 international ECG databases:
- PTB-XL, Chapman-Shaoxing, CPSC-2018, Georgia, MIMIC-IV-ECG, CODE-15

Multiple approaches:
1. Direct: QTc = f(QT, RR, features)
2. Factor: QTc = QT * f(RR)
3. Additive: QTc = QT + f(RR)

Goal: Find formulas with near-zero HR correlation that generalize across populations.

Usage:
    python scripts/08_0_sr_qtc_pooled_discovery.py
    
    # Specific approach
    python scripts/08_0_sr_qtc_pooled_discovery.py --approach factor
    
    # Custom complexity and iterations
    python scripts/08_0_sr_qtc_pooled_discovery.py --minsize 3 --maxsize 8 --iterations 150
    
    # Limit samples per dataset (for testing)
    python scripts/08_0_sr_qtc_pooled_discovery.py --samples-per-dataset 5000

Author: Kepler-ECG Project
Version: 1.0.0
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from scipy import stats
from sklearn.model_selection import train_test_split

import io

# Force UTF-8 encoding for stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ============================================================================
# Configuration
# ============================================================================

# All 6 datasets
ALL_DATASETS = [
    'ptb-xl',
    'chapman',
    'cpsc-2018',
    'georgia',
    'mimic-iv-ecg',
    'code-15',
]

# PySR Configuration (Anti-overfit settings)
PYSR_CONFIG = {
    # Evolution parameters
    "niterations": 100,          # Balanced iterations
    "populations": 20,           # Moderate populations
    "population_size": 33,       # Moderate size
    
    # Formula constraints - KEY FOR GENERALIZATION
    "maxsize": 7,                # Simple formulas
    "minsize": 3,                # Allow very simple formulas
    "parsimony": 0.05,           # Penalize complexity
    
    # Optimization
    "weight_optimize": 0.01,     # Less constant optimization
    "adaptive_parsimony_scaling": 1000.0,  # Aggressive scaling
    
    # Reproducibility
    "deterministic": True,
    "random_state": 42,
    "procs": 0,
    "multithreading": False,
    
    # Output
    "verbosity": 1,
    "progress": True,
}

# Operator configurations
OPERATORS_STANDARD = {
    "binary_operators": ["+", "-", "*", "/", "^"],
    "unary_operators": ["sqrt", "cbrt", "inv", "square", "cube"],
    
    "nested_constraints": {
        "sqrt": {"sqrt": 1, "cbrt": 1, "inv": 1},
        "cbrt": {"sqrt": 1, "cbrt": 1, "inv": 1},
        "inv": {"inv": 0},
        "square": {"square": 0, "cube": 0},
        "cube": {"square": 0, "cube": 0},
    },
    
    "complexity_of_operators": {
        "+": 1, "-": 1, "*": 1, "/": 1, "^": 2,
        "sqrt": 2, "cbrt": 2, "inv": 2, "square": 2, "cube": 2,
    },
}

OPERATORS_EXTENDED = {
    "binary_operators": ["+", "-", "*", "/", "^"],
    "unary_operators": ["sqrt", "cbrt", "inv", "square", "cube", "log", "exp"],
    
    "nested_constraints": {
        "sqrt": {"sqrt": 1, "cbrt": 1, "inv": 1},
        "cbrt": {"sqrt": 1, "cbrt": 1, "inv": 1},
        "inv": {"inv": 0},
        "square": {"square": 0, "cube": 0},
        "cube": {"square": 0, "cube": 0},
        "log": {"log": 0, "exp": 0},
        "exp": {"exp": 0, "log": 0},
    },
    
    "complexity_of_operators": {
        "+": 1, "-": 1, "*": 1, "/": 1, "^": 2,
        "sqrt": 2, "cbrt": 2, "inv": 2, "square": 2, "cube": 2,
        "log": 3, "exp": 3,  # Higher cost = less interpretable
    },
}

# Data split configuration
DATA_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
}

# Feature sets
FEATURE_SETS = {
    "minimal": ['QT_interval_ms', 'RR_interval_sec'],
    "standard": ['QT_interval_ms', 'RR_interval_sec', 'heart_rate_bpm'],
    "extended": ['QT_interval_ms', 'RR_interval_sec', 'heart_rate_bpm',
                 'QRS_duration_ms', 'PR_interval_ms'],
}

# Required columns for validation
REQUIRED_COLUMNS = ['QT_interval_ms', 'RR_interval_sec', 'QTc_reference_ms', 'heart_rate_bpm']


# ============================================================================
# Pooling Functions
# ============================================================================

def load_and_pool_datasets(
    datasets: List[str],
    results_dir: Path,
    samples_per_dataset: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load and pool QTc preparation data from multiple datasets.
    
    Returns:
        Tuple of (pooled_df, dataset_counts)
    """
    print("\n" + "="*70)
    print("LOADING AND POOLING DATASETS")
    print("="*70)
    
    all_dfs = []
    dataset_counts = {}
    
    for dataset in datasets:
        input_path = results_dir / dataset / 'qtc' / f'{dataset}_qtc_preparation.csv'
        
        if not input_path.exists():
            print(f"⚠️  {dataset}: File not found, skipping")
            continue
        
        df = pd.read_csv(input_path)
        original_count = len(df)
        
        # Add dataset source column
        df['source_dataset'] = dataset
        
        # Sample if requested
        if samples_per_dataset is not None and samples_per_dataset < len(df):
            np.random.seed(random_state)
            df = df.sample(n=samples_per_dataset, random_state=random_state)
            print(f"✅ {dataset}: {original_count:,} → {len(df):,} (sampled)")
        else:
            print(f"✅ {dataset}: {len(df):,} records")
        
        all_dfs.append(df)
        dataset_counts[dataset] = len(df)
    
    if not all_dfs:
        raise ValueError("No datasets loaded successfully")
    
    # Concatenate all
    pooled_df = pd.concat(all_dfs, ignore_index=True)
    
    print("-"*50)
    print(f"📊 POOLED TOTAL: {len(pooled_df):,} records from {len(dataset_counts)} datasets")
    
    return pooled_df, dataset_counts


# ============================================================================
# SR Pipeline
# ============================================================================

def run_symbolic_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    approach: str,
    config: Dict,
    dataset_name: str = "pool6",
) -> Tuple[Dict, object, pd.DataFrame]:
    """
    Run PySR for QTc discovery.
    
    Returns:
        Tuple of (result_dict, model, equations_df)
    """
    
    try:
        from pysr import PySRRegressor
    except ImportError:
        return {"error": "PySR not installed. Install with: pip install pysr"}, None, None
    
    print(f"\n{'='*50}")
    print(f"SR Approach: {approach.upper()}")
    print(f"{'='*50}")
    
    # Prepare target based on approach
    qt_idx = feature_names.index('QT_interval_ms')
    qt = X[:, qt_idx]
    
    if approach == 'direct':
        y_target = y
        target_name = 'QTc'
    elif approach == 'factor':
        y_target = y / qt
        target_name = 'QTc/QT'
    elif approach == 'additive':
        y_target = y - qt
        target_name = 'QTc-QT'
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    print(f"Target: {target_name}")
    print(f"Range: [{y_target.min():.2f}, {y_target.max():.2f}]")
    
    # Build PySR model
    model = PySRRegressor(
        niterations=config["niterations"],
        populations=config["populations"],
        population_size=config["population_size"],
        maxsize=config["maxsize"],
        parsimony=config["parsimony"],
        
        binary_operators=config["binary_operators"],
        unary_operators=config["unary_operators"],
        
        nested_constraints=config["nested_constraints"],
        complexity_of_operators=config["complexity_of_operators"],
        
        weight_optimize=config["weight_optimize"],
        adaptive_parsimony_scaling=config["adaptive_parsimony_scaling"],
        
        deterministic=config["deterministic"],
        random_state=config["random_state"],
        procs=config["procs"],
        multithreading=config["multithreading"],
        
        verbosity=config["verbosity"],
        progress=config["progress"],
    )
    
    print("\nFitting on pooled data...")
    model.fit(X, y_target, variable_names=feature_names)
    
    # Get equations dataframe
    equations = model.equations_.copy()
    equations['approach'] = approach
    
    # Find best by minimum loss
    best_idx = equations['loss'].idxmin()
    best_eq = equations.loc[best_idx, 'equation']
    
    result = {
        'approach': approach,
        'target': target_name,
        'best_equation': str(best_eq),
        'best_complexity': int(equations.loc[best_idx, 'complexity']),
        'best_loss': float(equations.loc[best_idx, 'loss']),
    }
    
    print(f"\nBest: {best_eq}")
    print(f"Complexity: {result['best_complexity']}, Loss: {result['best_loss']:.6f}")
    
    return result, model, equations


def evaluate_formula_from_equation(
    equation_row: pd.Series,
    X: np.ndarray,
    y_true: np.ndarray,
    hr: np.ndarray,
    qt: np.ndarray,
    rr: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """
    Evaluate a single equation from the pareto front.
    
    Returns metrics dict with r_vs_HR, MAE, RMSE.
    """
    approach = equation_row['approach']
    
    # Get the lambda function from PySR
    try:
        lambda_func = equation_row['lambda_format']
        
        # Build variable dict for evaluation
        var_dict = {}
        for i, name in enumerate(feature_names):
            var_dict[name] = X[:, i]
        
        # Evaluate
        y_pred = lambda_func(X)
        
        # Convert to QTc based on approach
        qt_idx = feature_names.index('QT_interval_ms')
        qt_vals = X[:, qt_idx]
        
        if approach == 'direct':
            qtc = y_pred
        elif approach == 'factor':
            qtc = y_pred * qt_vals
        else:  # additive
            qtc = y_pred + qt_vals
        
        # Clip to reasonable range
        qtc = np.clip(qtc, 200, 700)
        
        # Calculate metrics
        valid = ~(np.isnan(qtc) | np.isnan(y_true) | np.isnan(hr))
        if valid.sum() < 10:
            return None
            
        r_hr, _ = stats.pearsonr(qtc[valid], hr[valid])
        mae = np.mean(np.abs(qtc[valid] - y_true[valid]))
        rmse = np.sqrt(np.mean((qtc[valid] - y_true[valid])**2))
        
        return {
            'r_vs_HR': round(r_hr, 4),
            'abs_r_HR': round(abs(r_hr), 4),
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
        }
    except Exception as e:
        return None


def evaluate_formula(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    hr: np.ndarray,
    qt: np.ndarray,
    rr: np.ndarray,
    approach: str,
    feature_names: List[str],
) -> Tuple[Dict, np.ndarray]:
    """Evaluate discovered formula against standard corrections."""
    
    # Predict with discovered formula
    y_pred = model.predict(X)
    
    # Convert back to QTc based on approach
    qt_idx = feature_names.index('QT_interval_ms')
    qt_vals = X[:, qt_idx]
    
    if approach == 'direct':
        qtc_kepler = y_pred
    elif approach == 'factor':
        qtc_kepler = y_pred * qt_vals
    else:  # additive
        qtc_kepler = y_pred + qt_vals
    
    # Clip to physiologically reasonable range
    qtc_kepler = np.clip(qtc_kepler, 200, 700)
    
    def calc_metrics(qtc, name):
        """Calculate metrics for a QTc formula."""
        valid = ~(np.isnan(qtc) | np.isnan(y_true) | np.isnan(hr))
        r_hr, _ = stats.pearsonr(qtc[valid], hr[valid])
        r_ref, _ = stats.pearsonr(qtc[valid], y_true[valid])
        mae = np.mean(np.abs(qtc[valid] - y_true[valid]))
        rmse = np.sqrt(np.mean((qtc[valid] - y_true[valid])**2))
        
        return {
            'r_vs_HR': round(r_hr, 4),
            'r_vs_ref': round(r_ref, 4),
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'mean': round(np.mean(qtc[valid]), 2),
            'std': round(np.std(qtc[valid]), 2),
        }
    
    # Standard QTc formulas for comparison
    qtc_bazett = qt / np.sqrt(rr)
    qtc_fridericia = qt / np.cbrt(rr)
    qtc_framingham = qt + 154 * (1 - rr)
    
    metrics = {
        'Kepler': calc_metrics(qtc_kepler, 'Kepler'),
        'Bazett': calc_metrics(qtc_bazett, 'Bazett'),
        'Fridericia': calc_metrics(qtc_fridericia, 'Fridericia'),
        'Framingham': calc_metrics(qtc_framingham, 'Framingham'),
    }
    
    # Print comparison table
    print("\n" + "="*60)
    print("FORMULA COMPARISON (Pooled Test Set)")
    print("="*60)
    print(f"{'Formula':<15} {'MAE':>8} {'RMSE':>8} {'r(HR)':>10}")
    print("-"*45)
    for name, m in metrics.items():
        print(f"{name:<15} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['r_vs_HR']:>+10.4f}")
    
    return metrics, qtc_kepler


def evaluate_by_dataset(
    model,
    df_test: pd.DataFrame,
    X_test: np.ndarray,
    approach: str,
    feature_names: List[str],
) -> Dict:
    """Evaluate formula separately on each source dataset."""
    
    print("\n" + "="*60)
    print("PER-DATASET EVALUATION")
    print("="*60)
    
    # Predict
    y_pred = model.predict(X_test)
    qt_idx = feature_names.index('QT_interval_ms')
    qt_vals = X_test[:, qt_idx]
    
    if approach == 'direct':
        qtc_kepler = y_pred
    elif approach == 'factor':
        qtc_kepler = y_pred * qt_vals
    else:
        qtc_kepler = y_pred + qt_vals
    
    qtc_kepler = np.clip(qtc_kepler, 200, 700)
    
    # Also compute Bazett for comparison
    rr_idx = feature_names.index('RR_interval_sec')
    rr_vals = X_test[:, rr_idx]
    qtc_bazett = qt_vals / np.sqrt(rr_vals)
    
    results = {}
    
    print(f"{'Dataset':<15} {'N':>8} {'Kepler r(HR)':>14} {'Bazett r(HR)':>14} {'Improvement':>12}")
    print("-"*70)
    
    for dataset in df_test['source_dataset'].unique():
        mask = df_test['source_dataset'].values == dataset
        n = mask.sum()
        
        if n < 100:
            continue
        
        hr_ds = df_test['heart_rate_bpm'].values[mask]
        qtc_k = qtc_kepler[mask]
        qtc_b = qtc_bazett[mask]
        
        valid = ~(np.isnan(qtc_k) | np.isnan(hr_ds))
        
        if valid.sum() < 50:
            continue
        
        r_kepler, _ = stats.pearsonr(qtc_k[valid], hr_ds[valid])
        r_bazett, _ = stats.pearsonr(qtc_b[valid], hr_ds[valid])
        
        improvement = abs(r_bazett) / abs(r_kepler) if abs(r_kepler) > 0.0001 else float('inf')
        
        results[dataset] = {
            'n': int(n),
            'r_HR_kepler': round(r_kepler, 4),
            'r_HR_bazett': round(r_bazett, 4),
            'improvement': round(improvement, 1),
        }
        
        print(f"{dataset:<15} {n:>8,} {r_kepler:>+14.4f} {r_bazett:>+14.4f} {improvement:>11.1f}x")
    
    return results


def analyze_by_hr_bins(
    qtc_kepler: np.ndarray,
    qt: np.ndarray,
    rr: np.ndarray,
    hr: np.ndarray,
    y_true: np.ndarray,
) -> Dict:
    """Analyze QTc by heart rate bins."""
    
    qtc_bazett = qt / np.sqrt(rr)
    qtc_fridericia = qt / np.cbrt(rr)
    
    bins = [(40, 60), (60, 80), (80, 100), (100, 120), (120, 150)]
    results = {}
    
    for low, high in bins:
        mask = (hr >= low) & (hr < high)
        n = mask.sum()
        
        if n < 10:
            continue
        
        bin_name = f"{low}-{high}"
        results[bin_name] = {
            'n': int(n),
            'Kepler_mean': round(float(np.nanmean(qtc_kepler[mask])), 2),
            'Kepler_std': round(float(np.nanstd(qtc_kepler[mask])), 2),
            'Bazett_mean': round(float(np.nanmean(qtc_bazett[mask])), 2),
            'Fridericia_mean': round(float(np.nanmean(qtc_fridericia[mask])), 2),
        }
    
    return results


def compute_unified_score(abs_r_hr: float, complexity: int, maxsize: int) -> float:
    """
    Compute unified score balancing HR-independence and simplicity.
    
    Lower score = better formula.
    
    Score = |r(HR)| + 0.1 * (complexity / maxsize)
    """
    complexity_penalty = 0.1 * (complexity / maxsize)
    return abs_r_hr + complexity_penalty


def build_unified_equations_df(
    all_equations: List[pd.DataFrame],
    X_test: np.ndarray,
    y_test: np.ndarray,
    hr_test: np.ndarray,
    qt_test: np.ndarray,
    rr_test: np.ndarray,
    feature_names: List[str],
    minsize: int,
    maxsize: int,
) -> pd.DataFrame:
    """
    Build unified DataFrame with all equations from all approaches,
    evaluated and scored.
    """
    
    # Concatenate all equations
    unified = pd.concat(all_equations, ignore_index=True)
    
    # Filter by complexity range
    unified = unified[
        (unified['complexity'] >= minsize) & 
        (unified['complexity'] <= maxsize)
    ].copy()
    
    # Evaluate each equation
    metrics_list = []
    for idx, row in unified.iterrows():
        metrics = evaluate_formula_from_equation(
            row, X_test, y_test, hr_test, qt_test, rr_test, feature_names
        )
        if metrics:
            metrics_list.append({
                'idx': idx,
                **metrics
            })
        else:
            metrics_list.append({
                'idx': idx,
                'r_vs_HR': np.nan,
                'abs_r_HR': np.nan,
                'MAE': np.nan,
                'RMSE': np.nan,
            })
    
    metrics_df = pd.DataFrame(metrics_list).set_index('idx')
    
    # Join metrics to unified
    unified = unified.join(metrics_df)
    
    # Compute unified score
    unified['score'] = unified.apply(
        lambda row: compute_unified_score(
            row['abs_r_HR'] if pd.notna(row['abs_r_HR']) else 1.0,
            row['complexity'],
            maxsize
        ),
        axis=1
    )
    
    # Sort by score (lower = better)
    unified = unified.sort_values('score').reset_index(drop=True)
    
    return unified


def format_full_formula(row: pd.Series) -> str:
    """
    Format the full QTc formula based on approach.
    """
    approach = row['approach']
    eq = row['equation']
    
    if approach == 'direct':
        return f"QTc = {eq}"
    elif approach == 'factor':
        return f"QTc = QT × ({eq})"
    else:  # additive
        return f"QTc = QT + ({eq})"


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Kepler-ECG Pooled Symbolic Regression for QTc Discovery'
    )
    
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Base results directory (default: results)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory (default: results/pool6/sr_qtc)')
    parser.add_argument('--approach', type=str, default='all',
                        choices=['direct', 'factor', 'additive', 'all'],
                        help='SR approach (default: all)')
    parser.add_argument('--iterations', type=int, default=PYSR_CONFIG["niterations"],
                        help=f'Number of SR iterations (default: {PYSR_CONFIG["niterations"]})')
    parser.add_argument('--maxsize', type=int, default=PYSR_CONFIG["maxsize"],
                        help=f'Max formula complexity (default: {PYSR_CONFIG["maxsize"]})')
    parser.add_argument('--minsize', type=int, default=PYSR_CONFIG["minsize"],
                        help=f'Min formula complexity (default: {PYSR_CONFIG["minsize"]})')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top formulas to keep (default: 10)')
    parser.add_argument('--feature-set', type=str, default='minimal',
                        choices=['minimal', 'standard', 'extended'],
                        help='Feature set to use (default: minimal)')
    parser.add_argument('--use-log-exp', action='store_true',
                        help='Include log/exp operators (default: off)')
    parser.add_argument('--samples-per-dataset', type=int, default=None,
                        help='Limit samples per dataset for testing (default: use all)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel processes 1-8 (default: 1)')
    parser.add_argument('--datasets', type=str, nargs='+', default=ALL_DATASETS,
                        help=f'Datasets to include (default: all 6)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for sampling and split (default: 42)')
    
    args = parser.parse_args()
    
    # Validate minsize <= maxsize
    if args.minsize > args.maxsize:
        parser.error(f"minsize ({args.minsize}) cannot be greater than maxsize ({args.maxsize})")
    
    # Validate n-jobs range
    if args.n_jobs < 1 or args.n_jobs > 8:
        parser.error(f"n-jobs must be between 1 and 8, got {args.n_jobs}")
    
    results_dir = Path(args.results_dir)
    
    # -------------------------------------------------------------------------
    # Determine output directory and file suffix
    # -------------------------------------------------------------------------
    # Check for LODO mode (Leave-One-Dataset-Out)
    excluded_datasets = set(ALL_DATASETS) - set(args.datasets)
    if len(excluded_datasets) == 1:
        excluded_dataset = list(excluded_datasets)[0]
        lodo_suffix = f"_exclude_{excluded_dataset}"
    else:
        excluded_dataset = None
        lodo_suffix = ""
    
    # Build file suffix combining LODO and random_state
    file_suffix = lodo_suffix
    if args.random_state != 42:
        file_suffix += f"_rs{args.random_state}"
    
    if args.output:
        output_dir = Path(args.output)
    else:
        # Build output directory name
        dir_name = 'sr_qtc'
        if excluded_dataset:
            dir_name = f'sr_qtc_exclude_{excluded_dataset}'
        if args.random_state != 42:
            dir_name += f'_rs{args.random_state}'
        output_dir = results_dir / 'pool6' / dir_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print LODO info if applicable
    if excluded_dataset:
        print(f"🔬 LODO MODE: Excluding {excluded_dataset} (will be used for validation)")
    
    # -------------------------------------------------------------------------
    # Build configuration
    # -------------------------------------------------------------------------
    config = PYSR_CONFIG.copy()
    config["niterations"] = args.iterations
    config["maxsize"] = args.maxsize
    config["minsize"] = args.minsize
    
    # -------------------------------------------------------------------------
    # Select operator set
    # -------------------------------------------------------------------------
    if args.use_log_exp:
        operators = OPERATORS_EXTENDED
        print("Operators: extended (with log/exp)")
    else:
        operators = OPERATORS_STANDARD
        print("Operators: standard (no log/exp)")
    
    config.update(operators)
    
    # -------------------------------------------------------------------------
    # Configure parallelization
    # -------------------------------------------------------------------------
    if args.n_jobs > 1:
        config["procs"] = args.n_jobs
        config["multithreading"] = True
        config["deterministic"] = False
        parallel_mode = f"⚡ Parallel mode: {args.n_jobs} processes (non-deterministic)"
    else:
        config["procs"] = 0
        config["multithreading"] = False
        config["deterministic"] = True
        parallel_mode = "🔒 Sequential mode: deterministic results"
    
    # -------------------------------------------------------------------------
    # Print header
    # -------------------------------------------------------------------------
    print("="*70)
    print("KEPLER-ECG: POOLED SYMBOLIC REGRESSION FOR QTc DISCOVERY")
    print("="*70)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Output: {output_dir}")
    print(f"Iterations: {config['niterations']}")
    print(f"Complexity range: {config['minsize']} - {config['maxsize']}")
    print(f"Top-N formulas: {args.top_n}")
    print(f"Random state: {args.random_state}")
    if args.samples_per_dataset:
        print(f"Samples per dataset: {args.samples_per_dataset}")
    print(parallel_mode)
    
    # -------------------------------------------------------------------------
    # Load and pool datasets
    # -------------------------------------------------------------------------
    try:
        df, dataset_counts = load_and_pool_datasets(
            args.datasets,
            results_dir,
            samples_per_dataset=args.samples_per_dataset,
            random_state=args.random_state,
        )
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    
    # -------------------------------------------------------------------------
    # Validate required columns
    # -------------------------------------------------------------------------
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return 1
    
    # -------------------------------------------------------------------------
    # Feature selection
    # -------------------------------------------------------------------------
    feature_cols = FEATURE_SETS[args.feature_set].copy()
    
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"\nFeatures: {feature_cols}")
    
    # -------------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------------
    X = df[feature_cols].copy()
    y = df['QTc_reference_ms'].values
    hr = df['heart_rate_bpm'].values
    qt = df['QT_interval_ms'].values
    rr = df['RR_interval_sec'].values
    
    # Filter valid samples
    valid = X.notna().all(axis=1) & ~np.isnan(y)
    X = X[valid].values
    y = y[valid]
    hr = hr[valid]
    qt = qt[valid]
    rr = rr[valid]
    df_valid = df[valid].copy().reset_index(drop=True)
    
    print(f"Valid samples: {len(y):,}")
    
    if len(y) < 1000:
        print(f"❌ Insufficient data: {len(y)} samples (minimum: 1000 for pooled)")
        return 1
    
    # -------------------------------------------------------------------------
    # Train/test split (stratified by dataset if possible)
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test, hr_train, hr_test, qt_train, qt_test, rr_train, rr_test, df_train, df_test = \
        train_test_split(X, y, hr, qt, rr, df_valid,
                        test_size=DATA_CONFIG["test_size"], 
                        random_state=args.random_state)
    
    print(f"Train: {len(y_train):,}, Test: {len(y_test):,}")
    
    # Show distribution by dataset
    print("\nTest set distribution:")
    for ds in df_test['source_dataset'].unique():
        n = (df_test['source_dataset'] == ds).sum()
        print(f"  {ds}: {n:,}")
    
    # -------------------------------------------------------------------------
    # Determine approaches to run
    # -------------------------------------------------------------------------
    if args.approach == 'all':
        approaches = ['direct', 'factor', 'additive']
    else:
        approaches = [args.approach]
    
    # -------------------------------------------------------------------------
    # Run SR for each approach
    # -------------------------------------------------------------------------
    all_results = {}
    all_equations = []
    all_models = {}
    best_overall = {'abs_r_hr': float('inf'), 'approach': None}
    
    for approach in approaches:
        try:
            sr_result, model, equations_df = run_symbolic_regression(
                X_train, y_train, feature_cols, approach, config, dataset_name="pool6"
            )
            
            if 'error' in sr_result:
                print(f"❌ {approach}: {sr_result['error']}")
                continue
            
            all_models[approach] = model
            all_equations.append(equations_df)
            
            # Evaluate on test set
            metrics, qtc_kepler = evaluate_formula(
                model, X_test, y_test, hr_test, qt_test, rr_test,
                approach, feature_cols
            )
            
            # Per-dataset evaluation
            dataset_metrics = evaluate_by_dataset(
                model, df_test, X_test, approach, feature_cols
            )
            
            # HR bin analysis
            hr_bins = analyze_by_hr_bins(qtc_kepler, qt_test, rr_test, hr_test, y_test)
            
            all_results[approach] = {
                'sr_result': sr_result,
                'metrics': metrics,
                'dataset_metrics': dataset_metrics,
                'hr_bin_analysis': hr_bins,
            }
            
            # Track best by minimum |r(HR)|
            kepler_r_hr = abs(metrics['Kepler']['r_vs_HR'])
            if kepler_r_hr < best_overall['abs_r_hr']:
                best_overall = {
                    'abs_r_hr': kepler_r_hr,
                    'approach': approach,
                    'equation': sr_result['best_equation'],
                    'complexity': sr_result['best_complexity'],
                }
            
        except Exception as e:
            print(f"❌ {approach}: {e}")
            import traceback
            traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # Build unified equations table
    # -------------------------------------------------------------------------
    if all_equations:
        print("\n" + "="*70)
        print("BUILDING UNIFIED EQUATIONS TABLE")
        print("="*70)
        
        unified_df = build_unified_equations_df(
            all_equations,
            X_test, y_test, hr_test, qt_test, rr_test,
            feature_cols,
            args.minsize,
            args.maxsize
        )
        
        # Add full formula column
        unified_df['full_formula'] = unified_df.apply(format_full_formula, axis=1)
        
        # Select columns for output
        output_cols = [
            'approach', 'complexity', 'loss', 'equation', 'full_formula',
            'r_vs_HR', 'abs_r_HR', 'MAE', 'RMSE', 'score'
        ]
        output_cols = [c for c in output_cols if c in unified_df.columns]
        
        # Save full unified table
        unified_path = output_dir / f'pool6_equations_unified{file_suffix}.csv'
        unified_df[output_cols].to_csv(unified_path, index=False)
        print(f"✅ Saved unified equations: {unified_path}")
        
        # Save top-N
        top_n_df = unified_df.head(args.top_n)
        top_n_path = output_dir / f'pool6_equations_top{args.top_n}{file_suffix}.csv'
        top_n_df[output_cols].to_csv(top_n_path, index=False)
        print(f"✅ Saved top-{args.top_n} equations: {top_n_path}")
        
        # Print top-N summary
        print(f"\n{'='*70}")
        print(f"TOP {args.top_n} FORMULAS (by unified score)")
        print(f"{'='*70}")
        print(f"{'Rank':<5} {'Approach':<10} {'Cmplx':<6} {'|r(HR)|':<10} {'Score':<8} Formula")
        print("-"*70)
        for i, row in top_n_df.head(args.top_n).iterrows():
            rank = i + 1
            print(f"{rank:<5} {row['approach']:<10} {row['complexity']:<6} "
                  f"{row['abs_r_HR']:<10.4f} {row['score']:<8.4f} {row['equation'][:40]}")
    
    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("FINAL RESULTS - POOLED DISCOVERY")
    print("="*70)
    
    if best_overall['approach']:
        print(f"\nBest approach: {best_overall['approach']}")
        print(f"Best equation: {best_overall['equation']}")
        print(f"Complexity: {best_overall['complexity']}")
        print(f"|r(HR)|: {best_overall['abs_r_hr']:.4f}")
        
        # Comparison with Bazett
        if all_results:
            bazett_r = abs(list(all_results.values())[0]['metrics']['Bazett']['r_vs_HR'])
            if best_overall['abs_r_hr'] > 0:
                improvement = bazett_r / best_overall['abs_r_hr']
                print(f"\n📈 Improvement over Bazett: {improvement:.1f}x")
            else:
                print(f"\n📈 Perfect HR independence achieved!")
    else:
        print("\n⚠️  No successful SR runs")
    
    # -------------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------------
    report = {
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'dataset': 'pool6',
        'random_state': args.random_state,
        'lodo_mode': excluded_dataset is not None,
        'excluded_dataset': excluded_dataset,
        'datasets_included': list(dataset_counts.keys()),
        'dataset_counts': dataset_counts,
        'total_records': sum(dataset_counts.values()),
        'samples_per_dataset_limit': args.samples_per_dataset,
        'config': {
            'pysr': {k: v for k, v in config.items() if k not in ['binary_operators', 'unary_operators', 'nested_constraints', 'complexity_of_operators']},
            'operators': 'extended' if args.use_log_exp else 'standard',
            'n_jobs': args.n_jobs,
            'deterministic': config['deterministic'],
            'data': DATA_CONFIG,
        },
        'complexity_range': {
            'minsize': args.minsize,
            'maxsize': args.maxsize,
        },
        'feature_set': args.feature_set,
        'features_used': feature_cols,
        'train_size': len(y_train),
        'test_size': len(y_test),
        'best_overall': best_overall,
        'top_n': args.top_n,
        'approaches': {
            approach: {
                'equation': res['sr_result']['best_equation'],
                'complexity': res['sr_result']['best_complexity'],
                'loss': res['sr_result']['best_loss'],
                'metrics': res['metrics'],
                'dataset_metrics': res['dataset_metrics'],
                'hr_bin_analysis': res['hr_bin_analysis'],
            }
            for approach, res in all_results.items()
        },
    }
    
    report_path = output_dir / f'pool6_sr_qtc_report{file_suffix}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report saved: {report_path}")
    print(f"✅ Results saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
