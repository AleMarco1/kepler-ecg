#!/usr/bin/env python3
"""
Kepler-ECG Phase 4.5 - Task 4: Symbolic Regression for QTc Discovery
=====================================================================

Discovers interpretable QTc correction formulas using PySR.

Multiple approaches:
1. Direct prediction: QTc = f(QT, RR, features)
2. Correction factor: QTc = QT * f(RR, features)  
3. Additive correction: QTc = QT + f(RR, features)

Author: Kepler-ECG Project
Date: 2025-12-17

Usage:
    python task4_sr_qtc_discovery.py \
        --dataset ./results/stream_c/qtc_sr_dataset_all_v2.csv \
        --output_path ./results/stream_c/sr_results \
        --approach all \
        --iterations 150
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load SR dataset."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records from {filepath}")
    return df


def prepare_features(df: pd.DataFrame, feature_set: str = 'minimal') -> tuple:
    """
    Prepare feature matrix and targets.
    
    Feature sets:
    - minimal: QT, RR only (most interpretable)
    - standard: QT, RR, HR
    - extended: + QRS, PR, amplitudes
    - full: + age, sex
    """
    # Core targets
    y_ref = df['QTc_reference_ms'].values
    
    if feature_set == 'minimal':
        feature_cols = ['QT_interval_ms', 'RR_interval_sec']
    elif feature_set == 'standard':
        feature_cols = ['QT_interval_ms', 'RR_interval_sec', 'heart_rate_bpm']
    elif feature_set == 'extended':
        feature_cols = ['QT_interval_ms', 'RR_interval_sec', 'heart_rate_bpm',
                       'QRS_duration_ms', 'PR_interval_ms', 'T_amplitude_mV', 'R_amplitude_mV']
    else:  # full
        feature_cols = ['QT_interval_ms', 'RR_interval_sec', 'heart_rate_bpm',
                       'QRS_duration_ms', 'PR_interval_ms', 'T_amplitude_mV', 
                       'R_amplitude_mV', 'age', 'sex']
    
    # Filter to available columns
    available = [c for c in feature_cols if c in df.columns]
    
    # Create feature matrix, drop NaN
    X = df[available].copy()
    valid_mask = X.notna().all(axis=1) & ~np.isnan(y_ref)
    
    X = X[valid_mask].values
    y = y_ref[valid_mask]
    
    # Also get HR for correlation analysis
    hr = df.loc[valid_mask, 'heart_rate_bpm'].values
    qt = df.loc[valid_mask, 'QT_interval_ms'].values
    rr = df.loc[valid_mask, 'RR_interval_sec'].values
    
    logger.info(f"Features ({feature_set}): {available}")
    logger.info(f"Final dataset: {len(y)} samples")
    
    return X, y, hr, qt, rr, available


def run_symbolic_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    approach: str = 'direct',
    n_iterations: int = 150,
    maxsize: int = 15,
) -> dict:
    """
    Run PySR symbolic regression.
    
    Approaches:
    - direct: Predict QTc directly from features
    - factor: Predict QTc/QT (correction factor)
    - additive: Predict QTc - QT (correction term)
    """
    from pysr import PySRRegressor
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Running SR: approach={approach}, iterations={n_iterations}")
    logger.info(f"{'='*50}")
    
    # Prepare target based on approach
    qt_idx = feature_names.index('QT_interval_ms')
    qt = X[:, qt_idx]
    
    if approach == 'direct':
        y_target = y  # Predict QTc directly
        target_name = 'QTc'
    elif approach == 'factor':
        y_target = y / qt  # Predict correction factor
        target_name = 'QTc/QT'
    elif approach == 'additive':
        y_target = y - qt  # Predict correction term
        target_name = 'QTc-QT'
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    logger.info(f"Target: {target_name}, range: [{y_target.min():.2f}, {y_target.max():.2f}]")
    
    # Configure PySR
    model = PySRRegressor(
        niterations=n_iterations,
        populations=30,
        population_size=40,
        maxsize=maxsize,
        
        # Operators suitable for QTc formulas
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "sqrt",      # For Bazett-like: QT/sqrt(RR)
            "cbrt",      # For Fridericia-like: QT/cbrt(RR)
            "inv",       # 1/x
            "square",    # x^2
        ],
        
        # Constraints for stability
        nested_constraints={
            "sqrt": {"sqrt": 0, "cbrt": 0},
            "cbrt": {"sqrt": 0, "cbrt": 0},
            "inv": {"inv": 0},
            "square": {"square": 0},
        },
        
        # Complexity penalties
        complexity_of_operators={
            "+": 1, "-": 1, "*": 1, "/": 1,
            "sqrt": 2, "cbrt": 2, "inv": 2, "square": 2,
        },
        
        # Optimization
        weight_optimize=0.01,
        adaptive_parsimony_scaling=100.0,
        
        # Reproducibility
        deterministic=True,
        random_state=42,
        parallelism='serial',
        #procs=4,
        
        # Output
        temp_equation_file=True,
        verbosity=1,
    )
    
    # Fit
    logger.info("Starting PySR fitting...")
    model.fit(X, y_target, variable_names=feature_names)
    
    # Get results
    equations = model.equations_
    
    # Extract Pareto front
    pareto_front = []
    for idx, row in equations.iterrows():
        pareto_front.append({
            'complexity': int(row['complexity']),
            'loss': float(row['loss']),
            'equation': str(row['equation']),
            'score': float(row['score']) if 'score' in row else None,
        })
    
    # Get best equation
    best_eq = model.get_best()
    best_idx = equations[equations['equation'].astype(str) == str(best_eq)].index[0]
    
    result = {
        'approach': approach,
        'target': target_name,
        'best_equation': str(best_eq),
        'best_complexity': int(equations.loc[best_idx, 'complexity']),
        'best_loss': float(equations.loc[best_idx, 'loss']),
        'pareto_front': pareto_front,
        'feature_names': feature_names,
        'n_samples': len(y),
        'n_iterations': n_iterations,
    }
    
    logger.info(f"\nBest equation ({approach}): {best_eq}")
    logger.info(f"Complexity: {result['best_complexity']}, Loss: {result['best_loss']:.6f}")
    
    return result, model


def evaluate_formula(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    hr: np.ndarray,
    qt: np.ndarray,
    rr: np.ndarray,
    approach: str,
    feature_names: list,
) -> dict:
    """
    Evaluate discovered formula.
    
    Key metrics:
    - MAE, RMSE vs reference QTc
    - Correlation with HR (should be ~0)
    - Comparison with Bazett/Fridericia
    """
    # Predict
    y_pred_raw = model.predict(X)
    
    # Convert back to QTc based on approach
    qt_idx = feature_names.index('QT_interval_ms')
    qt_vals = X[:, qt_idx]
    
    if approach == 'direct':
        qtc_kepler = y_pred_raw
    elif approach == 'factor':
        qtc_kepler = qt_vals * y_pred_raw
    elif approach == 'additive':
        qtc_kepler = qt_vals + y_pred_raw
    
    # Calculate standard QTc formulas
    qtc_bazett = qt / np.sqrt(rr)
    qtc_fridericia = qt / np.cbrt(rr)
    qtc_framingham = qt + 154 * (1 - rr)
    
    # Metrics
    def calc_metrics(qtc_pred, name):
        mae = np.mean(np.abs(qtc_pred - y_true))
        rmse = np.sqrt(np.mean((qtc_pred - y_true)**2))
        r_hr, p_hr = stats.pearsonr(qtc_pred, hr)
        r_ref, _ = stats.pearsonr(qtc_pred, y_true)
        
        return {
            'name': name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'r_vs_HR': round(r_hr, 4),
            'p_vs_HR': round(p_hr, 6),
            'r_vs_ref': round(r_ref, 4),
        }
    
    metrics = {
        'Kepler': calc_metrics(qtc_kepler, 'Kepler'),
        'Bazett': calc_metrics(qtc_bazett, 'Bazett'),
        'Fridericia': calc_metrics(qtc_fridericia, 'Fridericia'),
        'Framingham': calc_metrics(qtc_framingham, 'Framingham'),
    }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("FORMULA EVALUATION")
    logger.info("="*60)
    logger.info(f"{'Formula':<15} {'MAE':>8} {'RMSE':>8} {'r(HR)':>10} {'r(ref)':>10}")
    logger.info("-"*60)
    for name, m in metrics.items():
        logger.info(f"{name:<15} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['r_vs_HR']:>+10.4f} {m['r_vs_ref']:>10.4f}")
    
    return metrics, qtc_kepler


def analyze_by_hr_bins(
    qtc_kepler: np.ndarray,
    qt: np.ndarray,
    rr: np.ndarray,
    hr: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Analyze QTc performance by HR bins."""
    
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
            'Kepler_mean': round(np.mean(qtc_kepler[mask]), 2),
            'Kepler_std': round(np.std(qtc_kepler[mask]), 2),
            'Bazett_mean': round(np.mean(qtc_bazett[mask]), 2),
            'Bazett_std': round(np.std(qtc_bazett[mask]), 2),
            'Fridericia_mean': round(np.mean(qtc_fridericia[mask]), 2),
            'Fridericia_std': round(np.std(qtc_fridericia[mask]), 2),
            'Reference_mean': round(np.mean(y_true[mask]), 2),
        }
    
    return results


def run_baseline_comparison(X, y, feature_names):
    """Run baseline Ridge regression for comparison."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    
    model = Ridge(alpha=1.0)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    model.fit(X, y)
    y_pred = model.predict(X)
    mae = np.mean(np.abs(y_pred - y))
    
    return {
        'cv_r2_mean': round(np.mean(scores), 4),
        'cv_r2_std': round(np.std(scores), 4),
        'mae': round(mae, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to SR dataset CSV')
    parser.add_argument('--output_path', type=str, default='./results/stream_c/sr_results',
                        help='Output directory')
    parser.add_argument('--approach', type=str, default='all',
                        choices=['direct', 'factor', 'additive', 'all'],
                        help='SR approach')
    parser.add_argument('--iterations', type=int, default=150,
                        help='Number of SR iterations')
    parser.add_argument('--feature_set', type=str, default='minimal',
                        choices=['minimal', 'standard', 'extended', 'full'],
                        help='Feature set to use')
    parser.add_argument('--maxsize', type=int, default=15,
                        help='Maximum formula complexity')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Kepler-ECG Phase 4.5 - Task 4: Symbolic Regression for QTc")
    print("=" * 70)
    
    # Load data
    df = load_dataset(Path(args.dataset))
    
    # Prepare features
    X, y, hr, qt, rr, feature_names = prepare_features(df, args.feature_set)
    
    # Train/test split
    X_train, X_test, y_train, y_test, hr_train, hr_test, qt_train, qt_test, rr_train, rr_test = \
        train_test_split(X, y, hr, qt, rr, test_size=0.2, random_state=42)
    
    print(f"\nTrain: {len(y_train)}, Test: {len(y_test)}")
    
    # Baseline
    print("\nRunning baseline Ridge regression...")
    baseline = run_baseline_comparison(X_train, y_train, feature_names)
    print(f"Baseline R²: {baseline['cv_r2_mean']:.4f} ± {baseline['cv_r2_std']:.4f}")
    
    # Determine approaches to run
    if args.approach == 'all':
        approaches = ['direct', 'factor', 'additive']
    else:
        approaches = [args.approach]
    
    # Run SR for each approach
    all_results = {}
    best_overall = {'r_hr': float('inf'), 'approach': None, 'model': None}
    
    for approach in approaches:
        print(f"\n{'='*70}")
        print(f"APPROACH: {approach.upper()}")
        print('='*70)
        
        try:
            sr_result, model = run_symbolic_regression(
                X_train, y_train, feature_names,
                approach=approach,
                n_iterations=args.iterations,
                maxsize=args.maxsize,
            )
            
            # Evaluate on test set
            metrics, qtc_kepler = evaluate_formula(
                model, X_test, y_test, hr_test, qt_test, rr_test,
                approach, feature_names
            )
            
            # HR bin analysis
            hr_bins = analyze_by_hr_bins(qtc_kepler, qt_test, rr_test, hr_test, y_test)
            
            # Store results
            all_results[approach] = {
                'sr_result': sr_result,
                'metrics': metrics,
                'hr_bin_analysis': hr_bins,
            }
            
            # Track best
            kepler_r_hr = abs(metrics['Kepler']['r_vs_HR'])
            if kepler_r_hr < best_overall['r_hr']:
                best_overall = {
                    'r_hr': kepler_r_hr,
                    'approach': approach,
                    'equation': sr_result['best_equation'],
                    'complexity': sr_result['best_complexity'],
                    'metrics': metrics['Kepler'],
                }
            
            # Save model equations
            eq_file = output_path / f'equations_{approach}.csv'
            model.equations_.to_csv(eq_file, index=False)
            print(f"Saved equations to {eq_file}")
            
        except Exception as e:
            logger.error(f"Error in approach {approach}: {e}")
            import traceback
            traceback.print_exc()
            all_results[approach] = {'error': str(e)}
    
    # Final report
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nBest approach: {best_overall['approach']}")
    print(f"Best equation: {best_overall.get('equation', 'N/A')}")
    print(f"Complexity: {best_overall.get('complexity', 'N/A')}")
    print(f"|r(HR)|: {best_overall['r_hr']:.4f}")
    
    # Comparison table
    print("\n" + "-" * 70)
    print(f"{'Formula':<20} {'Approach':<12} {'|r(HR)|':>10} {'MAE':>10}")
    print("-" * 70)
    
    for approach, res in all_results.items():
        if 'error' not in res:
            m = res['metrics']['Kepler']
            print(f"{'Kepler-' + approach:<20} {approach:<12} {abs(m['r_vs_HR']):>10.4f} {m['MAE']:>10.2f}")
    
    # Add standard formulas
    if all_results and 'error' not in list(all_results.values())[0]:
        first_res = list(all_results.values())[0]
        for formula in ['Bazett', 'Fridericia', 'Framingham']:
            m = first_res['metrics'][formula]
            print(f"{formula:<20} {'standard':<12} {abs(m['r_vs_HR']):>10.4f} {m['MAE']:>10.2f}")
    
    # Save comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'task': 'Task 4 - Symbolic Regression QTc Discovery',
        'parameters': {
            'feature_set': args.feature_set,
            'features': feature_names,
            'iterations': args.iterations,
            'maxsize': args.maxsize,
            'train_size': len(y_train),
            'test_size': len(y_test),
        },
        'baseline': baseline,
        'best_overall': best_overall,
        'approaches': {},
    }
    
    for approach, res in all_results.items():
        if 'error' not in res:
            report['approaches'][approach] = {
                'equation': res['sr_result']['best_equation'],
                'complexity': res['sr_result']['best_complexity'],
                'loss': res['sr_result']['best_loss'],
                'metrics': res['metrics'],
                'hr_bin_analysis': res['hr_bin_analysis'],
                'pareto_front': res['sr_result']['pareto_front'][:5],  # Top 5 only
            }
        else:
            report['approaches'][approach] = {'error': res['error']}
    
    report_file = output_path / 'task4_sr_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {report_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
