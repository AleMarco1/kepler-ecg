#!/usr/bin/env python3
"""
Kepler-ECG: Symbolic Regression for QTc Discovery

Discovers interpretable QTc correction formulas using PySR.

Multiple approaches:
1. Direct: QTc = f(QT, RR, features)
2. Factor: QTc = QT * f(RR)
3. Additive: QTc = QT + f(RR)

Goal: Find formulas with near-zero HR correlation (better than Bazett/Fridericia).

Usage:
    python scripts/sr_qtc_discovery.py --dataset ptb-xl
    
    # Specific approach
    python scripts/sr_qtc_discovery.py --input results/ptb-xl/qtc/qtc_sr_dataset_all.csv --approach factor
    
    # All approaches
    python scripts/sr_qtc_discovery.py --dataset ptb-xl --approach all

Author: Kepler-ECG Project
Version: 2.0.0
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "niterations": 150,
    "populations": 30,
    "population_size": 40,
    "maxsize": 15,
    "parsimony": 0.004,
    
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["sqrt", "cbrt", "inv", "square"],
    
    "test_size": 0.2,
    "random_state": 42,
}


# ============================================================================
# SR Pipeline
# ============================================================================

def run_symbolic_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    approach: str,
    config: Dict,
) -> Tuple[Dict, object]:
    """Run PySR for QTc discovery."""
    
    try:
        from pysr import PySRRegressor
    except ImportError:
        return {"error": "PySR not installed"}, None
    
    print(f"\n{'='*50}")
    print(f"SR Approach: {approach.upper()}")
    print(f"{'='*50}")
    
    # Prepare target
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
    
    model = PySRRegressor(
        niterations=config["niterations"],
        populations=config["populations"],
        population_size=config["population_size"],
        maxsize=config["maxsize"],
        
        binary_operators=config["binary_operators"],
        unary_operators=config["unary_operators"],
        
        nested_constraints={
            "sqrt": {"sqrt": 0, "cbrt": 0},
            "cbrt": {"sqrt": 0, "cbrt": 0},
            "inv": {"inv": 0},
            "square": {"square": 0},
        },
        
        complexity_of_operators={
            "+": 1, "-": 1, "*": 1, "/": 1,
            "sqrt": 2, "cbrt": 2, "inv": 2, "square": 2,
        },
        
        weight_optimize=0.01,
        adaptive_parsimony_scaling=100.0,
        
        # Deterministic mode
        deterministic=True,
        random_state=config["random_state"],
        procs=0,
        multithreading=False,
        
        verbosity=1,
        progress=True,
    )
    
    print("\nFitting...")
    model.fit(X, y_target, variable_names=feature_names)
    
    # Get best equation
    best_eq = model.get_best()
    equations = model.equations_
    best_idx = equations[equations['equation'].astype(str) == str(best_eq)].index[0]
    
    result = {
        'approach': approach,
        'target': target_name,
        'best_equation': str(best_eq),
        'best_complexity': int(equations.loc[best_idx, 'complexity']),
        'best_loss': float(equations.loc[best_idx, 'loss']),
        'pareto_front': [
            {
                'complexity': int(row['complexity']),
                'loss': float(row['loss']),
                'equation': str(row['equation']),
            }
            for _, row in equations.iterrows()
        ],
    }
    
    print(f"\nBest: {best_eq}")
    print(f"Complexity: {result['best_complexity']}, Loss: {result['best_loss']:.6f}")
    
    return result, model


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
    """Evaluate discovered formula."""
    
    # Predict
    y_pred = model.predict(X)
    
    # Convert back to QTc
    qt_idx = feature_names.index('QT_interval_ms')
    qt_vals = X[:, qt_idx]
    
    if approach == 'direct':
        qtc_kepler = y_pred
    elif approach == 'factor':
        qtc_kepler = y_pred * qt_vals
    else:  # additive
        qtc_kepler = y_pred + qt_vals
    
    # Clip to reasonable range
    qtc_kepler = np.clip(qtc_kepler, 200, 700)
    
    # Calculate metrics
    def calc_metrics(qtc, name):
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
    
    # Standard formulas
    qtc_bazett = qt / np.sqrt(rr)
    qtc_fridericia = qt / np.cbrt(rr)
    qtc_framingham = qt + 154 * (1 - rr)
    
    metrics = {
        'Kepler': calc_metrics(qtc_kepler, 'Kepler'),
        'Bazett': calc_metrics(qtc_bazett, 'Bazett'),
        'Fridericia': calc_metrics(qtc_fridericia, 'Fridericia'),
        'Framingham': calc_metrics(qtc_framingham, 'Framingham'),
    }
    
    # Print comparison
    print("\n" + "="*60)
    print("FORMULA COMPARISON")
    print("="*60)
    print(f"{'Formula':<15} {'MAE':>8} {'RMSE':>8} {'r(HR)':>10}")
    print("-"*45)
    for name, m in metrics.items():
        print(f"{name:<15} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['r_vs_HR']:>+10.4f}")
    
    return metrics, qtc_kepler


def analyze_by_hr_bins(
    qtc_kepler: np.ndarray,
    qt: np.ndarray,
    rr: np.ndarray,
    hr: np.ndarray,
    y_true: np.ndarray,
) -> Dict:
    """Analyze QTc by HR bins."""
    
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


def save_plots(
    metrics: Dict,
    qtc_kepler: np.ndarray,
    hr: np.ndarray,
    qt: np.ndarray,
    rr: np.ndarray,
    approach: str,
    output_dir: Path,
):
    """Save visualization plots."""
    
    qtc_bazett = qt / np.sqrt(rr)
    qtc_fridericia = qt / np.cbrt(rr)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sample for plotting
    n_plot = min(5000, len(hr))
    idx = np.random.choice(len(hr), n_plot, replace=False)
    
    formulas = [
        ('Kepler', qtc_kepler, 'steelblue'),
        ('Bazett', qtc_bazett, 'orange'),
        ('Fridericia', qtc_fridericia, 'green'),
    ]
    
    for ax, (name, qtc, color) in zip(axes, formulas):
        ax.scatter(hr[idx], qtc[idx], alpha=0.3, s=5, c=color)
        
        # Trend line
        z = np.polyfit(hr, qtc, 1)
        p = np.poly1d(z)
        hr_range = np.linspace(hr.min(), hr.max(), 100)
        ax.plot(hr_range, p(hr_range), 'r-', linewidth=2)
        
        r, _ = stats.pearsonr(qtc, hr)
        
        ax.set_xlabel('Heart Rate (bpm)')
        ax.set_ylabel('QTc (ms)')
        ax.set_title(f'{name}: r = {r:.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'sr_qtc_{approach}_comparison.png', dpi=150)
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Kepler-ECG Symbolic Regression for QTc Discovery'
    )
    
    parser.add_argument('--input', '-i', type=str,
                        help='Path to QTc SR dataset CSV')
    parser.add_argument('--dataset', '-d', type=str,
                        help='Dataset name')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory')
    parser.add_argument('--approach', type=str, default='all',
                        choices=['direct', 'factor', 'additive', 'all'],
                        help='SR approach')
    parser.add_argument('--iterations', type=int, default=150,
                        help='Number of SR iterations')
    parser.add_argument('--maxsize', type=int, default=15,
                        help='Max formula complexity')
    parser.add_argument('--feature-set', type=str, default='minimal',
                        choices=['minimal', 'standard', 'extended'],
                        help='Feature set to use')
    
    args = parser.parse_args()
    
    # Determine paths
    if args.input:
        input_path = Path(args.input)
    elif args.dataset:
        input_path = Path(f"results/{args.dataset}/qtc/qtc_sr_dataset_all.csv")
    else:
        parser.error("Must provide either --input or --dataset")
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1
    
    if args.output:
        output_dir = Path(args.output)
    elif args.dataset:
        output_dir = Path(f"results/{args.dataset}/sr_qtc")
    else:
        output_dir = Path("results/sr_qtc")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = DEFAULT_CONFIG.copy()
    config["niterations"] = args.iterations
    config["maxsize"] = args.maxsize
    
    print("="*70)
    print("KEPLER-ECG: SYMBOLIC REGRESSION FOR QTc DISCOVERY")
    print("="*70)
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded: {len(df)} records")
    
    # Feature selection
    if args.feature_set == 'minimal':
        feature_cols = ['QT_interval_ms', 'RR_interval_sec']
    elif args.feature_set == 'standard':
        feature_cols = ['QT_interval_ms', 'RR_interval_sec', 'heart_rate_bpm']
    else:
        feature_cols = ['QT_interval_ms', 'RR_interval_sec', 'heart_rate_bpm',
                       'QRS_duration_ms', 'PR_interval_ms']
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Features: {feature_cols}")
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df['QTc_reference_ms'].values
    hr = df['heart_rate_bpm'].values
    qt = df['QT_interval_ms'].values
    rr = df['RR_interval_sec'].values
    
    valid = X.notna().all(axis=1) & ~np.isnan(y)
    X = X[valid].values
    y = y[valid]
    hr = hr[valid]
    qt = qt[valid]
    rr = rr[valid]
    
    print(f"Valid samples: {len(y)}")
    
    # Split
    X_train, X_test, y_train, y_test, hr_train, hr_test, qt_train, qt_test, rr_train, rr_test = \
        train_test_split(X, y, hr, qt, rr, test_size=config["test_size"], 
                        random_state=config["random_state"])
    
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Approaches
    if args.approach == 'all':
        approaches = ['direct', 'factor', 'additive']
    else:
        approaches = [args.approach]
    
    # Run SR
    all_results = {}
    best_overall = {'abs_r_hr': float('inf'), 'approach': None}
    
    for approach in approaches:
        try:
            sr_result, model = run_symbolic_regression(
                X_train, y_train, feature_cols, approach, config
            )
            
            if 'error' in sr_result:
                print(f"❌ {approach}: {sr_result['error']}")
                continue
            
            # Evaluate
            metrics, qtc_kepler = evaluate_formula(
                model, X_test, y_test, hr_test, qt_test, rr_test,
                approach, feature_cols
            )
            
            # HR bin analysis
            hr_bins = analyze_by_hr_bins(qtc_kepler, qt_test, rr_test, hr_test, y_test)
            
            all_results[approach] = {
                'sr_result': sr_result,
                'metrics': metrics,
                'hr_bin_analysis': hr_bins,
            }
            
            # Track best
            kepler_r_hr = abs(metrics['Kepler']['r_vs_HR'])
            if kepler_r_hr < best_overall['abs_r_hr']:
                best_overall = {
                    'abs_r_hr': kepler_r_hr,
                    'approach': approach,
                    'equation': sr_result['best_equation'],
                    'complexity': sr_result['best_complexity'],
                }
            
            # Save plots
            save_plots(metrics, qtc_kepler, hr_test, qt_test, rr_test, approach, output_dir)
            
            # Save equations
            model.equations_.to_csv(output_dir / f'equations_{approach}.csv', index=False)
            
        except Exception as e:
            print(f"❌ {approach}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    if best_overall['approach']:
        print(f"\nBest approach: {best_overall['approach']}")
        print(f"Best equation: {best_overall['equation']}")
        print(f"Complexity: {best_overall['complexity']}")
        print(f"|r(HR)|: {best_overall['abs_r_hr']:.4f}")
        
        # Comparison with Bazett
        if all_results:
            bazett_r = abs(list(all_results.values())[0]['metrics']['Bazett']['r_vs_HR'])
            improvement = bazett_r / best_overall['abs_r_hr'] if best_overall['abs_r_hr'] > 0 else float('inf')
            print(f"\n📈 Improvement over Bazett: {improvement:.1f}x")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_file': str(input_path),
        'config': config,
        'feature_set': args.feature_set,
        'features': feature_cols,
        'train_size': len(y_train),
        'test_size': len(y_test),
        'best_overall': best_overall,
        'approaches': {
            approach: {
                'equation': res['sr_result']['best_equation'],
                'complexity': res['sr_result']['best_complexity'],
                'metrics': res['metrics'],
            }
            for approach, res in all_results.items()
        },
    }
    
    with open(output_dir / 'sr_qtc_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
