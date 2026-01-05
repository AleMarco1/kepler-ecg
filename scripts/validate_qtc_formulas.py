#!/usr/bin/env python3
"""
Kepler-ECG: QTc Formula Validation

Comprehensive validation of discovered QTc formulas:
- HR-independence analysis (correlation with HR)
- HR bin analysis (QTc stability across heart rate ranges)
- Cross-validation stability
- Clinical threshold analysis
- Comparison with standard formulas (Bazett, Fridericia, Framingham, Hodges)

Usage:
    python scripts/validate_qtc_formulas.py --dataset ptb-xl
    
    # Custom input
    python scripts/validate_qtc_formulas.py --input results/ptb-xl/qtc/qtc_sr_dataset_all.csv

Author: Kepler-ECG Project
Version: 2.0.0
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')


# ============================================================================
# QTc Formulas
# ============================================================================

def qtc_bazett(qt_ms: np.ndarray, rr_sec: np.ndarray) -> np.ndarray:
    """Bazett (1920): QTc = QT / sqrt(RR)"""
    return qt_ms / np.sqrt(rr_sec)

def qtc_fridericia(qt_ms: np.ndarray, rr_sec: np.ndarray) -> np.ndarray:
    """Fridericia (1920): QTc = QT / cbrt(RR)"""
    return qt_ms / np.cbrt(rr_sec)

def qtc_framingham(qt_ms: np.ndarray, rr_sec: np.ndarray) -> np.ndarray:
    """Framingham (1992): QTc = QT + 154*(1-RR)"""
    return qt_ms + 154 * (1 - rr_sec)

def qtc_hodges(qt_ms: np.ndarray, hr_bpm: np.ndarray) -> np.ndarray:
    """Hodges (1983): QTc = QT + 1.75*(HR-60)"""
    return qt_ms + 1.75 * (hr_bpm - 60)

# Kepler formulas (discovered via SR)
def qtc_kepler_linear(qt_ms: np.ndarray, rr_sec: np.ndarray) -> np.ndarray:
    """Kepler Linear: QTc = QT - 184.54*RR + 156.72"""
    return qt_ms - 184.54 * rr_sec + 156.72

def qtc_kepler_cubic(qt_ms: np.ndarray, rr_sec: np.ndarray) -> np.ndarray:
    """Kepler Cubic: QTc = QT - 495.11*cbrt(RR) + 466.81"""
    return qt_ms - 495.11 * np.cbrt(rr_sec) + 466.81

def qtc_kepler_factor(qt_ms: np.ndarray, rr_sec: np.ndarray) -> np.ndarray:
    """Kepler Factor: QTc = QT * (0.364/RR + 0.562)"""
    return qt_ms * (0.36369 / rr_sec + 0.56214)


# ============================================================================
# Validation Functions
# ============================================================================

def calculate_all_qtc(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate QTc using all formulas."""
    df = df.copy()
    
    qt = df['QT_interval_ms'].values
    rr = df['RR_interval_sec'].values
    hr = df['heart_rate_bpm'].values
    
    # Standard formulas
    df['QTc_Bazett'] = qtc_bazett(qt, rr)
    df['QTc_Fridericia'] = qtc_fridericia(qt, rr)
    df['QTc_Framingham'] = qtc_framingham(qt, rr)
    df['QTc_Hodges'] = qtc_hodges(qt, hr)
    
    # Kepler formulas
    df['QTc_Kepler_Linear'] = qtc_kepler_linear(qt, rr)
    df['QTc_Kepler_Cubic'] = qtc_kepler_cubic(qt, rr)
    df['QTc_Kepler_Factor'] = qtc_kepler_factor(qt, rr)
    
    return df


def analyze_hr_independence(df: pd.DataFrame, qtc_cols: List[str]) -> Dict:
    """Analyze QTc independence from HR."""
    hr = df['heart_rate_bpm'].values
    
    results = {}
    for col in qtc_cols:
        if col not in df.columns:
            continue
        
        qtc = df[col].values
        valid = ~(np.isnan(qtc) | np.isnan(hr))
        
        if valid.sum() < 100:
            continue
        
        r, p = stats.pearsonr(qtc[valid], hr[valid])
        rho, p_spearman = stats.spearmanr(qtc[valid], hr[valid])
        
        results[col] = {
            'pearson_r': round(r, 4),
            'pearson_p': p,
            'spearman_rho': round(rho, 4),
            'abs_r': round(abs(r), 4),
        }
    
    return results


def analyze_by_hr_bins(df: pd.DataFrame, qtc_cols: List[str]) -> Dict:
    """Analyze QTc statistics by HR bins."""
    bins = [(40, 60), (60, 80), (80, 100), (100, 120), (120, 150)]
    hr = df['heart_rate_bpm'].values
    
    results = {}
    
    for col in qtc_cols:
        if col not in df.columns:
            continue
        
        qtc = df[col].values
        col_results = {}
        
        for low, high in bins:
            mask = (hr >= low) & (hr < high)
            n = mask.sum()
            
            if n < 20:
                continue
            
            col_results[f"{low}-{high}"] = {
                'n': int(n),
                'mean': round(float(np.nanmean(qtc[mask])), 2),
                'std': round(float(np.nanstd(qtc[mask])), 2),
                'median': round(float(np.nanmedian(qtc[mask])), 2),
            }
        
        # Cross-bin variation
        means = [v['mean'] for v in col_results.values()]
        if len(means) >= 2:
            col_results['cross_bin_variation'] = {
                'mean_of_means': round(np.mean(means), 2),
                'std_of_means': round(np.std(means), 2),
                'range_of_means': round(max(means) - min(means), 2),
            }
        
        results[col] = col_results
    
    return results


def cross_validation_stability(df: pd.DataFrame, qtc_cols: List[str], n_folds: int = 5) -> Dict:
    """Assess stability of HR-independence across CV folds."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    hr = df['heart_rate_bpm'].values
    
    results = {}
    
    for col in qtc_cols:
        if col not in df.columns:
            continue
        
        qtc = df[col].values
        fold_correlations = []
        
        for _, test_idx in kf.split(df):
            valid = ~(np.isnan(qtc[test_idx]) | np.isnan(hr[test_idx]))
            if valid.sum() > 10:
                r, _ = stats.pearsonr(qtc[test_idx][valid], hr[test_idx][valid])
                fold_correlations.append(r)
        
        if fold_correlations:
            results[col] = {
                'mean_r': round(np.mean(fold_correlations), 4),
                'std_r': round(np.std(fold_correlations), 4),
                'min_r': round(min(fold_correlations), 4),
                'max_r': round(max(fold_correlations), 4),
            }
    
    return results


def clinical_threshold_analysis(df: pd.DataFrame, qtc_cols: List[str]) -> Dict:
    """Analyze clinical QTc thresholds."""
    results = {}
    
    for col in qtc_cols:
        if col not in df.columns:
            continue
        
        qtc = df[col].values
        valid = ~np.isnan(qtc)
        n_total = valid.sum()
        
        if n_total == 0:
            continue
        
        # Thresholds
        prolonged_450 = (qtc > 450).sum()
        prolonged_460 = (qtc > 460).sum()
        high_risk_500 = (qtc > 500).sum()
        short_340 = (qtc < 340).sum()
        
        results[col] = {
            'n_total': int(n_total),
            'mean_qtc': round(float(np.nanmean(qtc)), 2),
            'std_qtc': round(float(np.nanstd(qtc)), 2),
            'prolonged_450ms': {
                'n': int(prolonged_450),
                'pct': round(100 * prolonged_450 / n_total, 2),
            },
            'prolonged_460ms': {
                'n': int(prolonged_460),
                'pct': round(100 * prolonged_460 / n_total, 2),
            },
            'high_risk_500ms': {
                'n': int(high_risk_500),
                'pct': round(100 * high_risk_500 / n_total, 2),
            },
            'short_340ms': {
                'n': int(short_340),
                'pct': round(100 * short_340 / n_total, 2),
            },
        }
    
    return results


def create_validation_plots(df: pd.DataFrame, output_path: Path):
    """Create validation plots."""
    
    qtc_cols = [
        'QTc_Bazett', 'QTc_Fridericia', 'QTc_Framingham',
        'QTc_Kepler_Linear', 'QTc_Kepler_Cubic', 'QTc_Kepler_Factor'
    ]
    qtc_cols = [c for c in qtc_cols if c in df.columns]
    
    hr = df['heart_rate_bpm'].values
    
    # Plot 1: QTc vs HR for all formulas
    n_cols = min(len(qtc_cols), 3)
    n_rows = (len(qtc_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    # Sample for plotting
    n_plot = min(5000, len(hr))
    idx = np.random.choice(len(hr), n_plot, replace=False)
    
    for i, col in enumerate(qtc_cols):
        ax = axes[i]
        qtc = df[col].values
        
        ax.scatter(hr[idx], qtc[idx], alpha=0.3, s=5)
        
        # Regression line
        valid = ~(np.isnan(hr) | np.isnan(qtc))
        z = np.polyfit(hr[valid], qtc[valid], 1)
        p = np.poly1d(z)
        hr_range = np.linspace(hr.min(), hr.max(), 100)
        ax.plot(hr_range, p(hr_range), 'r-', linewidth=2)
        
        r, _ = stats.pearsonr(qtc[valid], hr[valid])
        
        ax.set_xlabel('Heart Rate (bpm)')
        ax.set_ylabel('QTc (ms)')
        ax.set_title(f'{col}\nr = {r:.4f}')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for j in range(len(qtc_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path / 'qtc_vs_hr_comparison.png', dpi=150)
    plt.close()
    
    # Plot 2: Box plots by HR bin
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    bins = [(40, 60), (60, 80), (80, 100), (100, 120)]
    bin_labels = ['40-60', '60-80', '80-100', '100-120']
    
    for i, col in enumerate(qtc_cols):
        ax = axes[i]
        qtc = df[col].values
        
        data_by_bin = []
        for low, high in bins:
            mask = (hr >= low) & (hr < high) & ~np.isnan(qtc)
            data_by_bin.append(qtc[mask])
        
        bp = ax.boxplot(data_by_bin, labels=bin_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_xlabel('Heart Rate Bin (bpm)')
        ax.set_ylabel('QTc (ms)')
        ax.set_title(col)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Mean line
        means = [np.nanmean(d) for d in data_by_bin]
        ax.plot(range(1, len(bins)+1), means, 'ro-', markersize=8, linewidth=2)
    
    for j in range(len(qtc_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path / 'qtc_hr_bins_comparison.png', dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Kepler-ECG QTc Formula Validation'
    )
    
    parser.add_argument('--input', '-i', type=str,
                        help='Path to QTc dataset CSV')
    parser.add_argument('--dataset', '-d', type=str,
                        help='Dataset name')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples for analysis')
    
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
        output_path = Path(args.output)
    elif args.dataset:
        output_path = Path(f"results/{args.dataset}/qtc_validation")
    else:
        output_path = Path("results/qtc_validation")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("KEPLER-ECG QTc FORMULA VALIDATION")
    print("="*70)
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded: {len(df)} records")
    
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)
        print(f"Sampled to {len(df)} records")
    
    # Calculate all QTc values
    print("\nCalculating QTc with all formulas...")
    df = calculate_all_qtc(df)
    
    qtc_cols = [
        'QTc_Bazett', 'QTc_Fridericia', 'QTc_Framingham', 'QTc_Hodges',
        'QTc_Kepler_Linear', 'QTc_Kepler_Cubic', 'QTc_Kepler_Factor'
    ]
    
    # Run analyses
    print("\n1. HR independence analysis...")
    hr_results = analyze_hr_independence(df, qtc_cols)
    
    print("2. HR bin analysis...")
    bin_results = analyze_by_hr_bins(df, qtc_cols)
    
    print("3. Cross-validation stability...")
    cv_results = cross_validation_stability(df, qtc_cols)
    
    print("4. Clinical threshold analysis...")
    clinical_results = clinical_threshold_analysis(df, qtc_cols)
    
    print("5. Creating plots...")
    create_validation_plots(df, output_path)
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    print("\n### HR Independence (target: |r| ≈ 0)")
    print(f"{'Formula':<25} {'r(HR)':>12} {'|r|':>10}")
    print("-"*50)
    
    sorted_by_r = sorted(hr_results.items(), key=lambda x: abs(x[1]['pearson_r']))
    for formula, data in sorted_by_r:
        r = data['pearson_r']
        print(f"{formula:<25} {r:>+12.4f} {abs(r):>10.4f}")
    
    # Best formula
    if sorted_by_r:
        best_formula = sorted_by_r[0][0]
        best_r = sorted_by_r[0][1]['abs_r']
        bazett_r = hr_results.get('QTc_Bazett', {}).get('abs_r', 0)
        
        if bazett_r > 0 and best_r > 0:
            improvement = bazett_r / best_r
            print(f"\n📈 Best: {best_formula}")
            print(f"   Improvement over Bazett: {improvement:.1f}x")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_file': str(input_path),
        'n_samples': len(df),
        'hr_independence': hr_results,
        'hr_bin_analysis': bin_results,
        'cv_stability': cv_results,
        'clinical_thresholds': clinical_results,
        'ranking': [
            {'rank': i+1, 'formula': f, 'abs_r': d['abs_r']}
            for i, (f, d) in enumerate(sorted_by_r)
        ],
    }
    
    with open(output_path / 'qtc_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
