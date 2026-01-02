#!/usr/bin/env python3
"""
Kepler-ECG Phase 4.5 - Task 5: Validation & Comparison
======================================================

Comprehensive validation of discovered QTc formulas against:
- Standard formulas (Bazett, Fridericia, Framingham, Hodges)
- HR-independence analysis by bins
- Cross-validation stability
- Clinical threshold analysis

Author: Kepler-ECG Project
Date: 2025-12-17

Usage:
    python task5_validation.py \
        --dataset ./results/stream_c/qtc_sr_dataset_all_v2.csv \
        --output_path ./results/stream_c/validation
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DISCOVERED FORMULAS FROM TASK 4
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

# Kepler formulas discovered in Task 4 (from Pareto front)
def qtc_kepler_linear(qt_ms: np.ndarray, rr_sec: np.ndarray) -> np.ndarray:
    """Kepler Linear (complexity 7): QTc = QT - 184.54*RR + 156.72"""
    return qt_ms - 184.54 * rr_sec + 156.72

def qtc_kepler_cubic(qt_ms: np.ndarray, rr_sec: np.ndarray) -> np.ndarray:
    """Kepler Cubic (complexity 9): QTc = QT - 495.1*cbrt(RR) + 466.8"""
    return qt_ms - 495.11 * np.cbrt(rr_sec) + 466.81

def qtc_kepler_factor(qt_ms: np.ndarray, rr_sec: np.ndarray) -> np.ndarray:
    """Kepler Factor (complexity 5): QTc = QT * (0.364/RR + 0.562)"""
    return qt_ms * (0.36369 / rr_sec + 0.56214)


# ============================================================================
# VALIDATION FUNCTIONS  
# ============================================================================

def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load dataset."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records")
    return df


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
    """
    Analyze QTc independence from HR.
    
    Key metric: Pearson correlation between QTc and HR
    Target: r ≈ 0 (no correlation)
    """
    hr = df['heart_rate_bpm'].values
    
    results = {}
    for col in qtc_cols:
        qtc = df[col].values
        valid = ~(np.isnan(qtc) | np.isnan(hr))
        
        if valid.sum() < 100:
            continue
            
        r, p = stats.pearsonr(qtc[valid], hr[valid])
        
        # Spearman for robustness
        rho, p_spearman = stats.spearmanr(qtc[valid], hr[valid])
        
        results[col] = {
            'pearson_r': round(r, 4),
            'pearson_p': round(p, 8),
            'spearman_rho': round(rho, 4),
            'spearman_p': round(p_spearman, 8),
            'abs_r': round(abs(r), 4),
        }
    
    return results


def analyze_by_hr_bins(df: pd.DataFrame, qtc_cols: List[str]) -> Dict:
    """
    Analyze QTc statistics by HR bins.
    
    Goal: QTc should be stable across HR ranges.
    """
    bins = [(40, 60), (60, 80), (80, 100), (100, 120), (120, 150)]
    hr = df['heart_rate_bpm'].values
    
    results = {}
    
    for col in qtc_cols:
        qtc = df[col].values
        col_results = {}
        
        for low, high in bins:
            mask = (hr >= low) & (hr < high)
            n = mask.sum()
            
            if n < 20:
                continue
            
            col_results[f"{low}-{high}"] = {
                'n': int(n),
                'mean': round(np.nanmean(qtc[mask]), 2),
                'std': round(np.nanstd(qtc[mask]), 2),
                'median': round(np.nanmedian(qtc[mask]), 2),
            }
        
        # Calculate variation across bins (should be minimal for good formula)
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
    """
    Assess stability of HR-independence across CV folds.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    hr = df['heart_rate_bpm'].values
    
    results = {}
    
    for col in qtc_cols:
        qtc = df[col].values
        fold_correlations = []
        
        for train_idx, test_idx in kf.split(df):
            r, _ = stats.pearsonr(qtc[test_idx], hr[test_idx])
            fold_correlations.append(r)
        
        results[col] = {
            'mean_r': round(np.mean(fold_correlations), 4),
            'std_r': round(np.std(fold_correlations), 4),
            'min_r': round(min(fold_correlations), 4),
            'max_r': round(max(fold_correlations), 4),
        }
    
    return results


def clinical_threshold_analysis(df: pd.DataFrame, qtc_cols: List[str]) -> Dict:
    """
    Analyze clinical QTc thresholds.
    
    Standard thresholds:
    - QTc > 450ms (men) / 460ms (women): prolonged
    - QTc > 500ms: high risk
    - QTc < 340ms: short QT
    """
    results = {}
    
    # Use sex-specific thresholds if available
    has_sex = 'sex' in df.columns
    
    for col in qtc_cols:
        qtc = df[col].values
        valid = ~np.isnan(qtc)
        n_total = valid.sum()
        
        # General thresholds
        prolonged_450 = (qtc > 450).sum()
        prolonged_460 = (qtc > 460).sum()
        high_risk_500 = (qtc > 500).sum()
        short_340 = (qtc < 340).sum()
        
        results[col] = {
            'n_total': int(n_total),
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
            'mean_qtc': round(np.nanmean(qtc), 2),
            'std_qtc': round(np.nanstd(qtc), 2),
        }
        
        # Sex-specific if available
        if has_sex:
            male_mask = df['sex'] == 0
            female_mask = df['sex'] == 1
            
            # Men: >450ms is prolonged
            male_prolonged = ((qtc > 450) & male_mask).sum()
            male_total = (male_mask & valid).sum()
            
            # Women: >460ms is prolonged
            female_prolonged = ((qtc > 460) & female_mask).sum()
            female_total = (female_mask & valid).sum()
            
            if male_total > 0:
                results[col]['male_prolonged'] = {
                    'n': int(male_prolonged),
                    'total': int(male_total),
                    'pct': round(100 * male_prolonged / male_total, 2),
                }
            if female_total > 0:
                results[col]['female_prolonged'] = {
                    'n': int(female_prolonged),
                    'total': int(female_total),
                    'pct': round(100 * female_prolonged / female_total, 2),
                }
    
    return results


def formula_agreement_analysis(df: pd.DataFrame, qtc_cols: List[str]) -> Dict:
    """
    Analyze agreement between formulas.
    """
    results = {'correlations': {}, 'mean_differences': {}}
    
    for i, col1 in enumerate(qtc_cols):
        for col2 in qtc_cols[i+1:]:
            qtc1 = df[col1].values
            qtc2 = df[col2].values
            valid = ~(np.isnan(qtc1) | np.isnan(qtc2))
            
            # Correlation
            r, _ = stats.pearsonr(qtc1[valid], qtc2[valid])
            
            # Mean difference (bias)
            diff = qtc1[valid] - qtc2[valid]
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            
            key = f"{col1}_vs_{col2}"
            results['correlations'][key] = round(r, 4)
            results['mean_differences'][key] = {
                'mean': round(mean_diff, 2),
                'std': round(std_diff, 2),
            }
    
    return results


def generate_comparison_table(hr_results: Dict, clinical_results: Dict) -> str:
    """Generate markdown comparison table."""
    
    lines = [
        "## QTc Formula Comparison",
        "",
        "| Formula | |r(HR)| | Mean QTc | SD | >450ms (%) | >500ms (%) |",
        "|---------|--------|----------|-----|------------|------------|",
    ]
    
    for formula in hr_results.keys():
        r = abs(hr_results[formula]['pearson_r'])
        clinical = clinical_results.get(formula, {})
        mean_qtc = clinical.get('mean_qtc', 'N/A')
        std_qtc = clinical.get('std_qtc', 'N/A')
        prol_450 = clinical.get('prolonged_450ms', {}).get('pct', 'N/A')
        high_500 = clinical.get('high_risk_500ms', {}).get('pct', 'N/A')
        
        lines.append(f"| {formula} | {r:.4f} | {mean_qtc} | {std_qtc} | {prol_450} | {high_500} |")
    
    return "\n".join(lines)


def create_validation_plots(df: pd.DataFrame, output_path: Path):
    """Create validation plots."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    qtc_cols = [
        'QTc_Bazett', 'QTc_Fridericia', 'QTc_Framingham',
        'QTc_Kepler_Linear', 'QTc_Kepler_Cubic', 'QTc_Kepler_Factor'
    ]
    
    hr = df['heart_rate_bpm'].values
    
    # Plot 1: QTc vs HR for all formulas
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(qtc_cols):
        ax = axes[i]
        qtc = df[col].values
        
        # Sample for plotting (max 5000 points)
        n_plot = min(5000, len(hr))
        idx = np.random.choice(len(hr), n_plot, replace=False)
        
        ax.scatter(hr[idx], qtc[idx], alpha=0.3, s=5)
        
        # Add regression line
        z = np.polyfit(hr, qtc, 1)
        p = np.poly1d(z)
        hr_range = np.linspace(hr.min(), hr.max(), 100)
        ax.plot(hr_range, p(hr_range), 'r-', linewidth=2)
        
        # Correlation
        r, _ = stats.pearsonr(qtc, hr)
        
        ax.set_xlabel('Heart Rate (bpm)')
        ax.set_ylabel('QTc (ms)')
        ax.set_title(f'{col}\nr = {r:.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'qtc_vs_hr_comparison.png', dpi=150)
    plt.close()
    
    # Plot 2: HR bin analysis (box plots)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    bins = [(40, 60), (60, 80), (80, 100), (100, 120)]
    bin_labels = ['40-60', '60-80', '80-100', '100-120']
    
    for i, col in enumerate(qtc_cols):
        ax = axes[i]
        qtc = df[col].values
        
        data_by_bin = []
        for low, high in bins:
            mask = (hr >= low) & (hr < high)
            data_by_bin.append(qtc[mask])
        
        bp = ax.boxplot(data_by_bin, labels=bin_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_xlabel('Heart Rate Bin (bpm)')
        ax.set_ylabel('QTc (ms)')
        ax.set_title(col)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        means = [np.mean(d) for d in data_by_bin]
        ax.plot(range(1, len(bins)+1), means, 'ro-', markersize=8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_path / 'qtc_hr_bins_comparison.png', dpi=150)
    plt.close()
    
    # Plot 3: Formula comparison scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    comparisons = [
        ('QTc_Bazett', 'QTc_Kepler_Cubic'),
        ('QTc_Fridericia', 'QTc_Kepler_Cubic'),
        ('QTc_Kepler_Linear', 'QTc_Kepler_Cubic'),
    ]
    
    for ax, (col1, col2) in zip(axes, comparisons):
        qtc1 = df[col1].values
        qtc2 = df[col2].values
        
        n_plot = min(5000, len(qtc1))
        idx = np.random.choice(len(qtc1), n_plot, replace=False)
        
        ax.scatter(qtc1[idx], qtc2[idx], alpha=0.3, s=5)
        
        # Identity line
        lims = [min(qtc1.min(), qtc2.min()), max(qtc1.max(), qtc2.max())]
        ax.plot(lims, lims, 'r--', linewidth=2)
        
        r, _ = stats.pearsonr(qtc1, qtc2)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f'r = {r:.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'formula_agreement.png', dpi=150)
    plt.close()
    
    logger.info(f"Plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='./results/stream_c/validation')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples for analysis (None=all)')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Kepler-ECG Phase 4.5 - Task 5: Validation & Comparison")
    print("=" * 70)
    
    # Load data
    df = load_dataset(Path(args.dataset))
    
    # Optional sampling for faster analysis
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)
        logger.info(f"Sampled to {len(df)} records")
    
    # Calculate all QTc values
    print("\nCalculating QTc with all formulas...")
    df = calculate_all_qtc(df)
    
    # Define columns for analysis
    qtc_cols = [
        'QTc_Bazett', 'QTc_Fridericia', 'QTc_Framingham', 'QTc_Hodges',
        'QTc_Kepler_Linear', 'QTc_Kepler_Cubic', 'QTc_Kepler_Factor'
    ]
    
    # Run analyses
    print("\n1. Analyzing HR independence...")
    hr_results = analyze_hr_independence(df, qtc_cols)
    
    print("2. Analyzing by HR bins...")
    bin_results = analyze_by_hr_bins(df, qtc_cols)
    
    print("3. Cross-validation stability...")
    cv_results = cross_validation_stability(df, qtc_cols)
    
    print("4. Clinical threshold analysis...")
    clinical_results = clinical_threshold_analysis(df, qtc_cols)
    
    print("5. Formula agreement analysis...")
    agreement_results = formula_agreement_analysis(df, qtc_cols)
    
    print("6. Creating validation plots...")
    create_validation_plots(df, output_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n### HR Independence (target: |r| ≈ 0)")
    print(f"{'Formula':<25} {'Pearson r':>12} {'|r|':>10} {'Rank':>8}")
    print("-" * 60)
    
    sorted_by_r = sorted(hr_results.items(), key=lambda x: abs(x[1]['pearson_r']))
    for rank, (formula, data) in enumerate(sorted_by_r, 1):
        r = data['pearson_r']
        print(f"{formula:<25} {r:>+12.4f} {abs(r):>10.4f} {rank:>8}")
    
    print("\n### Cross-Bin Variation (QTc mean across HR bins)")
    print(f"{'Formula':<25} {'Range (ms)':>12} {'SD (ms)':>10}")
    print("-" * 50)
    
    for formula, data in bin_results.items():
        if 'cross_bin_variation' in data:
            var = data['cross_bin_variation']
            print(f"{formula:<25} {var['range_of_means']:>12.1f} {var['std_of_means']:>10.1f}")
    
    print("\n### Clinical Thresholds")
    print(f"{'Formula':<25} {'Mean QTc':>10} {'>450ms (%)':>12} {'>500ms (%)':>12}")
    print("-" * 65)
    
    for formula, data in clinical_results.items():
        mean_qtc = data['mean_qtc']
        prol = data['prolonged_450ms']['pct']
        high = data['high_risk_500ms']['pct']
        print(f"{formula:<25} {mean_qtc:>10.1f} {prol:>12.1f} {high:>12.1f}")
    
    # Save comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'task': 'Task 5 - Validation & Comparison',
        'n_samples': len(df),
        'hr_independence': hr_results,
        'hr_bin_analysis': bin_results,
        'cv_stability': cv_results,
        'clinical_thresholds': clinical_results,
        'formula_agreement': agreement_results,
        'ranking': {
            'by_hr_independence': [
                {'rank': i+1, 'formula': f, 'abs_r': abs(d['pearson_r'])}
                for i, (f, d) in enumerate(sorted_by_r)
            ]
        },
        'kepler_formulas': {
            'linear': 'QTc = QT - 184.54*RR + 156.72',
            'cubic': 'QTc = QT - 495.11*cbrt(RR) + 466.81',
            'factor': 'QTc = QT * (0.364/RR + 0.562)',
        }
    }
    
    report_file = output_path / 'task5_validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown summary
    md_content = generate_comparison_table(hr_results, clinical_results)
    md_file = output_path / 'validation_summary.md'
    with open(md_file, 'w') as f:
        f.write(md_content)
    
    print(f"\nReports saved to {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
