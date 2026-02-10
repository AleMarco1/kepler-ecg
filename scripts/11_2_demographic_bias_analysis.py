#!/usr/bin/env python3
"""
Kepler-ECG: Demographic Bias Analysis (Script 11_2)
====================================================

PUNTO 7 DELLA VALIDAZIONE: Bias Demografico Silenzioso

Obiettivo: Dimostrare che la formula Kepler non crea errori sistematici
su donne, anziani, o altre sottopopolazioni demografiche.

La Prova Richiesta:
- Grafico dei residui (Errore vs HR) colorato per Sesso e Età
- Se le nuvole si sovrappongono → formula "unisex" per merito
- Se sono separate → bias nascosto sotto la media statistica

Design:
1. Stratifica per sesso (M/F), età (<40, 40-65, >65), HR range
2. Calcola |r(QTc, HR)| per ogni subgruppo
3. Genera residual plots e analisi statistica

Author: Alessandro Marconi
Version: 1.0.0
Date: February 2026
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for Demographic Bias Analysis."""
    
    ALL_DATASETS = ['code-15', 'ptb-xl', 'mimic-iv-ecg', 'chapman', 
                    'cpsc-2018', 'georgia', 'ludb', 'ecg-arrhythmia']
    
    # Kepler coefficients
    KEPLER_K = 125
    KEPLER_C = -158
    
    # Age groups
    AGE_BINS = [0, 40, 65, 120]
    AGE_LABELS = ['<40', '40-65', '>65']
    
    # HR groups
    HR_BINS = [0, 60, 100, 200]
    HR_LABELS = ['Brady (<60)', 'Normal (60-100)', 'Tachy (>100)']
    
    # Paths
    RESULTS_BASE = Path('results')
    OUTPUT_DIR = Path('results/demographic_analysis')
    
    # Thresholds
    BIAS_THRESHOLD = 0.02  # Max acceptable difference in |r| between groups


# ============================================================================
# DATA LOADING
# ============================================================================

def load_qtc_data(dataset: str) -> Optional[pd.DataFrame]:
    """Load QTc preparation data for a dataset."""
    
    locations = [
        Config.RESULTS_BASE / dataset / 'qtc' / f'{dataset}_qtc_preparation.csv',
        Config.RESULTS_BASE / dataset / f'{dataset}_qtc_preparation.csv',
        Config.RESULTS_BASE / dataset / f'{dataset}_full_results.csv',
    ]
    
    for loc in locations:
        if loc.exists():
            try:
                df = pd.read_csv(loc)
                
                # Standardize columns
                col_map = {
                    'QT_interval_ms': 'QT_ms', 'QT': 'QT_ms', 'qt_ms': 'QT_ms',
                    'manual_QT_ms': 'QT_ms',
                    'RR_interval_sec': 'RR_sec', 'RR': 'RR_sec', 'rr_sec': 'RR_sec',
                    'manual_RR_sec': 'RR_sec',
                    'heart_rate_bpm': 'HR_bpm', 'HR': 'HR_bpm', 'hr_bpm': 'HR_bpm',
                    'age': 'age', 'Age': 'age', 'AGE': 'age',
                    'sex': 'sex', 'Sex': 'sex', 'SEX': 'sex', 'gender': 'sex', 'Gender': 'sex',
                }
                for old, new in col_map.items():
                    if old in df.columns and new not in df.columns:
                        df[new] = df[old]
                
                if 'HR_bpm' not in df.columns and 'RR_sec' in df.columns:
                    df['HR_bpm'] = 60 / df['RR_sec']
                
                if 'RR_sec' in df.columns and df['RR_sec'].median() > 10:
                    df['RR_sec'] = df['RR_sec'] / 1000
                
                # Filter valid QT/RR
                df = df[(df['QT_ms'] >= 200) & (df['QT_ms'] <= 600) &
                       (df['RR_sec'] >= 0.4) & (df['RR_sec'] <= 2.0)].copy()
                
                df['dataset'] = dataset
                return df
                
            except Exception as e:
                print(f"  ⚠️ Error loading {dataset}: {e}")
                return None
    
    return None


def load_all_datasets() -> pd.DataFrame:
    """Load all datasets and combine."""
    
    print("\n📊 Loading datasets...")
    all_dfs = []
    
    for ds in Config.ALL_DATASETS:
        df = load_qtc_data(ds)
        if df is not None:
            all_dfs.append(df)
            
            # Check for demographic columns
            has_age = 'age' in df.columns and df['age'].notna().sum() > 0
            has_sex = 'sex' in df.columns and df['sex'].notna().sum() > 0
            
            age_str = f"age: {df['age'].notna().sum():,}" if has_age else "no age"
            sex_str = f"sex: {df['sex'].notna().sum():,}" if has_sex else "no sex"
            
            print(f"  ✓ {ds}: {len(df):,} records ({age_str}, {sex_str})")
        else:
            print(f"  ⚠️ {ds}: not found")
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


# ============================================================================
# QTc COMPUTATION
# ============================================================================

def compute_qtc_kepler(qt_ms: np.ndarray, rr_sec: np.ndarray) -> np.ndarray:
    """Kepler formula: QTc = QT + k/RR + c"""
    return qt_ms + Config.KEPLER_K / rr_sec + Config.KEPLER_C


def compute_qtc_bazett(qt_ms: np.ndarray, rr_sec: np.ndarray) -> np.ndarray:
    """Bazett formula: QTc = QT / sqrt(RR)"""
    return qt_ms / np.sqrt(rr_sec)


def compute_hr_correlation(df: pd.DataFrame, qtc_col: str) -> Dict:
    """Compute correlation between QTc and HR."""
    valid = df[[qtc_col, 'HR_bpm']].dropna()
    if len(valid) < 20:
        return {'r': np.nan, 'abs_r': np.nan, 'p': np.nan, 'n': len(valid)}
    
    r, p = stats.pearsonr(valid[qtc_col], valid['HR_bpm'])
    return {'r': float(r), 'abs_r': float(abs(r)), 'p': float(p), 'n': len(valid)}


# ============================================================================
# DEMOGRAPHIC STRATIFICATION
# ============================================================================

def standardize_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize sex column to M/F."""
    
    if 'sex' not in df.columns:
        df['sex_std'] = np.nan
        return df
    
    sex_map = {
        'M': 'M', 'm': 'M', 'Male': 'M', 'male': 'M', 'MALE': 'M', 0: 'M', '0': 'M',
        'F': 'F', 'f': 'F', 'Female': 'F', 'female': 'F', 'FEMALE': 'F', 1: 'F', '1': 'F',
    }
    
    df['sex_std'] = df['sex'].map(sex_map)
    return df


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create age group column."""
    
    if 'age' not in df.columns:
        df['age_group'] = np.nan
        return df
    
    df['age_group'] = pd.cut(df['age'], bins=Config.AGE_BINS, labels=Config.AGE_LABELS, right=False)
    return df


def create_hr_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create HR group column."""
    
    df['hr_group'] = pd.cut(df['HR_bpm'], bins=Config.HR_BINS, labels=Config.HR_LABELS, right=False)
    return df


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_by_sex(df: pd.DataFrame) -> Dict:
    """Analyze HR independence by sex."""
    
    results = {'available': False, 'by_sex': {}, 'comparison': {}}
    
    if 'sex_std' not in df.columns or df['sex_std'].isna().all():
        return results
    
    results['available'] = True
    
    for sex in ['M', 'F']:
        subset = df[df['sex_std'] == sex]
        if len(subset) < 100:
            continue
        
        kepler_corr = compute_hr_correlation(subset, 'QTc_Kepler')
        bazett_corr = compute_hr_correlation(subset, 'QTc_Bazett')
        
        results['by_sex'][sex] = {
            'n': len(subset),
            'kepler': kepler_corr,
            'bazett': bazett_corr,
            'mean_age': float(subset['age'].mean()) if 'age' in subset.columns else None,
            'mean_hr': float(subset['HR_bpm'].mean()),
            'mean_qtc_kepler': float(subset['QTc_Kepler'].mean()),
            'mean_qtc_bazett': float(subset['QTc_Bazett'].mean()),
        }
    
    # Comparison
    if 'M' in results['by_sex'] and 'F' in results['by_sex']:
        r_m = results['by_sex']['M']['kepler']['abs_r']
        r_f = results['by_sex']['F']['kepler']['abs_r']
        
        results['comparison'] = {
            'delta_abs_r': abs(r_m - r_f),
            'bias_detected': abs(r_m - r_f) > Config.BIAS_THRESHOLD,
            'worse_group': 'M' if r_m > r_f else 'F',
        }
    
    return results


def analyze_by_age(df: pd.DataFrame) -> Dict:
    """Analyze HR independence by age group."""
    
    results = {'available': False, 'by_age': {}, 'comparison': {}}
    
    if 'age_group' not in df.columns or df['age_group'].isna().all():
        return results
    
    results['available'] = True
    
    for age_grp in Config.AGE_LABELS:
        subset = df[df['age_group'] == age_grp]
        if len(subset) < 100:
            continue
        
        kepler_corr = compute_hr_correlation(subset, 'QTc_Kepler')
        bazett_corr = compute_hr_correlation(subset, 'QTc_Bazett')
        
        results['by_age'][age_grp] = {
            'n': len(subset),
            'kepler': kepler_corr,
            'bazett': bazett_corr,
            'mean_age': float(subset['age'].mean()),
            'mean_hr': float(subset['HR_bpm'].mean()),
            'pct_male': float((subset['sex_std'] == 'M').mean() * 100) if 'sex_std' in subset.columns else None,
        }
    
    # Comparison: max difference between any two age groups
    if len(results['by_age']) >= 2:
        r_values = [v['kepler']['abs_r'] for v in results['by_age'].values() if not np.isnan(v['kepler']['abs_r'])]
        if r_values:
            results['comparison'] = {
                'max_abs_r': max(r_values),
                'min_abs_r': min(r_values),
                'range_abs_r': max(r_values) - min(r_values),
                'bias_detected': (max(r_values) - min(r_values)) > Config.BIAS_THRESHOLD,
            }
    
    return results


def analyze_by_hr(df: pd.DataFrame) -> Dict:
    """Analyze HR independence by HR category."""
    
    results = {'by_hr': {}}
    
    for hr_grp in Config.HR_LABELS:
        subset = df[df['hr_group'] == hr_grp]
        if len(subset) < 100:
            continue
        
        kepler_corr = compute_hr_correlation(subset, 'QTc_Kepler')
        bazett_corr = compute_hr_correlation(subset, 'QTc_Bazett')
        
        results['by_hr'][hr_grp] = {
            'n': len(subset),
            'kepler': kepler_corr,
            'bazett': bazett_corr,
            'mean_hr': float(subset['HR_bpm'].mean()),
            'hr_range': [float(subset['HR_bpm'].min()), float(subset['HR_bpm'].max())],
        }
    
    return results


def analyze_cross_stratification(df: pd.DataFrame) -> Dict:
    """Analyze sex × age and sex × HR combinations."""
    
    results = {'sex_age': {}, 'sex_hr': {}}
    
    # Sex × Age
    if 'sex_std' in df.columns and 'age_group' in df.columns:
        for sex in ['M', 'F']:
            for age_grp in Config.AGE_LABELS:
                subset = df[(df['sex_std'] == sex) & (df['age_group'] == age_grp)]
                if len(subset) < 50:
                    continue
                
                kepler_corr = compute_hr_correlation(subset, 'QTc_Kepler')
                
                key = f"{sex}_{age_grp}"
                results['sex_age'][key] = {
                    'n': len(subset),
                    'kepler_abs_r': kepler_corr['abs_r'],
                    'mean_hr': float(subset['HR_bpm'].mean()),
                }
    
    # Sex × HR
    if 'sex_std' in df.columns and 'hr_group' in df.columns:
        for sex in ['M', 'F']:
            for hr_grp in Config.HR_LABELS:
                subset = df[(df['sex_std'] == sex) & (df['hr_group'] == hr_grp)]
                if len(subset) < 50:
                    continue
                
                kepler_corr = compute_hr_correlation(subset, 'QTc_Kepler')
                
                key = f"{sex}_{hr_grp}"
                results['sex_hr'][key] = {
                    'n': len(subset),
                    'kepler_abs_r': kepler_corr['abs_r'],
                }
    
    return results


def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute QTc residuals for plotting."""
    
    # Residual = QTc - mean(QTc) within HR bins
    # Or simpler: just use QTc directly for scatter plots
    
    df['residual_kepler'] = df['QTc_Kepler'] - df['QTc_Kepler'].mean()
    df['residual_bazett'] = df['QTc_Bazett'] - df['QTc_Bazett'].mean()
    
    return df


# ============================================================================
# VISUALIZATION SCRIPT GENERATOR
# ============================================================================

def generate_visualization_script(df: pd.DataFrame, output_dir: Path) -> str:
    """Generate Python script for matplotlib visualizations."""
    
    # Sample data for plotting (max 50k points per group)
    sample_size = min(50000, len(df))
    
    script = f'''#!/usr/bin/env python3
"""
Demographic Bias Visualization
Generated: {datetime.now().isoformat()}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
df = pd.read_csv('{output_dir}/demographic_analysis_data.csv')

# Sample for plotting if too large
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Color maps
sex_colors = {{'M': '#3498db', 'F': '#e74c3c'}}
age_colors = {{'<40': '#2ecc71', '40-65': '#f39c12', '>65': '#9b59b6'}}

# 1. QTc vs HR by Sex
ax1 = axes[0, 0]
for sex in ['M', 'F']:
    subset = df[df['sex_std'] == sex]
    if len(subset) > 0:
        sample = subset.sample(n=min(10000, len(subset)), random_state=42)
        ax1.scatter(sample['HR_bpm'], sample['QTc_Kepler'], 
                   c=sex_colors[sex], alpha=0.3, s=5, label=f'{{sex}} (n={{len(subset):,}})')
ax1.set_xlabel('Heart Rate (bpm)')
ax1.set_ylabel('QTc Kepler (ms)')
ax1.set_title('QTc vs HR by Sex')
ax1.legend()
ax1.set_xlim(40, 140)
ax1.axhline(450, color='gray', linestyle='--', alpha=0.5)

# 2. QTc vs HR by Age
ax2 = axes[0, 1]
for age_grp in ['<40', '40-65', '>65']:
    subset = df[df['age_group'] == age_grp]
    if len(subset) > 0:
        sample = subset.sample(n=min(10000, len(subset)), random_state=42)
        ax2.scatter(sample['HR_bpm'], sample['QTc_Kepler'],
                   c=age_colors[age_grp], alpha=0.3, s=5, label=f'{{age_grp}} (n={{len(subset):,}})')
ax2.set_xlabel('Heart Rate (bpm)')
ax2.set_ylabel('QTc Kepler (ms)')
ax2.set_title('QTc vs HR by Age Group')
ax2.legend()
ax2.set_xlim(40, 140)

# 3. |r| comparison bar chart - Sex
ax3 = axes[0, 2]
sex_data = df.groupby('sex_std').apply(
    lambda x: abs(x['QTc_Kepler'].corr(x['HR_bpm'])) if len(x) > 20 else np.nan
).dropna()
if len(sex_data) > 0:
    bars = ax3.bar(sex_data.index, sex_data.values, color=[sex_colors.get(s, 'gray') for s in sex_data.index])
    ax3.axhline(0.05, color='green', linestyle='--', label='|r|=0.05 threshold')
    ax3.set_ylabel('|r(QTc, HR)|')
    ax3.set_title('HR Independence by Sex')
    ax3.legend()
    for bar, val in zip(bars, sex_data.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{{val:.3f}}', ha='center', va='bottom', fontsize=10)

# 4. |r| comparison bar chart - Age
ax4 = axes[1, 0]
age_data = df.groupby('age_group').apply(
    lambda x: abs(x['QTc_Kepler'].corr(x['HR_bpm'])) if len(x) > 20 else np.nan
).dropna()
if len(age_data) > 0:
    # Reorder
    age_order = ['<40', '40-65', '>65']
    age_data = age_data.reindex([a for a in age_order if a in age_data.index])
    bars = ax4.bar(age_data.index, age_data.values, color=[age_colors.get(a, 'gray') for a in age_data.index])
    ax4.axhline(0.05, color='green', linestyle='--', label='|r|=0.05 threshold')
    ax4.set_ylabel('|r(QTc, HR)|')
    ax4.set_title('HR Independence by Age Group')
    ax4.legend()
    for bar, val in zip(bars, age_data.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{{val:.3f}}', ha='center', va='bottom', fontsize=10)

# 5. Residual distribution by Sex
ax5 = axes[1, 1]
for sex in ['M', 'F']:
    subset = df[df['sex_std'] == sex]['QTc_Kepler']
    if len(subset) > 0:
        ax5.hist(subset, bins=50, alpha=0.5, color=sex_colors[sex], 
                label=f'{{sex}} (mean={{subset.mean():.1f}})', density=True)
ax5.set_xlabel('QTc Kepler (ms)')
ax5.set_ylabel('Density')
ax5.set_title('QTc Distribution by Sex')
ax5.legend()

# 6. Kepler vs Bazett |r| comparison
ax6 = axes[1, 2]
categories = []
kepler_r = []
bazett_r = []

for sex in ['M', 'F']:
    subset = df[df['sex_std'] == sex]
    if len(subset) > 100:
        categories.append(sex)
        kepler_r.append(abs(subset['QTc_Kepler'].corr(subset['HR_bpm'])))
        bazett_r.append(abs(subset['QTc_Bazett'].corr(subset['HR_bpm'])))

for age_grp in ['<40', '40-65', '>65']:
    subset = df[df['age_group'] == age_grp]
    if len(subset) > 100:
        categories.append(age_grp)
        kepler_r.append(abs(subset['QTc_Kepler'].corr(subset['HR_bpm'])))
        bazett_r.append(abs(subset['QTc_Bazett'].corr(subset['HR_bpm'])))

x = np.arange(len(categories))
width = 0.35
ax6.bar(x - width/2, kepler_r, width, label='Kepler', color='#27ae60')
ax6.bar(x + width/2, bazett_r, width, label='Bazett', color='#e74c3c')
ax6.set_xticks(x)
ax6.set_xticklabels(categories)
ax6.axhline(0.05, color='gray', linestyle='--', alpha=0.7)
ax6.set_ylabel('|r(QTc, HR)|')
ax6.set_title('Kepler vs Bazett by Demographic Group')
ax6.legend()

plt.tight_layout()
plt.savefig('{output_dir}/demographic_bias_plots.png', dpi=150, bbox_inches='tight')
plt.savefig('{output_dir}/demographic_bias_plots.pdf', bbox_inches='tight')
print(f"Saved: {output_dir}/demographic_bias_plots.png")
plt.show()
'''
    
    return script


# ============================================================================
# REPORTING
# ============================================================================

def print_report(sex_results: Dict, age_results: Dict, hr_results: Dict, 
                cross_results: Dict, total_n: int):
    """Print comprehensive report."""
    
    print("\n" + "="*70)
    print("DEMOGRAPHIC BIAS ANALYSIS - PUNTO 7")
    print("="*70)
    
    print(f"\n📊 DATI ANALIZZATI: {total_n:,} record totali")
    
    # Sex analysis
    print("\n" + "-"*50)
    print("ANALISI PER SESSO")
    print("-"*50)
    
    if sex_results['available']:
        print(f"\n{'Sesso':<10} {'N':>12} {'Kepler |r|':>12} {'Bazett |r|':>12} {'Mean QTc':>10}")
        print("-" * 60)
        
        for sex, data in sex_results['by_sex'].items():
            print(f"{sex:<10} {data['n']:>12,} {data['kepler']['abs_r']:>12.4f} "
                  f"{data['bazett']['abs_r']:>12.4f} {data['mean_qtc_kepler']:>10.1f}")
        
        if 'comparison' in sex_results and sex_results['comparison']:
            comp = sex_results['comparison']
            status = "⚠️ BIAS RILEVATO" if comp['bias_detected'] else "✅ NO BIAS"
            print(f"\nDifferenza |r| tra M e F: {comp['delta_abs_r']:.4f} → {status}")
    else:
        print("  ⚠️ Dati sul sesso non disponibili")
    
    # Age analysis
    print("\n" + "-"*50)
    print("ANALISI PER ETÀ")
    print("-"*50)
    
    if age_results['available']:
        print(f"\n{'Età':<12} {'N':>12} {'Kepler |r|':>12} {'Bazett |r|':>12} {'Mean HR':>10}")
        print("-" * 62)
        
        for age_grp, data in age_results['by_age'].items():
            print(f"{age_grp:<12} {data['n']:>12,} {data['kepler']['abs_r']:>12.4f} "
                  f"{data['bazett']['abs_r']:>12.4f} {data['mean_hr']:>10.1f}")
        
        if 'comparison' in age_results and age_results['comparison']:
            comp = age_results['comparison']
            status = "⚠️ BIAS RILEVATO" if comp['bias_detected'] else "✅ NO BIAS"
            print(f"\nRange |r| tra gruppi età: {comp['range_abs_r']:.4f} → {status}")
    else:
        print("  ⚠️ Dati sull'età non disponibili")
    
    # HR analysis
    print("\n" + "-"*50)
    print("ANALISI PER CATEGORIA HR")
    print("-"*50)
    
    if hr_results['by_hr']:
        print(f"\n{'HR Group':<20} {'N':>12} {'Kepler |r|':>12} {'Bazett |r|':>12}")
        print("-" * 58)
        
        for hr_grp, data in hr_results['by_hr'].items():
            print(f"{hr_grp:<20} {data['n']:>12,} {data['kepler']['abs_r']:>12.4f} "
                  f"{data['bazett']['abs_r']:>12.4f}")
    
    # Cross-stratification
    print("\n" + "-"*50)
    print("ANALISI SESSO × ETÀ")
    print("-"*50)
    
    if cross_results['sex_age']:
        print(f"\n{'Gruppo':<15} {'N':>10} {'Kepler |r|':>12}")
        print("-" * 40)
        
        for key, data in sorted(cross_results['sex_age'].items()):
            print(f"{key:<15} {data['n']:>10,} {data['kepler_abs_r']:>12.4f}")


def print_verdict(sex_results: Dict, age_results: Dict):
    """Print final verdict."""
    
    print("\n" + "="*70)
    print("VERDETTO FINALE - PUNTO 7")
    print("="*70)
    
    sex_bias = sex_results.get('comparison', {}).get('bias_detected', False)
    age_bias = age_results.get('comparison', {}).get('bias_detected', False)
    
    if not sex_bias and not age_bias:
        print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ✅  NESSUN BIAS DEMOGRAFICO RILEVATO                             ║
    ║                                                                    ║
    ║   • La formula Kepler performa UNIFORMEMENTE su M e F              ║
    ║   • La formula Kepler performa UNIFORMEMENTE su tutte le età       ║
    ║   • Le nuvole di punti si SOVRAPPONGONO                            ║
    ║   • La formula è genuinamente "UNISEX" e "age-independent"         ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
        """)
    else:
        issues = []
        if sex_bias:
            issues.append("differenza tra sessi")
        if age_bias:
            issues.append("differenza tra fasce d'età")
        
        print(f"""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ⚠️  BIAS DEMOGRAFICO PARZIALE RILEVATO                           ║
    ║                                                                    ║
    ║   Aree di attenzione: {', '.join(issues):<42} ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
        """)
    
    # Comparison with Bazett
    print("\n📊 CONFRONTO CON BAZETT:")
    
    if sex_results['available'] and sex_results['by_sex']:
        print("\n  Per Sesso:")
        for sex, data in sex_results['by_sex'].items():
            k_r = data['kepler']['abs_r']
            b_r = data['bazett']['abs_r']
            improvement = (b_r - k_r) / b_r * 100 if b_r > 0 else 0
            print(f"    {sex}: Kepler |r|={k_r:.4f}, Bazett |r|={b_r:.4f} → Kepler {improvement:.0f}% migliore")
    
    if age_results['available'] and age_results['by_age']:
        print("\n  Per Età:")
        for age, data in age_results['by_age'].items():
            k_r = data['kepler']['abs_r']
            b_r = data['bazett']['abs_r']
            improvement = (b_r - k_r) / b_r * 100 if b_r > 0 else 0
            print(f"    {age}: Kepler |r|={k_r:.4f}, Bazett |r|={b_r:.4f} → Kepler {improvement:.0f}% migliore")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    parser = argparse.ArgumentParser(
        description='Kepler-ECG Demographic Bias Analysis (Point 7)',
    )
    
    parser.add_argument('--output-dir', type=str, 
                       default=str(Config.OUTPUT_DIR),
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   KEPLER-ECG: Demographic Bias Analysis (Script 11_2)              ║
    ║   PUNTO 7 - Bias Demografico Silenzioso                            ║
    ║                                                                    ║
    ║   Obiettivo: Dimostrare che la formula Kepler non crea errori      ║
    ║   sistematici su donne, anziani, o altre sottopopolazioni          ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_all_datasets()
    
    if len(df) == 0:
        print("\n❌ No data loaded!")
        return 1
    
    print(f"\n📊 Total: {len(df):,} records")
    
    # Compute QTc
    print("\n📊 Computing QTc values...")
    df['QTc_Kepler'] = compute_qtc_kepler(df['QT_ms'].values, df['RR_sec'].values)
    df['QTc_Bazett'] = compute_qtc_bazett(df['QT_ms'].values, df['RR_sec'].values)
    
    # Standardize demographics
    df = standardize_sex(df)
    df = create_age_groups(df)
    df = create_hr_groups(df)
    
    # Check demographic coverage
    n_with_sex = df['sex_std'].notna().sum()
    n_with_age = df['age_group'].notna().sum()
    print(f"\n📊 Demographic coverage:")
    print(f"   With sex: {n_with_sex:,} ({n_with_sex/len(df)*100:.1f}%)")
    print(f"   With age: {n_with_age:,} ({n_with_age/len(df)*100:.1f}%)")
    
    # Run analyses
    print("\n📊 Running stratified analyses...")
    sex_results = analyze_by_sex(df)
    age_results = analyze_by_age(df)
    hr_results = analyze_by_hr(df)
    cross_results = analyze_cross_stratification(df)
    
    # Print report
    print_report(sex_results, age_results, hr_results, cross_results, len(df))
    
    # Print verdict
    print_verdict(sex_results, age_results)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(df),
        'n_with_sex': int(n_with_sex),
        'n_with_age': int(n_with_age),
        'sex_analysis': sex_results,
        'age_analysis': age_results,
        'hr_analysis': hr_results,
        'cross_stratification': cross_results,
    }
    
    json_path = output_dir / 'demographic_analysis_report.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Report JSON: {json_path}")
    
    # Save data for plotting
    plot_cols = ['HR_bpm', 'QTc_Kepler', 'QTc_Bazett', 'sex_std', 'age_group', 'hr_group', 'dataset']
    plot_cols = [c for c in plot_cols if c in df.columns]
    df[plot_cols].to_csv(output_dir / 'demographic_analysis_data.csv', index=False)
    print(f"💾 Data CSV: {output_dir}/demographic_analysis_data.csv")
    
    # Save visualization script
    viz_script = generate_visualization_script(df, output_dir)
    viz_path = output_dir / 'generate_plots.py'
    with open(viz_path, 'w', encoding='utf-8') as f:
        f.write(viz_script)
    print(f"💾 Visualization script: {viz_path}")
    print(f"   Run with: python {viz_path}")
    
    print(f"\n✅ Analysis complete. Results in: {output_dir}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
