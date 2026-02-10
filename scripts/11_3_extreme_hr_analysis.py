#!/usr/bin/env python3
"""
Kepler-ECG: Extreme HR Analysis (Script 11_3)
==============================================

PUNTO 4 DELLA VALIDAZIONE: Il Tallone d'Achille della Bradicardia

Obiettivo: Verificare che la formula Kepler non sottocorregga in bradicardia
estrema (HR < 50 bpm) e non sovracorregga in tachicardia (HR > 100 bpm).

La Prova Richiesta:
- Isolare record con HR < 50 (bradicardia severa) e HR > 100 (tachicardia)
- Calcolare concordanza diagnostica Kepler vs Bazett
- Se Kepler non peggiora rispetto a Bazett → punto superato

Metriche chiave:
- Falsi normali: QTc < soglia quando dovrebbe essere lungo
- Falsi allungati: QTc ≥ soglia quando dovrebbe essere normale
- HR-independence in ogni strato

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
    """Configuration for Extreme HR Analysis."""
    
    ALL_DATASETS = ['code-15', 'ptb-xl', 'mimic-iv-ecg', 'chapman', 
                    'cpsc-2018', 'georgia', 'ludb', 'ecg-arrhythmia']
    
    # Kepler coefficients
    KEPLER_K = 125
    KEPLER_C = -158
    
    # HR stratification (fine-grained)
    HR_STRATA = {
        'Brady_severe': (0, 50),      # Bradicardia severa
        'Brady_mild': (50, 60),       # Bradicardia lieve
        'Normal_low': (60, 75),       # Normale basso
        'Normal_high': (75, 100),     # Normale alto
        'Tachy_mild': (100, 120),     # Tachicardia lieve
        'Tachy_severe': (120, 200),   # Tachicardia severa
    }
    
    # Clinical thresholds for QTc prolongation
    QTC_THRESHOLD_MALE = 450      # ms
    QTC_THRESHOLD_FEMALE = 460    # ms
    QTC_THRESHOLD_UNIVERSAL = 450 # ms (conservative)
    
    # Severity levels
    QTC_BORDERLINE = 450
    QTC_PROLONGED = 470
    QTC_HIGH_RISK = 500
    
    # Paths
    RESULTS_BASE = Path('results')
    OUTPUT_DIR = Path('results/extreme_hr_analysis')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_qtc_data(dataset: str) -> Optional[pd.DataFrame]:
    """Load QTc preparation data for a dataset."""
    
    locations = [
        Config.RESULTS_BASE / dataset / 'qtc' / f'{dataset}_qtc_preparation.csv',
        Config.RESULTS_BASE / dataset / f'{dataset}_qtc_preparation.csv',
    ]
    
    for loc in locations:
        if loc.exists():
            try:
                df = pd.read_csv(loc)
                
                # Standardize columns
                col_map = {
                    'QT_interval_ms': 'QT_ms', 'QT': 'QT_ms',
                    'RR_interval_sec': 'RR_sec', 'RR': 'RR_sec',
                    'heart_rate_bpm': 'HR_bpm', 'HR': 'HR_bpm',
                    'sex': 'sex', 'Sex': 'sex', 'gender': 'sex',
                }
                for old, new in col_map.items():
                    if old in df.columns and new not in df.columns:
                        df[new] = df[old]
                
                if 'HR_bpm' not in df.columns and 'RR_sec' in df.columns:
                    df['HR_bpm'] = 60 / df['RR_sec']
                
                if 'RR_sec' in df.columns and df['RR_sec'].median() > 10:
                    df['RR_sec'] = df['RR_sec'] / 1000
                
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
            print(f"  ✓ {ds}: {len(df):,} records")
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


# ============================================================================
# QTc COMPUTATION
# ============================================================================

def compute_all_qtc(df: pd.DataFrame) -> pd.DataFrame:
    """Compute QTc with multiple formulas."""
    
    qt = df['QT_ms'].values
    rr = df['RR_sec'].values
    
    # Kepler
    df['QTc_Kepler'] = qt + Config.KEPLER_K / rr + Config.KEPLER_C
    
    # Bazett
    df['QTc_Bazett'] = qt / np.sqrt(rr)
    
    # Fridericia
    df['QTc_Fridericia'] = qt / np.cbrt(rr)
    
    # Framingham
    df['QTc_Framingham'] = qt + 154 * (1 - rr)
    
    # Hodges
    hr = df['HR_bpm'].values
    df['QTc_Hodges'] = qt + 1.75 * (hr - 60)
    
    return df


def assign_hr_stratum(df: pd.DataFrame) -> pd.DataFrame:
    """Assign HR stratum to each record."""
    
    conditions = []
    choices = []
    
    for name, (low, high) in Config.HR_STRATA.items():
        conditions.append((df['HR_bpm'] >= low) & (df['HR_bpm'] < high))
        choices.append(name)
    
    df['hr_stratum'] = np.select(conditions, choices, default='Unknown')
    
    return df


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_hr_stratum(df: pd.DataFrame, stratum_name: str) -> Dict:
    """Analyze a single HR stratum."""
    
    subset = df[df['hr_stratum'] == stratum_name]
    n = len(subset)
    
    if n < 50:
        return {'n': n, 'insufficient_data': True}
    
    result = {
        'n': n,
        'hr_range': [float(subset['HR_bpm'].min()), float(subset['HR_bpm'].max())],
        'hr_mean': float(subset['HR_bpm'].mean()),
        'hr_std': float(subset['HR_bpm'].std()),
        'formulas': {},
    }
    
    # Analyze each formula
    for formula in ['Kepler', 'Bazett', 'Fridericia', 'Framingham', 'Hodges']:
        col = f'QTc_{formula}'
        if col not in subset.columns:
            continue
        
        qtc = subset[col].values
        hr = subset['HR_bpm'].values
        
        # HR correlation
        r, p = stats.pearsonr(qtc, hr)
        
        # QTc statistics
        result['formulas'][formula] = {
            'r': float(r),
            'abs_r': float(abs(r)),
            'p': float(p),
            'qtc_mean': float(np.mean(qtc)),
            'qtc_std': float(np.std(qtc)),
            'qtc_median': float(np.median(qtc)),
            'qtc_p5': float(np.percentile(qtc, 5)),
            'qtc_p95': float(np.percentile(qtc, 95)),
        }
        
        # Prolongation rates
        result['formulas'][formula]['pct_above_450'] = float((qtc >= 450).mean() * 100)
        result['formulas'][formula]['pct_above_470'] = float((qtc >= 470).mean() * 100)
        result['formulas'][formula]['pct_above_500'] = float((qtc >= 500).mean() * 100)
    
    return result


def analyze_diagnostic_concordance(df: pd.DataFrame, threshold: float = 450) -> Dict:
    """
    Analyze diagnostic concordance between Kepler and Bazett.
    
    Categories:
    - Both normal: Kepler < threshold AND Bazett < threshold
    - Both prolonged: Kepler >= threshold AND Bazett >= threshold
    - Kepler only prolonged: Kepler >= threshold AND Bazett < threshold (potential false positive)
    - Bazett only prolonged: Kepler < threshold AND Bazett >= threshold (potential false negative for Kepler)
    """
    
    results = {}
    
    for stratum_name in Config.HR_STRATA.keys():
        subset = df[df['hr_stratum'] == stratum_name]
        n = len(subset)
        
        if n < 50:
            results[stratum_name] = {'n': n, 'insufficient_data': True}
            continue
        
        kepler = subset['QTc_Kepler'].values
        bazett = subset['QTc_Bazett'].values
        
        both_normal = ((kepler < threshold) & (bazett < threshold)).sum()
        both_prolonged = ((kepler >= threshold) & (bazett >= threshold)).sum()
        kepler_only = ((kepler >= threshold) & (bazett < threshold)).sum()
        bazett_only = ((kepler < threshold) & (bazett >= threshold)).sum()
        
        results[stratum_name] = {
            'n': n,
            'threshold': threshold,
            'both_normal': int(both_normal),
            'both_prolonged': int(both_prolonged),
            'kepler_only_prolonged': int(kepler_only),  # Kepler more sensitive
            'bazett_only_prolonged': int(bazett_only),  # Potential Kepler false negative
            'pct_both_normal': float(both_normal / n * 100),
            'pct_both_prolonged': float(both_prolonged / n * 100),
            'pct_kepler_only': float(kepler_only / n * 100),
            'pct_bazett_only': float(bazett_only / n * 100),
            'concordance_rate': float((both_normal + both_prolonged) / n * 100),
            'kepler_prolongation_rate': float((kepler >= threshold).mean() * 100),
            'bazett_prolongation_rate': float((bazett >= threshold).mean() * 100),
        }
        
        # Clinical interpretation
        # In bradycardia: bazett_only means Bazett flags it but Kepler doesn't
        # This could be a Kepler false negative OR a Bazett false positive (overcorrection)
        
        # In tachycardia: kepler_only means Kepler flags it but Bazett doesn't
        # This could be a Kepler false positive OR a Bazett false negative (undercorrection)
    
    return results


def analyze_extreme_cases(df: pd.DataFrame) -> Dict:
    """Analyze the most extreme HR cases."""
    
    results = {}
    
    # Very severe bradycardia (HR < 45)
    severe_brady = df[df['HR_bpm'] < 45]
    if len(severe_brady) >= 20:
        results['HR_below_45'] = {
            'n': len(severe_brady),
            'hr_range': [float(severe_brady['HR_bpm'].min()), float(severe_brady['HR_bpm'].max())],
            'kepler_r': float(abs(severe_brady['QTc_Kepler'].corr(severe_brady['HR_bpm']))),
            'bazett_r': float(abs(severe_brady['QTc_Bazett'].corr(severe_brady['HR_bpm']))),
            'kepler_qtc_mean': float(severe_brady['QTc_Kepler'].mean()),
            'bazett_qtc_mean': float(severe_brady['QTc_Bazett'].mean()),
            'kepler_pct_above_450': float((severe_brady['QTc_Kepler'] >= 450).mean() * 100),
            'bazett_pct_above_450': float((severe_brady['QTc_Bazett'] >= 450).mean() * 100),
        }
    
    # Very severe tachycardia (HR > 130)
    severe_tachy = df[df['HR_bpm'] > 130]
    if len(severe_tachy) >= 20:
        results['HR_above_130'] = {
            'n': len(severe_tachy),
            'hr_range': [float(severe_tachy['HR_bpm'].min()), float(severe_tachy['HR_bpm'].max())],
            'kepler_r': float(abs(severe_tachy['QTc_Kepler'].corr(severe_tachy['HR_bpm']))),
            'bazett_r': float(abs(severe_tachy['QTc_Bazett'].corr(severe_tachy['HR_bpm']))),
            'kepler_qtc_mean': float(severe_tachy['QTc_Kepler'].mean()),
            'bazett_qtc_mean': float(severe_tachy['QTc_Bazett'].mean()),
            'kepler_pct_above_450': float((severe_tachy['QTc_Kepler'] >= 450).mean() * 100),
            'bazett_pct_above_450': float((severe_tachy['QTc_Bazett'] >= 450).mean() * 100),
        }
    
    return results


def compute_sensitivity_specificity(df: pd.DataFrame, reference: str = 'Framingham') -> Dict:
    """
    Compute sensitivity/specificity using another formula as reference.
    
    This is tricky because we don't have true gold standard.
    Using Framingham as reference since it's widely validated.
    """
    
    results = {}
    threshold = 450
    
    ref_col = f'QTc_{reference}'
    if ref_col not in df.columns:
        return results
    
    ref_prolonged = df[ref_col] >= threshold
    
    for formula in ['Kepler', 'Bazett']:
        col = f'QTc_{formula}'
        test_prolonged = df[col] >= threshold
        
        # Confusion matrix
        tp = ((test_prolonged) & (ref_prolonged)).sum()
        tn = ((~test_prolonged) & (~ref_prolonged)).sum()
        fp = ((test_prolonged) & (~ref_prolonged)).sum()
        fn = ((~test_prolonged) & (ref_prolonged)).sum()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        
        results[formula] = {
            'reference': reference,
            'threshold': threshold,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
        }
    
    return results


# ============================================================================
# VISUALIZATION SCRIPT
# ============================================================================

def generate_visualization_script(output_dir: Path) -> str:
    """Generate Python script for matplotlib visualizations."""
    
    script = f'''#!/usr/bin/env python3
"""
Extreme HR Analysis Visualization
Generated: {datetime.now().isoformat()}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
df = pd.read_csv('{output_dir}/extreme_hr_analysis_data.csv')

# Sample for plotting
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Color by HR stratum
stratum_colors = {{
    'Brady_severe': '#e74c3c',
    'Brady_mild': '#f39c12', 
    'Normal_low': '#27ae60',
    'Normal_high': '#2ecc71',
    'Tachy_mild': '#3498db',
    'Tachy_severe': '#9b59b6',
}}

# 1. Kepler QTc vs HR
ax1 = axes[0, 0]
for stratum, color in stratum_colors.items():
    subset = df[df['hr_stratum'] == stratum]
    if len(subset) > 0:
        sample = subset.sample(n=min(5000, len(subset)), random_state=42)
        ax1.scatter(sample['HR_bpm'], sample['QTc_Kepler'], 
                   c=color, alpha=0.3, s=5, label=stratum)
ax1.axhline(450, color='red', linestyle='--', alpha=0.7, label='450ms threshold')
ax1.set_xlabel('Heart Rate (bpm)')
ax1.set_ylabel('QTc Kepler (ms)')
ax1.set_title('Kepler QTc vs HR by Stratum')
ax1.legend(loc='upper right', fontsize=8)
ax1.set_xlim(30, 150)
ax1.set_ylim(250, 550)

# 2. Bazett QTc vs HR
ax2 = axes[0, 1]
for stratum, color in stratum_colors.items():
    subset = df[df['hr_stratum'] == stratum]
    if len(subset) > 0:
        sample = subset.sample(n=min(5000, len(subset)), random_state=42)
        ax2.scatter(sample['HR_bpm'], sample['QTc_Bazett'],
                   c=color, alpha=0.3, s=5, label=stratum)
ax2.axhline(450, color='red', linestyle='--', alpha=0.7)
ax2.set_xlabel('Heart Rate (bpm)')
ax2.set_ylabel('QTc Bazett (ms)')
ax2.set_title('Bazett QTc vs HR by Stratum')
ax2.set_xlim(30, 150)
ax2.set_ylim(250, 550)

# 3. |r| by HR stratum
ax3 = axes[0, 2]
strata_order = ['Brady_severe', 'Brady_mild', 'Normal_low', 'Normal_high', 'Tachy_mild', 'Tachy_severe']
kepler_r = []
bazett_r = []
labels = []
for stratum in strata_order:
    subset = df[df['hr_stratum'] == stratum]
    if len(subset) > 50:
        labels.append(stratum.replace('_', '\\n'))
        kepler_r.append(abs(subset['QTc_Kepler'].corr(subset['HR_bpm'])))
        bazett_r.append(abs(subset['QTc_Bazett'].corr(subset['HR_bpm'])))

x = np.arange(len(labels))
width = 0.35
ax3.bar(x - width/2, kepler_r, width, label='Kepler', color='#27ae60')
ax3.bar(x + width/2, bazett_r, width, label='Bazett', color='#e74c3c')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=8)
ax3.axhline(0.05, color='gray', linestyle='--', alpha=0.7)
ax3.set_ylabel('|r(QTc, HR)|')
ax3.set_title('HR Independence by Stratum')
ax3.legend()

# 4. Prolongation rate by HR stratum
ax4 = axes[1, 0]
kepler_prol = []
bazett_prol = []
for stratum in strata_order:
    subset = df[df['hr_stratum'] == stratum]
    if len(subset) > 50:
        kepler_prol.append((subset['QTc_Kepler'] >= 450).mean() * 100)
        bazett_prol.append((subset['QTc_Bazett'] >= 450).mean() * 100)

ax4.bar(x - width/2, kepler_prol, width, label='Kepler', color='#27ae60')
ax4.bar(x + width/2, bazett_prol, width, label='Bazett', color='#e74c3c')
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=8)
ax4.set_ylabel('% QTc >= 450ms')
ax4.set_title('Prolongation Rate by Stratum')
ax4.legend()

# 5. Kepler vs Bazett scatter (extreme HR only)
ax5 = axes[1, 1]
extreme = df[(df['HR_bpm'] < 50) | (df['HR_bpm'] > 120)]
if len(extreme) > 0:
    sample = extreme.sample(n=min(10000, len(extreme)), random_state=42)
    colors = ['#e74c3c' if hr < 50 else '#9b59b6' for hr in sample['HR_bpm']]
    ax5.scatter(sample['QTc_Bazett'], sample['QTc_Kepler'], c=colors, alpha=0.3, s=10)
    ax5.plot([300, 550], [300, 550], 'k--', alpha=0.5, label='y=x')
    ax5.axhline(450, color='green', linestyle=':', alpha=0.7)
    ax5.axvline(450, color='green', linestyle=':', alpha=0.7)
    ax5.set_xlabel('QTc Bazett (ms)')
    ax5.set_ylabel('QTc Kepler (ms)')
    ax5.set_title('Kepler vs Bazett (Extreme HR only)')
    ax5.legend(['y=x', 'Brady (<50)', 'Tachy (>120)'])

# 6. Concordance summary
ax6 = axes[1, 2]
# Calculate concordance for extreme strata
brady_severe = df[df['hr_stratum'] == 'Brady_severe']
tachy_severe = df[df['hr_stratum'] == 'Tachy_severe']

categories = ['Brady\\nsevere', 'Tachy\\nsevere']
concordance = []
kepler_only = []
bazett_only = []

for subset in [brady_severe, tachy_severe]:
    if len(subset) > 50:
        k = subset['QTc_Kepler'].values
        b = subset['QTc_Bazett'].values
        both = (((k >= 450) & (b >= 450)) | ((k < 450) & (b < 450))).mean() * 100
        k_only = ((k >= 450) & (b < 450)).mean() * 100
        b_only = ((k < 450) & (b >= 450)).mean() * 100
        concordance.append(both)
        kepler_only.append(k_only)
        bazett_only.append(b_only)
    else:
        concordance.append(0)
        kepler_only.append(0)
        bazett_only.append(0)

x = np.arange(len(categories))
ax6.bar(x, concordance, label='Concordant', color='#27ae60')
ax6.bar(x, kepler_only, bottom=concordance, label='Kepler only prolonged', color='#3498db')
ax6.bar(x, bazett_only, bottom=[c+k for c,k in zip(concordance, kepler_only)], 
        label='Bazett only prolonged', color='#e74c3c')
ax6.set_xticks(x)
ax6.set_xticklabels(categories)
ax6.set_ylabel('Percentage')
ax6.set_title('Diagnostic Concordance (Extreme HR)')
ax6.legend(fontsize=8)

plt.tight_layout()
plt.savefig('{output_dir}/extreme_hr_analysis_plots.png', dpi=150, bbox_inches='tight')
plt.savefig('{output_dir}/extreme_hr_analysis_plots.pdf', bbox_inches='tight')
print(f"Saved: {output_dir}/extreme_hr_analysis_plots.png")
plt.show()
'''
    
    return script


# ============================================================================
# REPORTING
# ============================================================================

def print_report(stratum_results: Dict, concordance: Dict, extreme_cases: Dict):
    """Print comprehensive report."""
    
    print("\n" + "="*70)
    print("EXTREME HR ANALYSIS - PUNTO 4")
    print("="*70)
    
    # HR Stratum analysis
    print("\n" + "-"*50)
    print("ANALISI PER STRATO HR")
    print("-"*50)
    
    print(f"\n{'Stratum':<15} {'N':>10} {'HR range':>15} {'Kepler |r|':>12} {'Bazett |r|':>12}")
    print("-" * 70)
    
    for stratum in ['Brady_severe', 'Brady_mild', 'Normal_low', 'Normal_high', 'Tachy_mild', 'Tachy_severe']:
        if stratum in stratum_results and not stratum_results[stratum].get('insufficient_data'):
            data = stratum_results[stratum]
            hr_range = f"{data['hr_range'][0]:.0f}-{data['hr_range'][1]:.0f}"
            k_r = data['formulas']['Kepler']['abs_r']
            b_r = data['formulas']['Bazett']['abs_r']
            k_status = "✓" if k_r < 0.05 else "✗"
            b_status = "✓" if b_r < 0.05 else "✗"
            print(f"{stratum:<15} {data['n']:>10,} {hr_range:>15} {k_r:>10.4f} {k_status} {b_r:>10.4f} {b_status}")
    
    # Concordance analysis
    print("\n" + "-"*50)
    print("CONCORDANZA DIAGNOSTICA (soglia 450ms)")
    print("-"*50)
    
    print(f"\n{'Stratum':<15} {'Concord%':>10} {'Kepler+':>10} {'Bazett+':>10} {'K prol%':>10} {'B prol%':>10}")
    print("-" * 70)
    
    for stratum in ['Brady_severe', 'Brady_mild', 'Normal_low', 'Normal_high', 'Tachy_mild', 'Tachy_severe']:
        if stratum in concordance and not concordance[stratum].get('insufficient_data'):
            data = concordance[stratum]
            print(f"{stratum:<15} {data['concordance_rate']:>10.1f} {data['pct_kepler_only']:>10.1f} "
                  f"{data['pct_bazett_only']:>10.1f} {data['kepler_prolongation_rate']:>10.1f} "
                  f"{data['bazett_prolongation_rate']:>10.1f}")
    
    # Extreme cases
    if extreme_cases:
        print("\n" + "-"*50)
        print("CASI ESTREMI")
        print("-"*50)
        
        for case_name, data in extreme_cases.items():
            print(f"\n{case_name} (n={data['n']:,}):")
            print(f"  HR range: {data['hr_range'][0]:.0f}-{data['hr_range'][1]:.0f} bpm")
            print(f"  Kepler: |r|={data['kepler_r']:.4f}, QTc={data['kepler_qtc_mean']:.1f}ms, "
                  f">{450}ms: {data['kepler_pct_above_450']:.1f}%")
            print(f"  Bazett: |r|={data['bazett_r']:.4f}, QTc={data['bazett_qtc_mean']:.1f}ms, "
                  f">{450}ms: {data['bazett_pct_above_450']:.1f}%")


def print_verdict(stratum_results: Dict, concordance: Dict):
    """Print final verdict."""
    
    print("\n" + "="*70)
    print("VERDETTO FINALE - PUNTO 4")
    print("="*70)
    
    # Check bradycardia
    brady_ok = True
    brady_issues = []
    
    for stratum in ['Brady_severe', 'Brady_mild']:
        if stratum in stratum_results and not stratum_results[stratum].get('insufficient_data'):
            k_r = stratum_results[stratum]['formulas']['Kepler']['abs_r']
            b_r = stratum_results[stratum]['formulas']['Bazett']['abs_r']
            
            if k_r > b_r + 0.02:  # Kepler significantly worse
                brady_ok = False
                brady_issues.append(f"{stratum}: Kepler |r|={k_r:.3f} > Bazett |r|={b_r:.3f}")
    
    # Check tachycardia
    tachy_ok = True
    tachy_issues = []
    
    for stratum in ['Tachy_mild', 'Tachy_severe']:
        if stratum in stratum_results and not stratum_results[stratum].get('insufficient_data'):
            k_r = stratum_results[stratum]['formulas']['Kepler']['abs_r']
            b_r = stratum_results[stratum]['formulas']['Bazett']['abs_r']
            
            if k_r > b_r + 0.02:
                tachy_ok = False
                tachy_issues.append(f"{stratum}: Kepler |r|={k_r:.3f} > Bazett |r|={b_r:.3f}")
    
    # Check false negative rate in bradycardia
    brady_fn_ok = True
    if 'Brady_severe' in concordance and not concordance['Brady_severe'].get('insufficient_data'):
        bazett_only_pct = concordance['Brady_severe']['pct_bazett_only']
        if bazett_only_pct > 10:  # More than 10% potential false negatives
            brady_fn_ok = False
    
    if brady_ok and tachy_ok and brady_fn_ok:
        print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ✅  KEPLER NON PEGGIORA RISPETTO A BAZETT AGLI ESTREMI HR        ║
    ║                                                                    ║
    ║   • Bradicardia severa: Kepler mantiene HR-independence            ║
    ║   • Tachicardia severa: Kepler mantiene HR-independence            ║
    ║   • Tasso di falsi negativi accettabile                            ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
        """)
    else:
        print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ⚠️  ALCUNE CRITICITÀ RILEVATE AGLI ESTREMI HR                    ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
        """)
        
        if not brady_ok:
            print("  Problemi in bradicardia:")
            for issue in brady_issues:
                print(f"    - {issue}")
        
        if not tachy_ok:
            print("  Problemi in tachicardia:")
            for issue in tachy_issues:
                print(f"    - {issue}")
        
        if not brady_fn_ok:
            print("  ⚠️ Alto tasso di potenziali falsi negativi in bradicardia severa")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    parser = argparse.ArgumentParser(
        description='Kepler-ECG Extreme HR Analysis (Point 4)',
    )
    
    parser.add_argument('--output-dir', type=str, 
                       default=str(Config.OUTPUT_DIR),
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   KEPLER-ECG: Extreme HR Analysis (Script 11_3)                    ║
    ║   PUNTO 4 - Il Tallone d'Achille della Bradicardia                 ║
    ║                                                                    ║
    ║   Obiettivo: Verificare che Kepler non sottocorregga in            ║
    ║   bradicardia e non sovracorregga in tachicardia                   ║
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
    df = compute_all_qtc(df)
    df = assign_hr_stratum(df)
    
    # Show distribution
    print("\n📊 HR distribution:")
    for stratum in ['Brady_severe', 'Brady_mild', 'Normal_low', 'Normal_high', 'Tachy_mild', 'Tachy_severe']:
        n = (df['hr_stratum'] == stratum).sum()
        pct = n / len(df) * 100
        print(f"  {stratum:<15}: {n:>10,} ({pct:>5.1f}%)")
    
    # Run analyses
    print("\n📊 Analyzing HR strata...")
    stratum_results = {}
    for stratum in Config.HR_STRATA.keys():
        stratum_results[stratum] = analyze_hr_stratum(df, stratum)
    
    print("📊 Analyzing diagnostic concordance...")
    concordance = analyze_diagnostic_concordance(df)
    
    print("📊 Analyzing extreme cases...")
    extreme_cases = analyze_extreme_cases(df)
    
    # Print report
    print_report(stratum_results, concordance, extreme_cases)
    
    # Print verdict
    print_verdict(stratum_results, concordance)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(df),
        'hr_distribution': {stratum: int((df['hr_stratum'] == stratum).sum()) 
                           for stratum in Config.HR_STRATA.keys()},
        'stratum_analysis': stratum_results,
        'concordance': concordance,
        'extreme_cases': extreme_cases,
    }
    
    json_path = output_dir / 'extreme_hr_analysis_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Report JSON: {json_path}")
    
    # Save data for plotting
    plot_cols = ['HR_bpm', 'QTc_Kepler', 'QTc_Bazett', 'QTc_Fridericia', 'hr_stratum', 'dataset']
    df[plot_cols].to_csv(output_dir / 'extreme_hr_analysis_data.csv', index=False)
    print(f"💾 Data CSV: {output_dir}/extreme_hr_analysis_data.csv")
    
    # Save visualization script
    viz_script = generate_visualization_script(output_dir)
    viz_path = output_dir / 'generate_plots.py'
    with open(viz_path, 'w', encoding='utf-8') as f:
        f.write(viz_script)
    print(f"💾 Visualization script: {viz_path}")
    
    print(f"\n✅ Analysis complete. Results in: {output_dir}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
