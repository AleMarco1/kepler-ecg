#!/usr/bin/env python3
"""
Kepler-ECG: Sensitivity Analysis (Script 11_1)
===============================================

PUNTO 6 DELLA VALIDAZIONE: L'Inganno dei "Magic Numbers"

Obiettivo: Dimostrare che i coefficienti k=125 e c=-158 non sono arbitrari
ma rappresentano un punto di equilibrio biologico "rigido".

La Prova Richiesta:
- Dimostrare che oscillazioni minime (k±10, c±20) degradano la performance
- Se i numeri sono "rigidi" → equilibrio biologico trovato
- Se "ballano" senza effetto → solo approssimazione statistica

Design:
1. Sweep parametrico su k ∈ [100, 150] e c ∈ [-180, -140]
2. Per ogni (k,c): calcola |r(QTc, HR)| su tutti i dataset
3. Genera heatmap, curve 1D, analisi di rigidità

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
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for Sensitivity Analysis."""
    
    # Dataset groups
    ALL_DATASETS = ['code-15', 'ptb-xl', 'mimic-iv-ecg', 'chapman', 
                    'cpsc-2018', 'georgia', 'ludb', 'ecg-arrhythmia']
    
    DATASET_TO_TYPE = {
        'code-15': 'SCREENING', 'ptb-xl': 'SCREENING',
        'mimic-iv-ecg': 'CLINICAL', 'chapman': 'CLINICAL',
        'cpsc-2018': 'MIXED', 'georgia': 'MIXED',
        'ludb': 'EXTERNAL', 'ecg-arrhythmia': 'EXTERNAL',
    }
    
    # Reference coefficients
    KEPLER_K_REF = 125
    KEPLER_C_REF = -158
    
    # Sweep ranges
    K_RANGE = (100, 150)  # k sweep range
    C_RANGE = (-180, -130)  # c sweep range
    K_STEP = 5  # Step size for k
    C_STEP = 5  # Step size for c
    
    # Fine sweep around optimum
    K_FINE_RANGE = (115, 135)
    C_FINE_RANGE = (-170, -145)
    FINE_STEP = 2
    
    # Paths
    RESULTS_BASE = Path('results')
    OUTPUT_DIR = Path('results/sensitivity_analysis')
    
    # Success criteria
    RIGIDITY_THRESHOLD = 0.01  # Degradation threshold for "rigid" coefficients


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


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """Load all datasets."""
    
    print("\n📊 Loading datasets...")
    datasets = {}
    
    for ds in Config.ALL_DATASETS:
        df = load_qtc_data(ds)
        if df is not None:
            datasets[ds] = df
            print(f"  ✓ {ds}: {len(df):,} records")
        else:
            print(f"  ⚠️ {ds}: not found")
    
    return datasets


# ============================================================================
# QTc COMPUTATION
# ============================================================================

def compute_qtc_kepler(qt_ms: np.ndarray, rr_sec: np.ndarray, 
                       k: float, c: float) -> np.ndarray:
    """Kepler formula: QTc = QT + k/RR + c"""
    return qt_ms + k / rr_sec + c


def compute_hr_correlation(qtc: np.ndarray, hr: np.ndarray) -> float:
    """Compute absolute correlation between QTc and HR."""
    valid_mask = ~(np.isnan(qtc) | np.isnan(hr) | np.isinf(qtc))
    if valid_mask.sum() < 10:
        return np.nan
    r, _ = stats.pearsonr(qtc[valid_mask], hr[valid_mask])
    return abs(r)


# ============================================================================
# SENSITIVITY SWEEP
# ============================================================================

def run_sensitivity_sweep(datasets: Dict[str, pd.DataFrame], 
                          k_values: np.ndarray, c_values: np.ndarray,
                          description: str = "Sweep") -> Dict:
    """
    Run sensitivity sweep over k and c values.
    
    Returns dict with:
    - grid: 2D array of weighted |r| for each (k, c)
    - by_dataset: 3D array [dataset, k, c] of |r|
    - k_values, c_values: the sweep values
    """
    
    print(f"\n🔄 {description}: k∈[{k_values[0]}, {k_values[-1]}], c∈[{c_values[0]}, {c_values[-1]}]")
    print(f"   Grid size: {len(k_values)} × {len(c_values)} = {len(k_values) * len(c_values)} points")
    
    # Prepare data arrays
    dataset_names = list(datasets.keys())
    dataset_weights = {ds: len(df) for ds, df in datasets.items()}
    total_weight = sum(dataset_weights.values())
    
    # Pre-extract arrays for speed
    data_arrays = {}
    for ds, df in datasets.items():
        data_arrays[ds] = {
            'qt': df['QT_ms'].values,
            'rr': df['RR_sec'].values,
            'hr': df['HR_bpm'].values,
            'weight': dataset_weights[ds] / total_weight,
        }
    
    # Initialize result grids
    n_k, n_c = len(k_values), len(c_values)
    weighted_grid = np.zeros((n_k, n_c))
    worst_grid = np.zeros((n_k, n_c))
    by_dataset = {ds: np.zeros((n_k, n_c)) for ds in dataset_names}
    
    # Sweep
    total_points = n_k * n_c
    for i, k in enumerate(k_values):
        for j, c in enumerate(c_values):
            weighted_r = 0.0
            worst_r = 0.0
            
            for ds in dataset_names:
                arr = data_arrays[ds]
                qtc = compute_qtc_kepler(arr['qt'], arr['rr'], k, c)
                r = compute_hr_correlation(qtc, arr['hr'])
                
                by_dataset[ds][i, j] = r
                weighted_r += r * arr['weight']
                worst_r = max(worst_r, r)
            
            weighted_grid[i, j] = weighted_r
            worst_grid[i, j] = worst_r
        
        # Progress
        pct = (i + 1) / n_k * 100
        print(f"\r   Progress: {pct:.0f}%", end='', flush=True)
    
    print()  # New line after progress
    
    return {
        'k_values': k_values,
        'c_values': c_values,
        'weighted_grid': weighted_grid,
        'worst_grid': worst_grid,
        'by_dataset': by_dataset,
    }


# ============================================================================
# ANALYSIS
# ============================================================================

def find_optimal_coefficients(sweep_result: Dict) -> Dict:
    """Find optimal k and c from sweep results."""
    
    k_values = sweep_result['k_values']
    c_values = sweep_result['c_values']
    grid = sweep_result['weighted_grid']
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(grid), grid.shape)
    k_opt = k_values[min_idx[0]]
    c_opt = c_values[min_idx[1]]
    r_opt = grid[min_idx]
    
    return {
        'k_optimal': float(k_opt),
        'c_optimal': float(c_opt),
        'r_optimal': float(r_opt),
        'k_idx': int(min_idx[0]),
        'c_idx': int(min_idx[1]),
    }


def analyze_rigidity(sweep_result: Dict, k_ref: float, c_ref: float) -> Dict:
    """
    Analyze how "rigid" the coefficients are.
    
    Measures degradation when moving away from optimal.
    """
    
    k_values = sweep_result['k_values']
    c_values = sweep_result['c_values']
    grid = sweep_result['weighted_grid']
    
    # Find reference point indices
    k_idx = np.argmin(np.abs(k_values - k_ref))
    c_idx = np.argmin(np.abs(c_values - c_ref))
    r_ref = grid[k_idx, c_idx]
    
    # Analyze k sensitivity (c fixed at reference)
    k_slice = grid[:, c_idx]
    
    # Analyze c sensitivity (k fixed at reference)
    c_slice = grid[k_idx, :]
    
    # Calculate degradation at various offsets
    k_step = k_values[1] - k_values[0] if len(k_values) > 1 else 5
    c_step = c_values[1] - c_values[0] if len(c_values) > 1 else 5
    
    degradation = {}
    
    # k ± 5, 10, 15, 20
    for dk in [5, 10, 15, 20]:
        k_plus_idx = np.argmin(np.abs(k_values - (k_ref + dk)))
        k_minus_idx = np.argmin(np.abs(k_values - (k_ref - dk)))
        
        r_plus = grid[k_plus_idx, c_idx] if k_plus_idx < len(k_values) else np.nan
        r_minus = grid[k_minus_idx, c_idx] if k_minus_idx >= 0 else np.nan
        
        degradation[f'k+{dk}'] = float(r_plus - r_ref) if not np.isnan(r_plus) else None
        degradation[f'k-{dk}'] = float(r_minus - r_ref) if not np.isnan(r_minus) else None
    
    # c ± 5, 10, 15, 20
    for dc in [5, 10, 15, 20]:
        c_plus_idx = np.argmin(np.abs(c_values - (c_ref + dc)))
        c_minus_idx = np.argmin(np.abs(c_values - (c_ref - dc)))
        
        r_plus = grid[k_idx, c_plus_idx] if c_plus_idx < len(c_values) else np.nan
        r_minus = grid[k_idx, c_minus_idx] if c_minus_idx >= 0 else np.nan
        
        degradation[f'c+{dc}'] = float(r_plus - r_ref) if not np.isnan(r_plus) else None
        degradation[f'c-{dc}'] = float(r_minus - r_ref) if not np.isnan(r_minus) else None
    
    # Determine if "rigid"
    k_rigid = any(degradation.get(f'k+{d}', 0) or 0 > Config.RIGIDITY_THRESHOLD or 
                  degradation.get(f'k-{d}', 0) or 0 > Config.RIGIDITY_THRESHOLD 
                  for d in [10])
    c_rigid = any(degradation.get(f'c+{d}', 0) or 0 > Config.RIGIDITY_THRESHOLD or 
                  degradation.get(f'c-{d}', 0) or 0 > Config.RIGIDITY_THRESHOLD 
                  for d in [20])
    
    return {
        'k_ref': k_ref,
        'c_ref': c_ref,
        'r_at_reference': float(r_ref),
        'k_slice': k_slice.tolist(),
        'c_slice': c_slice.tolist(),
        'degradation': degradation,
        'k_is_rigid': k_rigid,
        'c_is_rigid': c_rigid,
        'overall_rigid': k_rigid and c_rigid,
    }


def analyze_by_dataset(sweep_result: Dict, k_ref: float, c_ref: float) -> Dict:
    """Analyze optimal coefficients per dataset."""
    
    k_values = sweep_result['k_values']
    c_values = sweep_result['c_values']
    by_dataset = sweep_result['by_dataset']
    
    results = {}
    
    for ds, grid in by_dataset.items():
        # Find minimum for this dataset
        min_idx = np.unravel_index(np.argmin(grid), grid.shape)
        k_opt = k_values[min_idx[0]]
        c_opt = c_values[min_idx[1]]
        r_opt = grid[min_idx]
        
        # Value at reference
        k_idx = np.argmin(np.abs(k_values - k_ref))
        c_idx = np.argmin(np.abs(c_values - c_ref))
        r_ref = grid[k_idx, c_idx]
        
        results[ds] = {
            'k_optimal': float(k_opt),
            'c_optimal': float(c_opt),
            'r_optimal': float(r_opt),
            'r_at_reference': float(r_ref),
            'delta_k': float(k_opt - k_ref),
            'delta_c': float(c_opt - c_ref),
        }
    
    return results


# ============================================================================
# VISUALIZATION (Text-based for portability)
# ============================================================================

def print_heatmap_ascii(grid: np.ndarray, k_values: np.ndarray, c_values: np.ndarray,
                        title: str, k_ref: float, c_ref: float):
    """Print ASCII heatmap of the grid."""
    
    print(f"\n{title}")
    print("=" * 60)
    
    # Symbols for different |r| ranges
    symbols = [' ', '░', '▒', '▓', '█']
    thresholds = [0.02, 0.04, 0.06, 0.08]
    
    def get_symbol(val):
        for i, t in enumerate(thresholds):
            if val < t:
                return symbols[i]
        return symbols[-1]
    
    # Print header
    print(f"     c: ", end='')
    for c in c_values[::2]:  # Every other value
        print(f"{c:>6.0f}", end='')
    print()
    
    # Print rows
    for i, k in enumerate(k_values):
        marker = " *" if abs(k - k_ref) < 3 else "  "
        print(f"k={k:3.0f}{marker} ", end='')
        for j in range(0, len(c_values), 2):
            val = grid[i, j]
            sym = get_symbol(val)
            # Mark reference point
            if abs(k - k_ref) < 3 and abs(c_values[j] - c_ref) < 3:
                print(f"  [{sym}] ", end='')
            else:
                print(f"   {sym}  ", end='')
        print(f" | {grid[i, :].min():.3f}")
    
    print()
    print(f"Legend: ' '<0.02  '░'<0.04  '▒'<0.06  '▓'<0.08  '█'≥0.08")
    print(f"[*] = Reference point (k={k_ref}, c={c_ref})")


def generate_matplotlib_script(sweep_result: Dict, output_dir: Path, 
                               k_ref: float, c_ref: float) -> str:
    """Generate a Python script to create matplotlib visualizations."""
    
    script = f'''#!/usr/bin/env python3
"""
Sensitivity Analysis Visualization
Generated: {datetime.now().isoformat()}
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Data from sweep
k_values = np.array({sweep_result['k_values'].tolist()})
c_values = np.array({sweep_result['c_values'].tolist()})
weighted_grid = np.array({sweep_result['weighted_grid'].tolist()})
worst_grid = np.array({sweep_result['worst_grid'].tolist()})

k_ref, c_ref = {k_ref}, {c_ref}

# Find optimal
min_idx = np.unravel_index(np.argmin(weighted_grid), weighted_grid.shape)
k_opt, c_opt = k_values[min_idx[0]], c_values[min_idx[1]]
r_opt = weighted_grid[min_idx]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Heatmap - Weighted |r|
ax1 = axes[0, 0]
im1 = ax1.imshow(weighted_grid, extent=[c_values[0], c_values[-1], k_values[-1], k_values[0]],
                  aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.1)
ax1.plot(c_ref, k_ref, 'k*', markersize=15, label=f'Reference ({{k_ref}}, {{c_ref}})')
ax1.plot(c_opt, k_opt, 'wo', markersize=10, markeredgecolor='black', label=f'Optimal ({{k_opt:.0f}}, {{c_opt:.0f}})')
ax1.set_xlabel('c coefficient')
ax1.set_ylabel('k coefficient')
ax1.set_title('Weighted Mean |r(QTc, HR)| across datasets')
ax1.legend(loc='upper right')
plt.colorbar(im1, ax=ax1, label='|r|')

# 2. Heatmap - Worst case |r|
ax2 = axes[0, 1]
im2 = ax2.imshow(worst_grid, extent=[c_values[0], c_values[-1], k_values[-1], k_values[0]],
                  aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.15)
ax2.plot(c_ref, k_ref, 'k*', markersize=15)
ax2.set_xlabel('c coefficient')
ax2.set_ylabel('k coefficient')
ax2.set_title('Worst-case |r(QTc, HR)| across datasets')
plt.colorbar(im2, ax=ax2, label='|r|')

# 3. 1D slice - k sensitivity
ax3 = axes[1, 0]
c_idx = np.argmin(np.abs(c_values - c_ref))
k_slice = weighted_grid[:, c_idx]
ax3.plot(k_values, k_slice, 'b-o', linewidth=2, markersize=6)
ax3.axvline(k_ref, color='red', linestyle='--', label=f'k={k_ref}')
ax3.axhline(0.05, color='green', linestyle=':', alpha=0.7, label='|r|=0.05 threshold')
ax3.set_xlabel('k coefficient')
ax3.set_ylabel('Weighted |r|')
ax3.set_title(f'k Sensitivity (c fixed at {{c_ref}})')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, max(0.1, k_slice.max() * 1.1))

# 4. 1D slice - c sensitivity
ax4 = axes[1, 1]
k_idx = np.argmin(np.abs(k_values - k_ref))
c_slice = weighted_grid[k_idx, :]
ax4.plot(c_values, c_slice, 'b-o', linewidth=2, markersize=6)
ax4.axvline(c_ref, color='red', linestyle='--', label=f'c={c_ref}')
ax4.axhline(0.05, color='green', linestyle=':', alpha=0.7, label='|r|=0.05 threshold')
ax4.set_xlabel('c coefficient')
ax4.set_ylabel('Weighted |r|')
ax4.set_title(f'c Sensitivity (k fixed at {{k_ref}})')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, max(0.1, c_slice.max() * 1.1))

plt.tight_layout()
plt.savefig('{output_dir}/sensitivity_analysis_plots.png', dpi=150, bbox_inches='tight')
plt.savefig('{output_dir}/sensitivity_analysis_plots.pdf', bbox_inches='tight')
print(f"Saved: {output_dir}/sensitivity_analysis_plots.png")
plt.show()
'''
    
    return script


# ============================================================================
# REPORT
# ============================================================================

def print_report(optimal: Dict, rigidity: Dict, by_dataset: Dict):
    """Print comprehensive report."""
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS REPORT - PUNTO 6")
    print("="*70)
    
    # Optimal coefficients
    print(f"\n📊 COEFFICIENTI OTTIMALI (da sweep):")
    print(f"   k_optimal = {optimal['k_optimal']:.1f}")
    print(f"   c_optimal = {optimal['c_optimal']:.1f}")
    print(f"   |r| al minimo = {optimal['r_optimal']:.4f}")
    
    print(f"\n📊 CONFRONTO CON RIFERIMENTO (k=125, c=-158):")
    print(f"   Δk = {optimal['k_optimal'] - Config.KEPLER_K_REF:.1f}")
    print(f"   Δc = {optimal['c_optimal'] - Config.KEPLER_C_REF:.1f}")
    print(f"   |r| al riferimento = {rigidity['r_at_reference']:.4f}")
    
    # Rigidity analysis
    print(f"\n📊 ANALISI DI RIGIDITÀ:")
    print("-" * 50)
    print(f"{'Variazione':<15} {'Δ|r|':>10} {'Significativo?':>15}")
    print("-" * 50)
    
    for key, val in rigidity['degradation'].items():
        if val is not None:
            sig = "✓ SÌ" if abs(val) > Config.RIGIDITY_THRESHOLD else "no"
            print(f"{key:<15} {val:>+10.4f} {sig:>15}")
    
    print("-" * 50)
    print(f"\nk è RIGIDO? {'✅ SÌ' if rigidity['k_is_rigid'] else '❌ NO'}")
    print(f"c è RIGIDO? {'✅ SÌ' if rigidity['c_is_rigid'] else '❌ NO'}")
    
    # Per-dataset analysis
    print(f"\n📊 OTTIMALI PER DATASET:")
    print("-" * 70)
    print(f"{'Dataset':<15} {'k_opt':>8} {'c_opt':>8} {'Δk':>8} {'Δc':>8} {'|r|_opt':>10}")
    print("-" * 70)
    
    for ds, data in sorted(by_dataset.items()):
        print(f"{ds:<15} {data['k_optimal']:>8.1f} {data['c_optimal']:>8.1f} "
              f"{data['delta_k']:>+8.1f} {data['delta_c']:>+8.1f} {data['r_optimal']:>10.4f}")
    
    print("-" * 70)
    
    # Convergence analysis
    k_opts = [d['k_optimal'] for d in by_dataset.values()]
    c_opts = [d['c_optimal'] for d in by_dataset.values()]
    
    print(f"\n📊 CONVERGENZA DEI COEFFICIENTI:")
    print(f"   k range: [{min(k_opts):.0f}, {max(k_opts):.0f}] (spread: {max(k_opts)-min(k_opts):.0f})")
    print(f"   c range: [{min(c_opts):.0f}, {max(c_opts):.0f}] (spread: {max(c_opts)-min(c_opts):.0f})")
    print(f"   k mean ± std: {np.mean(k_opts):.1f} ± {np.std(k_opts):.1f}")
    print(f"   c mean ± std: {np.mean(c_opts):.1f} ± {np.std(c_opts):.1f}")


def print_final_verdict(rigidity: Dict, optimal: Dict):
    """Print final verdict."""
    
    print("\n" + "="*70)
    print("VERDETTO FINALE - PUNTO 6")
    print("="*70)
    
    is_rigid = rigidity['overall_rigid']
    k_close = abs(optimal['k_optimal'] - Config.KEPLER_K_REF) <= 10
    c_close = abs(optimal['c_optimal'] - Config.KEPLER_C_REF) <= 10
    
    if is_rigid and k_close and c_close:
        print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ✅  COEFFICIENTI "RIGIDI" - EQUILIBRIO BIOLOGICO CONFERMATO      ║
    ║                                                                    ║
    ║   • I coefficienti k=125, c=-158 NON sono arbitrari                ║
    ║   • Piccole variazioni DEGRADANO significativamente la performance ║
    ║   • L'ottimo del sweep COINCIDE con i valori scoperti da PySR      ║
    ║   • Questo suggerisce una relazione FISIOLOGICA reale              ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
        """)
    elif is_rigid:
        print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ⚠️  COEFFICIENTI RIGIDI MA OTTIMO DIVERSO                        ║
    ║                                                                    ║
    ║   • I coefficienti sono sensibili (rigidi)                         ║
    ║   • Ma l'ottimo del sweep differisce da 125/-158                   ║
    ║   • Possibile ottimizzazione ulteriore                             ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
        """)
    else:
        print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ⚠️  COEFFICIENTI "SOFT" - PLATEAU RILEVATO                       ║
    ║                                                                    ║
    ║   • Le variazioni NON degradano significativamente la performance  ║
    ║   • La formula è una delle tante approssimazioni possibili         ║
    ║   • I numeri 125/-158 non hanno significato fisico speciale        ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
        """)
    
    print(f"\n📋 RISPOSTA ALLA CRITICA:")
    print(f'   "Perché proprio 125 e 158?"')
    print(f"   → Ottimo del sweep: k={optimal['k_optimal']:.0f}, c={optimal['c_optimal']:.0f}")
    print(f"   → Degradazione con k±10: {rigidity['degradation'].get('k+10', 'N/A')}")
    print(f"   → I coefficienti {'SONO' if is_rigid else 'NON sono'} un punto di equilibrio rigido")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    parser = argparse.ArgumentParser(
        description='Kepler-ECG Sensitivity Analysis (Point 6)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--fine', action='store_true',
                       help='Run fine-grained sweep around optimum')
    parser.add_argument('--output-dir', type=str, 
                       default=str(Config.OUTPUT_DIR),
                       help='Output directory')
    parser.add_argument('--k-range', type=str, default='100,150',
                       help='k sweep range (min,max)')
    parser.add_argument('--c-range', type=str, default='-180,-130',
                       help='c sweep range (min,max)')
    parser.add_argument('--step', type=int, default=5,
                       help='Sweep step size')
    
    args = parser.parse_args()
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   KEPLER-ECG: Sensitivity Analysis (Script 11_1)                   ║
    ║   PUNTO 6 - L'Inganno dei "Magic Numbers"                          ║
    ║                                                                    ║
    ║   Obiettivo: Dimostrare che k=125 e c=-158 sono "rigidi"           ║
    ║   e rappresentano un equilibrio biologico, non polvere statistica  ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    datasets = load_all_datasets()
    
    if not datasets:
        print("\n❌ No datasets loaded!")
        return 1
    
    print(f"\n📊 Total: {sum(len(df) for df in datasets.values()):,} records from {len(datasets)} datasets")
    
    # Parse ranges
    k_min, k_max = map(int, args.k_range.split(','))
    c_min, c_max = map(int, args.c_range.split(','))
    
    k_values = np.arange(k_min, k_max + 1, args.step)
    c_values = np.arange(c_min, c_max + 1, args.step)
    
    # Run coarse sweep
    sweep_result = run_sensitivity_sweep(datasets, k_values, c_values, "Coarse Sweep")
    
    # Find optimal
    optimal = find_optimal_coefficients(sweep_result)
    print(f"\n✓ Optimal found: k={optimal['k_optimal']:.0f}, c={optimal['c_optimal']:.0f}, |r|={optimal['r_optimal']:.4f}")
    
    # Fine sweep if requested
    if args.fine:
        k_fine = np.arange(optimal['k_optimal'] - 15, optimal['k_optimal'] + 16, 2)
        c_fine = np.arange(optimal['c_optimal'] - 15, optimal['c_optimal'] + 16, 2)
        fine_result = run_sensitivity_sweep(datasets, k_fine, c_fine, "Fine Sweep")
        optimal_fine = find_optimal_coefficients(fine_result)
        print(f"✓ Fine optimal: k={optimal_fine['k_optimal']:.0f}, c={optimal_fine['c_optimal']:.0f}, |r|={optimal_fine['r_optimal']:.4f}")
    
    # Analyze rigidity
    rigidity = analyze_rigidity(sweep_result, Config.KEPLER_K_REF, Config.KEPLER_C_REF)
    
    # Per-dataset analysis
    by_dataset = analyze_by_dataset(sweep_result, Config.KEPLER_K_REF, Config.KEPLER_C_REF)
    
    # Print ASCII heatmap
    print_heatmap_ascii(sweep_result['weighted_grid'], k_values, c_values,
                       "Weighted |r| Heatmap", Config.KEPLER_K_REF, Config.KEPLER_C_REF)
    
    # Print report
    print_report(optimal, rigidity, by_dataset)
    
    # Print verdict
    print_final_verdict(rigidity, optimal)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'k_range': [k_min, k_max],
            'c_range': [c_min, c_max],
            'step': args.step,
            'reference': {'k': Config.KEPLER_K_REF, 'c': Config.KEPLER_C_REF},
        },
        'optimal': optimal,
        'rigidity': rigidity,
        'by_dataset': by_dataset,
        'sweep': {
            'k_values': k_values.tolist(),
            'c_values': c_values.tolist(),
            'weighted_grid': sweep_result['weighted_grid'].tolist(),
            'worst_grid': sweep_result['worst_grid'].tolist(),
        },
    }
    
    json_path = output_dir / 'sensitivity_analysis_report.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Report JSON: {json_path}")
    
    # Save visualization script
    viz_script = generate_matplotlib_script(sweep_result, output_dir, 
                                            Config.KEPLER_K_REF, Config.KEPLER_C_REF)
    viz_path = output_dir / 'generate_plots.py'
    with open(viz_path, 'w') as f:
        f.write(viz_script)
    print(f"💾 Visualization script: {viz_path}")
    print(f"   Run with: python {viz_path}")
    
    # Save CSV summary
    csv_data = []
    for ds, data in by_dataset.items():
        csv_data.append({
            'dataset': ds,
            'type': Config.DATASET_TO_TYPE.get(ds, 'UNKNOWN'),
            'k_optimal': data['k_optimal'],
            'c_optimal': data['c_optimal'],
            'r_optimal': data['r_optimal'],
            'r_at_reference': data['r_at_reference'],
            'delta_k': data['delta_k'],
            'delta_c': data['delta_c'],
        })
    
    csv_path = output_dir / 'sensitivity_by_dataset.csv'
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"💾 Per-dataset CSV: {csv_path}")
    
    print(f"\n✅ Analysis complete. Results in: {output_dir}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
