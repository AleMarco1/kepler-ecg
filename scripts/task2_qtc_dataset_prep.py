#!/usr/bin/env python3
"""
Kepler-ECG Phase 4.5 - Task 2 FIX: QTc Dataset Preparation
==========================================================

Fixed version with robust QTc reference calculation.

Usage:
    python task2_qtc_dataset_prep_fix.py \
        --wave_features results/stream_c/wave_features.cvs \
        --output_path ./results/stream_c
"""

import argparse
import ast
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath: Path) -> pd.DataFrame:
    """Load wave features."""
    if filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records")
    return df


def parse_scp_codes(scp_str) -> Dict:
    """Parse SCP codes from string."""
    try:
        if pd.isna(scp_str):
            return {}
        if isinstance(scp_str, dict):
            return scp_str
        return ast.literal_eval(str(scp_str))
    except:
        return {}


def extract_diagnostic_class(scp_codes: Dict) -> str:
    """Extract diagnostic class."""
    if not scp_codes:
        return 'UNKNOWN'
    
    hyp_codes = ['LVH', 'RVH', 'LVOLT', 'HVOLT', 'HYP']
    mi_codes = ['IMI', 'AMI', 'PMI']
    cd_codes = ['LAFB', 'LPFB', 'LBBB', 'RBBB', 'IRBBB', 'IVCD']
    sttc_codes = ['STTC', 'STD', 'STE', 'INVT']
    
    for code in scp_codes.keys():
        if code in mi_codes or code.startswith('MI'):
            return 'MI'
    for code in scp_codes.keys():
        if code in hyp_codes:
            return 'HYP'
    for code in scp_codes.keys():
        if code in cd_codes:
            return 'CD'
    for code in scp_codes.keys():
        if code in sttc_codes or code.startswith('NST'):
            return 'STTC'
    if 'NORM' in scp_codes and scp_codes['NORM'] >= 50:
        return 'NORM'
    return 'OTHER'


def apply_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Apply quality filters."""
    n_initial = len(df)
    stats = {'initial': n_initial}
    
    # Filter QT valid
    df = df[df['QT_interval_ms'].notna()].copy()
    stats['after_qt_valid'] = len(df)
    
    # Filter QT range (200-600 ms)
    df = df[(df['QT_interval_ms'] >= 200) & (df['QT_interval_ms'] <= 600)]
    stats['after_qt_range'] = len(df)
    
    # Filter RR range (400-2000 ms)
    df = df[(df['RR_interval_ms'] >= 400) & (df['RR_interval_ms'] <= 2000)]
    stats['after_rr_range'] = len(df)
    
    # Filter HR range (30-150 bpm)
    df = df[(df['heart_rate_bpm'] >= 30) & (df['heart_rate_bpm'] <= 150)]
    stats['after_hr_range'] = len(df)
    
    # Filter signal quality
    df = df[df['signal_quality'] >= 0.3]
    stats['after_quality'] = len(df)
    
    # Filter T detection
    df = df[df['t_wave_detection_rate'] >= 0.5]
    stats['after_t_detection'] = len(df)
    
    stats['final'] = len(df)
    stats['retention_rate'] = round(100 * len(df) / n_initial, 2)
    
    return df, stats


def calculate_qtc_reference_robust(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate data-driven QTc reference using polynomial regression.
    
    This is more robust than spline fitting.
    Uses quadratic fit: QT_expected = a*RR^2 + b*RR + c
    Then: QTc_ref = QT - QT_expected + mean(QT)
    """
    df = df.copy()
    
    rr_sec = df['RR_interval_sec'].values
    qt_ms = df['QT_interval_ms'].values
    
    # Remove any NaN
    valid_mask = ~(np.isnan(rr_sec) | np.isnan(qt_ms))
    rr_valid = rr_sec[valid_mask]
    qt_valid = qt_ms[valid_mask]
    
    logger.info(f"Fitting QT-RR relationship on {len(rr_valid)} valid points")
    
    # Fit polynomial (degree 2)
    coeffs = np.polyfit(rr_valid, qt_valid, deg=2)
    logger.info(f"Polynomial coefficients: {coeffs}")
    
    # Calculate expected QT for all points
    qt_expected = np.polyval(coeffs, rr_sec)
    
    # Mean QT for recentering
    qt_mean = np.nanmean(qt_ms)
    logger.info(f"Mean QT: {qt_mean:.2f} ms")
    
    # QTc reference = QT - expected + mean
    qtc_ref = qt_ms - qt_expected + qt_mean
    
    df['QTc_reference_ms'] = qtc_ref
    df['QT_expected_ms'] = qt_expected
    
    # Verify correlations
    valid_ref = df['QTc_reference_ms'].notna()
    if valid_ref.sum() > 10:
        r_ref = np.corrcoef(df.loc[valid_ref, 'QTc_reference_ms'], 
                           df.loc[valid_ref, 'heart_rate_bpm'])[0, 1]
        r_baz = np.corrcoef(df.loc[valid_ref, 'QTc_Bazett_ms'], 
                           df.loc[valid_ref, 'heart_rate_bpm'])[0, 1]
        r_frid = np.corrcoef(df.loc[valid_ref, 'QTc_Fridericia_ms'], 
                            df.loc[valid_ref, 'heart_rate_bpm'])[0, 1]
        
        logger.info(f"QTc-HR correlations:")
        logger.info(f"  Reference: r = {r_ref:.4f}")
        logger.info(f"  Bazett:    r = {r_baz:.4f}")
        logger.info(f"  Fridericia: r = {r_frid:.4f}")
    
    return df


def prepare_sr_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset for Symbolic Regression."""
    feature_cols = [
        'ecg_id',
        'QT_interval_ms',
        'RR_interval_ms', 
        'RR_interval_sec',
        'heart_rate_bpm',
        'QRS_duration_ms',
        'PR_interval_ms',
        'T_amplitude_mV',
        'R_amplitude_mV',
        'P_amplitude_mV',
        'signal_quality',
        'age',
        'sex',
        'QTc_reference_ms',
        'QTc_Bazett_ms',
        'QTc_Fridericia_ms',
        'QTc_Framingham_ms',
        'QTc_Hodges_ms',
    ]
    
    # Keep only existing columns
    cols = [c for c in feature_cols if c in df.columns]
    df_sr = df[cols].copy()
    
    # Drop rows with missing essential values
    essential = ['QT_interval_ms', 'RR_interval_sec', 'heart_rate_bpm', 'QTc_reference_ms']
    df_sr = df_sr.dropna(subset=essential)
    
    logger.info(f"SR dataset: {len(df_sr)} records")
    
    return df_sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave_features', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='./results/stream_c')
    args = parser.parse_args()
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Kepler-ECG Phase 4.5 - Task 2 FIX: QTc Dataset")
    print("=" * 60)
    
    # Load
    df = load_data(Path(args.wave_features))
    
    # Filter
    df_filtered, filter_stats = apply_filters(df)
    print(f"\nFiltering: {filter_stats['initial']} -> {filter_stats['final']} ({filter_stats['retention_rate']}%)")
    
    # Classify
    df_filtered['scp_dict'] = df_filtered['scp_codes'].apply(parse_scp_codes)
    df_filtered['diagnostic_class'] = df_filtered['scp_dict'].apply(extract_diagnostic_class)
    
    print(f"\nDiagnostic distribution:")
    print(df_filtered['diagnostic_class'].value_counts())
    
    # Calculate QTc reference
    print("\nCalculating QTc reference (polynomial method)...")
    df_filtered = calculate_qtc_reference_robust(df_filtered)
    
    # Create NORM subset
    df_norm = df_filtered[df_filtered['diagnostic_class'] == 'NORM'].copy()
    print(f"\nNORM subset: {len(df_norm)} ECGs")
    
    # Prepare SR dataset (using ALL data, not just NORM)
    df_sr = prepare_sr_dataset(df_filtered)
    
    # Also prepare NORM-only SR dataset
    df_sr_norm = prepare_sr_dataset(df_norm)
    
    # Save
    print("\nSaving datasets...")
    df_filtered.to_csv(output_path / 'qtc_dataset_filtered_v2.csv', index=False)
    df_norm.to_csv(output_path / 'qtc_dataset_norm_v2.csv', index=False)
    df_sr.to_csv(output_path / 'qtc_sr_dataset_all_v2.csv', index=False)
    df_sr_norm.to_csv(output_path / 'qtc_sr_dataset_norm_v2.csv', index=False)
    
    # Calculate final correlations
    print("\n" + "=" * 60)
    print("FINAL QTc-HR CORRELATIONS")
    print("=" * 60)
    
    for name, data in [('All filtered', df_sr), ('NORM only', df_sr_norm)]:
        print(f"\n{name} (n={len(data)}):")
        hr = data['heart_rate_bpm']
        for col in ['QTc_Bazett_ms', 'QTc_Fridericia_ms', 'QTc_Framingham_ms', 'QTc_reference_ms']:
            r, p = stats.pearsonr(data[col], hr)
            print(f"  {col:25s}: r = {r:+.4f} (p={p:.2e})")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'task': 'Task 2 FIX - QTc Dataset Preparation',
        'dataset_sizes': {
            'after_filters': len(df_filtered),
            'norm_only': len(df_norm),
            'sr_all': len(df_sr),
            'sr_norm': len(df_sr_norm),
        },
        'filter_statistics': filter_stats,
        'qtc_hr_correlations_all': {
            'Bazett': round(np.corrcoef(df_sr['QTc_Bazett_ms'], df_sr['heart_rate_bpm'])[0,1], 4),
            'Fridericia': round(np.corrcoef(df_sr['QTc_Fridericia_ms'], df_sr['heart_rate_bpm'])[0,1], 4),
            'Framingham': round(np.corrcoef(df_sr['QTc_Framingham_ms'], df_sr['heart_rate_bpm'])[0,1], 4),
            'Reference': round(np.corrcoef(df_sr['QTc_reference_ms'], df_sr['heart_rate_bpm'])[0,1], 4),
        },
        'qtc_hr_correlations_norm': {
            'Bazett': round(np.corrcoef(df_sr_norm['QTc_Bazett_ms'], df_sr_norm['heart_rate_bpm'])[0,1], 4),
            'Fridericia': round(np.corrcoef(df_sr_norm['QTc_Fridericia_ms'], df_sr_norm['heart_rate_bpm'])[0,1], 4),
            'Framingham': round(np.corrcoef(df_sr_norm['QTc_Framingham_ms'], df_sr_norm['heart_rate_bpm'])[0,1], 4),
            'Reference': round(np.corrcoef(df_sr_norm['QTc_reference_ms'], df_sr_norm['heart_rate_bpm'])[0,1], 4),
        },
        'interval_summary': {
            'QT_ms': {'mean': round(df_sr['QT_interval_ms'].mean(), 2), 
                     'std': round(df_sr['QT_interval_ms'].std(), 2)},
            'RR_sec': {'mean': round(df_sr['RR_interval_sec'].mean(), 4),
                      'std': round(df_sr['RR_interval_sec'].std(), 4)},
            'HR_bpm': {'mean': round(df_sr['heart_rate_bpm'].mean(), 2),
                      'std': round(df_sr['heart_rate_bpm'].std(), 2)},
        }
    }
    
    with open(output_path / 'task2_report_v2.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TASK 2 COMPLETED")
    print("=" * 60)
    print(f"SR dataset (all): {len(df_sr)} records")
    print(f"SR dataset (NORM): {len(df_sr_norm)} records")
    print(f"\nOutput files in {output_path}:")
    print("  - qtc_sr_dataset_all_v2.csv")
    print("  - qtc_sr_dataset_norm_v2.csv")
    print("  - task2_report_v2.json")


if __name__ == '__main__':
    main()
