#!/usr/bin/env python3
"""
Kepler-ECG: QTc Dataset Preparation

Prepares wave features for QTc formula discovery via Symbolic Regression.

Steps:
1. Quality filtering (QT/RR ranges, signal quality)
2. Diagnostic classification (SCP/SNOMED codes)
3. QTc reference calculation (polynomial regression)
4. SR-ready dataset generation

Usage:
    python scripts/prepare_qtc_dataset.py \\
        --input results/ptb-xl/waves/wave_features.csv \\
        --output results/ptb-xl/qtc
    
    # Or using dataset name
    python scripts/prepare_qtc_dataset.py --dataset ptb-xl

Author: Kepler-ECG Project
Version: 2.0.0
"""

import argparse
import ast
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ============================================================================
# Filtering
# ============================================================================

def apply_quality_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Apply quality filters to wave features."""
    n_initial = len(df)
    filter_stats = {'initial': n_initial}
    
    # QT valid
    df = df[df['QT_interval_ms'].notna()].copy()
    filter_stats['after_qt_valid'] = len(df)
    
    # QT physiological range (200-600 ms)
    df = df[(df['QT_interval_ms'] >= 200) & (df['QT_interval_ms'] <= 600)]
    filter_stats['after_qt_range'] = len(df)
    
    # RR range (400-2000 ms, i.e., 30-150 bpm)
    df = df[(df['RR_interval_ms'] >= 400) & (df['RR_interval_ms'] <= 2000)]
    filter_stats['after_rr_range'] = len(df)
    
    # HR range
    df = df[(df['heart_rate_bpm'] >= 30) & (df['heart_rate_bpm'] <= 150)]
    filter_stats['after_hr_range'] = len(df)
    
    # Signal quality
    if 'signal_quality' in df.columns:
        df = df[df['signal_quality'] >= 0.3]
        filter_stats['after_quality'] = len(df)
    
    # T wave detection
    if 't_wave_detection_rate' in df.columns:
        df = df[df['t_wave_detection_rate'] >= 0.5]
        filter_stats['after_t_detection'] = len(df)
    
    filter_stats['final'] = len(df)
    filter_stats['retention_rate'] = round(100 * len(df) / n_initial, 2) if n_initial > 0 else 0
    
    return df, filter_stats


# ============================================================================
# Diagnostic Classification
# ============================================================================

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
    """Extract diagnostic class from SCP codes."""
    if not scp_codes:
        return 'UNKNOWN'
    
    # Priority order for classification
    mi_codes = ['IMI', 'AMI', 'PMI', 'MI']
    hyp_codes = ['LVH', 'RVH', 'LVOLT', 'HVOLT', 'HYP']
    cd_codes = ['LAFB', 'LPFB', 'LBBB', 'RBBB', 'IRBBB', 'IVCD']
    sttc_codes = ['STTC', 'STD', 'STE', 'INVT', 'NST_']
    
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


# ============================================================================
# QTc Reference Calculation
# ============================================================================

def calculate_qtc_reference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate data-driven QTc reference using polynomial regression.
    
    Uses quadratic fit: QT_expected = a*RR² + b*RR + c
    QTc_ref = QT - QT_expected + mean(QT)
    
    This provides an HR-independent QTc reference for SR to learn from.
    """
    df = df.copy()
    
    rr_sec = df['RR_interval_sec'].values
    qt_ms = df['QT_interval_ms'].values
    
    # Valid data only
    valid_mask = ~(np.isnan(rr_sec) | np.isnan(qt_ms))
    rr_valid = rr_sec[valid_mask]
    qt_valid = qt_ms[valid_mask]
    
    logger.info(f"Fitting QT-RR relationship on {len(rr_valid)} valid points")
    
    # Polynomial fit (degree 2)
    coeffs = np.polyfit(rr_valid, qt_valid, deg=2)
    logger.info(f"Polynomial: {coeffs[0]:.4f}*RR² + {coeffs[1]:.4f}*RR + {coeffs[2]:.4f}")
    
    # Expected QT
    qt_expected = np.polyval(coeffs, rr_sec)
    
    # Mean QT for recentering
    qt_mean = np.nanmean(qt_ms)
    logger.info(f"Mean QT: {qt_mean:.2f} ms")
    
    # QTc reference
    df['QTc_reference_ms'] = qt_ms - qt_expected + qt_mean
    df['QT_expected_ms'] = qt_expected
    
    # Verify HR independence
    valid = df['QTc_reference_ms'].notna()
    if valid.sum() > 10:
        r_ref = np.corrcoef(df.loc[valid, 'QTc_reference_ms'],
                            df.loc[valid, 'heart_rate_bpm'])[0, 1]
        r_baz = np.corrcoef(df.loc[valid, 'QTc_Bazett_ms'],
                           df.loc[valid, 'heart_rate_bpm'])[0, 1]
        
        logger.info(f"QTc-HR correlations:")
        logger.info(f"  Reference: r = {r_ref:.4f}")
        logger.info(f"  Bazett:    r = {r_baz:.4f}")
    
    return df


# ============================================================================
# Dataset Preparation
# ============================================================================

def prepare_sr_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare minimal dataset for Symbolic Regression."""
    
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
    essential = [c for c in essential if c in df_sr.columns]
    df_sr = df_sr.dropna(subset=essential)
    
    logger.info(f"SR dataset: {len(df_sr)} records")
    
    return df_sr


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Kepler-ECG QTc Dataset Preparation'
    )
    
    parser.add_argument('--input', '-i', type=str,
                        help='Path to wave_features.csv')
    parser.add_argument('--dataset', '-d', type=str,
                        help='Dataset name (looks for results/{dataset}/waves/)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Determine paths
    if args.input:
        input_path = Path(args.input)
    elif args.dataset:
        input_path = Path(f"results/{args.dataset}/waves/wave_features.csv")
        if not input_path.exists():
            input_path = Path(f"results/{args.dataset}/waves/wave_features.parquet")
    else:
        parser.error("Must provide either --input or --dataset")
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1
    
    if args.output:
        output_path = Path(args.output)
    elif args.dataset:
        output_path = Path(f"results/{args.dataset}/qtc")
    else:
        output_path = Path("results/qtc")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("KEPLER-ECG QTc DATASET PREPARATION")
    print("="*60)
    
    # Load data
    print(f"\nLoading: {input_path}")
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    print(f"Loaded: {len(df)} records")
    
    # Apply filters
    print("\nApplying quality filters...")
    df_filtered, filter_stats = apply_quality_filters(df)
    print(f"Retained: {filter_stats['final']} records ({filter_stats['retention_rate']}%)")
    
    # Diagnostic classification
    if 'scp_codes' in df_filtered.columns:
        print("\nClassifying diagnoses...")
        df_filtered['scp_dict'] = df_filtered['scp_codes'].apply(parse_scp_codes)
        df_filtered['diagnostic_class'] = df_filtered['scp_dict'].apply(extract_diagnostic_class)
        
        print("Distribution:")
        print(df_filtered['diagnostic_class'].value_counts())
    
    # Calculate QTc reference
    print("\nCalculating QTc reference (polynomial method)...")
    df_filtered = calculate_qtc_reference(df_filtered)
    
    # Create NORM subset
    if 'diagnostic_class' in df_filtered.columns:
        df_norm = df_filtered[df_filtered['diagnostic_class'] == 'NORM'].copy()
        print(f"\nNORM subset: {len(df_norm)} records")
    else:
        df_norm = df_filtered.copy()
    
    # Prepare SR datasets
    print("\nPreparing SR datasets...")
    df_sr_all = prepare_sr_dataset(df_filtered)
    df_sr_norm = prepare_sr_dataset(df_norm)
    
    # Save outputs
    print("\nSaving datasets...")
    df_filtered.to_csv(output_path / 'qtc_dataset_filtered.csv', index=False)
    df_norm.to_csv(output_path / 'qtc_dataset_norm.csv', index=False)
    df_sr_all.to_csv(output_path / 'qtc_sr_dataset_all.csv', index=False)
    df_sr_norm.to_csv(output_path / 'qtc_sr_dataset_norm.csv', index=False)
    
    # Calculate final correlations
    print("\n" + "="*60)
    print("QTc-HR CORRELATIONS")
    print("="*60)
    
    for name, data in [('All', df_sr_all), ('NORM', df_sr_norm)]:
        if len(data) > 10:
            print(f"\n{name} (n={len(data)}):")
            hr = data['heart_rate_bpm']
            for col in ['QTc_Bazett_ms', 'QTc_Fridericia_ms', 'QTc_reference_ms']:
                if col in data.columns:
                    valid = ~(data[col].isna() | hr.isna())
                    r, p = stats.pearsonr(data.loc[valid, col], hr[valid])
                    print(f"  {col:25s}: r = {r:+.4f}")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_file': str(input_path),
        'output_dir': str(output_path),
        'filter_statistics': filter_stats,
        'dataset_sizes': {
            'filtered': len(df_filtered),
            'norm': len(df_norm),
            'sr_all': len(df_sr_all),
            'sr_norm': len(df_sr_norm),
        },
        'qtc_hr_correlations': {},
    }
    
    for col in ['QTc_Bazett_ms', 'QTc_Fridericia_ms', 'QTc_reference_ms']:
        if col in df_sr_all.columns:
            valid = ~(df_sr_all[col].isna() | df_sr_all['heart_rate_bpm'].isna())
            r, _ = stats.pearsonr(df_sr_all.loc[valid, col], 
                                  df_sr_all.loc[valid, 'heart_rate_bpm'])
            report['qtc_hr_correlations'][col] = round(r, 4)
    
    with open(output_path / 'qtc_preparation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"SR dataset (all): {len(df_sr_all)} records")
    print(f"SR dataset (NORM): {len(df_sr_norm)} records")
    print(f"Output: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
