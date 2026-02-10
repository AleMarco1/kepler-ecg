#!/usr/bin/env python3
"""
Kepler-ECG: QTc Dataset Preparation

Prepares wave features for QTc formula discovery via Symbolic Regression.

Steps:
1. Load wave features and merge with features.csv (for demographics + diagnostics)
2. Quality filtering (QT/RR ranges, signal quality)
3. QTc reference calculation (polynomial regression)
4. SR-ready dataset generation

Usage:
    python scripts/04_0_prepare_qtc_dataset.py --dataset ptb-xl
    python scripts/04_0_prepare_qtc_dataset.py --dataset chapman
    python scripts/04_0_prepare_qtc_dataset.py --dataset code-15
    python scripts/04_0_prepare_qtc_dataset.py --dataset cpsc-2018
    python scripts/04_0_prepare_qtc_dataset.py --dataset georgia
    python scripts/04_0_prepare_qtc_dataset.py --dataset mimic-iv-ecg

Output:
    results/{dataset}/qtc/{dataset}_qtc_preparation.csv
    results/{dataset}/qtc/{dataset}_qtc_preparation_report.json

Author: Kepler-ECG Project
Version: 3.0.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================

def load_wave_features(dataset_name: str) -> Optional[pd.DataFrame]:
    """
    Load wave features from extract_waves output.
    
    Tries multiple possible paths and formats.
    """
    possible_paths = [
        Path(f"results/{dataset_name}/waves/{dataset_name}_wave_features.csv"),
        Path(f"results/{dataset_name}/waves/{dataset_name}_wave_features.parquet"),
        Path(f"results/{dataset_name}/waves/wave_features.csv"),
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Loading wave features from: {path}")
            if path.suffix == '.parquet':
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, sep=';', decimal=',')
            logger.info(f"  Loaded {len(df)} records, {len(df.columns)} columns")
            return df
    
    logger.error(f"Wave features not found for dataset '{dataset_name}'")
    logger.error(f"  Tried: {[str(p) for p in possible_paths]}")
    return None


def load_features_metadata(dataset_name: str) -> Optional[pd.DataFrame]:
    """
    Load features.csv from process_dataset output.
    
    Contains: primary_superclass, age, sex, label_NORM, label_MI, etc.
    """
    possible_paths = [
        Path(f"results/{dataset_name}/preprocess/{dataset_name}_features.csv"),
        Path(f"results/{dataset_name}/{dataset_name}_features.csv"),
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Loading features metadata from: {path}")
            df = pd.read_csv(path, sep=';', decimal=',')
            logger.info(f"  Loaded {len(df)} records")
            
            # Log available columns
            key_cols = ['ecg_id', 'age', 'sex', 'primary_superclass', 
                       'label_NORM', 'label_MI', 'label_STTC', 'label_CD', 'label_HYP']
            available = [c for c in key_cols if c in df.columns]
            logger.info(f"  Key columns available: {available}")
            
            return df
    
    logger.warning(f"Features metadata not found for dataset '{dataset_name}'")
    return None


def merge_wave_and_features(df_waves: pd.DataFrame, 
                            df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Merge wave features with features metadata.
    
    Adds: primary_superclass, age, sex, label_* columns from features.csv
    """
    # Columns to merge from features (exclude those already in waves)
    merge_cols = ['ecg_id']
    
    # Demographics
    for col in ['age', 'sex']:
        if col in df_features.columns and col not in df_waves.columns:
            merge_cols.append(col)
    
    # Diagnostics
    if 'primary_superclass' in df_features.columns:
        merge_cols.append('primary_superclass')
    
    # Label columns
    label_cols = [c for c in df_features.columns if c.startswith('label_')]
    merge_cols.extend([c for c in label_cols if c not in df_waves.columns])
    
    # Perform merge
    df_features_subset = df_features[merge_cols].drop_duplicates(subset=['ecg_id'])
    
    n_before = len(df_waves)
    df_merged = df_waves.merge(df_features_subset, on='ecg_id', how='left')
    n_after = len(df_merged)
    
    if n_after != n_before:
        logger.warning(f"Merge changed row count: {n_before} -> {n_after}")
    
    # Log merge success
    if 'primary_superclass' in df_merged.columns:
        n_matched = df_merged['primary_superclass'].notna().sum()
        logger.info(f"  Diagnostic match rate: {n_matched}/{len(df_merged)} ({100*n_matched/len(df_merged):.1f}%)")
    
    if 'age' in df_merged.columns:
        n_age = df_merged['age'].notna().sum()
        logger.info(f"  Age match rate: {n_age}/{len(df_merged)} ({100*n_age/len(df_merged):.1f}%)")
    
    return df_merged


# ============================================================================
# Quality Filtering
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
    
    # Signal quality (if available)
    if 'signal_quality' in df.columns:
        df = df[df['signal_quality'] >= 0.3]
        filter_stats['after_quality'] = len(df)
    
    # T wave detection (if available)
    if 't_wave_detection_rate' in df.columns:
        df = df[df['t_wave_detection_rate'] >= 0.5]
        filter_stats['after_t_detection'] = len(df)
    
    filter_stats['final'] = len(df)
    filter_stats['retention_rate'] = round(100 * len(df) / n_initial, 2) if n_initial > 0 else 0
    
    return df, filter_stats


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

def prepare_qtc_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare final QTc dataset for Symbolic Regression and validation.
    
    Includes all relevant columns for SR discovery and clinical validation.
    """
    
    # Core columns (always needed)
    core_cols = [
        'ecg_id',
        'age',
        'sex',
        'QT_interval_ms',
        'RR_interval_ms',
        'RR_interval_sec',
        'heart_rate_bpm',
    ]
    
    # Wave morphology (useful for SR)
    morphology_cols = [
        'QRS_duration_ms',
        'PR_interval_ms',
        'T_amplitude_mV',
        'R_amplitude_mV',
        'P_amplitude_mV',
    ]
    
    # Quality metrics
    quality_cols = [
        'signal_quality',
        't_wave_detection_rate',
    ]
    
    # Diagnostics (from features.csv)
    diag_cols = [
        'primary_superclass',
    ]
    
    # Label columns (from features.csv)
    label_cols = [c for c in df.columns if c.startswith('label_')]
    
    # QTc formulas (calculated by extract_waves)
    qtc_cols = [
        'QTc_reference_ms',
        'QT_expected_ms',
        'QTc_Bazett_ms',
        'QTc_Fridericia_ms',
        'QTc_Framingham_ms',
        'QTc_Hodges_ms',
    ]
    
    # Combine all and keep only existing
    all_cols = (core_cols + morphology_cols + quality_cols + 
            diag_cols + label_cols + qtc_cols)
    cols = [c for c in all_cols if c in df.columns]
    
    df_out = df[cols].copy()
    
    # Drop rows with missing essential values
    essential = ['QT_interval_ms', 'RR_interval_sec', 'heart_rate_bpm', 'QTc_reference_ms']
    essential = [c for c in essential if c in df_out.columns]
    df_out = df_out.dropna(subset=essential)
    
    logger.info(f"Final dataset: {len(df_out)} records, {len(cols)} columns")
    
    return df_out


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Kepler-ECG QTc Dataset Preparation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/04_0_prepare_qtc_dataset.py --dataset ptb-xl
    python scripts/04_0_prepare_qtc_dataset.py --dataset chapman
    python scripts/04_0_prepare_qtc_dataset.py --dataset code-15
    python scripts/04_0_prepare_qtc_dataset.py --dataset cpsc-2018
    python scripts/04_0_prepare_qtc_dataset.py --dataset georgia
    python scripts/04_0_prepare_qtc_dataset.py --dataset mimic-iv-ecg
    
    # Test with limited samples
    python scripts/04_0_prepare_qtc_dataset.py --dataset ptb-xl --n_samples 1000
        """
    )
    
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='Dataset name (ptb-xl, chapman, code-15, cpsc-2018, georgia, mimic-iv-ecg)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory (default: results/{dataset}/qtc)')
    parser.add_argument('--n_samples', '-n', type=int, default=None,
                        help='Limit number of samples (default: all). Useful for testing.')
    
    args = parser.parse_args()
    
    dataset_name = args.dataset
    n_samples = args.n_samples
    
    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"results/{dataset_name}/qtc")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("KEPLER-ECG QTc DATASET PREPARATION")
    print("=" * 60)
    print(f"Dataset:   {dataset_name}")
    print(f"Output:    {output_path}")
    if n_samples:
        print(f"N samples: {n_samples:,} (limited)")
    else:
        print(f"N samples: all")
    
    # -------------------------------------------------------------------------
    # Step 1: Load wave features
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 1: Loading wave features")
    print("-" * 60)
    
    df_waves = load_wave_features(dataset_name)
    if df_waves is None:
        print(f"❌ Cannot proceed without wave features")
        return 1
    
    print(f"Loaded: {len(df_waves):,} records")
    
    # Apply sample limit if specified
    if n_samples and n_samples < len(df_waves):
        df_waves = df_waves.sample(n=n_samples, random_state=42)
        print(f"Sampled: {len(df_waves):,} records (random_state=42)")
    
    # -------------------------------------------------------------------------
    # Step 2: Load and merge features metadata
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 2: Loading features metadata (demographics + diagnostics)")
    print("-" * 60)
    
    df_features = load_features_metadata(dataset_name)
    
    if df_features is not None:
        print("Merging with wave features...")
        df = merge_wave_and_features(df_waves, df_features)
    else:
        print("⚠️  No features metadata found, proceeding without demographics/diagnostics")
        df = df_waves.copy()
    
    # -------------------------------------------------------------------------
    # Step 3: Apply quality filters
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 3: Applying quality filters")
    print("-" * 60)
    
    df_filtered, filter_stats = apply_quality_filters(df)
    
    print(f"Initial:  {filter_stats['initial']:,} records")
    print(f"After QT valid:     {filter_stats.get('after_qt_valid', 'N/A'):,}")
    print(f"After QT range:     {filter_stats.get('after_qt_range', 'N/A'):,}")
    print(f"After RR range:     {filter_stats.get('after_rr_range', 'N/A'):,}")
    print(f"After HR range:     {filter_stats.get('after_hr_range', 'N/A'):,}")
    if 'after_quality' in filter_stats:
        print(f"After quality:      {filter_stats['after_quality']:,}")
    if 'after_t_detection' in filter_stats:
        print(f"After T detection:  {filter_stats['after_t_detection']:,}")
    print(f"Final:    {filter_stats['final']:,} records ({filter_stats['retention_rate']}%)")
    
    # -------------------------------------------------------------------------
    # Step 4: Calculate QTc reference
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 4: Calculating QTc reference (polynomial method)")
    print("-" * 60)
    
    df_filtered = calculate_qtc_reference(df_filtered)
    
    # -------------------------------------------------------------------------
    # Step 5: Prepare final dataset
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 5: Preparing final dataset")
    print("-" * 60)
    
    df_final = prepare_qtc_dataset(df_filtered)
    
    # Show diagnostic distribution if available
    if 'primary_superclass' in df_final.columns:
        print("\nDiagnostic distribution (primary_superclass):")
        dist = df_final['primary_superclass'].value_counts()
        for cls, count in dist.items():
            pct = 100 * count / len(df_final)
            print(f"  {cls:10s}: {count:6,} ({pct:5.1f}%)")
    
    # -------------------------------------------------------------------------
    # Step 6: Save output
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 6: Saving output")
    print("-" * 60)
    
    output_file = output_path / f"{dataset_name}_qtc_preparation.csv"
    df_final.to_csv(output_file, index=False, sep=';', decimal=',')
    print(f"✅ Saved: {output_file}")
    print(f"   Records: {len(df_final):,}")
    print(f"   Columns: {len(df_final.columns)}")
    
    # -------------------------------------------------------------------------
    # Step 7: Calculate and display correlations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QTc-HR CORRELATIONS")
    print("=" * 60)
    
    hr = df_final['heart_rate_bpm']
    correlations = {}
    
    for col in ['QTc_Bazett_ms', 'QTc_Fridericia_ms', 'QTc_Framingham_ms', 
                'QTc_Hodges_ms', 'QTc_reference_ms']:
        if col in df_final.columns:
            valid = ~(df_final[col].isna() | hr.isna())
            if valid.sum() > 10:
                r, p = stats.pearsonr(df_final.loc[valid, col], hr[valid])
                correlations[col] = round(r, 4)
                print(f"  {col:25s}: r = {r:+.4f}")
    
    # Show by diagnostic class if available
    if 'primary_superclass' in df_final.columns:
        print("\nBy diagnostic class (Bazett):")
        for cls in df_final['primary_superclass'].dropna().unique():
            subset = df_final[df_final['primary_superclass'] == cls]
            if len(subset) > 10 and 'QTc_Bazett_ms' in subset.columns:
                valid = ~(subset['QTc_Bazett_ms'].isna() | subset['heart_rate_bpm'].isna())
                if valid.sum() > 10:
                    r, _ = stats.pearsonr(subset.loc[valid, 'QTc_Bazett_ms'], 
                                         subset.loc[valid, 'heart_rate_bpm'])
                    print(f"  {cls:10s} (n={len(subset):,}): r = {r:+.4f}")
    
    # -------------------------------------------------------------------------
    # Step 8: Save report
    # -------------------------------------------------------------------------
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_name,
        'n_samples_requested': n_samples,
        'output_file': str(output_file),
        'filter_statistics': filter_stats,
        'final_dataset': {
            'records': len(df_final),
            'columns': len(df_final.columns),
            'column_names': df_final.columns.tolist(),
        },
        'qtc_hr_correlations': correlations,
    }
    
    # Add diagnostic distribution
    if 'primary_superclass' in df_final.columns:
        report['diagnostic_distribution'] = df_final['primary_superclass'].value_counts().to_dict()
    
    # Add demographic summary
    if 'age' in df_final.columns:
        report['demographics'] = {
            'age_mean': round(df_final['age'].mean(), 1) if df_final['age'].notna().any() else None,
            'age_std': round(df_final['age'].std(), 1) if df_final['age'].notna().any() else None,
            'age_completeness': round(100 * df_final['age'].notna().mean(), 1),
        }
        if 'sex' in df_final.columns:
            report['demographics']['sex_completeness'] = round(100 * df_final['sex'].notna().mean(), 1)
            sex_dist = df_final['sex'].value_counts().to_dict()
            report['demographics']['sex_distribution'] = {str(k): v for k, v in sex_dist.items()}
    
    report_file = output_path / f"{dataset_name}_qtc_preparation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Report: {report_file}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Dataset:  {dataset_name}")
    print(f"Records:  {len(df_final):,}")
    print(f"Output:   {output_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
