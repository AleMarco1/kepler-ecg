#!/usr/bin/env python3
"""
09_1_subpopulation_stratification.py

Kepler-ECG Pipeline - Phase 09_1: Subpopulation Stratification Analysis

This script analyzes the clinical performance of QTc correction formulas
across different patient subgroups to identify where Kepler provides
the greatest clinical benefit.

Subgroups analyzed:
1. Age: <40, 40-65, >65 years
2. Sex: Male vs Female  
3. Heart Rate: Bradycardia (<60), Normal (60-100), Tachycardia (>100)
4. Dataset/Geographic origin

For each subgroup, we compute:
- QTc-HR correlation (heart rate independence)
- False positive rate (QTc>threshold in NORM patients)
- Triage distribution (GREEN/YELLOW/RED)
- Improvement vs Bazett

Author: Alessandro Marconi
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json
import logging
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

__version__ = '2.0'

# Clinical thresholds (ms)
THRESHOLD_MALE = 450
THRESHOLD_FEMALE = 460

# Age groups
AGE_BINS = [0, 40, 65, 120]
AGE_LABELS = ['<40', '40-65', '>65']

# HR groups  
HR_BINS = [0, 60, 100, 200]
HR_LABELS = ['Bradycardia (<60 bpm)', 'Normal (60-100 bpm)', 'Tachycardia (>100 bpm)']

# Datasets
DATASETS = ['ptb-xl', 'chapman', 'cpsc-2018', 'georgia', 'mimic-iv-ecg', 'code-15']

# Kepler formula coefficients
KEPLER_K = 125
KEPLER_C = -158

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# SEX NORMALIZATION (single source of truth — aligned with 09_0)
# =============================================================================

def normalize_sex(sex_series):
    """
    Normalize sex values to 'M'/'F'/'unknown'.

    Handles all dataset conventions:
      PTB-XL:  int 0/1         (0=M, 1=F)
      Chapman/CPSC/Georgia: float 1.0/0.0  (1=M, 0=F — reversed!)
      Code-15: string 'Male'/'Female'
      MIMIC:   string 'M'/'F'
      CSV edge cases: '0', '1', '0.0', '1.0' as strings
    """
    SEX_MAP = {
        # Numeric conventions (PTB-XL: 0=M, 1=F)
        0: 'M', 0.0: 'M',
        1: 'F', 1.0: 'F',
        # String versions of numeric (CSV edge cases)
        '0': 'M', '0.0': 'M',
        '1': 'F', '1.0': 'F',
        # String labels
        'male': 'M', 'Male': 'M', 'MALE': 'M', 'm': 'M', 'M': 'M',
        'female': 'F', 'Female': 'F', 'FEMALE': 'F', 'f': 'F', 'F': 'F',
    }
    return sex_series.map(SEX_MAP).fillna('unknown')


# =============================================================================
# QTc CALCULATION FUNCTIONS (5 formulas)
# =============================================================================

def calc_qtc_bazett(qt_ms, rr_s):
    """Bazett formula: QTc = QT / sqrt(RR)"""
    return qt_ms / np.sqrt(rr_s)

def calc_qtc_fridericia(qt_ms, rr_s):
    """Fridericia formula: QTc = QT / RR^(1/3)"""
    return qt_ms / np.cbrt(rr_s)

def calc_qtc_framingham(qt_ms, rr_s):
    """Framingham formula: QTc = QT + 154*(1 - RR)"""
    return qt_ms + 154 * (1 - rr_s)

def calc_qtc_hodges(qt_ms, hr_bpm):
    """Hodges formula: QTc = QT + 1.75*(HR - 60)"""
    return qt_ms + 1.75 * (hr_bpm - 60)

def calc_qtc_kepler(qt_ms, rr_s, k=KEPLER_K, c=KEPLER_C):
    """Kepler formula: QTc = QT + k/RR + c"""
    return qt_ms + k / rr_s + c


# =============================================================================
# CLASSIFICATION FUNCTIONS (vectorized)
# =============================================================================

def classify_qtc_binary(qtc_series, sex_series):
    """
    Vectorized QTc classification: 'normal' or 'prolonged'.

    Uses sex-specific thresholds: Male > 450ms, Female > 460ms.
    """
    sex_norm = normalize_sex(sex_series)
    threshold = np.where(sex_norm == 'M', THRESHOLD_MALE,
                np.where(sex_norm == 'F', THRESHOLD_FEMALE, 455))
    result = np.where(qtc_series > threshold, 'prolonged', 'normal')
    # NaN where QTc or sex is invalid
    mask = qtc_series.isna() | (sex_norm == 'unknown')
    result = pd.Series(result, index=qtc_series.index)
    result[mask] = np.nan
    return result


def classify_triage_vectorized(qtc_bazett, qtc_kepler, sex_series):
    """
    Vectorized 3-level triage:
      GREEN:  Bazett normal → Discharge
      YELLOW: Bazett prolonged, Kepler normal → Follow-up
      RED:    Both prolonged → Urgent referral
    """
    class_baz = classify_qtc_binary(qtc_bazett, sex_series)
    class_kep = classify_qtc_binary(qtc_kepler, sex_series)
    result = np.where(class_baz == 'normal', 'GREEN',
             np.where(class_kep == 'normal', 'YELLOW', 'RED'))
    mask = class_baz.isna() | class_kep.isna()
    result = pd.Series(result, index=qtc_bazett.index)
    result[mask] = np.nan
    return result


# =============================================================================
# COLUMN NAME MAPPING
# =============================================================================

# Map from standardized names to possible column names in the CSV files
COLUMN_MAPPING = {
    'QT': ['QT_interval_ms', 'QT_ms', 'QT', 'qt_interval_ms', 'qt'],
    'RR': ['RR_interval_sec', 'RR_s', 'RR_sec', 'RR', 'rr_interval_sec'],
    'HR': ['heart_rate_bpm', 'HR', 'heart_rate', 'hr_bpm', 'hr'],
    'age': ['age', 'Age', 'AGE'],
    'sex': ['sex', 'Sex', 'SEX', 'gender', 'Gender'],
    'superclass': ['primary_superclass', 'superclass', 'Superclass', 'diagnosis_superclass']
}


def find_column(df, standard_name):
    """Find the actual column name in the dataframe for a standardized name."""
    if standard_name in COLUMN_MAPPING:
        for possible_name in COLUMN_MAPPING[standard_name]:
            if possible_name in df.columns:
                return possible_name
    # Fallback: return standard name if it exists
    if standard_name in df.columns:
        return standard_name
    return None


def standardize_columns(df):
    """Rename columns to standardized names."""
    rename_map = {}
    for standard_name in COLUMN_MAPPING.keys():
        actual_name = find_column(df, standard_name)
        if actual_name and actual_name != standard_name:
            rename_map[actual_name] = standard_name
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(base_path, dataset_name):
    """
    Load QTc preparation data and merge with demographics.
    
    Returns DataFrame with: QT, RR, HR, age, sex, superclass
    """
    base = Path(base_path)
    
    # QTc preparation file
    qtc_file = base / 'results' / dataset_name / 'qtc' / f'{dataset_name}_qtc_preparation.csv'
    if not qtc_file.exists():
        print(f"  Warning: {qtc_file} not found")
        return None
    
    df = pd.read_csv(qtc_file)
    
    # Standardize column names
    df = standardize_columns(df)
    
    # Check if we already have all needed columns from qtc_preparation
    needed_cols = ['QT', 'RR', 'HR', 'age', 'sex', 'superclass']
    missing_cols = [col for col in needed_cols if col not in df.columns]
    
    # If missing columns, try to load from features file
    if missing_cols:
        features_file = base / 'results' / dataset_name / 'preprocess' / f'{dataset_name}_features.csv'
        
        if features_file.exists():
            features = pd.read_csv(features_file)
            features = standardize_columns(features)
            
            # Identify common key
            possible_keys = ['ecg_id', 'record_id', 'study_id', 'id']
            key = None
            for k in possible_keys:
                if k in df.columns and k in features.columns:
                    key = k
                    break
            
            if key:
                # Select only needed columns from features
                feature_cols = [key]
                for col in missing_cols:
                    if col in features.columns:
                        feature_cols.append(col)
                
                if len(feature_cols) > 1:  # More than just the key
                    df = df.merge(features[feature_cols], on=key, how='left')
    
    # Calculate HR if still not present
    if 'HR' not in df.columns and 'RR' in df.columns:
        df['HR'] = 60 / df['RR']
    
    # Add dataset identifier
    df['dataset'] = dataset_name
    
    # Print column summary
    print(f"  Columns available: {[c for c in needed_cols if c in df.columns]}")
    
    return df


def load_all_datasets(base_path):
    """Load and combine all datasets."""
    all_data = []
    
    for dataset in DATASETS:
        print(f"Loading {dataset}...")
        df = load_dataset(base_path, dataset)
        if df is not None:
            print(f"  Loaded {len(df):,} records")
            all_data.append(df)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal: {len(combined):,} records")
        return combined
    return None


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_subgroup_metrics(df, subgroup_name, subgroup_value):
    """
    Compute all metrics for a specific subgroup.
    
    Returns dict with:
    - n: sample size
    - hr_corr_{bazett,fridericia,framingham,hodges,kepler}: |r(QTc, HR)|
    - hr_corr_{formula}_p: p-value for each correlation
    - fp_rate_bazett/kepler: false positive rate in NORM patients (PRIMARY metric)
    - triage_green/yellow/red: triage distribution
    - improvement_corr: correlation improvement factor (SECONDARY — can be misleading)
    - improvement_fp: false positive improvement factor (PRIMARY)
    - gs_*: gold standard metrics when QTc_reference_ms available
    """
    results = {
        'subgroup': subgroup_name,
        'value': subgroup_value,
        'n': len(df)
    }
    
    if len(df) < 100:
        return results
    
    # Calculate QTc values (vectorized)
    df = df.copy()
    df['sex_norm'] = normalize_sex(df['sex']) if 'sex' in df.columns else 'unknown'
    df['QTc_Bazett'] = calc_qtc_bazett(df['QT'], df['RR'])
    df['QTc_Kepler'] = calc_qtc_kepler(df['QT'], df['RR'])
    df['QTc_Fridericia'] = calc_qtc_fridericia(df['QT'], df['RR'])
    df['QTc_Framingham'] = calc_qtc_framingham(df['QT'], df['RR'])
    if 'HR' in df.columns:
        df['QTc_Hodges'] = calc_qtc_hodges(df['QT'], df['HR'])
    
    # HR correlations (all 5 formulas)
    valid = df.dropna(subset=['HR', 'QTc_Bazett', 'QTc_Kepler'])
    if len(valid) > 10:
        for fname, col in [('bazett', 'QTc_Bazett'), ('fridericia', 'QTc_Fridericia'),
                           ('framingham', 'QTc_Framingham'), ('hodges', 'QTc_Hodges'),
                           ('kepler', 'QTc_Kepler')]:
            if col in valid.columns:
                r, p = stats.pearsonr(valid['HR'], valid[col])
                results[f'hr_corr_{fname}'] = abs(r)
                results[f'hr_corr_{fname}_p'] = p
        
        # Improvement factor (Bazett / Kepler) — SECONDARY metric
        # NOTE: Can be misleading when errors cancel (see MIMIC anomaly)
        if results.get('hr_corr_kepler', 0) > 0.001:
            results['improvement_corr'] = results['hr_corr_bazett'] / results['hr_corr_kepler']
        else:
            results['improvement_corr'] = float('inf')
    
    # False positive analysis — PRIMARY metric (immune to cancellation)
    if 'superclass' in df.columns:
        # Case-insensitive NORM filter
        norm_df = df[df['superclass'].astype(str).str.upper() == 'NORM'].copy()
        results['n_norm'] = len(norm_df)
        
        if len(norm_df) > 10 and 'sex_norm' in norm_df.columns:
            # Vectorized classification
            norm_df['class_bazett'] = classify_qtc_binary(norm_df['QTc_Bazett'], norm_df['sex'])
            norm_df['class_kepler'] = classify_qtc_binary(norm_df['QTc_Kepler'], norm_df['sex'])
            norm_df['triage'] = classify_triage_vectorized(
                norm_df['QTc_Bazett'], norm_df['QTc_Kepler'], norm_df['sex'])
            
            # False positive rates
            valid_class = norm_df['class_bazett'].notna()
            n_valid = valid_class.sum()
            if n_valid > 0:
                fp_bazett = (norm_df.loc[valid_class, 'class_bazett'] == 'prolonged').sum()
                fp_kepler = (norm_df.loc[valid_class, 'class_kepler'] == 'prolonged').sum()
                
                results['fp_bazett'] = int(fp_bazett)
                results['fp_kepler'] = int(fp_kepler)
                results['fp_rate_bazett'] = 100 * fp_bazett / n_valid
                results['fp_rate_kepler'] = 100 * fp_kepler / n_valid
                
                # FP improvement — PRIMARY metric
                if fp_kepler > 0:
                    results['improvement_fp'] = fp_bazett / fp_kepler
                else:
                    results['improvement_fp'] = float('inf')
                
                # Triage distribution
                triage_valid = norm_df.loc[valid_class, 'triage']
                triage_counts = triage_valid.value_counts()
                for level in ['GREEN', 'YELLOW', 'RED']:
                    results[f'triage_{level.lower()}'] = int(triage_counts.get(level, 0))
                    results[f'triage_{level.lower()}_pct'] = 100 * triage_counts.get(level, 0) / n_valid
    
    # Gold standard analysis (when QTc_reference_ms is available)
    if 'QTc_reference_ms' in df.columns:
        ref = df.dropna(subset=['QTc_reference_ms', 'QTc_Bazett', 'QTc_Kepler', 'sex_norm'])
        ref = ref[ref['sex_norm'].isin(['M', 'F'])]
        if len(ref) > 10:
            threshold = np.where(ref['sex_norm'] == 'M', THRESHOLD_MALE, THRESHOLD_FEMALE)
            ref_prolonged = ref['QTc_reference_ms'].values > threshold
            
            for fname, col in [('bazett', 'QTc_Bazett'), ('kepler', 'QTc_Kepler')]:
                pred_prolonged = ref[col].values > threshold
                tp = int((pred_prolonged & ref_prolonged).sum())
                tn = int((~pred_prolonged & ~ref_prolonged).sum())
                fp = int((pred_prolonged & ~ref_prolonged).sum())
                fn = int((~pred_prolonged & ref_prolonged).sum())
                
                results[f'gs_{fname}_tp'] = tp
                results[f'gs_{fname}_tn'] = tn
                results[f'gs_{fname}_fp'] = fp
                results[f'gs_{fname}_fn'] = fn
                results[f'gs_{fname}_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                results[f'gs_{fname}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                results[f'gs_{fname}_ppv'] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                results[f'gs_{fname}_npv'] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
                sens = results[f'gs_{fname}_sensitivity']
                ppv = results[f'gs_{fname}_ppv']
                if pd.notna(sens) and pd.notna(ppv) and (sens + ppv) > 0:
                    results[f'gs_{fname}_f1'] = 2 * sens * ppv / (sens + ppv)
                else:
                    results[f'gs_{fname}_f1'] = np.nan
    
    return results


def analyze_by_age(df):
    """Stratify analysis by age groups."""
    results = []
    
    if 'age' not in df.columns:
        logger.warning("'age' column not found")
        return results
    
    df = df.copy()
    df['age_group'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    
    for age_group in AGE_LABELS:
        subdf = df[df['age_group'] == age_group]
        if len(subdf) > 0:
            metrics = compute_subgroup_metrics(subdf, 'Age', age_group)
            results.append(metrics)
    
    return results


def analyze_by_sex(df):
    """Stratify analysis by sex."""
    results = []
    
    if 'sex' not in df.columns:
        logger.warning("'sex' column not found")
        return results
    
    df = df.copy()
    df['sex_normalized'] = normalize_sex(df['sex'])
    
    for sex_code, sex_label in [('M', 'Male'), ('F', 'Female')]:
        subdf = df[df['sex_normalized'] == sex_code]
        if len(subdf) > 0:
            metrics = compute_subgroup_metrics(subdf, 'Sex', sex_label)
            results.append(metrics)
    
    return results


def analyze_by_hr(df):
    """Stratify analysis by heart rate groups."""
    results = []
    
    if 'HR' not in df.columns:
        logger.warning("'HR' column not found")
        return results
    
    df = df.copy()
    df['hr_group'] = pd.cut(df['HR'], bins=HR_BINS, labels=HR_LABELS, right=False)
    
    for hr_group in HR_LABELS:
        subdf = df[df['hr_group'] == hr_group]
        if len(subdf) > 0:
            metrics = compute_subgroup_metrics(subdf, 'Heart Rate', hr_group)
            results.append(metrics)
    
    return results


def analyze_by_dataset(df):
    """Stratify analysis by dataset/geographic origin."""
    results = []
    
    if 'dataset' not in df.columns:
        logger.warning("'dataset' column not found")
        return results
    
    for dataset in sorted(df['dataset'].unique()):
        subdf = df[df['dataset'] == dataset]
        if len(subdf) > 0:
            metrics = compute_subgroup_metrics(subdf, 'Dataset', dataset)
            results.append(metrics)
    
    return results


def analyze_cross_hr_sex(df):
    """Cross-stratification: HR × Sex (6 groups)."""
    results = []
    
    if 'HR' not in df.columns or 'sex' not in df.columns:
        logger.warning("HR or sex column not found for cross-stratification")
        return results
    
    df = df.copy()
    df['hr_group'] = pd.cut(df['HR'], bins=HR_BINS, labels=['Bradycardia', 'Normal', 'Tachycardia'], right=False)
    df['sex_normalized'] = normalize_sex(df['sex'])
    
    for hr in ['Bradycardia', 'Normal', 'Tachycardia']:
        for sex_code, sex_label in [('M', 'Male'), ('F', 'Female')]:
            subdf = df[(df['hr_group'] == hr) & (df['sex_normalized'] == sex_code)]
            if len(subdf) > 0:
                label = f"{hr} × {sex_label}"
                metrics = compute_subgroup_metrics(subdf, 'HR × Sex', label)
                results.append(metrics)
    
    return results


def analyze_cross_hr_age(df):
    """Cross-stratification: HR × Age (9 groups)."""
    results = []
    
    if 'HR' not in df.columns or 'age' not in df.columns:
        logger.warning("HR or age column not found for cross-stratification")
        return results
    
    df = df.copy()
    df['hr_group'] = pd.cut(df['HR'], bins=HR_BINS, labels=['Bradycardia', 'Normal', 'Tachycardia'], right=False)
    df['age_group'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    
    for hr in ['Bradycardia', 'Normal', 'Tachycardia']:
        for age in AGE_LABELS:
            subdf = df[(df['hr_group'] == hr) & (df['age_group'] == age)]
            if len(subdf) > 0:
                label = f"{hr} × {age}"
                metrics = compute_subgroup_metrics(subdf, 'HR × Age', label)
                results.append(metrics)
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(base_path, output_path=None):
    """
    Run complete subpopulation stratification analysis.
    """
    print("=" * 70)
    print(f"KEPLER-ECG: Subpopulation Stratification Analysis v{__version__}")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading datasets...")
    df = load_all_datasets(base_path)
    
    if df is None or len(df) == 0:
        print("ERROR: No data loaded")
        return None
    
    # Run analyses
    print("\n2. Analyzing subpopulations...")
    
    all_results = []
    
    print("\n  2.1 By Age...")
    age_results = analyze_by_age(df)
    all_results.extend(age_results)
    
    print("\n  2.2 By Sex...")
    sex_results = analyze_by_sex(df)
    all_results.extend(sex_results)
    
    print("\n  2.3 By Heart Rate...")
    hr_results = analyze_by_hr(df)
    all_results.extend(hr_results)
    
    print("\n  2.4 By Dataset...")
    dataset_results = analyze_by_dataset(df)
    all_results.extend(dataset_results)
    
    print("\n  2.5 Cross-stratification HR × Sex...")
    hr_sex_results = analyze_cross_hr_sex(df)
    all_results.extend(hr_sex_results)
    
    print("\n  2.6 Cross-stratification HR × Age...")
    hr_age_results = analyze_cross_hr_age(df)
    all_results.extend(hr_age_results)
    
    # Overall
    print("\n  2.7 Overall (pooled)...")
    overall = compute_subgroup_metrics(df, 'Overall', 'All datasets')
    all_results.append(overall)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_path:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = out_dir / f'subpopulation_stratification_{timestamp}.csv'
        results_df.to_csv(csv_file, index=False)
        print(f"\n3. Results saved to {csv_file}")
        
        # JSON with metadata
        json_file = out_dir / f'subpopulation_stratification_{timestamp}.json'
        report = {
            'generated': datetime.now().isoformat(),
            'script': '09_1_subpopulation_stratification.py',
            'version': __version__,
            'metadata': {
                'kepler_formula': f'QTc = QT + {KEPLER_K}/RR + ({KEPLER_C})',
                'kepler_k': KEPLER_K,
                'kepler_c': KEPLER_C,
                'total_records': len(df),
                'n_subgroups_analyzed': len(all_results),
                'datasets': sorted(df['dataset'].unique().tolist()),
                'primary_metric': 'fp_rate (immune to cancellation bias)',
                'secondary_metric': 'hr_corr (WARNING: can mask bidirectional errors)',
            },
            'results': all_results,
        }
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"   JSON saved to {json_file}")
    
    # =========================================================================
    # SUMMARY — FP rate as PRIMARY metric
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS (PRIMARY metric: FP rate)")
    print("=" * 70)
    
    def _fmt_fp(r):
        fp_b = r.get('fp_rate_bazett', None)
        fp_k = r.get('fp_rate_kepler', None)
        if fp_b is not None and fp_k is not None:
            imp = r.get('improvement_fp', 0)
            imp_s = f"{imp:.0f}x" if np.isfinite(imp) and imp < 1000 else "∞"
            return f"FP: {fp_b:.1f}% → {fp_k:.1f}% ({imp_s})"
        return "—"
    
    def _fmt_corr(r):
        rb = r.get('hr_corr_bazett', None)
        rk = r.get('hr_corr_kepler', None)
        if rb is not None and rk is not None:
            return f"|r|: Baz={rb:.4f}, Kep={rk:.4f}"
        return ""
    
    for section, results_list in [
        ("BY AGE", age_results),
        ("BY SEX", sex_results),
        ("BY HEART RATE", hr_results),
        ("BY DATASET", dataset_results),
        ("CROSS: HR × SEX", hr_sex_results),
        ("CROSS: HR × AGE", hr_age_results),
    ]:
        print(f"\n--- {section} ---")
        for r in results_list:
            print(f"  {r['value']:30s}: n={r['n']:>8,}  {_fmt_fp(r):35s}  {_fmt_corr(r)}")
    
    print(f"\n--- OVERALL ---")
    print(f"  {overall['value']:30s}: n={overall['n']:>8,}  {_fmt_fp(overall):35s}  {_fmt_corr(overall)}")
    
    # =========================================================================
    # CANCELLATION WARNING for subgroups where |r| is misleading
    # =========================================================================
    misleading = []
    for r in all_results:
        rb = r.get('hr_corr_bazett', 999)
        rk = r.get('hr_corr_kepler', 999)
        fp_b = r.get('fp_rate_bazett', 0)
        fp_k = r.get('fp_rate_kepler', 0)
        if rk > rb and fp_b > fp_k:
            misleading.append(r)
    
    if misleading:
        print(f"\n⚠️  CANCELLATION WARNING: {len(misleading)} subgroup(s) where |r|_Kepler > |r|_Bazett")
        print(f"   but Kepler STILL wins on FP rate — correlation is misleading here:")
        for r in misleading:
            print(f"   • {r['subgroup']}={r['value']}: "
                  f"|r| Baz={r['hr_corr_bazett']:.4f} < Kep={r['hr_corr_kepler']:.4f}, "
                  f"but FP Baz={r['fp_rate_bazett']:.1f}% > Kep={r['fp_rate_kepler']:.1f}%")
    
    return results_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Kepler-ECG: Subpopulation Stratification Analysis v2.0')
    parser.add_argument('--results-dir', '--base-path', type=str, required=True,
                        help='Base path to Kepler-ECG project')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        help='Datasets to analyze')
    
    args = parser.parse_args()
    
    output = args.output or Path(args.results_dir) / 'results' / 'clinical_analysis'
    Path(output).mkdir(parents=True, exist_ok=True)
    
    main(args.results_dir, output)
