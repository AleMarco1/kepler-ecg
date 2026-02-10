#!/usr/bin/env python3
"""
09_3_diagnostic_concordance.py

Kepler-ECG Pipeline - Phase 09_3: Diagnostic Concordance Analysis

This script analyzes the concordance between QTc formula classifications
and actual clinical diagnoses (SNOMED/ICD-10 labels) present in the datasets.

Key analyses:
1. Overall diagnostic concordance (Sensitivity/Specificity/PPV/NPV/F1)
2. Concordance stratified by dataset
3. Concordance stratified by heart rate
4. Concordance stratified by age
5. Triage safety analysis on true positives
6. Concordance by QTc severity level
7. ROC analysis and optimal threshold determination

Author: Alessandro Marconi
Date: January 2026
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json
import warnings
import re
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION (defaults, overridable via CLI)
# =============================================================================

# Clinical thresholds (ms)
THRESHOLD_MALE = 450
THRESHOLD_FEMALE = 460
THRESHOLD_UNKNOWN = 455  # Midpoint when sex is unavailable

# Kepler formula coefficients
KEPLER_K = 125
KEPLER_C = -158

# HR bins for stratification
HR_BINS = [0, 60, 100, 300]
HR_LABELS = ['Bradycardia (<60)', 'Normal (60-100)', 'Tachycardia (>100)']

# Age bins
AGE_BINS = [0, 40, 65, 120]
AGE_LABELS = ['<40', '40-65', '>65']

# QTc severity bins (relative to threshold)
QTC_SEVERITY_BINS = [-1000, -20, 0, 20, 50, 1000]
QTC_SEVERITY_LABELS = ['Normal', 'Borderline-low', 'Borderline-high',
                       'Prolonged', 'Severely prolonged']

# Datasets
DATASETS = ['ptb-xl', 'chapman', 'cpsc-2018', 'georgia', 'mimic-iv-ecg', 'code-15']

# Sex value mappings (no overlap)
# Pipeline convention from 02_0_process_dataset.py: 0=male, 1=female
# String datasets (Chapman/Georgia/MIMIC) use 'Male'/'Female' or 'M'/'F'
# Note: pandas reads integer columns with NaN as float64 (0.0, 1.0)
#       and .astype(str) produces '0.0'/'1.0', so we must cover those too
SEX_MALE_VALUES = ['m', 'male', '0', '0.0']
SEX_FEMALE_VALUES = ['f', 'female', '1', '1.0']

# SNOMED/diagnosis patterns for prolonged QT
PROLONGED_QT_PATTERNS = [
    r'prolong.*qt',
    r'qt.*prolong',
    r'long.*qt',
    r'qt.*long',
    r'lqts\d*\b',           # LQTS, LQTS1, LQTS2, LQTS3...
    r'\blqt\d*\b',          # LQT, LQT1, LQT2, LQT3...
    r'long qt syndrome',
    r'qt interval prolongation',
    r'prolonged qt interval',
    r'acquired long qt',
    r'congenital long qt',
    # ICD-10 codes (lowercase — text is lowered before matching)
    r'\bi45\.81\b',          # Long QT syndrome (exact match)
    # SNOMED CT codes
    r'\b9651000119103\b',    # Prolongation of QT interval (SNOMED)
    r'\b77867006\b',         # Long QT syndrome (SNOMED)
]

# Column mapping (single source of truth)
COLUMN_MAPPING = {
    'QT': ['QT_interval_ms', 'QT_ms', 'QT', 'qt_interval_ms', 'qt'],
    'RR': ['RR_interval_sec', 'RR_s', 'RR_sec', 'RR', 'rr_interval_sec'],
    'HR': ['heart_rate_bpm', 'HR', 'heart_rate', 'hr_bpm', 'hr'],
    'age': ['age', 'Age', 'AGE'],
    'sex': ['sex', 'Sex', 'SEX', 'gender', 'Gender'],
    'superclass': ['primary_superclass', 'superclass', 'Superclass',
                   'diagnosis_superclass'],
}

# Diagnosis columns to search (single source of truth, used for both
# features merge and true-positive identification)
DIAGNOSIS_COLUMNS = [
    'scp_codes', 'scp_codes_str', 'diagnosis', 'diagnoses',
    'report', 'machine_report', 'labels', 'diagnostic_superclass'
]

# Minimum sample sizes for analyses
MIN_SAMPLES_STRATUM = 10     # Min true positives per stratum (dataset/HR/age)
MIN_SAMPLES_SEVERITY = 50    # Min total records per severity bin
MIN_SAMPLES_TRIAGE_HR = 5    # Min true positives per triage HR subgroup

# ROC defaults
DEFAULT_ROC_CONFIG = {
    'threshold_min': 350,
    'threshold_max': 550,
    'threshold_step': 5,
}


# =============================================================================
# HELPER: SEX IDENTIFICATION (centralized, no overlap)
# =============================================================================

def is_male(sex_value):
    """Check if sex value indicates male."""
    if pd.isna(sex_value):
        return False
    return str(sex_value).lower().strip() in SEX_MALE_VALUES


def is_female(sex_value):
    """Check if sex value indicates female."""
    if pd.isna(sex_value):
        return False
    return str(sex_value).lower().strip() in SEX_FEMALE_VALUES


# =============================================================================
# QTc CALCULATION FUNCTIONS
# =============================================================================

def calc_qtc_bazett(qt_ms, rr_s):
    """Bazett formula: QTc = QT / sqrt(RR)"""
    return qt_ms / np.sqrt(rr_s)

def calc_qtc_fridericia(qt_ms, rr_s):
    """Fridericia formula: QTc = QT / RR^(1/3)"""
    return qt_ms / np.cbrt(rr_s)

def calc_qtc_kepler(qt_ms, rr_s):
    """Kepler formula: QTc = QT + k/RR + c"""
    return qt_ms + KEPLER_K / rr_s + KEPLER_C


# =============================================================================
# CLASSIFICATION FUNCTIONS (vectorized)
# =============================================================================

def get_thresholds_vectorized(sex_series):
    """Vectorized threshold lookup for a pandas Series."""
    sex_lower = sex_series.astype(str).str.lower().str.strip()
    thresholds = pd.Series(THRESHOLD_UNKNOWN, index=sex_series.index, dtype=float)
    thresholds[sex_lower.isin(SEX_MALE_VALUES)] = THRESHOLD_MALE
    thresholds[sex_lower.isin(SEX_FEMALE_VALUES)] = THRESHOLD_FEMALE
    thresholds[sex_series.isna()] = THRESHOLD_UNKNOWN
    return thresholds


def classify_prolonged_vec(qtc_series, sex_series):
    """Vectorized binary classification: True if prolonged."""
    thresholds = get_thresholds_vectorized(sex_series)
    return qtc_series > thresholds


def assign_triage_vec(pred_bazett, pred_kepler):
    """
    Vectorized triage assignment from boolean prediction arrays.
    GREEN:  Bazett normal  → Discharge
    YELLOW: Bazett prolonged, Kepler normal → Follow-up
    RED:    Both prolonged → Urgent referral
    """
    triage = pd.Series('RED', index=pred_bazett.index)
    triage[~pred_bazett] = 'GREEN'
    triage[pred_bazett & ~pred_kepler] = 'YELLOW'
    return triage


# =============================================================================
# DIAGNOSIS DETECTION
# =============================================================================

def has_prolonged_qt_diagnosis(diagnosis_text):
    """Check if a diagnosis string contains prolonged QT indicators."""
    if pd.isna(diagnosis_text):
        return False
    diagnosis_lower = str(diagnosis_text).lower()
    for pattern in PROLONGED_QT_PATTERNS:
        if re.search(pattern, diagnosis_lower):
            return True
    return False


def identify_true_prolonged_qt(df):
    """
    Identify patients with clinical diagnosis of prolonged QT.
    Searches through all columns listed in DIAGNOSIS_COLUMNS.
    """
    df['has_qt_prolonged_dx'] = False

    for col in DIAGNOSIS_COLUMNS:
        if col in df.columns:
            mask = df[col].apply(has_prolonged_qt_diagnosis)
            df['has_qt_prolonged_dx'] = df['has_qt_prolonged_dx'] | mask
            n_found = mask.sum()
            if n_found > 0:
                print(f"    Found {n_found} prolonged QT cases in column '{col}'")

    return df


# =============================================================================
# DATA LOADING
# =============================================================================

def find_column(df, standard_name):
    """Find the actual column name in the dataframe."""
    if standard_name in COLUMN_MAPPING:
        for possible_name in COLUMN_MAPPING[standard_name]:
            if possible_name in df.columns:
                return possible_name
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


def load_dataset(base_path, dataset_name):
    """Load QTc preparation data with diagnosis information."""
    base = Path(base_path)

    # QTc preparation file
    qtc_file = base / 'results' / dataset_name / 'qtc' / f'{dataset_name}_qtc_preparation.csv'
    if not qtc_file.exists():
        print(f"  Warning: {qtc_file} not found")
        return None

    df = pd.read_csv(qtc_file)
    df = standardize_columns(df)

    # Also load features file for additional diagnosis info
    features_file = base / 'results' / dataset_name / 'preprocess' / f'{dataset_name}_features.csv'

    if features_file.exists():
        features = pd.read_csv(features_file)

        # Find common key
        possible_keys = ['ecg_id', 'record_id', 'study_id', 'id']
        key = None
        for k in possible_keys:
            if k in df.columns and k in features.columns:
                key = k
                break

        if key:
            # Get diagnosis columns from features that are not already in df
            diag_cols = [key]
            for col in features.columns:
                if any(x in col.lower() for x in ['diag', 'scp', 'report', 'label', 'code']):
                    if col not in df.columns:
                        diag_cols.append(col)

            if len(diag_cols) > 1:
                df = df.merge(features[diag_cols], on=key, how='left')

    # Calculate HR if not present
    if 'HR' not in df.columns and 'RR' in df.columns:
        df['HR'] = 60 / df['RR']

    df['dataset'] = dataset_name
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
# CENTRALIZED QTc ENRICHMENT
# =============================================================================

def enrich_qtc(df):
    """
    Calculate all QTc values, predictions, and triage once centrally.
    Returns the same DataFrame with added columns.
    """
    print("\nEnriching dataset with QTc values and classifications...")

    # QTc calculations
    df['QTc_Bazett'] = calc_qtc_bazett(df['QT'], df['RR'])
    df['QTc_Kepler'] = calc_qtc_kepler(df['QT'], df['RR'])
    df['QTc_Fridericia'] = calc_qtc_fridericia(df['QT'], df['RR'])

    # Vectorized binary classification
    df['pred_bazett'] = classify_prolonged_vec(df['QTc_Bazett'], df['sex'])
    df['pred_kepler'] = classify_prolonged_vec(df['QTc_Kepler'], df['sex'])
    df['pred_fridericia'] = classify_prolonged_vec(df['QTc_Fridericia'], df['sex'])

    # Triage
    df['triage'] = assign_triage_vec(df['pred_bazett'], df['pred_kepler'])

    # Thresholds (for severity analysis)
    df['threshold'] = get_thresholds_vectorized(df['sex'])

    print(f"  Columns added: QTc_Bazett, QTc_Kepler, QTc_Fridericia, "
          f"pred_*, triage, threshold")

    return df


# =============================================================================
# DIAGNOSTIC METRICS CALCULATION
# =============================================================================

def calculate_diagnostic_metrics(y_true, y_pred):
    """
    Calculate sensitivity, specificity, PPV, NPV, F1.

    y_true: boolean array (True = has prolonged QT diagnosis)
    y_pred: boolean array (True = formula predicts prolonged)
    """
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true = np.array(y_true[mask], dtype=bool)
    y_pred = np.array(y_pred[mask], dtype=bool)

    if len(y_true) == 0:
        return {}

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return {
        'n_total': len(y_true),
        'n_true_positive_dx': int(np.sum(y_true)),
        'n_pred_positive': int(np.sum(y_pred)),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1,
        'accuracy': accuracy
    }


# =============================================================================
# ANALYSIS 1: OVERALL CONCORDANCE
# =============================================================================

def analyze_overall_concordance(df):
    """Analyze overall concordance between formulas and clinical diagnoses."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Overall Diagnostic Concordance")
    print("=" * 70)

    n_true_qt_long = df['has_qt_prolonged_dx'].sum()
    print(f"\n  Total patients with prolonged QT diagnosis: "
          f"{n_true_qt_long:,} ({100 * n_true_qt_long / len(df):.2f}%)")

    if n_true_qt_long == 0:
        print("  WARNING: No prolonged QT diagnoses found. Check diagnosis columns.")
        return None

    results = {}

    for formula in ['bazett', 'kepler', 'fridericia']:
        metrics = calculate_diagnostic_metrics(
            df['has_qt_prolonged_dx'], df[f'pred_{formula}'])
        results[formula] = metrics

        print(f"\n  {formula.upper()}:")
        print(f"    Sensitivity: {metrics['sensitivity'] * 100:.1f}%")
        print(f"    Specificity: {metrics['specificity'] * 100:.1f}%")
        print(f"    PPV: {metrics['ppv'] * 100:.1f}%")
        print(f"    NPV: {metrics['npv'] * 100:.1f}%")
        print(f"    F1 Score: {metrics['f1_score']:.3f}")

    # Triage analysis on true positives
    true_positives = df[df['has_qt_prolonged_dx']]
    triage_dist = true_positives['triage'].value_counts()
    n_tp = len(true_positives)

    results['triage_on_true_positives'] = {
        'n_true_positives': n_tp,
        'green': int(triage_dist.get('GREEN', 0)),
        'yellow': int(triage_dist.get('YELLOW', 0)),
        'red': int(triage_dist.get('RED', 0)),
        'green_pct': 100 * triage_dist.get('GREEN', 0) / n_tp if n_tp > 0 else 0,
        'yellow_pct': 100 * triage_dist.get('YELLOW', 0) / n_tp if n_tp > 0 else 0,
        'red_pct': 100 * triage_dist.get('RED', 0) / n_tp if n_tp > 0 else 0,
        'captured_pct': 100 * (triage_dist.get('YELLOW', 0) + triage_dist.get('RED', 0)) / n_tp if n_tp > 0 else 0
    }

    t = results['triage_on_true_positives']
    print(f"\n  TRIAGE on True Prolonged QT (n={n_tp:,}):")
    print(f"    GREEN (missed): {t['green_pct']:.1f}%")
    print(f"    YELLOW (follow-up): {t['yellow_pct']:.1f}%")
    print(f"    RED (urgent): {t['red_pct']:.1f}%")
    print(f"    CAPTURED (YELLOW+RED): {t['captured_pct']:.1f}%")

    return results


# =============================================================================
# ANALYSIS 2: CONCORDANCE BY DATASET
# =============================================================================

def analyze_by_dataset(df):
    """Analyze concordance stratified by dataset."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Concordance by Dataset")
    print("=" * 70)

    results = {}

    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        n_true = dataset_df['has_qt_prolonged_dx'].sum()

        if n_true < MIN_SAMPLES_STRATUM:
            print(f"\n  {dataset}: {n_true} true positives (insufficient, "
                  f"min={MIN_SAMPLES_STRATUM})")
            continue

        print(f"\n  {dataset} (n={len(dataset_df):,}, true QT long={n_true:,}):")

        dataset_results = {}
        for formula in ['bazett', 'kepler']:
            metrics = calculate_diagnostic_metrics(
                dataset_df['has_qt_prolonged_dx'],
                dataset_df[f'pred_{formula}'])
            dataset_results[formula] = metrics
            print(f"    {formula}: Sens={metrics['sensitivity'] * 100:.1f}%, "
                  f"Spec={metrics['specificity'] * 100:.1f}%, "
                  f"F1={metrics['f1_score']:.3f}")

        # Triage
        true_pos = dataset_df[dataset_df['has_qt_prolonged_dx']]
        triage_dist = true_pos['triage'].value_counts()
        captured = (triage_dist.get('YELLOW', 0) + triage_dist.get('RED', 0)) / len(true_pos) * 100 if len(true_pos) > 0 else 0
        dataset_results['triage_captured_pct'] = captured
        print(f"    Triage captured: {captured:.1f}%")

        results[dataset] = dataset_results

    return results


# =============================================================================
# ANALYSIS 3: CONCORDANCE BY HEART RATE
# =============================================================================

def analyze_by_hr(df):
    """Analyze concordance stratified by heart rate."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Concordance by Heart Rate")
    print("=" * 70)

    df = df.copy()
    df['hr_group'] = pd.cut(df['HR'], bins=HR_BINS, labels=HR_LABELS, right=False)

    results = {}

    for hr_group in HR_LABELS:
        hr_df = df[df['hr_group'] == hr_group]
        n_true = hr_df['has_qt_prolonged_dx'].sum()

        if n_true < MIN_SAMPLES_STRATUM:
            print(f"\n  {hr_group}: {n_true} true positives (insufficient)")
            continue

        print(f"\n  {hr_group} (n={len(hr_df):,}, true QT long={n_true:,}):")

        hr_results = {}
        for formula in ['bazett', 'kepler']:
            metrics = calculate_diagnostic_metrics(
                hr_df['has_qt_prolonged_dx'],
                hr_df[f'pred_{formula}'])
            hr_results[formula] = metrics
            print(f"    {formula}: Sens={metrics['sensitivity'] * 100:.1f}%, "
                  f"Spec={metrics['specificity'] * 100:.1f}%, "
                  f"F1={metrics['f1_score']:.3f}")

        # Triage
        true_pos = hr_df[hr_df['has_qt_prolonged_dx']]
        triage_dist = true_pos['triage'].value_counts()
        captured = (triage_dist.get('YELLOW', 0) + triage_dist.get('RED', 0)) / len(true_pos) * 100 if len(true_pos) > 0 else 0
        missed = triage_dist.get('GREEN', 0) / len(true_pos) * 100 if len(true_pos) > 0 else 0
        hr_results['triage_captured_pct'] = captured
        hr_results['triage_missed_pct'] = missed
        print(f"    Triage: Captured={captured:.1f}%, Missed={missed:.1f}%")

        results[hr_group] = hr_results

    return results


# =============================================================================
# ANALYSIS 4: CONCORDANCE BY AGE
# =============================================================================

def analyze_by_age(df):
    """Analyze concordance stratified by age."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Concordance by Age")
    print("=" * 70)

    if 'age' not in df.columns:
        print("  Age column not available")
        return None

    df = df.copy()
    df['age_group'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS,
                             right=False)

    results = {}

    for age_group in AGE_LABELS:
        age_df = df[df['age_group'] == age_group]
        n_true = age_df['has_qt_prolonged_dx'].sum()

        if n_true < MIN_SAMPLES_STRATUM:
            print(f"\n  {age_group}: {n_true} true positives (insufficient)")
            continue

        print(f"\n  Age {age_group} (n={len(age_df):,}, true QT long={n_true:,}):")

        age_results = {}
        for formula in ['bazett', 'kepler']:
            metrics = calculate_diagnostic_metrics(
                age_df['has_qt_prolonged_dx'],
                age_df[f'pred_{formula}'])
            age_results[formula] = metrics
            print(f"    {formula}: Sens={metrics['sensitivity'] * 100:.1f}%, "
                  f"Spec={metrics['specificity'] * 100:.1f}%, "
                  f"F1={metrics['f1_score']:.3f}")

        # Triage
        true_pos = age_df[age_df['has_qt_prolonged_dx']]
        triage_dist = true_pos['triage'].value_counts()
        captured = (triage_dist.get('YELLOW', 0) + triage_dist.get('RED', 0)) / len(true_pos) * 100 if len(true_pos) > 0 else 0
        age_results['triage_captured_pct'] = captured
        print(f"    Triage captured: {captured:.1f}%")

        results[age_group] = age_results

    return results


# =============================================================================
# ANALYSIS 5: TRIAGE SAFETY
# =============================================================================

def analyze_triage_safety(df):
    """Detailed analysis of triage system safety on true positives."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Triage Safety Analysis")
    print("=" * 70)

    true_positives = df[df['has_qt_prolonged_dx']].copy()

    if len(true_positives) == 0:
        print("  No true positives found")
        return None

    print(f"\n  Total true prolonged QT cases: {len(true_positives):,}")

    results = {'n_true_positives': len(true_positives)}

    # Overall triage distribution
    triage_dist = true_positives['triage'].value_counts()
    n_tp = len(true_positives)
    results['overall'] = {
        'green': int(triage_dist.get('GREEN', 0)),
        'yellow': int(triage_dist.get('YELLOW', 0)),
        'red': int(triage_dist.get('RED', 0)),
        'green_pct': 100 * triage_dist.get('GREEN', 0) / n_tp,
        'yellow_pct': 100 * triage_dist.get('YELLOW', 0) / n_tp,
        'red_pct': 100 * triage_dist.get('RED', 0) / n_tp
    }

    o = results['overall']
    print(f"\n  Overall Triage Distribution:")
    print(f"    GREEN (missed by both): {o['green']:,} ({o['green_pct']:.1f}%)")
    print(f"    YELLOW (Bazett caught, Kepler filtered): "
          f"{o['yellow']:,} ({o['yellow_pct']:.1f}%)")
    print(f"    RED (both caught): {o['red']:,} ({o['red_pct']:.1f}%)")

    safety_pct = o['yellow_pct'] + o['red_pct']
    print(f"\n  SAFETY METRIC: {safety_pct:.1f}% of true prolonged QT "
          f"captured by triage (YELLOW+RED)")
    results['safety_captured_pct'] = safety_pct

    # Analyze GREEN cases (missed) — what are their characteristics?
    green_missed = true_positives[true_positives['triage'] == 'GREEN']

    if len(green_missed) > 0:
        print(f"\n  Analysis of MISSED cases (GREEN triage, n={len(green_missed)}):")

        results['missed_analysis'] = {
            'n': len(green_missed),
            'mean_hr': float(green_missed['HR'].mean()),
            'mean_qtc_bazett': float(green_missed['QTc_Bazett'].mean()),
            'mean_qtc_kepler': float(green_missed['QTc_Kepler'].mean())
        }

        if 'age' in green_missed.columns:
            results['missed_analysis']['mean_age'] = float(
                green_missed['age'].mean())
            print(f"    Mean age: {green_missed['age'].mean():.1f} years")

        print(f"    Mean HR: {green_missed['HR'].mean():.1f} bpm")
        print(f"    Mean QTc Bazett: {green_missed['QTc_Bazett'].mean():.1f} ms")
        print(f"    Mean QTc Kepler: {green_missed['QTc_Kepler'].mean():.1f} ms")

    # By HR zone
    print("\n  Triage Safety by HR Zone:")
    if 'HR' in true_positives.columns:
        true_positives['hr_group'] = pd.cut(
            true_positives['HR'], bins=HR_BINS, labels=HR_LABELS, right=False)

        results['by_hr'] = {}
        for hr_group in HR_LABELS:
            hr_df = true_positives[true_positives['hr_group'] == hr_group]
            if len(hr_df) < MIN_SAMPLES_TRIAGE_HR:
                continue

            triage_hr = hr_df['triage'].value_counts()
            captured = (triage_hr.get('YELLOW', 0) + triage_hr.get('RED', 0)) / len(hr_df) * 100

            results['by_hr'][hr_group] = {
                'n': len(hr_df),
                'captured_pct': captured,
                'missed_pct': 100 - captured
            }
            print(f"    {hr_group}: n={len(hr_df)}, "
                  f"Captured={captured:.1f}%, Missed={100 - captured:.1f}%")

    return results


# =============================================================================
# ANALYSIS 6: CONCORDANCE BY QTc SEVERITY
# =============================================================================

def analyze_qtc_severity(df):
    """Analyze concordance by QTc severity level."""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Concordance by QTc Severity")
    print("=" * 70)

    df = df.copy()

    # Distance from threshold (using pre-computed thresholds)
    df['qtc_distance_bazett'] = df['QTc_Bazett'] - df['threshold']
    df['qtc_distance_kepler'] = df['QTc_Kepler'] - df['threshold']

    # Categorize by Kepler QTc distance
    df['severity'] = pd.cut(df['qtc_distance_kepler'],
                            bins=QTC_SEVERITY_BINS,
                            labels=QTC_SEVERITY_LABELS,
                            right=False)

    results = {}

    print("\n  Triage Performance by QTc Severity (Kepler-based):")

    for severity in QTC_SEVERITY_LABELS:
        sev_df = df[df['severity'] == severity]
        n_true = int(sev_df['has_qt_prolonged_dx'].sum())

        if len(sev_df) < MIN_SAMPLES_SEVERITY:
            continue

        triage_dist = sev_df['triage'].value_counts()
        n_sev = len(sev_df)

        results[severity] = {
            'n_total': n_sev,
            'n_true_positive': n_true,
            'prevalence': 100 * n_true / n_sev,
            'triage_green_pct': 100 * triage_dist.get('GREEN', 0) / n_sev,
            'triage_yellow_pct': 100 * triage_dist.get('YELLOW', 0) / n_sev,
            'triage_red_pct': 100 * triage_dist.get('RED', 0) / n_sev
        }

        true_pos = sev_df[sev_df['has_qt_prolonged_dx']]
        if len(true_pos) > 0:
            tp_triage = true_pos['triage'].value_counts()
            results[severity]['true_pos_captured_pct'] = 100 * (
                tp_triage.get('YELLOW', 0) + tp_triage.get('RED', 0)
            ) / len(true_pos)

        r = results[severity]
        print(f"\n    {severity}:")
        print(f"      N={n_sev:,}, True QT long={n_true} "
              f"({r['prevalence']:.2f}%)")
        print(f"      Triage: GREEN={r['triage_green_pct']:.1f}%, "
              f"YELLOW={r['triage_yellow_pct']:.1f}%, "
              f"RED={r['triage_red_pct']:.1f}%")
        if 'true_pos_captured_pct' in r:
            print(f"      True positives captured: "
                  f"{r['true_pos_captured_pct']:.1f}%")

    return results


# =============================================================================
# ANALYSIS 7: ROC ANALYSIS AND OPTIMAL THRESHOLD
# =============================================================================

def analyze_roc(df, roc_config=None):
    """
    ROC analysis: sweep QTc thresholds and compute sensitivity/specificity
    for each formula. Determine optimal threshold (Youden's J).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 7: ROC Analysis & Optimal Thresholds")
    print("=" * 70)

    cfg = roc_config or DEFAULT_ROC_CONFIG

    y_true = df['has_qt_prolonged_dx'].values.astype(bool)
    n_positive = int(y_true.sum())
    n_negative = int((~y_true).sum())

    if n_positive == 0:
        print("  No true positives — cannot compute ROC")
        return None

    thresholds = np.arange(cfg['threshold_min'],
                           cfg['threshold_max'] + cfg['threshold_step'],
                           cfg['threshold_step'])

    results = {}

    for formula_name, qtc_col in [('Bazett', 'QTc_Bazett'),
                                   ('Kepler', 'QTc_Kepler'),
                                   ('Fridericia', 'QTc_Fridericia')]:
        qtc_values = df[qtc_col].values
        roc_points = []
        best_j = -1
        best_threshold = 0
        best_sens = 0
        best_spec = 0

        for t in thresholds:
            pred = qtc_values > t
            tp = int(np.sum(y_true & pred))
            fp = int(np.sum(~y_true & pred))
            fn = int(np.sum(y_true & ~pred))
            tn = int(np.sum(~y_true & ~pred))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            j = sens + spec - 1  # Youden's J

            roc_points.append({
                'threshold': int(t),
                'sensitivity': round(sens, 4),
                'specificity': round(spec, 4),
                'youden_j': round(j, 4),
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
            })

            if j > best_j:
                best_j = j
                best_threshold = int(t)
                best_sens = sens
                best_spec = spec

        results[formula_name] = {
            'optimal_threshold': best_threshold,
            'optimal_sensitivity': best_sens,
            'optimal_specificity': best_spec,
            'optimal_youden_j': best_j,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'roc_points': roc_points
        }

        print(f"\n  {formula_name}:")
        print(f"    Optimal threshold (Youden's J): {best_threshold} ms")
        print(f"    Sensitivity: {best_sens * 100:.1f}%")
        print(f"    Specificity: {best_spec * 100:.1f}%")
        print(f"    Youden's J: {best_j:.3f}")

    # Compare optimal vs standard thresholds
    print("\n  COMPARISON: Standard vs Optimal Thresholds")
    print("  " + "-" * 60)
    for formula_name in ['Bazett', 'Kepler', 'Fridericia']:
        opt = results[formula_name]['optimal_threshold']
        std = THRESHOLD_MALE  # Use male as reference
        delta = opt - std
        print(f"    {formula_name}: Standard={std}ms, "
              f"Optimal={opt}ms (delta={delta:+d}ms)")

    return results


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_results(all_results, output_dir):
    """Save all results to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full JSON
    # Strip roc_points from JSON to keep it readable (saved separately as CSV)
    json_results = {}
    for key, value in all_results.items():
        if key == 'roc':
            if value is None:
                json_results[key] = None
                continue
            json_results[key] = {}
            for formula, roc_data in value.items():
                json_results[key][formula] = {
                    k: v for k, v in roc_data.items() if k != 'roc_points'
                }
        else:
            json_results[key] = value

    json_file = output_dir / 'diagnostic_concordance.json'
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nJSON report saved to {json_file}")

    # CSV: Overall metrics per formula
    if all_results.get('overall'):
        rows = []
        for formula in ['bazett', 'kepler', 'fridericia']:
            if formula in all_results['overall']:
                row = {'formula': formula}
                row.update(all_results['overall'][formula])
                rows.append(row)
        if rows:
            csv_file = output_dir / 'concordance_overall_metrics.csv'
            pd.DataFrame(rows).to_csv(csv_file, index=False)
            print(f"Overall metrics saved to {csv_file}")

    # CSV: By dataset
    if all_results.get('by_dataset'):
        rows = []
        for dataset, data in all_results['by_dataset'].items():
            for formula in ['bazett', 'kepler']:
                if formula in data:
                    row = {'dataset': dataset, 'formula': formula}
                    row.update(data[formula])
                    row['triage_captured_pct'] = data.get(
                        'triage_captured_pct', None)
                    rows.append(row)
        if rows:
            csv_file = output_dir / 'concordance_by_dataset.csv'
            pd.DataFrame(rows).to_csv(csv_file, index=False)
            print(f"By-dataset metrics saved to {csv_file}")

    # CSV: By HR
    if all_results.get('by_hr'):
        rows = []
        for hr_group, data in all_results['by_hr'].items():
            for formula in ['bazett', 'kepler']:
                if formula in data:
                    row = {'hr_group': hr_group, 'formula': formula}
                    row.update(data[formula])
                    row['triage_captured_pct'] = data.get(
                        'triage_captured_pct', None)
                    row['triage_missed_pct'] = data.get(
                        'triage_missed_pct', None)
                    rows.append(row)
        if rows:
            csv_file = output_dir / 'concordance_by_hr.csv'
            pd.DataFrame(rows).to_csv(csv_file, index=False)
            print(f"By-HR metrics saved to {csv_file}")

    # CSV: By age
    if all_results.get('by_age'):
        rows = []
        for age_group, data in all_results['by_age'].items():
            for formula in ['bazett', 'kepler']:
                if formula in data:
                    row = {'age_group': age_group, 'formula': formula}
                    row.update(data[formula])
                    row['triage_captured_pct'] = data.get(
                        'triage_captured_pct', None)
                    rows.append(row)
        if rows:
            csv_file = output_dir / 'concordance_by_age.csv'
            pd.DataFrame(rows).to_csv(csv_file, index=False)
            print(f"By-age metrics saved to {csv_file}")

    # CSV: By severity
    if all_results.get('by_severity'):
        rows = []
        for severity, data in all_results['by_severity'].items():
            row = {'severity': severity}
            row.update(data)
            rows.append(row)
        if rows:
            csv_file = output_dir / 'concordance_by_severity.csv'
            pd.DataFrame(rows).to_csv(csv_file, index=False)
            print(f"By-severity metrics saved to {csv_file}")

    # CSV: ROC points (per formula)
    if all_results.get('roc'):
        for formula_name, roc_data in all_results['roc'].items():
            if roc_data and 'roc_points' in roc_data:
                csv_file = output_dir / f'concordance_roc_{formula_name.lower()}.csv'
                pd.DataFrame(roc_data['roc_points']).to_csv(
                    csv_file, index=False)
        print(f"ROC curves saved to {output_dir}/concordance_roc_*.csv")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(base_path, output_path=None, roc_config=None):
    """Run complete diagnostic concordance analysis."""
    print("=" * 70)
    print("KEPLER-ECG: Diagnostic Concordance Analysis (Phase 09_3)")
    print("=" * 70)

    # Load data
    print("\nLoading datasets...")
    df = load_all_datasets(base_path)

    if df is None or len(df) == 0:
        print("ERROR: No data loaded")
        return None

    # Centralized QTc enrichment (calculate once, use everywhere)
    df = enrich_qtc(df)

    # Identify true prolonged QT
    print("\nIdentifying patients with prolonged QT diagnosis...")
    df = identify_true_prolonged_qt(df)

    n_true = df['has_qt_prolonged_dx'].sum()
    print(f"\n  Total with QT prolongation diagnosis: {n_true:,} "
          f"({100 * n_true / len(df):.2f}%)")

    if n_true == 0:
        print("\n  WARNING: No prolonged QT diagnoses found in any dataset.")
        print("  Check that diagnosis columns exist in the features files.")
        return None

    # Run all analyses
    all_results = {}

    all_results['overall'] = analyze_overall_concordance(df)
    all_results['by_dataset'] = analyze_by_dataset(df)
    all_results['by_hr'] = analyze_by_hr(df)
    all_results['by_age'] = analyze_by_age(df)
    all_results['triage_safety'] = analyze_triage_safety(df)
    all_results['by_severity'] = analyze_qtc_severity(df)
    all_results['roc'] = analyze_roc(df, roc_config)

    # Save results
    if output_path:
        save_results(all_results, output_path)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Kepler-ECG Phase 09_3: Diagnostic Concordance Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base path to Kepler-ECG project')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')

    # Stratification bins
    strat_group = parser.add_argument_group('Stratification parameters')
    strat_group.add_argument('--hr-bins', type=float, nargs='+',
                             default=HR_BINS,
                             help='HR bin edges for stratification')
    strat_group.add_argument('--age-bins', type=float, nargs='+',
                             default=AGE_BINS,
                             help='Age bin edges for stratification')

    # Minimum sample sizes
    sample_group = parser.add_argument_group('Minimum sample sizes')
    sample_group.add_argument('--min-samples-stratum', type=int,
                              default=MIN_SAMPLES_STRATUM,
                              help='Min true positives per stratum')
    sample_group.add_argument('--min-samples-severity', type=int,
                              default=MIN_SAMPLES_SEVERITY,
                              help='Min total records per severity bin')

    # ROC parameters
    roc_group = parser.add_argument_group('ROC analysis parameters')
    roc_group.add_argument('--roc-threshold-min', type=int,
                           default=DEFAULT_ROC_CONFIG['threshold_min'],
                           help='Min QTc threshold for ROC sweep (ms)')
    roc_group.add_argument('--roc-threshold-max', type=int,
                           default=DEFAULT_ROC_CONFIG['threshold_max'],
                           help='Max QTc threshold for ROC sweep (ms)')
    roc_group.add_argument('--roc-threshold-step', type=int,
                           default=DEFAULT_ROC_CONFIG['threshold_step'],
                           help='Step size for ROC sweep (ms)')

    args = parser.parse_args()

    # Override globals from CLI
    HR_BINS[:] = args.hr_bins
    AGE_BINS[:] = args.age_bins
    MIN_SAMPLES_STRATUM = args.min_samples_stratum
    MIN_SAMPLES_SEVERITY = args.min_samples_severity

    # Recompute labels from bins
    HR_LABELS.clear()
    for i in range(len(HR_BINS) - 1):
        if i == 0:
            HR_LABELS.append(f'<{HR_BINS[1]}')
        elif i == len(HR_BINS) - 2:
            HR_LABELS.append(f'>={HR_BINS[i]}')
        else:
            HR_LABELS.append(f'{HR_BINS[i]}-{HR_BINS[i+1]}')

    AGE_LABELS.clear()
    for i in range(len(AGE_BINS) - 1):
        if i == 0:
            AGE_LABELS.append(f'<{AGE_BINS[1]}')
        elif i == len(AGE_BINS) - 2:
            AGE_LABELS.append(f'>={AGE_BINS[i]}')
        else:
            AGE_LABELS.append(f'{AGE_BINS[i]}-{AGE_BINS[i+1]}')

    output = args.output or str(
        Path(args.base_path) / 'results' / 'clinical_analysis')

    roc_config = {
        'threshold_min': args.roc_threshold_min,
        'threshold_max': args.roc_threshold_max,
        'threshold_step': args.roc_threshold_step,
    }

    main(args.base_path, output, roc_config)
