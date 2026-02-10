#!/usr/bin/env python3
"""
09_2_hr_extremes_analysis.py

Kepler-ECG Pipeline - Phase 09_2: Heart Rate Extremes Analysis

This script performs deep analysis of QTc formula performance at heart rate
extremes, where clinical decisions are most critical and where Bazett's
systematic bias causes the most harm.

Analyses included:
1. Fine-grained HR stratification (8 zones)
2. Systematic bias quantification (Bazett vs Kepler)
3. Discordant case analysis (diagnosis changes)
4. Special populations (athletes proxy, febrile proxy, elderly bradycardic)
5. Triage by HR zone (3-level)
5b. Triage for special populations
6. Number Needed to Screen calculations
7. Adaptive thresholds exploration
8. Alternative correction approaches

Author: Alessandro Marconi
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION (defaults, overridable via CLI)
# =============================================================================

# Clinical thresholds (ms)
THRESHOLD_MALE = 450
THRESHOLD_FEMALE = 460
THRESHOLD_UNKNOWN = 455  # Midpoint when sex is unavailable
BORDERLINE_MARGIN = 10  # 440-450 for males, 450-460 for females

# Fine-grained HR bins
HR_FINE_BINS = [0, 50, 60, 75, 90, 100, 110, 120, 300]
HR_FINE_LABELS = [
    'Severe Bradycardia (<50)',
    'Mild Bradycardia (50-60)',
    'Normal-Low (60-75)',
    'Normal-High (75-90)',
    'Pre-Tachycardia (90-100)',
    'Mild Tachycardia (100-110)',
    'Moderate Tachycardia (110-120)',
    'Severe Tachycardia (>120)'
]

# Kepler formula coefficients
KEPLER_K = 125
KEPLER_C = -158

# Column mapping
COLUMN_MAPPING = {
    'QT': ['QT_interval_ms', 'QT_ms', 'QT', 'qt_interval_ms', 'qt'],
    'RR': ['RR_interval_sec', 'RR_s', 'RR_sec', 'RR', 'rr_interval_sec'],
    'HR': ['heart_rate_bpm', 'HR', 'heart_rate', 'hr_bpm', 'hr'],
    'age': ['age', 'Age', 'AGE'],
    'sex': ['sex', 'Sex', 'SEX', 'gender', 'Gender'],
    'superclass': ['primary_superclass', 'superclass', 'Superclass', 'diagnosis_superclass']
}

# Datasets
DATASETS = ['ptb-xl', 'chapman', 'cpsc-2018', 'georgia', 'mimic-iv-ecg', 'code-15']

# Sex value mappings (no overlap)
# Pipeline convention from 02_0_process_dataset.py: 0=male, 1=female
# String datasets (Chapman/Georgia/MIMIC) use 'Male'/'Female' or 'M'/'F'
# Note: pandas reads integer columns with NaN as float64 (0.0, 1.0)
#       and .astype(str) produces '0.0'/'1.0', so we must cover those too
SEX_MALE_VALUES = ['m', 'male', '0', '0.0']
SEX_FEMALE_VALUES = ['f', 'female', '1', '1.0']

# Special population defaults
DEFAULT_POP_CONFIG = {
    'athletes_max_hr': 55,
    'athletes_max_age': 45,
    'febrile_min_hr': 100,
    'elderly_min_age': 70,
    'elderly_max_hr': 60,
    'young_tachy_min_hr': 100,
    'young_tachy_max_age': 40,
    'extreme_brady_max_hr': 50,
    'extreme_tachy_min_hr': 120,
}

# Adaptive threshold defaults
DEFAULT_ADAPTIVE_CONFIG = {
    'reference_hr': 75,
    'target_fp_rate': 2.0,
    'threshold_adj_min': -20,
    'threshold_adj_max': 21,
    'threshold_adj_step': 2,
    'min_fp_rate': 0.5,
}

# Confidence scoring weights
DEFAULT_CONFIDENCE_CONFIG = {
    'hr_extreme_penalty': 20,       # HR < 50 or > 120
    'hr_borderline_penalty': 10,    # HR 50-55 or 110-120
    'disagree_high_penalty': 25,    # |Bazett - Kepler| > 40ms
    'disagree_low_penalty': 10,     # |Bazett - Kepler| > 20ms
    'near_threshold_high_penalty': 20,  # within 10ms of threshold
    'near_threshold_low_penalty': 10,   # within 20ms of threshold
    'high_confidence_cutoff': 70,
    'medium_confidence_cutoff': 40,
}

# Minimum sample sizes for analyses
MIN_SAMPLES_ZONE = 100
MIN_SAMPLES_SUB = 10
MIN_SAMPLES_SEX = 50


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

def calc_qtc_kepler(qt_ms, rr_s, k=KEPLER_K, c=KEPLER_C):
    """Kepler formula: QTc = QT + k/RR + c"""
    return qt_ms + k/rr_s + c


# =============================================================================
# CLASSIFICATION FUNCTIONS (vectorized)
# =============================================================================

def get_threshold(sex):
    """Get QTc threshold based on sex (scalar)."""
    if pd.isna(sex):
        return THRESHOLD_UNKNOWN
    if is_male(sex):
        return THRESHOLD_MALE
    elif is_female(sex):
        return THRESHOLD_FEMALE
    return THRESHOLD_UNKNOWN


def get_thresholds_vectorized(sex_series):
    """Vectorized threshold lookup for a pandas Series."""
    sex_lower = sex_series.astype(str).str.lower().str.strip()
    thresholds = pd.Series(THRESHOLD_UNKNOWN, index=sex_series.index, dtype=float)
    thresholds[sex_lower.isin(SEX_MALE_VALUES)] = THRESHOLD_MALE
    thresholds[sex_lower.isin(SEX_FEMALE_VALUES)] = THRESHOLD_FEMALE
    # Restore NaN → unknown
    thresholds[sex_series.isna()] = THRESHOLD_UNKNOWN
    return thresholds


def classify_qtc_binary_vec(qtc_series, sex_series):
    """Vectorized binary classification: 'normal' or 'prolonged'."""
    thresholds = get_thresholds_vectorized(sex_series)
    result = pd.Series('normal', index=qtc_series.index)
    result[qtc_series > thresholds] = 'prolonged'
    result[qtc_series.isna() | sex_series.isna()] = np.nan
    return result


def classify_qtc_detailed_vec(qtc_series, sex_series):
    """Vectorized detailed classification: 'normal', 'borderline', 'prolonged'."""
    thresholds = get_thresholds_vectorized(sex_series)
    borderline_lower = thresholds - BORDERLINE_MARGIN
    result = pd.Series('normal', index=qtc_series.index)
    result[qtc_series > borderline_lower] = 'borderline'
    result[qtc_series > thresholds] = 'prolonged'
    result[qtc_series.isna() | sex_series.isna()] = np.nan
    return result


def is_male_vec(sex_series):
    """Vectorized male check."""
    return sex_series.astype(str).str.lower().str.strip().isin(SEX_MALE_VALUES)


# =============================================================================
# DATA LOADING (same as Phase 2)
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
    """Load QTc preparation data."""
    base = Path(base_path)
    qtc_file = base / 'results' / dataset_name / 'qtc' / f'{dataset_name}_qtc_preparation.csv'

    if not qtc_file.exists():
        print(f"  Warning: {qtc_file} not found")
        return None

    df = pd.read_csv(qtc_file)
    df = standardize_columns(df)

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
    Calculate all QTc values and HR zones once, centrally.
    Returns a copy with added columns: QTc_Bazett, QTc_Kepler, QTc_Fridericia, hr_zone.
    """
    df = df.copy()
    df['QTc_Bazett'] = calc_qtc_bazett(df['QT'], df['RR'])
    df['QTc_Kepler'] = calc_qtc_kepler(df['QT'], df['RR'])
    df['QTc_Fridericia'] = calc_qtc_fridericia(df['QT'], df['RR'])
    df['hr_zone'] = pd.cut(df['HR'], bins=HR_FINE_BINS, labels=HR_FINE_LABELS, right=False)
    return df


# =============================================================================
# ANALYSIS 1: FINE-GRAINED HR STRATIFICATION
# =============================================================================

def analyze_fine_hr_stratification(df):
    """
    Analyze performance across 8 fine-grained HR zones.
    Expects pre-enriched DataFrame (QTc columns + hr_zone already present).
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: Fine-Grained HR Stratification")
    print("="*70)

    results = []

    for zone in HR_FINE_LABELS:
        zone_df = df[df['hr_zone'] == zone].copy()

        if len(zone_df) < MIN_SAMPLES_ZONE:
            continue

        result = {
            'zone': zone,
            'n': len(zone_df),
            'hr_mean': zone_df['HR'].mean(),
            'hr_std': zone_df['HR'].std()
        }

        # HR correlations
        valid = zone_df.dropna(subset=['HR', 'QTc_Bazett', 'QTc_Kepler', 'QTc_Fridericia'])
        if len(valid) > MIN_SAMPLES_SUB:
            result['r_bazett'] = stats.pearsonr(valid['HR'], valid['QTc_Bazett'])[0]
            result['r_kepler'] = stats.pearsonr(valid['HR'], valid['QTc_Kepler'])[0]
            result['r_fridericia'] = stats.pearsonr(valid['HR'], valid['QTc_Fridericia'])[0]
            result['abs_r_bazett'] = abs(result['r_bazett'])
            result['abs_r_kepler'] = abs(result['r_kepler'])
            result['abs_r_fridericia'] = abs(result['r_fridericia'])

            if result['abs_r_kepler'] > 0.001:
                result['improvement'] = result['abs_r_bazett'] / result['abs_r_kepler']
            else:
                result['improvement'] = float('inf')

        # False positive analysis (NORM only)
        if 'superclass' in zone_df.columns:
            norm_df = zone_df[zone_df['superclass'] == 'NORM'].copy()
            result['n_norm'] = len(norm_df)

            if len(norm_df) > MIN_SAMPLES_SUB and 'sex' in norm_df.columns:
                norm_df['fp_bazett'] = classify_qtc_binary_vec(
                    norm_df['QTc_Bazett'], norm_df['sex']) == 'prolonged'
                norm_df['fp_kepler'] = classify_qtc_binary_vec(
                    norm_df['QTc_Kepler'], norm_df['sex']) == 'prolonged'

                result['fp_bazett'] = int(norm_df['fp_bazett'].sum())
                result['fp_kepler'] = int(norm_df['fp_kepler'].sum())
                result['fp_rate_bazett'] = 100 * result['fp_bazett'] / len(norm_df)
                result['fp_rate_kepler'] = 100 * result['fp_kepler'] / len(norm_df)

                if result['fp_kepler'] > 0:
                    result['fp_reduction'] = result['fp_bazett'] / result['fp_kepler']
                else:
                    result['fp_reduction'] = float('inf')

        results.append(result)

        # Print summary
        imp = result.get('improvement', 0)
        imp_str = f"{imp:.1f}x" if imp < 100 else ">>100x"
        print(f"  {zone:30s}: n={result['n']:>7,}, |r|_B={result.get('abs_r_bazett', 0):.3f}, "
              f"|r|_K={result.get('abs_r_kepler', 0):.3f}, Imp={imp_str}")

    return results


# =============================================================================
# ANALYSIS 2: SYSTEMATIC BIAS QUANTIFICATION
# =============================================================================

def analyze_systematic_bias(df):
    """
    Quantify the systematic bias of Bazett vs Kepler across HR range.
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: Systematic Bias Quantification")
    print("="*70)

    # Calculate bias: positive = Bazett overcorrects
    bias = df['QTc_Bazett'] - df['QTc_Kepler']

    results = []

    for zone in HR_FINE_LABELS:
        zone_mask = df['hr_zone'] == zone
        zone_df = df[zone_mask]

        if len(zone_df) < MIN_SAMPLES_ZONE:
            continue

        bias_values = bias[zone_mask].dropna()

        result = {
            'zone': zone,
            'n': len(zone_df),
            'hr_mean': zone_df['HR'].mean(),
            'bias_mean': bias_values.mean(),
            'bias_std': bias_values.std(),
            'bias_median': bias_values.median(),
            'bias_p5': bias_values.quantile(0.05),
            'bias_p95': bias_values.quantile(0.95),
            'pct_bias_gt_20ms': 100 * (abs(bias_values) > 20).sum() / len(bias_values),
            'pct_bias_gt_40ms': 100 * (abs(bias_values) > 40).sum() / len(bias_values),
            'pct_overcorrection': 100 * (bias_values > 0).sum() / len(bias_values),
            'pct_undercorrection': 100 * (bias_values < 0).sum() / len(bias_values)
        }

        results.append(result)

        direction = "OVER" if result['bias_mean'] > 0 else "UNDER"
        print(f"  {zone:30s}: bias={result['bias_mean']:+6.1f}ms ({direction}), "
              f">20ms: {result['pct_bias_gt_20ms']:.1f}%, >40ms: {result['pct_bias_gt_40ms']:.1f}%")

    # Overall bias correlation with HR
    valid = df.dropna(subset=['HR', 'QTc_Bazett', 'QTc_Kepler'])
    bias_valid = valid['QTc_Bazett'] - valid['QTc_Kepler']
    r_bias_hr, p_bias_hr = stats.pearsonr(valid['HR'], bias_valid)
    print(f"\n  Bias-HR correlation: r={r_bias_hr:.4f} (p={p_bias_hr:.2e})")
    print(f"  Interpretation: {'Higher HR = Bazett overcorrects MORE' if r_bias_hr > 0 else 'Higher HR = Bazett undercorrects MORE'}")

    return results, {'r_bias_hr': r_bias_hr, 'p_bias_hr': p_bias_hr}


# =============================================================================
# ANALYSIS 3: DISCORDANT CASES
# =============================================================================

def analyze_discordant_cases(df):
    """
    Identify and analyze cases where Bazett and Kepler disagree on diagnosis.
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: Discordant Case Analysis")
    print("="*70)

    # Vectorized classification
    class_bazett = classify_qtc_detailed_vec(df['QTc_Bazett'], df['sex'])
    class_kepler = classify_qtc_detailed_vec(df['QTc_Kepler'], df['sex'])
    binary_bazett = classify_qtc_binary_vec(df['QTc_Bazett'], df['sex'])
    binary_kepler = classify_qtc_binary_vec(df['QTc_Kepler'], df['sex'])

    # Define discordance scenarios
    scenarios = {
        'A_FP_saved': (binary_bazett == 'prolonged') & (binary_kepler == 'normal'),
        'B_FN_risk': (binary_bazett == 'normal') & (binary_kepler == 'prolonged'),
        'C_borderline_resolved': (class_bazett == 'borderline') & (class_kepler == 'normal'),
        'D_downgraded': (class_bazett == 'prolonged') & (class_kepler == 'borderline')
    }

    results = {}

    for scenario_name, mask in scenarios.items():
        scenario_df = df[mask].copy()

        result = {
            'n': len(scenario_df),
            'pct_of_total': 100 * len(scenario_df) / len(df)
        }

        if len(scenario_df) > 0:
            result['hr_mean'] = scenario_df['HR'].mean()
            result['hr_std'] = scenario_df['HR'].std()
            result['hr_min'] = scenario_df['HR'].min()
            result['hr_max'] = scenario_df['HR'].max()
            result['age_mean'] = scenario_df['age'].mean() if 'age' in scenario_df.columns else None

            # Distribution by superclass
            if 'superclass' in scenario_df.columns:
                superclass_dist = scenario_df['superclass'].value_counts().to_dict()
                result['superclass_dist'] = superclass_dist
                result['pct_norm'] = 100 * superclass_dist.get('NORM', 0) / len(scenario_df)

            # HR zone distribution
            hr_zone_dist = scenario_df['hr_zone'].value_counts().to_dict()
            result['hr_zone_dist'] = {str(k): v for k, v in hr_zone_dist.items()}

        results[scenario_name] = result

        print(f"\n  Scenario {scenario_name}:")
        print(f"    N = {result['n']:,} ({result['pct_of_total']:.2f}% of total)")
        if result['n'] > 0:
            print(f"    Mean HR = {result.get('hr_mean', 0):.1f} bpm (range: {result.get('hr_min', 0):.0f}-{result.get('hr_max', 0):.0f})")
            print(f"    % NORM patients = {result.get('pct_norm', 0):.1f}%")

    # Summary interpretation
    print("\n  CLINICAL INTERPRETATION:")
    print(f"    - {results['A_FP_saved']['n']:,} patients SAVED from false positive (Bazett+, Kepler-)")
    print(f"    - {results['B_FN_risk']['n']:,} patients at RISK of false negative (Bazett-, Kepler+)")
    print(f"    - Ratio (saved/risk): {results['A_FP_saved']['n'] / max(results['B_FN_risk']['n'], 1):.1f}:1")

    return results


# =============================================================================
# ANALYSIS 4: SPECIAL POPULATIONS
# =============================================================================

def analyze_special_populations(df, pop_config=None):
    """
    Analyze specific high-risk populations.
    """
    print("\n" + "="*70)
    print("ANALYSIS 4: Special Populations Analysis")
    print("="*70)

    cfg = pop_config or DEFAULT_POP_CONFIG

    populations = {}

    # 4.1 Athletes proxy
    if 'age' in df.columns and 'superclass' in df.columns:
        athletes = df[
            (df['HR'] < cfg['athletes_max_hr']) &
            (df['age'] < cfg['athletes_max_age']) &
            (df['superclass'] == 'NORM')
        ]
        populations['athletes_proxy'] = {
            'criteria': f"HR < {cfg['athletes_max_hr']} bpm AND age < {cfg['athletes_max_age']} AND NORM",
            'description': 'Potential athletes with physiological bradycardia'
        }
        populations['athletes_proxy'].update(analyze_population(athletes, 'Athletes Proxy'))

    # 4.2 Febrile/stressed proxy
    if 'superclass' in df.columns:
        febrile = df[(df['HR'] > cfg['febrile_min_hr']) & (df['superclass'] == 'NORM')]
        populations['febrile_proxy'] = {
            'criteria': f"HR > {cfg['febrile_min_hr']} bpm AND NORM",
            'description': 'Tachycardic but otherwise healthy (fever, anxiety, etc.)'
        }
        populations['febrile_proxy'].update(analyze_population(febrile, 'Febrile/Stressed Proxy'))

    # 4.3 Elderly bradycardic
    if 'age' in df.columns:
        elderly_brady = df[
            (df['HR'] < cfg['elderly_max_hr']) &
            (df['age'] > cfg['elderly_min_age'])
        ]
        populations['elderly_bradycardic'] = {
            'criteria': f"HR < {cfg['elderly_max_hr']} bpm AND age > {cfg['elderly_min_age']}",
            'description': 'High-risk elderly with bradycardia'
        }
        populations['elderly_bradycardic'].update(analyze_population(elderly_brady, 'Elderly Bradycardic'))

    # 4.4 Young tachycardic
    if 'age' in df.columns:
        young_tachy = df[
            (df['HR'] > cfg['young_tachy_min_hr']) &
            (df['age'] < cfg['young_tachy_max_age'])
        ]
        populations['young_tachycardic'] = {
            'criteria': f"HR > {cfg['young_tachy_min_hr']} bpm AND age < {cfg['young_tachy_max_age']}",
            'description': 'Young patients with tachycardia (maximum Bazett bias)'
        }
        populations['young_tachycardic'].update(analyze_population(young_tachy, 'Young Tachycardic'))

    # 4.5 Extreme bradycardia
    extreme_brady = df[df['HR'] < cfg['extreme_brady_max_hr']]
    populations['extreme_bradycardia'] = {
        'criteria': f"HR < {cfg['extreme_brady_max_hr']} bpm",
        'description': 'Severe bradycardia (athletes, heart block, medications)'
    }
    populations['extreme_bradycardia'].update(analyze_population(extreme_brady, 'Extreme Bradycardia'))

    # 4.6 Extreme tachycardia
    extreme_tachy = df[df['HR'] > cfg['extreme_tachy_min_hr']]
    populations['extreme_tachycardia'] = {
        'criteria': f"HR > {cfg['extreme_tachy_min_hr']} bpm",
        'description': 'Severe tachycardia (emergency, arrhythmia)'
    }
    populations['extreme_tachycardia'].update(analyze_population(extreme_tachy, 'Extreme Tachycardia'))

    return populations


def analyze_population(pop_df, name):
    """Helper function to analyze a specific population."""
    result = {'n': len(pop_df)}

    if len(pop_df) < MIN_SAMPLES_SUB:
        print(f"\n  {name}: n={len(pop_df)} (insufficient data)")
        return result

    print(f"\n  {name}: n={len(pop_df):,}")

    # Basic stats
    result['hr_mean'] = pop_df['HR'].mean()
    result['hr_std'] = pop_df['HR'].std()

    if 'age' in pop_df.columns:
        result['age_mean'] = pop_df['age'].mean()

    # HR correlations
    valid = pop_df.dropna(subset=['HR', 'QTc_Bazett', 'QTc_Kepler'])
    if len(valid) > MIN_SAMPLES_SUB:
        result['r_bazett'] = abs(stats.pearsonr(valid['HR'], valid['QTc_Bazett'])[0])
        result['r_kepler'] = abs(stats.pearsonr(valid['HR'], valid['QTc_Kepler'])[0])

        if result['r_kepler'] > 0.001:
            result['improvement'] = result['r_bazett'] / result['r_kepler']
        else:
            result['improvement'] = float('inf')

        print(f"    |r| Bazett: {result['r_bazett']:.4f}, |r| Kepler: {result['r_kepler']:.4f}")
        imp = result['improvement']
        print(f"    Improvement: {imp:.1f}x" if imp < 100 else f"    Improvement: >>100x")

    # Bias analysis
    bias = pop_df['QTc_Bazett'] - pop_df['QTc_Kepler']
    result['bias_mean'] = bias.mean()
    result['bias_std'] = bias.std()

    direction = "overcorrects" if result['bias_mean'] > 0 else "undercorrects"
    print(f"    Bazett {direction} by {abs(result['bias_mean']):.1f} ms on average")

    # FP analysis if NORM patients
    if 'superclass' in pop_df.columns and 'sex' in pop_df.columns:
        norm_pop = pop_df[pop_df['superclass'] == 'NORM']
        if len(norm_pop) > MIN_SAMPLES_SUB:
            fp_bazett = classify_qtc_binary_vec(
                norm_pop['QTc_Bazett'], norm_pop['sex']) == 'prolonged'
            fp_kepler = classify_qtc_binary_vec(
                norm_pop['QTc_Kepler'], norm_pop['sex']) == 'prolonged'

            result['n_norm'] = len(norm_pop)
            result['fp_bazett'] = int(fp_bazett.sum())
            result['fp_kepler'] = int(fp_kepler.sum())
            result['fp_rate_bazett'] = 100 * result['fp_bazett'] / len(norm_pop)
            result['fp_rate_kepler'] = 100 * result['fp_kepler'] / len(norm_pop)

            print(f"    FP rate: Bazett {result['fp_rate_bazett']:.1f}% → Kepler {result['fp_rate_kepler']:.1f}%")

    return result


# =============================================================================
# ANALYSIS 5: TRIAGE SYSTEM BY HR ZONE
# =============================================================================

def assign_triage_vec(class_bazett, class_kepler):
    """
    Vectorized triage assignment.
    GREEN: Bazett normal → Discharge
    YELLOW: Bazett prolonged, Kepler normal → Follow-up
    RED: Both prolonged → Urgent referral
    """
    triage = pd.Series('RED', index=class_bazett.index)
    triage[class_bazett == 'normal'] = 'GREEN'
    triage[(class_bazett == 'prolonged') & (class_kepler == 'normal')] = 'YELLOW'
    return triage


def analyze_triage_by_hr(df):
    """
    Analyze 3-level triage distribution across HR zones.
    """
    print("\n" + "="*70)
    print("ANALYSIS 5: 3-Level Triage by HR Zone")
    print("="*70)

    if 'superclass' not in df.columns or 'sex' not in df.columns:
        print("  Cannot analyze triage without superclass and sex columns")
        return None, None

    norm_df = df[df['superclass'] == 'NORM'].copy()

    # Vectorized classification
    norm_df['class_bazett'] = classify_qtc_binary_vec(norm_df['QTc_Bazett'], norm_df['sex'])
    norm_df['class_kepler'] = classify_qtc_binary_vec(norm_df['QTc_Kepler'], norm_df['sex'])
    norm_df['triage'] = assign_triage_vec(norm_df['class_bazett'], norm_df['class_kepler'])

    results = []

    print("\n  Triage Distribution by HR Zone (NORM patients):")
    print("  " + "-"*90)
    print(f"  {'HR Zone':<30} {'N':>8} {'GREEN':>10} {'YELLOW':>10} {'RED':>10} {'Urg.Red.':>10}")
    print("  " + "-"*90)

    for zone in HR_FINE_LABELS:
        zone_df = norm_df[norm_df['hr_zone'] == zone]

        if len(zone_df) < MIN_SAMPLES_SUB:
            continue

        triage_counts = zone_df['triage'].value_counts()
        n_green = triage_counts.get('GREEN', 0)
        n_yellow = triage_counts.get('YELLOW', 0)
        n_red = triage_counts.get('RED', 0)
        n_total = len(zone_df)

        # Calculate urgent reduction: (Bazett positives - RED) / Bazett positives
        bazett_positive = n_yellow + n_red
        urgent_reduction = 100 * n_yellow / bazett_positive if bazett_positive > 0 else 0

        result = {
            'zone': zone,
            'n': n_total,
            'green': n_green,
            'yellow': n_yellow,
            'red': n_red,
            'green_pct': 100 * n_green / n_total,
            'yellow_pct': 100 * n_yellow / n_total,
            'red_pct': 100 * n_red / n_total,
            'urgent_reduction_pct': urgent_reduction,
            'bazett_positive': bazett_positive
        }
        results.append(result)

        print(f"  {zone:<30} {n_total:>8,} {result['green_pct']:>9.1f}% {result['yellow_pct']:>9.1f}% "
              f"{result['red_pct']:>9.1f}% {urgent_reduction:>9.1f}%")

    # Overall
    triage_overall = norm_df['triage'].value_counts()
    n_total = len(norm_df)
    overall = {
        'zone': 'OVERALL',
        'n': n_total,
        'green': triage_overall.get('GREEN', 0),
        'yellow': triage_overall.get('YELLOW', 0),
        'red': triage_overall.get('RED', 0),
        'green_pct': 100 * triage_overall.get('GREEN', 0) / n_total,
        'yellow_pct': 100 * triage_overall.get('YELLOW', 0) / n_total,
        'red_pct': 100 * triage_overall.get('RED', 0) / n_total
    }
    bazett_pos_overall = overall['yellow'] + overall['red']
    overall['urgent_reduction_pct'] = 100 * overall['yellow'] / bazett_pos_overall if bazett_pos_overall > 0 else 0

    print("  " + "-"*90)
    print(f"  {'OVERALL':<30} {n_total:>8,} {overall['green_pct']:>9.1f}% {overall['yellow_pct']:>9.1f}% "
          f"{overall['red_pct']:>9.1f}% {overall['urgent_reduction_pct']:>9.1f}%")

    return results, overall


def analyze_triage_special_populations(df, pop_config=None):
    """
    Analyze triage distribution for special populations.
    """
    print("\n" + "="*70)
    print("ANALYSIS 5b: Triage for Special Populations")
    print("="*70)

    cfg = pop_config or DEFAULT_POP_CONFIG

    if 'superclass' not in df.columns or 'sex' not in df.columns:
        print("  Cannot analyze triage without superclass and sex columns")
        return None

    # Define populations
    populations = {}

    if 'age' in df.columns:
        populations['athletes_proxy'] = {
            'filter': (df['HR'] < cfg['athletes_max_hr']) & (df['age'] < cfg['athletes_max_age']) & (df['superclass'] == 'NORM'),
            'description': f"Athletes (HR<{cfg['athletes_max_hr']}, Age<{cfg['athletes_max_age']}, NORM)"
        }
        populations['young_tachycardic'] = {
            'filter': (df['HR'] > cfg['young_tachy_min_hr']) & (df['age'] < cfg['young_tachy_max_age']),
            'description': f"Young Tachycardic (HR>{cfg['young_tachy_min_hr']}, Age<{cfg['young_tachy_max_age']})"
        }
        populations['elderly_bradycardic'] = {
            'filter': (df['HR'] < cfg['elderly_max_hr']) & (df['age'] > cfg['elderly_min_age']),
            'description': f"Elderly Bradycardic (HR<{cfg['elderly_max_hr']}, Age>{cfg['elderly_min_age']})"
        }

    populations['febrile_proxy'] = {
        'filter': (df['HR'] > cfg['febrile_min_hr']) & (df['superclass'] == 'NORM'),
        'description': f"Febrile/Stressed (HR>{cfg['febrile_min_hr']}, NORM)"
    }
    populations['extreme_bradycardia'] = {
        'filter': df['HR'] < cfg['extreme_brady_max_hr'],
        'description': f"Extreme Bradycardia (HR<{cfg['extreme_brady_max_hr']})"
    }
    populations['extreme_tachycardia'] = {
        'filter': df['HR'] > cfg['extreme_tachy_min_hr'],
        'description': f"Extreme Tachycardia (HR>{cfg['extreme_tachy_min_hr']})"
    }

    results = {}

    for pop_name, pop_info in populations.items():
        pop_df = df[pop_info['filter']].copy()

        if len(pop_df) < MIN_SAMPLES_SUB:
            continue

        # Vectorized classification and triage
        pop_df['class_bazett'] = classify_qtc_binary_vec(pop_df['QTc_Bazett'], pop_df['sex'])
        pop_df['class_kepler'] = classify_qtc_binary_vec(pop_df['QTc_Kepler'], pop_df['sex'])
        pop_df['triage'] = assign_triage_vec(pop_df['class_bazett'], pop_df['class_kepler'])

        triage_counts = pop_df['triage'].value_counts()
        n_total = len(pop_df)

        result = {
            'description': pop_info['description'],
            'n': n_total,
            'green': triage_counts.get('GREEN', 0),
            'yellow': triage_counts.get('YELLOW', 0),
            'red': triage_counts.get('RED', 0),
            'green_pct': 100 * triage_counts.get('GREEN', 0) / n_total,
            'yellow_pct': 100 * triage_counts.get('YELLOW', 0) / n_total,
            'red_pct': 100 * triage_counts.get('RED', 0) / n_total
        }

        bazett_pos = result['yellow'] + result['red']
        result['urgent_reduction_pct'] = 100 * result['yellow'] / bazett_pos if bazett_pos > 0 else 0

        results[pop_name] = result

        print(f"\n  {pop_info['description']}:")
        print(f"    N = {n_total:,}")
        print(f"    GREEN: {result['green_pct']:.1f}% | YELLOW: {result['yellow_pct']:.1f}% | RED: {result['red_pct']:.1f}%")
        print(f"    Urgent Reduction: {result['urgent_reduction_pct']:.1f}%")

    return results


# =============================================================================
# ANALYSIS 6: NUMBER NEEDED TO SCREEN (NNS)
# =============================================================================

def calculate_nns(df):
    """
    Calculate Number Needed to Screen to avoid one false positive.
    """
    print("\n" + "="*70)
    print("ANALYSIS 6: Number Needed to Screen (NNS)")
    print("="*70)

    # Only NORM patients
    if 'superclass' not in df.columns:
        print("  Cannot calculate NNS without superclass column")
        return None, None

    norm_df = df[df['superclass'] == 'NORM'].copy()

    if 'sex' not in norm_df.columns:
        print("  Cannot calculate NNS without sex column")
        return None, None

    # Vectorized FP classification
    norm_df['fp_bazett'] = classify_qtc_binary_vec(
        norm_df['QTc_Bazett'], norm_df['sex']) == 'prolonged'
    norm_df['fp_kepler'] = classify_qtc_binary_vec(
        norm_df['QTc_Kepler'], norm_df['sex']) == 'prolonged'

    results = []

    print("\n  Per-Zone NNS Analysis:")
    print("  " + "-"*80)
    print(f"  {'HR Zone':<35} {'N':>8} {'FP_B':>7} {'FP_K':>7} {'Avoided':>8} {'NNS':>10}")
    print("  " + "-"*80)

    for zone in HR_FINE_LABELS:
        zone_df = norm_df[norm_df['hr_zone'] == zone]

        if len(zone_df) < MIN_SAMPLES_SUB:
            continue

        fp_b = int(zone_df['fp_bazett'].sum())
        fp_k = int(zone_df['fp_kepler'].sum())
        avoided = fp_b - fp_k

        # NNS = Number of patients to screen with Kepler instead of Bazett to avoid 1 FP
        nns = len(zone_df) / avoided if avoided > 0 else float('inf')

        result = {
            'zone': zone,
            'n': len(zone_df),
            'fp_bazett': fp_b,
            'fp_kepler': fp_k,
            'fp_avoided': avoided,
            'nns': nns,
            'fp_rate_bazett': 100 * fp_b / len(zone_df),
            'fp_rate_kepler': 100 * fp_k / len(zone_df)
        }
        results.append(result)

        nns_str = f"{nns:.1f}" if nns < 1000 else "N/A"
        print(f"  {zone:<35} {len(zone_df):>8,} {fp_b:>7,} {fp_k:>7,} {avoided:>8,} {nns_str:>10}")

    # Overall
    fp_b_total = int(norm_df['fp_bazett'].sum())
    fp_k_total = int(norm_df['fp_kepler'].sum())
    avoided_total = fp_b_total - fp_k_total
    nns_overall = len(norm_df) / avoided_total if avoided_total > 0 else float('inf')

    print("  " + "-"*80)
    print(f"  {'OVERALL':<35} {len(norm_df):>8,} {fp_b_total:>7,} {fp_k_total:>7,} {avoided_total:>8,} {nns_overall:>10.1f}")

    print(f"\n  INTERPRETATION:")
    print(f"    Using Kepler instead of Bazett, we avoid 1 false positive for every {nns_overall:.1f} patients screened.")
    print(f"    Total false positives avoided: {avoided_total:,} out of {len(norm_df):,} NORM patients.")

    return results, {
        'overall_nns': nns_overall,
        'total_fp_avoided': avoided_total,
        'total_norm': len(norm_df)
    }


# =============================================================================
# ANALYSIS 7: ADAPTIVE THRESHOLDS EXPLORATION
# =============================================================================

def explore_adaptive_thresholds(df, adaptive_config=None):
    """
    Explore HR-adaptive thresholds for QTc classification.

    Approaches tested:
    1. Linear HR-adjusted threshold: threshold = base + slope * (HR - 75)
    2. Zone-specific fixed thresholds
    3. Percentile-based thresholds (e.g., 95th percentile of NORM)
    4. Combined formula+threshold optimization
    """
    print("\n" + "="*70)
    print("ANALYSIS 7: Adaptive Thresholds Exploration")
    print("="*70)

    cfg = adaptive_config or DEFAULT_ADAPTIVE_CONFIG

    if 'superclass' not in df.columns or 'sex' not in df.columns:
        print("  Cannot explore thresholds without superclass and sex columns")
        return None

    norm_df = df[df['superclass'] == 'NORM'].copy()

    # Pre-compute vectorized thresholds and sex masks
    norm_thresholds = get_thresholds_vectorized(norm_df['sex'])
    norm_is_male = is_male_vec(norm_df['sex'])

    results = {
        'approach_1_linear': {},
        'approach_2_zone_specific': {},
        'approach_3_percentile': {},
        'approach_4_optimal_kepler_threshold': {}
    }

    # =========================================================================
    # Approach 1: Linear HR-adjusted threshold for Bazett
    # threshold = base + slope * (HR - reference_HR)
    # =========================================================================
    print("\n  Approach 1: Linear HR-Adjusted Threshold for Bazett")
    print("  " + "-"*60)

    reference_hr = cfg['reference_hr']
    best_slope = 0
    best_fp_rate = float('inf')
    hr_offset = norm_df['HR'].values - reference_hr

    for slope in np.arange(-0.5, 0.5, 0.05):
        adaptive_thresh = norm_thresholds.values + slope * hr_offset
        fp = (norm_df['QTc_Bazett'].values > adaptive_thresh).sum()
        fp_rate = 100 * fp / len(norm_df)

        if fp_rate < best_fp_rate:
            best_fp_rate = fp_rate
            best_slope = slope

    # Fixed threshold FP rate
    fp_fixed = (norm_df['QTc_Bazett'].values > norm_thresholds.values).sum()
    fp_rate_fixed = 100 * fp_fixed / len(norm_df)

    results['approach_1_linear'] = {
        'optimal_slope': best_slope,
        'reference_hr': reference_hr,
        'fp_rate_adaptive': best_fp_rate,
        'fp_rate_fixed': fp_rate_fixed,
        'formula': f"threshold = base + {best_slope:.2f} × (HR - {reference_hr})"
    }

    print(f"    Optimal slope: {best_slope:.2f} ms/bpm")
    print(f"    Formula: threshold = base + {best_slope:.2f} × (HR - {reference_hr})")
    print(f"    FP rate: {fp_rate_fixed:.2f}% → {best_fp_rate:.2f}%")

    # =========================================================================
    # Approach 2: Zone-specific fixed thresholds
    # =========================================================================
    print("\n  Approach 2: Zone-Specific Fixed Thresholds")
    print("  " + "-"*60)

    target_fp_rate = cfg['target_fp_rate']
    zone_thresholds = {}

    for zone in HR_FINE_LABELS:
        zone_df = norm_df[norm_df['hr_zone'] == zone]
        if len(zone_df) < MIN_SAMPLES_ZONE:
            continue

        # Calculate male/female separately
        for sex_label, sex_mask_fn in [('male', lambda s: is_male_vec(s)), ('female', lambda s: ~is_male_vec(s) & s.notna())]:
            sex_mask = sex_mask_fn(zone_df['sex'])
            sex_df = zone_df[sex_mask]
            if len(sex_df) < MIN_SAMPLES_SEX:
                continue

            qtc_values = sex_df['QTc_Bazett'].dropna().sort_values()
            optimal_threshold = qtc_values.quantile((100 - target_fp_rate) / 100)

            zone_thresholds[f"{zone}_{sex_label}"] = {
                'threshold': optimal_threshold,
                'n': len(sex_df),
                'achieved_fp_rate': 100 * (sex_df['QTc_Bazett'] > optimal_threshold).sum() / len(sex_df)
            }

    results['approach_2_zone_specific'] = zone_thresholds

    print(f"    Zone-specific thresholds (targeting ~{target_fp_rate}% FP rate):")
    for zone in HR_FINE_LABELS[:4]:
        male_key = f"{zone}_male"
        female_key = f"{zone}_female"
        if male_key in zone_thresholds:
            female_val = zone_thresholds.get(female_key, {}).get('threshold', None)
            female_str = f"{female_val:.0f}ms" if female_val is not None else "N/A"
            print(f"      {zone}: Male={zone_thresholds[male_key]['threshold']:.0f}ms, Female={female_str}")

    # =========================================================================
    # Approach 3: Percentile-based thresholds from NORM distribution
    # =========================================================================
    print("\n  Approach 3: Percentile-Based Thresholds (95th, 97.5th, 99th)")
    print("  " + "-"*60)

    percentile_results = {}

    male_mask = norm_is_male
    female_mask = ~male_mask & norm_df['sex'].notna()

    for formula_name, qtc_col in [('Bazett', 'QTc_Bazett'), ('Kepler', 'QTc_Kepler')]:
        percentile_results[formula_name] = {}

        for pct in [95, 97.5, 99]:
            male_threshold = norm_df.loc[male_mask, qtc_col].quantile(pct / 100)
            female_threshold = norm_df.loc[female_mask, qtc_col].quantile(pct / 100)

            percentile_results[formula_name][f"p{pct}"] = {
                'male_threshold': male_threshold,
                'female_threshold': female_threshold,
                'expected_fp_rate': 100 - pct
            }

    results['approach_3_percentile'] = percentile_results

    print(f"    Bazett 95th percentile: Male={percentile_results['Bazett']['p95']['male_threshold']:.0f}ms, "
          f"Female={percentile_results['Bazett']['p95']['female_threshold']:.0f}ms")
    print(f"    Kepler 95th percentile: Male={percentile_results['Kepler']['p95']['male_threshold']:.0f}ms, "
          f"Female={percentile_results['Kepler']['p95']['female_threshold']:.0f}ms")

    # =========================================================================
    # Approach 4: Find optimal threshold for Kepler (vectorized)
    # =========================================================================
    print("\n  Approach 4: Optimal Threshold for Kepler Formula")
    print("  " + "-"*60)

    kepler_values = norm_df['QTc_Kepler'].values
    threshold_values = norm_thresholds.values

    # Current performance
    current_fp = (kepler_values > threshold_values).sum()
    current_fp_rate = 100 * current_fp / len(norm_df)

    # Test different threshold adjustments (vectorized)
    best_adjustment = 0
    best_balanced_score = 0
    min_fp_rate = cfg['min_fp_rate']

    for adj in range(cfg['threshold_adj_min'], cfg['threshold_adj_max'], cfg['threshold_adj_step']):
        fp = (kepler_values > (threshold_values + adj)).sum()
        fp_rate = 100 * fp / len(norm_df)
        score = (current_fp_rate - fp_rate) - abs(adj) * 0.1

        if score > best_balanced_score and fp_rate > min_fp_rate:
            best_balanced_score = score
            best_adjustment = adj

    # Final metrics with best adjustment
    final_fp = (kepler_values > (threshold_values + best_adjustment)).sum()
    optimal_fp_rate = 100 * final_fp / len(norm_df)

    results['approach_4_optimal_kepler_threshold'] = {
        'current_thresholds': {'male': THRESHOLD_MALE, 'female': THRESHOLD_FEMALE},
        'optimal_adjustment': best_adjustment,
        'optimal_thresholds': {'male': THRESHOLD_MALE + best_adjustment, 'female': THRESHOLD_FEMALE + best_adjustment},
        'current_fp_rate': current_fp_rate,
        'optimal_fp_rate': optimal_fp_rate,
        'recommendation': f"Consider threshold = {THRESHOLD_MALE + best_adjustment}/{THRESHOLD_FEMALE + best_adjustment} ms for Kepler"
    }

    print(f"    Current thresholds: {THRESHOLD_MALE}/{THRESHOLD_FEMALE} ms → FP rate: {current_fp_rate:.2f}%")
    print(f"    Optimal adjustment: {best_adjustment:+d} ms")
    print(f"    Suggested thresholds: {THRESHOLD_MALE + best_adjustment}/{THRESHOLD_FEMALE + best_adjustment} ms → FP rate: {optimal_fp_rate:.2f}%")

    # =========================================================================
    # Summary comparison
    # =========================================================================
    print("\n  SUMMARY: Approaches Comparison")
    print("  " + "-"*60)
    print(f"    Baseline Bazett (fixed threshold):     FP rate = {fp_rate_fixed:.2f}%")
    print(f"    Bazett + Linear adaptive threshold:    FP rate = {best_fp_rate:.2f}%")
    print(f"    Kepler (fixed threshold):              FP rate = {current_fp_rate:.2f}%")
    print(f"    Kepler (optimized threshold):          FP rate = {optimal_fp_rate:.2f}%")

    return results


# =============================================================================
# ANALYSIS 8: ALTERNATIVE CORRECTION APPROACHES
# =============================================================================

def explore_alternative_approaches(df, pop_config=None, confidence_config=None):
    """
    Explore alternative QTc correction approaches beyond formula changes.

    Approaches:
    1. Hybrid formula: Kepler in normal range, Bazett at extremes
    2. Confidence scoring: flag uncertain cases
    3. Enhanced 4-level triage with confidence
    """
    print("\n" + "="*70)
    print("ANALYSIS 8: Alternative Correction Approaches")
    print("="*70)

    pcfg = pop_config or DEFAULT_POP_CONFIG
    ccfg = confidence_config or DEFAULT_CONFIDENCE_CONFIG

    if 'superclass' not in df.columns or 'sex' not in df.columns:
        print("  Cannot explore alternatives without superclass and sex columns")
        return None

    norm_df = df[df['superclass'] == 'NORM'].copy()

    # Pre-compute vectorized thresholds
    norm_thresholds = get_thresholds_vectorized(norm_df['sex']).values
    kepler_values = norm_df['QTc_Kepler'].values
    bazett_values = norm_df['QTc_Bazett'].values
    hr_values = norm_df['HR'].values

    results = {}

    # =========================================================================
    # Approach 1: Hybrid formula based on HR
    # =========================================================================
    print("\n  Approach 1: Hybrid Formula (HR-dependent selection)")
    print("  " + "-"*60)

    extreme_mask = (hr_values < pcfg['extreme_brady_max_hr']) | (hr_values > pcfg['extreme_tachy_min_hr'])
    hybrid_qtc = np.where(extreme_mask, bazett_values, kepler_values)

    fp_hybrid = (hybrid_qtc > norm_thresholds).sum()
    fp_kepler = (kepler_values > norm_thresholds).sum()

    n_extreme = int(extreme_mask.sum())
    results['hybrid_formula'] = {
        'description': f"Bazett if HR<{pcfg['extreme_brady_max_hr']} or HR>{pcfg['extreme_tachy_min_hr']}, else Kepler",
        'fp_rate_hybrid': 100 * fp_hybrid / len(norm_df),
        'fp_rate_kepler': 100 * fp_kepler / len(norm_df),
        'n_extreme_hr': n_extreme,
        'pct_extreme_hr': 100 * n_extreme / len(norm_df)
    }

    print(f"    Hybrid FP rate: {results['hybrid_formula']['fp_rate_hybrid']:.2f}%")
    print(f"    Pure Kepler FP rate: {results['hybrid_formula']['fp_rate_kepler']:.2f}%")
    print(f"    Patients using Bazett (extreme HR): {results['hybrid_formula']['pct_extreme_hr']:.1f}%")

    # =========================================================================
    # Approach 2: Confidence scoring (vectorized)
    # =========================================================================
    print("\n  Approach 2: Confidence Scoring System")
    print("  " + "-"*60)

    score = np.full(len(norm_df), 100.0)

    # HR penalty
    score[extreme_mask] -= ccfg['hr_extreme_penalty']
    borderline_hr = ((hr_values >= pcfg['extreme_brady_max_hr']) & (hr_values < 55)) | \
                    ((hr_values > 110) & (hr_values <= pcfg['extreme_tachy_min_hr']))
    score[borderline_hr] -= ccfg['hr_borderline_penalty']

    # Disagreement penalty
    disagreement = np.abs(bazett_values - kepler_values)
    score[disagreement > 40] -= ccfg['disagree_high_penalty']
    score[(disagreement > 20) & (disagreement <= 40)] -= ccfg['disagree_low_penalty']

    # Near-threshold penalty
    distance_from_threshold = np.abs(kepler_values - norm_thresholds)
    score[distance_from_threshold < 10] -= ccfg['near_threshold_high_penalty']
    score[(distance_from_threshold >= 10) & (distance_from_threshold < 20)] -= ccfg['near_threshold_low_penalty']

    score = np.maximum(score, 0)
    norm_df['confidence'] = score

    high_cutoff = ccfg['high_confidence_cutoff']
    med_cutoff = ccfg['medium_confidence_cutoff']

    confidence_dist = {
        'high_confidence_pct': 100 * (score >= high_cutoff).sum() / len(norm_df),
        'medium_confidence_pct': 100 * ((score >= med_cutoff) & (score < high_cutoff)).sum() / len(norm_df),
        'low_confidence_pct': 100 * (score < med_cutoff).sum() / len(norm_df),
        'mean_confidence': float(score.mean())
    }

    results['confidence_scoring'] = confidence_dist

    print(f"    High confidence (≥{high_cutoff}): {confidence_dist['high_confidence_pct']:.1f}%")
    print(f"    Medium confidence ({med_cutoff}-{high_cutoff}): {confidence_dist['medium_confidence_pct']:.1f}%")
    print(f"    Low confidence (<{med_cutoff}): {confidence_dist['low_confidence_pct']:.1f}%")
    print(f"    Mean confidence: {confidence_dist['mean_confidence']:.1f}")

    # =========================================================================
    # Approach 3: 4-Level Triage with Confidence (vectorized)
    # =========================================================================
    print("\n  Approach 3: Enhanced 4-Level Triage")
    print("  " + "-"*60)

    bazett_pos = bazett_values > norm_thresholds
    kepler_pos = kepler_values > norm_thresholds

    enhanced = np.full(len(norm_df), 'RED', dtype=object)
    # Both negative, high confidence
    both_neg = ~bazett_pos & ~kepler_pos
    enhanced[both_neg & (score >= high_cutoff)] = 'GREEN'
    enhanced[both_neg & (score < high_cutoff)] = 'GREEN-WATCH'
    # Discordant: Bazett positive, Kepler negative
    enhanced[bazett_pos & ~kepler_pos] = 'YELLOW'
    # Both positive stays RED

    norm_df['enhanced_triage'] = enhanced

    enhanced_counts = pd.Series(enhanced).value_counts()
    n = len(norm_df)
    enhanced_results = {
        'green': int(enhanced_counts.get('GREEN', 0)),
        'green_watch': int(enhanced_counts.get('GREEN-WATCH', 0)),
        'yellow': int(enhanced_counts.get('YELLOW', 0)),
        'red': int(enhanced_counts.get('RED', 0)),
        'green_pct': 100 * enhanced_counts.get('GREEN', 0) / n,
        'green_watch_pct': 100 * enhanced_counts.get('GREEN-WATCH', 0) / n,
        'yellow_pct': 100 * enhanced_counts.get('YELLOW', 0) / n,
        'red_pct': 100 * enhanced_counts.get('RED', 0) / n
    }

    results['enhanced_triage'] = enhanced_results

    print(f"    GREEN (confident negative): {enhanced_results['green_pct']:.1f}%")
    print(f"    GREEN-WATCH (uncertain negative): {enhanced_results['green_watch_pct']:.1f}%")
    print(f"    YELLOW (discordant): {enhanced_results['yellow_pct']:.1f}%")
    print(f"    RED (both positive): {enhanced_results['red_pct']:.1f}%")

    return results


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_results(all_results, output_dir):
    """Save all results to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full JSON
    json_file = output_dir / 'hr_extremes_analysis.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_file}")

    # Save CSV: fine HR stratification
    if all_results.get('fine_hr_stratification'):
        csv_file = output_dir / 'hr_extremes_fine_stratification.csv'
        pd.DataFrame(all_results['fine_hr_stratification']).to_csv(csv_file, index=False)
        print(f"Fine stratification saved to {csv_file}")

    # Save CSV: bias analysis
    if all_results.get('systematic_bias'):
        csv_file = output_dir / 'hr_extremes_bias_analysis.csv'
        pd.DataFrame(all_results['systematic_bias']).to_csv(csv_file, index=False)
        print(f"Bias analysis saved to {csv_file}")

    # Save CSV: triage by HR
    if all_results.get('triage_by_hr'):
        csv_file = output_dir / 'hr_extremes_triage_by_hr.csv'
        pd.DataFrame(all_results['triage_by_hr']).to_csv(csv_file, index=False)
        print(f"Triage by HR saved to {csv_file}")

    # Save CSV: discordant cases
    if all_results.get('discordant_cases'):
        csv_file = output_dir / 'hr_extremes_discordant_cases.csv'
        rows = []
        for scenario, data in all_results['discordant_cases'].items():
            row = {'scenario': scenario}
            for k, v in data.items():
                if not isinstance(v, dict):
                    row[k] = v
            rows.append(row)
        pd.DataFrame(rows).to_csv(csv_file, index=False)
        print(f"Discordant cases saved to {csv_file}")

    # Save CSV: special populations
    if all_results.get('special_populations'):
        csv_file = output_dir / 'hr_extremes_special_populations.csv'
        rows = []
        for pop_name, data in all_results['special_populations'].items():
            row = {'population': pop_name}
            for k, v in data.items():
                if not isinstance(v, dict):
                    row[k] = v
            rows.append(row)
        pd.DataFrame(rows).to_csv(csv_file, index=False)
        print(f"Special populations saved to {csv_file}")

    # Save CSV: NNS by zone
    if all_results.get('nns_by_zone'):
        csv_file = output_dir / 'hr_extremes_nns_by_zone.csv'
        pd.DataFrame(all_results['nns_by_zone']).to_csv(csv_file, index=False)
        print(f"NNS by zone saved to {csv_file}")

    # Save CSV: adaptive thresholds (approach 4 summary)
    if all_results.get('adaptive_thresholds'):
        csv_file = output_dir / 'hr_extremes_adaptive_thresholds.csv'
        at = all_results['adaptive_thresholds']
        summary_rows = []
        for approach_key, approach_data in at.items():
            if isinstance(approach_data, dict) and not any(isinstance(v, dict) for v in approach_data.values()):
                row = {'approach': approach_key}
                row.update(approach_data)
                summary_rows.append(row)
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(csv_file, index=False)
            print(f"Adaptive thresholds saved to {csv_file}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(base_path, output_path=None, pop_config=None, adaptive_config=None, confidence_config=None):
    """Run complete HR extremes analysis."""
    print("="*70)
    print("KEPLER-ECG: Heart Rate Extremes Analysis (Phase 09_2)")
    print("="*70)

    # Load data
    print("\nLoading datasets...")
    df = load_all_datasets(base_path)

    if df is None or len(df) == 0:
        print("ERROR: No data loaded")
        return None

    # Centralized QTc enrichment (calculate once, use everywhere)
    print("\nEnriching dataset with QTc values...")
    df = enrich_qtc(df)
    print(f"  Columns added: QTc_Bazett, QTc_Kepler, QTc_Fridericia, hr_zone")

    # Run all analyses
    all_results = {}

    # Analysis 1: Fine-grained HR stratification
    all_results['fine_hr_stratification'] = analyze_fine_hr_stratification(df)

    # Analysis 2: Systematic bias
    bias_results, bias_summary = analyze_systematic_bias(df)
    all_results['systematic_bias'] = bias_results
    all_results['bias_summary'] = bias_summary

    # Analysis 3: Discordant cases
    all_results['discordant_cases'] = analyze_discordant_cases(df)

    # Analysis 4: Special populations
    all_results['special_populations'] = analyze_special_populations(df, pop_config)

    # Analysis 5: Triage by HR zone
    triage_results, triage_overall = analyze_triage_by_hr(df)
    all_results['triage_by_hr'] = triage_results
    all_results['triage_overall'] = triage_overall

    # Analysis 5b: Triage for special populations
    all_results['triage_special_populations'] = analyze_triage_special_populations(df, pop_config)

    # Analysis 6: NNS
    nns_results, nns_summary = calculate_nns(df)
    all_results['nns_by_zone'] = nns_results
    all_results['nns_summary'] = nns_summary

    # Analysis 7: Adaptive thresholds
    all_results['adaptive_thresholds'] = explore_adaptive_thresholds(df, adaptive_config)

    # Analysis 8: Alternative approaches
    all_results['alternative_approaches'] = explore_alternative_approaches(df, pop_config, confidence_config)

    # Save results
    if output_path:
        save_results(all_results, output_path)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Kepler-ECG Phase 09_2: Heart Rate Extremes Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base path to Kepler-ECG project')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')

    # Special population thresholds
    pop_group = parser.add_argument_group('Special population thresholds')
    pop_group.add_argument('--athletes-max-hr', type=int, default=DEFAULT_POP_CONFIG['athletes_max_hr'],
                           help='Max HR for athletes proxy')
    pop_group.add_argument('--athletes-max-age', type=int, default=DEFAULT_POP_CONFIG['athletes_max_age'],
                           help='Max age for athletes proxy')
    pop_group.add_argument('--febrile-min-hr', type=int, default=DEFAULT_POP_CONFIG['febrile_min_hr'],
                           help='Min HR for febrile proxy')
    pop_group.add_argument('--elderly-min-age', type=int, default=DEFAULT_POP_CONFIG['elderly_min_age'],
                           help='Min age for elderly bradycardic')
    pop_group.add_argument('--elderly-max-hr', type=int, default=DEFAULT_POP_CONFIG['elderly_max_hr'],
                           help='Max HR for elderly bradycardic')
    pop_group.add_argument('--extreme-brady-max-hr', type=int, default=DEFAULT_POP_CONFIG['extreme_brady_max_hr'],
                           help='Max HR for extreme bradycardia')
    pop_group.add_argument('--extreme-tachy-min-hr', type=int, default=DEFAULT_POP_CONFIG['extreme_tachy_min_hr'],
                           help='Min HR for extreme tachycardia')

    # Adaptive threshold parameters
    adapt_group = parser.add_argument_group('Adaptive threshold parameters')
    adapt_group.add_argument('--reference-hr', type=float, default=DEFAULT_ADAPTIVE_CONFIG['reference_hr'],
                             help='Reference HR for linear adaptive threshold')
    adapt_group.add_argument('--target-fp-rate', type=float, default=DEFAULT_ADAPTIVE_CONFIG['target_fp_rate'],
                             help='Target FP rate for zone-specific thresholds')

    args = parser.parse_args()

    output = args.output or str(Path(args.base_path) / 'results' / 'clinical_analysis')

    # Build config dicts from CLI args
    pop_config = {
        'athletes_max_hr': args.athletes_max_hr,
        'athletes_max_age': args.athletes_max_age,
        'febrile_min_hr': args.febrile_min_hr,
        'elderly_min_age': args.elderly_min_age,
        'elderly_max_hr': args.elderly_max_hr,
        'young_tachy_min_hr': args.febrile_min_hr,
        'young_tachy_max_age': DEFAULT_POP_CONFIG['young_tachy_max_age'],
        'extreme_brady_max_hr': args.extreme_brady_max_hr,
        'extreme_tachy_min_hr': args.extreme_tachy_min_hr,
    }

    adaptive_config = {
        'reference_hr': args.reference_hr,
        'target_fp_rate': args.target_fp_rate,
        'threshold_adj_min': DEFAULT_ADAPTIVE_CONFIG['threshold_adj_min'],
        'threshold_adj_max': DEFAULT_ADAPTIVE_CONFIG['threshold_adj_max'],
        'threshold_adj_step': DEFAULT_ADAPTIVE_CONFIG['threshold_adj_step'],
        'min_fp_rate': DEFAULT_ADAPTIVE_CONFIG['min_fp_rate'],
    }

    main(args.base_path, output, pop_config, adaptive_config)
