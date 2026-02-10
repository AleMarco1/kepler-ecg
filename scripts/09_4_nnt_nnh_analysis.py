#!/usr/bin/env python3
"""
09_4_nnt_nnh_analysis.py

Kepler-ECG Pipeline - Phase 09_4: NNT/NNH Clinical Translation

This script translates QTc formula performance into clinically actionable
metrics that physicians and healthcare administrators can use for
decision-making:

- NNT (Number Needed to Treat/Test): How many patients need to be screened
  with Kepler to avoid one false positive compared to Bazett
- NNH (Number Needed to Harm): How many patients screened with Kepler
  result in one missed diagnosis compared to Bazett
- Cost-effectiveness analysis (3 scenarios: Bazett-only, Kepler-only, Triage)
- Risk-benefit ratios by population (HR zone, age group, dataset)
- Clinical scenario modeling (preoperative, sports, drug monitoring)

Version: 2.0.0
Author: Alessandro Marconi
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Clinical thresholds (ms)
THRESHOLD_MALE = 450
THRESHOLD_FEMALE = 460
THRESHOLD_UNKNOWN = 455

# Kepler formula coefficients
KEPLER_K = 125
KEPLER_C = -158

# Sex value mapping (pipeline convention: 0=male, 1=female)
SEX_MALE_VALUES = ['m', 'male', '0', '0.0']
SEX_FEMALE_VALUES = ['f', 'female', '1', '1.0']

# Cost estimates (EUR) - conservative European estimates
DEFAULT_COSTS = {
    'ecg_screening': 25,              # Basic ECG
    'cardiology_referral': 150,       # Specialist consultation
    'holter_monitoring': 200,         # 24-48h Holter
    'echo': 300,                      # Echocardiogram
    'genetic_testing': 1500,          # LQTS genetic panel
    'false_positive_workup': 650,     # Referral + Holter + Echo (typical)
}

# Default HR and Age bins
DEFAULT_HR_BINS = [0, 60, 100, 300]
DEFAULT_AGE_BINS = [0, 40, 65, 120]

# Default clinical scenario sizes
DEFAULT_SCENARIO_PREOP = 1000
DEFAULT_SCENARIO_SPORTS = 10000
DEFAULT_SCENARIO_DRUG = 5000

# Default athletes max age
DEFAULT_ATHLETES_MAX_AGE = 35

# Minimum samples
MIN_SAMPLES_STRATUM = 100

# Column mapping
COLUMN_MAPPING = {
    'QT': ['QT_interval_ms', 'QT_ms', 'QT', 'qt_interval_ms'],
    'RR': ['RR_interval_sec', 'RR_s', 'RR', 'rr_interval_sec'],
    'HR': ['heart_rate_bpm', 'HR', 'heart_rate', 'hr_bpm'],
    'age': ['age', 'Age'],
    'sex': ['sex', 'Sex', 'gender'],
    'superclass': ['primary_superclass', 'superclass']
}

DATASETS = ['ptb-xl', 'chapman', 'cpsc-2018', 'georgia', 'mimic-iv-ecg', 'code-15']


# =============================================================================
# VECTORIZED QTc CALCULATION FUNCTIONS
# =============================================================================

def calc_qtc_bazett(qt_ms, rr_s):
    """QTc = QT / sqrt(RR)"""
    return qt_ms / np.sqrt(rr_s)

def calc_qtc_kepler(qt_ms, rr_s):
    """QTc = QT + 125/RR - 158"""
    return qt_ms + KEPLER_K / rr_s + KEPLER_C

def calc_qtc_fridericia(qt_ms, rr_s):
    """QTc = QT / cbrt(RR)"""
    return qt_ms / np.cbrt(rr_s)


# =============================================================================
# VECTORIZED CLASSIFICATION FUNCTIONS
# =============================================================================

def is_male(sex_value):
    """Check if a single sex value maps to male."""
    return str(sex_value).strip().lower() in SEX_MALE_VALUES

def is_female(sex_value):
    """Check if a single sex value maps to female."""
    return str(sex_value).strip().lower() in SEX_FEMALE_VALUES

def get_thresholds_vectorized(sex_series):
    """Return sex-specific thresholds for entire Series (vectorized)."""
    sex_lower = sex_series.astype(str).str.strip().str.lower()
    thresholds = pd.Series(THRESHOLD_UNKNOWN, index=sex_series.index, dtype=float)
    thresholds[sex_lower.isin(SEX_MALE_VALUES)] = THRESHOLD_MALE
    thresholds[sex_lower.isin(SEX_FEMALE_VALUES)] = THRESHOLD_FEMALE
    return thresholds

def classify_prolonged_vec(qtc_series, sex_series):
    """Vectorized classification: True if QTc > sex-specific threshold."""
    thresholds = get_thresholds_vectorized(sex_series)
    return qtc_series > thresholds

def assign_triage_vec(pred_bazett, pred_kepler):
    """
    Vectorized triage assignment.
    GREEN = Bazett normal (regardless of Kepler)
    YELLOW = Bazett prolonged + Kepler normal
    RED = Both prolonged
    """
    triage = pd.Series('GREEN', index=pred_bazett.index)
    bazett_pos = pred_bazett.fillna(False)
    kepler_pos = pred_kepler.fillna(False)
    triage[bazett_pos & ~kepler_pos] = 'YELLOW'
    triage[bazett_pos & kepler_pos] = 'RED'
    return triage


# =============================================================================
# CENTRALIZED ENRICHMENT
# =============================================================================

def enrich_qtc(df):
    """
    Calculate QTc for all formulas, classify, assign triage.
    Called ONCE, used everywhere.
    """
    # QTc calculations (vectorized)
    df['QTc_Bazett'] = calc_qtc_bazett(df['QT'], df['RR'])
    df['QTc_Kepler'] = calc_qtc_kepler(df['QT'], df['RR'])
    df['QTc_Fridericia'] = calc_qtc_fridericia(df['QT'], df['RR'])

    # Sex-specific threshold (vectorized)
    df['threshold'] = get_thresholds_vectorized(df['sex'])

    # Binary classification (vectorized)
    df['pred_bazett'] = classify_prolonged_vec(df['QTc_Bazett'], df['sex'])
    df['pred_kepler'] = classify_prolonged_vec(df['QTc_Kepler'], df['sex'])
    df['pred_fridericia'] = classify_prolonged_vec(df['QTc_Fridericia'], df['sex'])

    # Triage (vectorized)
    df['triage'] = assign_triage_vec(df['pred_bazett'], df['pred_kepler'])

    return df


# =============================================================================
# DATA LOADING
# =============================================================================

def find_column(df, standard_name):
    """Find the actual column name matching a standard name."""
    if standard_name in COLUMN_MAPPING:
        for possible_name in COLUMN_MAPPING[standard_name]:
            if possible_name in df.columns:
                return possible_name
    if standard_name in df.columns:
        return standard_name
    return None

def standardize_columns(df):
    """Rename columns to standard names."""
    rename_map = {}
    for standard_name in COLUMN_MAPPING:
        actual_name = find_column(df, standard_name)
        if actual_name and actual_name != standard_name:
            rename_map[actual_name] = standard_name
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def load_dataset(base_path, dataset_name):
    """Load a single dataset's QTc preparation CSV."""
    base = Path(base_path)
    qtc_file = base / 'results' / dataset_name / 'qtc' / f'{dataset_name}_qtc_preparation.csv'

    if not qtc_file.exists():
        print(f"  Warning: {qtc_file} not found")
        return None

    df = pd.read_csv(qtc_file)
    df = standardize_columns(df)

    if 'HR' not in df.columns and 'RR' in df.columns:
        df['HR'] = 60 / df['RR']

    df['dataset'] = dataset_name
    return df

def load_all_datasets(base_path):
    """Load and concatenate all 6 datasets."""
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
# HELPER: get NORM subset
# =============================================================================

def get_norm_subset(df):
    """Return NORM patients if superclass available, else all patients."""
    if 'superclass' in df.columns:
        norm_df = df[df['superclass'] == 'NORM']
        if len(norm_df) == 0:
            print("  Warning: No NORM patients found, using all patients")
            return df
        return norm_df
    else:
        print("  Warning: No superclass column, using all patients")
        return df


# =============================================================================
# HELPER: compute NNT for a subgroup
# =============================================================================

def compute_nnt_for_subgroup(sub_df):
    """
    Compute NNT metrics for a DataFrame subgroup.
    Returns dict with n, fp counts, nnt, benefit/harm ratio.
    """
    n = len(sub_df)
    fp_b = sub_df['pred_bazett'].sum()
    fp_k = sub_df['pred_kepler'].sum()
    fp_f = sub_df['pred_fridericia'].sum()
    fp_diff = fp_b - fp_k

    # Discordant cases
    scenario_a = (sub_df['pred_bazett'] & ~sub_df['pred_kepler']).sum()  # FP avoided
    scenario_b = (~sub_df['pred_bazett'] & sub_df['pred_kepler']).sum()  # Extra Kepler+

    nnt = n / fp_diff if fp_diff > 0 else float('inf')
    bh_ratio = scenario_a / scenario_b if scenario_b > 0 else float('inf')

    return {
        'n': int(n),
        'fp_bazett': int(fp_b),
        'fp_kepler': int(fp_k),
        'fp_fridericia': int(fp_f),
        'fp_avoided': int(fp_diff),
        'fp_avoided_pct': 100 * fp_diff / fp_b if fp_b > 0 else 0,
        'fp_rate_bazett': 100 * fp_b / n if n > 0 else 0,
        'fp_rate_kepler': 100 * fp_k / n if n > 0 else 0,
        'fp_rate_fridericia': 100 * fp_f / n if n > 0 else 0,
        'scenario_a_fp_avoided': int(scenario_a),
        'scenario_b_kepler_extra': int(scenario_b),
        'arr_fp': fp_diff / n if n > 0 else 0,
        'nnt': nnt,
        'benefit_harm_ratio': bh_ratio
    }


# =============================================================================
# ANALYSIS 1: Basic NNT/NNH
# =============================================================================

def calculate_nnt_nnh_basic(df):
    """
    Calculate basic NNT and NNH for Kepler vs Bazett on NORM patients.

    NNT (benefit): Number needed to screen with Kepler to avoid 1 false positive
    NNH (risk): Number needed to screen with Kepler to cause 1 additional false negative
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Basic NNT/NNH Calculation")
    print("=" * 70)

    norm_df = get_norm_subset(df)
    results = compute_nnt_for_subgroup(norm_df)

    print(f"\n  Population: {results['n']:,} NORM patients")
    print(f"\n  False Positives:")
    print(f"    Bazett:     {results['fp_bazett']:,} ({results['fp_rate_bazett']:.2f}%)")
    print(f"    Kepler:     {results['fp_kepler']:,} ({results['fp_rate_kepler']:.2f}%)")
    print(f"    Fridericia: {results['fp_fridericia']:,} ({results['fp_rate_fridericia']:.2f}%)")
    print(f"    Avoided (Bazett→Kepler): {results['fp_avoided']:,} ({results['fp_avoided_pct']:.1f}% reduction)")

    print(f"\n  NNT (Number Needed to Test with Kepler to avoid 1 FP):")
    print(f"    NNT = {results['nnt']:.1f}")
    print(f"    Interpretation: Screen {results['nnt']:.0f} patients with Kepler instead of Bazett")
    print(f"                    to prevent 1 unnecessary cardiology referral")

    print(f"\n  Discordant Cases:")
    print(f"    Scenario A (Bazett+, Kepler-): {results['scenario_a_fp_avoided']:,} - FP avoided")
    print(f"    Scenario B (Bazett-, Kepler+): {results['scenario_b_kepler_extra']:,} - Extra Kepler positives")
    print(f"    Benefit:Harm Ratio: {results['benefit_harm_ratio']:.1f}:1")

    return results


# =============================================================================
# ANALYSIS 2: NNT by Population Subgroup
# =============================================================================

def calculate_nnt_by_population(df, config):
    """
    Calculate NNT stratified by HR zone, age, and dataset.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: NNT by Population Subgroup")
    print("=" * 70)

    norm_df = get_norm_subset(df).copy()
    results = {}
    hr_bins = config['hr_bins']
    age_bins = config['age_bins']
    min_n = config['min_samples_stratum']

    # --- By HR zone ---
    print("\n  NNT by Heart Rate Zone:")
    hr_labels = _make_labels(hr_bins, 'bpm')
    norm_df['hr_group'] = pd.cut(norm_df['HR'], bins=hr_bins, labels=hr_labels, right=False)

    results['by_hr'] = {}
    for label in hr_labels:
        sub = norm_df[norm_df['hr_group'] == label]
        if len(sub) < min_n:
            continue
        r = compute_nnt_for_subgroup(sub)
        results['by_hr'][label] = r
        nnt_str = f"{r['nnt']:.1f}" if r['nnt'] < float('inf') else "N/A"
        print(f"    {label}: NNT = {nnt_str} (n={r['n']:,}, FP avoided={r['fp_avoided']:,})")

    # --- By Age ---
    if 'age' in norm_df.columns:
        print("\n  NNT by Age Group:")
        age_labels = _make_labels(age_bins, 'y')
        norm_df['age_group'] = pd.cut(norm_df['age'], bins=age_bins, labels=age_labels, right=False)

        results['by_age'] = {}
        for label in age_labels:
            sub = norm_df[norm_df['age_group'] == label]
            if len(sub) < min_n:
                continue
            r = compute_nnt_for_subgroup(sub)
            results['by_age'][label] = r
            nnt_str = f"{r['nnt']:.1f}" if r['nnt'] < float('inf') else "N/A"
            print(f"    {label}: NNT = {nnt_str} (n={r['n']:,}, FP avoided={r['fp_avoided']:,})")

    # --- By Dataset ---
    print("\n  NNT by Dataset:")
    results['by_dataset'] = {}
    for dataset in sorted(norm_df['dataset'].unique()):
        sub = norm_df[norm_df['dataset'] == dataset]
        if len(sub) < min_n:
            continue
        r = compute_nnt_for_subgroup(sub)
        results['by_dataset'][dataset] = r
        nnt_str = f"{r['nnt']:.1f}" if r['nnt'] < float('inf') else "N/A"
        print(f"    {dataset}: NNT = {nnt_str} (n={r['n']:,}, FP avoided={r['fp_avoided']:,})")

    return results


def _make_labels(bins, suffix=''):
    """Generate human-readable bin labels from bin edges."""
    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == 0 and lo == 0:
            labels.append(f"<{hi}")
        elif i == len(bins) - 2:
            labels.append(f">={lo}")
        else:
            labels.append(f"{lo}-{hi}")
    return labels


# =============================================================================
# ANALYSIS 3: Triage System NNT
# =============================================================================

def calculate_triage_nnt(df):
    """
    Calculate NNT for the 3-level triage system on NORM patients.

    The triage system converts some Bazett+ to YELLOW (follow-up)
    instead of RED (urgent referral).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Triage System NNT")
    print("=" * 70)

    norm_df = get_norm_subset(df)

    # Triage distribution (vectorized count)
    triage_counts = norm_df['triage'].value_counts()
    n_green = int(triage_counts.get('GREEN', 0))
    n_yellow = int(triage_counts.get('YELLOW', 0))
    n_red = int(triage_counts.get('RED', 0))
    n_total = len(norm_df)

    # Without triage (Bazett only): all Bazett+ would be referred urgently
    bazett_positive = n_yellow + n_red

    # With triage: only RED is referred urgently
    urgent_avoided = n_yellow

    # NNT for urgent referral avoidance
    nnt_urgent = n_total / urgent_avoided if urgent_avoided > 0 else float('inf')

    # Urgent reduction percentage
    urgent_reduction = 100 * urgent_avoided / bazett_positive if bazett_positive > 0 else 0

    results = {
        'n_total': n_total,
        'n_green': n_green,
        'n_yellow': n_yellow,
        'n_red': n_red,
        'green_pct': 100 * n_green / n_total,
        'yellow_pct': 100 * n_yellow / n_total,
        'red_pct': 100 * n_red / n_total,
        'bazett_positive': bazett_positive,
        'urgent_avoided': urgent_avoided,
        'urgent_reduction_pct': urgent_reduction,
        'nnt_avoid_urgent': nnt_urgent
    }

    print(f"\n  Population: {n_total:,} NORM patients")
    print(f"\n  Triage Distribution:")
    print(f"    GREEN:  {n_green:,} ({results['green_pct']:.1f}%) - Discharge")
    print(f"    YELLOW: {n_yellow:,} ({results['yellow_pct']:.1f}%) - Follow-up")
    print(f"    RED:    {n_red:,} ({results['red_pct']:.1f}%) - Urgent referral")

    print(f"\n  Urgent Referral Analysis:")
    print(f"    Bazett alone would refer urgently: {bazett_positive:,}")
    print(f"    With triage, urgent referrals: {n_red:,}")
    print(f"    Urgent referrals avoided: {urgent_avoided:,} ({urgent_reduction:.1f}% reduction)")

    print(f"\n  NNT (Number Needed to Triage to avoid 1 urgent referral):")
    print(f"    NNT = {nnt_urgent:.1f}")

    return results


# =============================================================================
# ANALYSIS 4: Cost-Effectiveness
# =============================================================================

def calculate_cost_effectiveness(df, basic_results, triage_results, costs):
    """
    Calculate cost-effectiveness of Kepler and triage vs Bazett alone.
    Uses costs passed from CLI/config (not global).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Cost-Effectiveness Analysis")
    print("=" * 70)

    n_total = basic_results['n']
    fp_bazett = basic_results['fp_bazett']
    fp_kepler = basic_results['fp_kepler']
    fp_avoided = basic_results['fp_avoided']

    urgent_avoided = triage_results['urgent_avoided']
    n_yellow = triage_results['n_yellow']
    n_red = triage_results['n_red']

    workup_cost = costs['false_positive_workup']
    referral_cost = costs['cardiology_referral']

    # Scenario 1: Bazett only (current practice) - all positives get full workup
    cost_bazett = fp_bazett * workup_cost

    # Scenario 2: Kepler only - fewer positives, all get full workup
    cost_kepler = fp_kepler * workup_cost

    # Scenario 3: 3-level triage - YELLOW gets referral only, RED gets full workup
    cost_triage = (n_yellow * referral_cost + n_red * workup_cost)

    # Savings
    savings_kepler = cost_bazett - cost_kepler
    savings_triage = cost_bazett - cost_triage

    # Per-patient costs
    cost_pp_bazett = cost_bazett / n_total if n_total > 0 else 0
    cost_pp_kepler = cost_kepler / n_total if n_total > 0 else 0
    cost_pp_triage = cost_triage / n_total if n_total > 0 else 0

    results = {
        'costs_used': costs,
        'scenario_bazett': {
            'total_cost': cost_bazett,
            'cost_per_patient': cost_pp_bazett,
            'n_workups': fp_bazett
        },
        'scenario_kepler': {
            'total_cost': cost_kepler,
            'cost_per_patient': cost_pp_kepler,
            'n_workups': fp_kepler,
            'savings': savings_kepler,
            'savings_pct': 100 * savings_kepler / cost_bazett if cost_bazett > 0 else 0
        },
        'scenario_triage': {
            'total_cost': cost_triage,
            'cost_per_patient': cost_pp_triage,
            'n_urgent_workups': n_red,
            'n_followup_only': n_yellow,
            'savings': savings_triage,
            'savings_pct': 100 * savings_triage / cost_bazett if cost_bazett > 0 else 0
        }
    }

    print(f"\n  Cost Assumptions (EUR):")
    print(f"    False positive workup: EUR {workup_cost}")
    print(f"    Cardiology follow-up only: EUR {referral_cost}")

    print(f"\n  Scenario 1: Bazett Only (Current Practice)")
    print(f"    Patients requiring workup: {fp_bazett:,}")
    print(f"    Total cost: EUR {cost_bazett:,.0f}")
    print(f"    Cost per screened patient: EUR {cost_pp_bazett:.2f}")

    print(f"\n  Scenario 2: Kepler Only")
    print(f"    Patients requiring workup: {fp_kepler:,}")
    print(f"    Total cost: EUR {cost_kepler:,.0f}")
    print(f"    Savings vs Bazett: EUR {savings_kepler:,.0f} ({results['scenario_kepler']['savings_pct']:.1f}%)")

    print(f"\n  Scenario 3: 3-Level Triage (Recommended)")
    print(f"    Urgent workups (RED): {n_red:,}")
    print(f"    Follow-up only (YELLOW): {n_yellow:,}")
    print(f"    Total cost: EUR {cost_triage:,.0f}")
    print(f"    Savings vs Bazett: EUR {savings_triage:,.0f} ({results['scenario_triage']['savings_pct']:.1f}%)")

    return results


# =============================================================================
# ANALYSIS 5: Clinical Scenario Modeling
# =============================================================================

def calculate_clinical_scenarios(df, basic_results, config, costs):
    """
    Model specific clinical scenarios with NNT/NNH.
    Scenario sizes and athletes age are configurable via CLI.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Clinical Scenario Modeling")
    print("=" * 70)

    norm_df = get_norm_subset(df)
    n_norm = len(norm_df)
    workup_cost = costs['false_positive_workup']

    scenarios = {}

    # --- Scenario A: Pre-operative screening ---
    n_preop = config['scenario_preop']
    print(f"\n  Scenario A: Pre-operative Cardiac Screening ({n_preop:,} patients)")

    scale = n_preop / n_norm if n_norm > 0 else 0
    fp_b_scaled = int(basic_results['fp_bazett'] * scale)
    fp_k_scaled = int(basic_results['fp_kepler'] * scale)
    fp_avoided = fp_b_scaled - fp_k_scaled

    scenarios['preoperative'] = {
        'n_patients': n_preop,
        'fp_bazett': fp_b_scaled,
        'fp_kepler': fp_k_scaled,
        'fp_avoided': fp_avoided,
        'surgeries_delayed_avoided': fp_avoided,
        'cost_saved': fp_avoided * workup_cost
    }

    print(f"    With Bazett: {fp_b_scaled} patients flagged -> surgery delayed")
    print(f"    With Kepler: {fp_k_scaled} patients flagged")
    print(f"    Unnecessary delays avoided: {fp_avoided}")
    print(f"    Cost saved: EUR {scenarios['preoperative']['cost_saved']:,.0f}")

    # --- Scenario B: Sports pre-participation ---
    n_sports = config['scenario_sports']
    athletes_max_age = config['athletes_max_age']
    print(f"\n  Scenario B: Sports Pre-participation Screening ({n_sports:,} athletes, age <{athletes_max_age})")

    # Use age-specific FP rates if available
    if 'age' in norm_df.columns:
        young_df = norm_df[norm_df['age'] < athletes_max_age]
        if len(young_df) >= MIN_SAMPLES_STRATUM:
            fp_rate_b_young = young_df['pred_bazett'].sum() / len(young_df)
            fp_rate_k_young = young_df['pred_kepler'].sum() / len(young_df)
        else:
            fp_rate_b_young = basic_results['fp_rate_bazett'] / 100
            fp_rate_k_young = basic_results['fp_rate_kepler'] / 100
    else:
        fp_rate_b_young = basic_results['fp_rate_bazett'] / 100
        fp_rate_k_young = basic_results['fp_rate_kepler'] / 100

    fp_b_sports = int(n_sports * fp_rate_b_young)
    fp_k_sports = int(n_sports * fp_rate_k_young)
    fp_avoided_sports = fp_b_sports - fp_k_sports

    scenarios['sports'] = {
        'n_patients': n_sports,
        'athletes_max_age': athletes_max_age,
        'fp_bazett': fp_b_sports,
        'fp_kepler': fp_k_sports,
        'fp_avoided': fp_avoided_sports,
        'disqualifications_avoided': fp_avoided_sports,
        'cost_saved': fp_avoided_sports * workup_cost
    }

    print(f"    With Bazett: {fp_b_sports} athletes flagged -> potential disqualification")
    print(f"    With Kepler: {fp_k_sports} athletes flagged")
    print(f"    Unnecessary disqualifications avoided: {fp_avoided_sports}")
    print(f"    Cost saved: EUR {scenarios['sports']['cost_saved']:,.0f}")

    # --- Scenario C: Drug safety monitoring ---
    n_drug = config['scenario_drug']
    print(f"\n  Scenario C: Drug Safety Monitoring ({n_drug:,} patients)")

    scale_drug = n_drug / n_norm if n_norm > 0 else 0
    fp_b_drug = int(basic_results['fp_bazett'] * scale_drug)
    fp_k_drug = int(basic_results['fp_kepler'] * scale_drug)
    fp_avoided_drug = fp_b_drug - fp_k_drug

    scenarios['drug_monitoring'] = {
        'n_patients': n_drug,
        'fp_bazett': fp_b_drug,
        'fp_kepler': fp_k_drug,
        'fp_avoided': fp_avoided_drug,
        'drug_discontinuations_avoided': fp_avoided_drug,
        'cost_saved': fp_avoided_drug * workup_cost
    }

    print(f"    With Bazett: {fp_b_drug} patients flagged -> drug discontinued")
    print(f"    With Kepler: {fp_k_drug} patients flagged")
    print(f"    Unnecessary discontinuations avoided: {fp_avoided_drug}")
    print(f"    Cost saved: EUR {scenarios['drug_monitoring']['cost_saved']:,.0f}")

    return scenarios


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def create_summary_table(basic_results, triage_results, cost_results, population_results):
    """Create summary table for easy reference."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Key NNT/NNH Metrics")
    print("=" * 70)

    summary = {
        'overall': {
            'nnt_avoid_fp': basic_results['nnt'],
            'nnt_avoid_urgent': triage_results['nnt_avoid_urgent'],
            'benefit_harm_ratio': basic_results['benefit_harm_ratio'],
            'fp_reduction_pct': basic_results['fp_avoided_pct'],
            'urgent_reduction_pct': triage_results['urgent_reduction_pct'],
            'cost_savings_pct': cost_results['scenario_triage']['savings_pct']
        },
        'by_hr': {},
        'by_age': {}
    }

    # Best NNT by HR
    if 'by_hr' in population_results and population_results['by_hr']:
        best_hr = min(population_results['by_hr'].items(),
                      key=lambda x: x[1]['nnt'] if x[1]['nnt'] < float('inf') else 9999)
        summary['best_hr_zone'] = {
            'zone': best_hr[0],
            'nnt': best_hr[1]['nnt']
        }

    # Best NNT by Age
    if 'by_age' in population_results and population_results['by_age']:
        best_age = min(population_results['by_age'].items(),
                       key=lambda x: x[1]['nnt'] if x[1]['nnt'] < float('inf') else 9999)
        summary['best_age_group'] = {
            'group': best_age[0],
            'nnt': best_age[1]['nnt']
        }

    nnt_fp = basic_results['nnt']
    nnt_urg = triage_results['nnt_avoid_urgent']

    print(f"\n  {'Metric':<45} {'Value':>15}")
    print("  " + "-" * 62)
    print(f"  {'NNT to avoid 1 false positive':<45} {nnt_fp:>15.1f}")
    print(f"  {'NNT to avoid 1 urgent referral (triage)':<45} {nnt_urg:>15.1f}")
    print(f"  {'Benefit:Harm ratio':<45} {basic_results['benefit_harm_ratio']:>14.1f}:1")
    print(f"  {'False positive reduction':<45} {basic_results['fp_avoided_pct']:>14.1f}%")
    print(f"  {'Urgent referral reduction':<45} {triage_results['urgent_reduction_pct']:>14.1f}%")
    print(f"  {'Cost savings (triage vs Bazett)':<45} {cost_results['scenario_triage']['savings_pct']:>14.1f}%")

    if 'best_hr_zone' in summary:
        print(f"  {'Best HR zone':<45} {summary['best_hr_zone']['zone']:>15}")
        print(f"  {'  NNT in best HR zone':<45} {summary['best_hr_zone']['nnt']:>15.1f}")

    if 'best_age_group' in summary:
        print(f"  {'Best age group':<45} {summary['best_age_group']['group']:>15}")
        print(f"  {'  NNT in best age group':<45} {summary['best_age_group']['nnt']:>15.1f}")

    return summary


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(all_results, output_dir):
    """Save JSON + CSV exports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    json_file = output_dir / 'nnt_nnh_analysis.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {json_file}")

    # --- CSV 1: Basic NNT/NNH ---
    basic = all_results['basic']
    pd.DataFrame([basic]).to_csv(output_dir / 'nnt_nnh_basic.csv', index=False)
    print(f"  Saved: {output_dir / 'nnt_nnh_basic.csv'}")

    # --- CSV 2: NNT by HR zone ---
    pop = all_results['by_population']
    if 'by_hr' in pop and pop['by_hr']:
        rows = []
        for label, data in pop['by_hr'].items():
            row = {'hr_zone': label}
            row.update(data)
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / 'nnt_by_hr.csv', index=False)
        print(f"  Saved: {output_dir / 'nnt_by_hr.csv'}")

    # --- CSV 3: NNT by Age group ---
    if 'by_age' in pop and pop['by_age']:
        rows = []
        for label, data in pop['by_age'].items():
            row = {'age_group': label}
            row.update(data)
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / 'nnt_by_age.csv', index=False)
        print(f"  Saved: {output_dir / 'nnt_by_age.csv'}")

    # --- CSV 4: NNT by Dataset ---
    if 'by_dataset' in pop and pop['by_dataset']:
        rows = []
        for label, data in pop['by_dataset'].items():
            row = {'dataset': label}
            row.update(data)
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / 'nnt_by_dataset.csv', index=False)
        print(f"  Saved: {output_dir / 'nnt_by_dataset.csv'}")

    # --- CSV 5: Triage NNT ---
    triage = all_results['triage']
    pd.DataFrame([triage]).to_csv(output_dir / 'nnt_triage.csv', index=False)
    print(f"  Saved: {output_dir / 'nnt_triage.csv'}")

    # --- CSV 6: Cost-effectiveness ---
    cost = all_results['cost_effectiveness']
    cost_rows = []
    for scenario_name in ['scenario_bazett', 'scenario_kepler', 'scenario_triage']:
        row = {'scenario': scenario_name.replace('scenario_', '')}
        row.update(cost[scenario_name])
        cost_rows.append(row)
    pd.DataFrame(cost_rows).to_csv(output_dir / 'cost_effectiveness.csv', index=False)
    print(f"  Saved: {output_dir / 'cost_effectiveness.csv'}")

    # --- CSV 7: Clinical scenarios ---
    scenarios = all_results['clinical_scenarios']
    scenario_rows = []
    for name, data in scenarios.items():
        row = {'scenario': name}
        row.update(data)
        scenario_rows.append(row)
    pd.DataFrame(scenario_rows).to_csv(output_dir / 'clinical_scenarios.csv', index=False)
    print(f"  Saved: {output_dir / 'clinical_scenarios.csv'}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(base_path, output_path, config, costs):
    """Run complete NNT/NNH analysis."""
    print("=" * 70)
    print("KEPLER-ECG: NNT/NNH Clinical Translation (Phase 09_4)")
    print("=" * 70)

    # Load data
    print("\nLoading datasets...")
    df = load_all_datasets(base_path)

    if df is None or len(df) == 0:
        print("ERROR: No data loaded")
        return None

    # Centralized enrichment (ONCE)
    print("\nEnriching dataset with QTc values and classifications...")
    df = enrich_qtc(df)
    print(f"  Columns added: QTc_Bazett, QTc_Kepler, QTc_Fridericia, "
          f"pred_*, triage, threshold")

    all_results = {}

    # Analysis 1: Basic NNT/NNH
    basic_results = calculate_nnt_nnh_basic(df)
    all_results['basic'] = basic_results

    # Analysis 2: NNT by population
    population_results = calculate_nnt_by_population(df, config)
    all_results['by_population'] = population_results

    # Analysis 3: Triage NNT
    triage_results = calculate_triage_nnt(df)
    all_results['triage'] = triage_results

    # Analysis 4: Cost-effectiveness
    cost_results = calculate_cost_effectiveness(df, basic_results, triage_results, costs)
    all_results['cost_effectiveness'] = cost_results

    # Analysis 5: Clinical scenarios
    scenario_results = calculate_clinical_scenarios(df, basic_results, config, costs)
    all_results['clinical_scenarios'] = scenario_results

    # Summary
    summary = create_summary_table(basic_results, triage_results, cost_results, population_results)
    all_results['summary'] = summary

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    save_results(all_results, output_path)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return all_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Kepler-ECG: NNT/NNH Clinical Translation (Phase 09_4)')

    # Required
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base path to Kepler-ECG project')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: {base-path}/results/clinical_analysis)')

    # Stratification bins
    parser.add_argument('--hr-bins', type=float, nargs='+',
                        default=DEFAULT_HR_BINS,
                        help=f'HR bin edges (default: {DEFAULT_HR_BINS})')
    parser.add_argument('--age-bins', type=float, nargs='+',
                        default=DEFAULT_AGE_BINS,
                        help=f'Age bin edges (default: {DEFAULT_AGE_BINS})')

    # Clinical scenario sizes
    parser.add_argument('--scenario-preop', type=int,
                        default=DEFAULT_SCENARIO_PREOP,
                        help=f'Preoperative scenario size (default: {DEFAULT_SCENARIO_PREOP})')
    parser.add_argument('--scenario-sports', type=int,
                        default=DEFAULT_SCENARIO_SPORTS,
                        help=f'Sports screening scenario size (default: {DEFAULT_SCENARIO_SPORTS})')
    parser.add_argument('--scenario-drug', type=int,
                        default=DEFAULT_SCENARIO_DRUG,
                        help=f'Drug monitoring scenario size (default: {DEFAULT_SCENARIO_DRUG})')
    parser.add_argument('--athletes-max-age', type=int,
                        default=DEFAULT_ATHLETES_MAX_AGE,
                        help=f'Max age for athletes scenario (default: {DEFAULT_ATHLETES_MAX_AGE})')

    # Cost overrides
    parser.add_argument('--cost-workup', type=float, default=None,
                        help=f'FP workup cost in EUR (default: {DEFAULT_COSTS["false_positive_workup"]})')
    parser.add_argument('--cost-referral', type=float, default=None,
                        help=f'Cardiology referral cost in EUR (default: {DEFAULT_COSTS["cardiology_referral"]})')

    # Minimum samples
    parser.add_argument('--min-samples-stratum', type=int,
                        default=MIN_SAMPLES_STRATUM,
                        help=f'Min samples per stratum (default: {MIN_SAMPLES_STRATUM})')

    args = parser.parse_args()

    # Build output path
    output = args.output or str(Path(args.base_path) / 'results' / 'clinical_analysis')

    # Build config dict
    config = {
        'hr_bins': [int(b) for b in args.hr_bins],
        'age_bins': [int(b) for b in args.age_bins],
        'scenario_preop': args.scenario_preop,
        'scenario_sports': args.scenario_sports,
        'scenario_drug': args.scenario_drug,
        'athletes_max_age': args.athletes_max_age,
        'min_samples_stratum': args.min_samples_stratum,
    }

    # Build costs dict (override defaults)
    costs = DEFAULT_COSTS.copy()
    if args.cost_workup is not None:
        costs['false_positive_workup'] = args.cost_workup
    if args.cost_referral is not None:
        costs['cardiology_referral'] = args.cost_referral

    main(args.base_path, output, config, costs)
