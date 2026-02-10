#!/usr/bin/env python3
"""
09_0_clinical_reclassification_analysis.py

Kepler-ECG Pipeline - Phase 09_0: Clinical Reclassification Analysis

This script analyzes the clinical impact of different QTc correction formulas
by examining how patients are reclassified between diagnostic categories
(normal, borderline, prolonged QT) when using different formulas.

The key clinical question: How many patients change diagnosis when switching
from Bazett to Kepler formula?

Author: Alessandro Marconi
Version: 2.0 - February 2026

Changelog v2.0:
- FIX: Unified sex mapping via normalize_sex() — resolves float 0.0/1.0 inconsistency
- FIX: Vectorized classify_qtc() and classify_qtc_binary() — ~50x speedup on 1.2M records
- FIX: NORM filter now case-insensitive (handles 'NORM', 'Norm', 'norm')
- FIX: Robust merge in load_dataset() — handles missing 'superclass'/'sex'/'age' gracefully
- ADD: --kepler-k / --kepler-c CLI arguments for formula coefficient override
- ADD: --bootstrap-n CLI for bootstrap confidence intervals on clinical metrics
- ADD: --min-samples CLI for stratification minimum sample threshold
- ADD: Bootstrap CI (95%) on sensitivity, specificity, PPV, NPV, F1, Youden's J, MCC
- ADD: SNOMED gold standard columns marked as [BACKLOG] with clear docstring
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS = ['ptb-xl', 'chapman', 'cpsc-2018', 'georgia', 'mimic-iv-ecg', 'code-15']

# Clinical QTc thresholds (ms) - AHA/ACC Guidelines
QTC_THRESHOLDS = {
    'male': {
        'normal': 450,      # <= 450 ms normal
        'borderline': 470,  # 451-470 ms borderline
        # > 470 ms prolonged
    },
    'female': {
        'normal': 460,      # <= 460 ms normal
        'borderline': 480,  # 461-480 ms borderline
        # > 480 ms prolonged
    },
    'unknown': {
        'normal': 450,      # Conservative: use male thresholds
        'borderline': 470,
    }
}

# Heart rate categories for stratification
HR_CATEGORIES = {
    'bradycardia': (0, 60),      # < 60 bpm
    'normal': (60, 100),         # 60-100 bpm
    'tachycardia': (100, 999),   # > 100 bpm
}

# Age categories
AGE_CATEGORIES = {
    'young': (0, 40),
    'middle': (40, 65),
    'elderly': (65, 999),
}

# Kepler formula coefficients (from pooled discovery)
KEPLER_K = 125  # ms·s
KEPLER_C = -158  # ms

# Minimum samples for stratified analyses
MIN_SAMPLES_STRATIFICATION = 100

# Bootstrap default iterations
BOOTSTRAP_N_DEFAULT = 0  # 0 = disabled; typical: 1000-10000


# =============================================================================
# QTc CALCULATION FUNCTIONS
# =============================================================================

def calc_qtc_bazett(qt_ms: np.ndarray, rr_s: np.ndarray) -> np.ndarray:
    """Bazett formula: QTc = QT / sqrt(RR)"""
    return qt_ms / np.sqrt(rr_s)


def calc_qtc_fridericia(qt_ms: np.ndarray, rr_s: np.ndarray) -> np.ndarray:
    """Fridericia formula: QTc = QT / RR^(1/3)"""
    return qt_ms / np.cbrt(rr_s)


def calc_qtc_framingham(qt_ms: np.ndarray, rr_s: np.ndarray) -> np.ndarray:
    """Framingham formula: QTc = QT + 154 * (1 - RR)"""
    return qt_ms + 154 * (1 - rr_s)


def calc_qtc_hodges(qt_ms: np.ndarray, rr_s: np.ndarray) -> np.ndarray:
    """Hodges formula: QTc = QT + 1.75 * (HR - 60)"""
    hr = 60 / rr_s
    return qt_ms + 1.75 * (hr - 60)


def calc_qtc_kepler(qt_ms: np.ndarray, rr_s: np.ndarray, 
                    k: float = KEPLER_K, c: float = KEPLER_C) -> np.ndarray:
    """Kepler formula: QTc = QT + k/RR + c"""
    return qt_ms + k / rr_s + c


# =============================================================================
# SEX NORMALIZATION (single source of truth)
# =============================================================================

def normalize_sex(sex: pd.Series) -> pd.Series:
    """
    Normalize sex values to a canonical set: 'M', 'F', or 'unknown'.
    
    Handles all known encodings across datasets:
      - String: 'M', 'F', 'male', 'female', 'Male', 'Female'
      - Numeric: 0 (male), 1 (female), 0.0, 1.0
      - Missing: NaN, None → 'unknown'
    
    Returns: pd.Series with values in {'M', 'F', 'unknown'}
    """
    mapping = {
        'M': 'M', 'male': 'M', 'Male': 'M',
        'F': 'F', 'female': 'F', 'Female': 'F',
        0: 'M', 0.0: 'M',
        1: 'F', 1.0: 'F',
    }
    normalized = sex.map(mapping)
    normalized = normalized.fillna('unknown')
    return normalized


# =============================================================================
# CLASSIFICATION FUNCTIONS (vectorized)
# =============================================================================

def classify_qtc(qtc_values: np.ndarray, sex: pd.Series) -> np.ndarray:
    """
    Classify QTc values into clinical categories based on sex-specific thresholds.
    Vectorized implementation — handles 1M+ records efficiently.
    
    Args:
        qtc_values: Array of QTc values in ms
        sex: Series of sex values (any encoding — will be normalized)
    
    Returns: array of categories ('normal', 'borderline', 'prolonged', 'invalid')
    """
    sex_norm = normalize_sex(sex)
    
    # Build threshold arrays based on normalized sex
    thresh_normal = np.where(
        sex_norm.values == 'F',
        QTC_THRESHOLDS['female']['normal'],
        QTC_THRESHOLDS['male']['normal']      # 'M' and 'unknown' both use male
    )
    thresh_borderline = np.where(
        sex_norm.values == 'F',
        QTC_THRESHOLDS['female']['borderline'],
        QTC_THRESHOLDS['male']['borderline']
    )
    
    # Vectorized classification
    categories = np.where(
        np.isnan(qtc_values), 'invalid',
        np.where(
            qtc_values <= thresh_normal, 'normal',
            np.where(
                qtc_values <= thresh_borderline, 'borderline',
                'prolonged'
            )
        )
    )
    
    return categories


def get_hr_category(hr: float) -> str:
    """Categorize heart rate."""
    for cat, (low, high) in HR_CATEGORIES.items():
        if low <= hr < high:
            return cat
    return 'unknown'


def get_age_category(age: float) -> str:
    """Categorize age."""
    if pd.isna(age):
        return 'unknown'
    for cat, (low, high) in AGE_CATEGORIES.items():
        if low <= age < high:
            return cat
    return 'unknown'


# =============================================================================
# RECLASSIFICATION ANALYSIS
# =============================================================================

def compute_reclassification_matrix(cat_ref: np.ndarray, cat_new: np.ndarray) -> dict:
    """
    Compute reclassification matrix between reference and new formula.
    
    Returns dict with:
    - matrix: 3x3 reclassification counts
    - upgraded: count moved to more severe category
    - downgraded: count moved to less severe category
    - unchanged: count staying in same category
    """
    categories = ['normal', 'borderline', 'prolonged']
    matrix = np.zeros((3, 3), dtype=int)
    
    for i, ref_cat in enumerate(categories):
        for j, new_cat in enumerate(categories):
            mask = (cat_ref == ref_cat) & (cat_new == new_cat)
            matrix[i, j] = mask.sum()
    
    # Calculate movements
    upgraded = 0    # normal->borderline, normal->prolonged, borderline->prolonged
    downgraded = 0  # prolonged->borderline, prolonged->normal, borderline->normal
    unchanged = 0
    
    for i in range(3):
        for j in range(3):
            if i == j:
                unchanged += matrix[i, j]
            elif j > i:
                upgraded += matrix[i, j]
            else:
                downgraded += matrix[i, j]
    
    return {
        'matrix': matrix.tolist(),
        'categories': categories,
        'upgraded': int(upgraded),
        'downgraded': int(downgraded),
        'unchanged': int(unchanged),
        'total': int(matrix.sum()),
    }


def analyze_clinical_impact(df: pd.DataFrame, 
                           filter_norm: bool = True,
                           superclass_col: str = 'superclass',
                           kepler_k: float = KEPLER_K,
                           kepler_c: float = KEPLER_C,
                           min_samples: int = MIN_SAMPLES_STRATIFICATION,
                           bootstrap_n: int = BOOTSTRAP_N_DEFAULT) -> dict:
    """
    Comprehensive clinical impact analysis.
    
    Args:
        df: DataFrame with QT, RR, sex, age, and optional superclass
        filter_norm: If True, analyze only NORM superclass patients
        superclass_col: Column name for superclass labels
        kepler_k: Kepler formula k coefficient (ms·s) — default from pooled discovery
        kepler_c: Kepler formula c constant (ms) — default from pooled discovery
        min_samples: Minimum samples required for stratified sub-analyses
        bootstrap_n: Number of bootstrap iterations for CI (0 = disabled)
    
    Returns:
        Dictionary with complete analysis results
    """
    results = {}
    
    # Filter to NORM patients if requested
    if filter_norm and superclass_col in df.columns:
        # Case-insensitive NORM matching
        norm_mask = df[superclass_col].astype(str).str.upper() == 'NORM'
        df_analysis = df[norm_mask].copy()
        results['population'] = 'NORM only'
        results['n_total'] = len(df)
        results['n_norm'] = len(df_analysis)
    else:
        df_analysis = df.copy()
        results['population'] = 'All patients'
        results['n_total'] = len(df)
        results['n_norm'] = len(df)
    
    if len(df_analysis) == 0:
        results['error'] = 'No records to analyze'
        return results
    
    # Ensure required columns
    required = ['QT_ms', 'RR_s']
    for col in required:
        if col not in df_analysis.columns:
            # Try alternative names
            if col == 'QT_ms' and 'qt_interval' in df_analysis.columns:
                df_analysis['QT_ms'] = df_analysis['qt_interval']
            elif col == 'RR_s' and 'rr_interval' in df_analysis.columns:
                df_analysis['RR_s'] = df_analysis['rr_interval'] / 1000  # ms to s
            else:
                results['error'] = f'Missing column: {col}'
                return results
    
    # Calculate all QTc formulas
    qt = df_analysis['QT_ms'].values
    rr = df_analysis['RR_s'].values
    
    qtc_bazett = calc_qtc_bazett(qt, rr)
    qtc_fridericia = calc_qtc_fridericia(qt, rr)
    qtc_framingham = calc_qtc_framingham(qt, rr)
    qtc_hodges = calc_qtc_hodges(qt, rr)
    qtc_kepler = calc_qtc_kepler(qt, rr, k=kepler_k, c=kepler_c)
    
    # Get sex for classification (handle various column names)
    sex_col = None
    for col in ['sex', 'Sex', 'gender', 'Gender']:
        if col in df_analysis.columns:
            sex_col = col
            break
    
    if sex_col:
        sex = df_analysis[sex_col]
    else:
        sex = pd.Series(['unknown'] * len(df_analysis))
    
    # Classify with each formula
    cat_bazett = classify_qtc(qtc_bazett, sex)
    cat_fridericia = classify_qtc(qtc_fridericia, sex)
    cat_framingham = classify_qtc(qtc_framingham, sex)
    cat_hodges = classify_qtc(qtc_hodges, sex)
    cat_kepler = classify_qtc(qtc_kepler, sex)
    
    # Store QTc statistics
    results['qtc_statistics'] = {
        'bazett': {
            'mean': float(np.nanmean(qtc_bazett)),
            'std': float(np.nanstd(qtc_bazett)),
            'median': float(np.nanmedian(qtc_bazett)),
        },
        'fridericia': {
            'mean': float(np.nanmean(qtc_fridericia)),
            'std': float(np.nanstd(qtc_fridericia)),
            'median': float(np.nanmedian(qtc_fridericia)),
        },
        'kepler': {
            'mean': float(np.nanmean(qtc_kepler)),
            'std': float(np.nanstd(qtc_kepler)),
            'median': float(np.nanmedian(qtc_kepler)),
        },
    }
    
    # Category distributions
    results['category_distribution'] = {}
    for name, cats in [('bazett', cat_bazett), ('fridericia', cat_fridericia),
                       ('framingham', cat_framingham), ('hodges', cat_hodges),
                       ('kepler', cat_kepler)]:
        unique, counts = np.unique(cats, return_counts=True)
        results['category_distribution'][name] = dict(zip(unique.tolist(), counts.tolist()))
    
    # Reclassification matrices (Kepler vs each historical formula)
    results['reclassification'] = {
        'kepler_vs_bazett': compute_reclassification_matrix(cat_bazett, cat_kepler),
        'kepler_vs_fridericia': compute_reclassification_matrix(cat_fridericia, cat_kepler),
        'kepler_vs_framingham': compute_reclassification_matrix(cat_framingham, cat_kepler),
        'kepler_vs_hodges': compute_reclassification_matrix(cat_hodges, cat_kepler),
    }
    
    # HR correlation analysis
    hr = 60 / rr  # beats per minute
    results['hr_correlation'] = {
        'bazett': float(np.corrcoef(qtc_bazett[~np.isnan(qtc_bazett)], 
                                     hr[~np.isnan(qtc_bazett)])[0, 1]),
        'fridericia': float(np.corrcoef(qtc_fridericia[~np.isnan(qtc_fridericia)], 
                                         hr[~np.isnan(qtc_fridericia)])[0, 1]),
        'kepler': float(np.corrcoef(qtc_kepler[~np.isnan(qtc_kepler)], 
                                     hr[~np.isnan(qtc_kepler)])[0, 1]),
    }
    
    # Stratified analysis by HR category
    df_analysis['hr'] = hr
    df_analysis['hr_category'] = df_analysis['hr'].apply(get_hr_category)
    df_analysis['cat_bazett'] = cat_bazett
    df_analysis['cat_kepler'] = cat_kepler
    
    results['stratified_by_hr'] = {}
    for hr_cat in ['bradycardia', 'normal', 'tachycardia']:
        mask = df_analysis['hr_category'] == hr_cat
        if mask.sum() > 0:
            results['stratified_by_hr'][hr_cat] = {
                'n': int(mask.sum()),
                'reclassification': compute_reclassification_matrix(
                    df_analysis.loc[mask, 'cat_bazett'].values,
                    df_analysis.loc[mask, 'cat_kepler'].values
                )
            }
    
    # Stratified analysis by sex
    results['stratified_by_sex'] = {}
    for sex_val in df_analysis[sex_col].unique() if sex_col else []:
        if pd.isna(sex_val):
            continue
        mask = df_analysis[sex_col] == sex_val
        if mask.sum() >= min_samples:  # Configurable via --min-samples
            results['stratified_by_sex'][str(sex_val)] = {
                'n': int(mask.sum()),
                'reclassification': compute_reclassification_matrix(
                    df_analysis.loc[mask, 'cat_bazett'].values,
                    df_analysis.loc[mask, 'cat_kepler'].values
                )
            }
    
    # Stratified by age (if available)
    age_col = None
    for col in ['age', 'Age', 'patient_age']:
        if col in df_analysis.columns:
            age_col = col
            break
    
    if age_col:
        df_analysis['age_category'] = df_analysis[age_col].apply(get_age_category)
        results['stratified_by_age'] = {}
        for age_cat in ['young', 'middle', 'elderly']:
            mask = df_analysis['age_category'] == age_cat
            if mask.sum() >= min_samples:
                results['stratified_by_age'][age_cat] = {
                    'n': int(mask.sum()),
                    'reclassification': compute_reclassification_matrix(
                        df_analysis.loc[mask, 'cat_bazett'].values,
                        df_analysis.loc[mask, 'cat_kepler'].values
                    )
                }
    
    # Clinical impact summary
    reclass = results['reclassification']['kepler_vs_bazett']
    
    # False positives avoided: patients classified as prolonged by Bazett but normal/borderline by Kepler
    # These are in row 2 (prolonged by Bazett), columns 0 and 1 (normal/borderline by Kepler)
    matrix = np.array(reclass['matrix'])
    
    # Bazett prolonged -> Kepler normal or borderline
    false_positives_avoided = matrix[2, 0] + matrix[2, 1]
    
    # Bazett normal/borderline -> Kepler prolonged (potential missed diagnoses)
    potential_missed = matrix[0, 2] + matrix[1, 2]
    
    results['clinical_impact_summary'] = {
        'false_positives_avoided': int(false_positives_avoided),
        'potential_missed_diagnoses': int(potential_missed),
        'net_benefit': int(false_positives_avoided - potential_missed),
        'ratio_avoided_to_missed': (
            float(false_positives_avoided / potential_missed) 
            if potential_missed > 0 else float('inf')
        ),
    }
    
    # =========================================================================
    # GOLD STANDARD ANALYSIS - Full Confusion Matrix
    # =========================================================================
    results['gold_standard_analysis'] = {}
    
    # Gold Standard 1: QTc_reference_ms (HR-independent polynomial regression)
    if 'QTc_reference_ms' in df_analysis.columns:
        qtc_ref = df_analysis['QTc_reference_ms'].values
        gs1_results = analyze_with_gold_standard(
            qtc_ref, sex, 
            {'bazett': qtc_bazett, 'fridericia': qtc_fridericia, 
             'framingham': qtc_framingham, 'hodges': qtc_hodges, 'kepler': qtc_kepler},
            gold_standard_name='QTc_reference (polynomial regression)',
            bootstrap_n=bootstrap_n
        )
        results['gold_standard_analysis']['qtc_reference'] = gs1_results
        
        # =====================================================================
        # CASCADE STRATEGY ANALYSIS (Bazett → Kepler)
        # =====================================================================
        cascade_results = analyze_cascade_strategy(
            qtc_ref, qtc_bazett, qtc_kepler, sex,
            gold_standard_name='QTc_reference (polynomial regression)'
        )
        results['gold_standard_analysis']['cascade_strategy'] = cascade_results
    
    # Gold Standard 2: SNOMED diagnosis of prolonged QT (if available)
    # [BACKLOG] No dataset in the current pipeline provides these columns.
    # This section activates automatically when a dataset includes one of:
    #   prolonged_qt, label_prolonged_qt, has_prolonged_qt,
    #   snomed_prolonged_qt, diagnosis_prolonged_qt
    # Target datasets: LUDB, Large Scale Arrhythmia DB, MIMIC-IV (with outcomes)
    prolonged_qt_col = None
    for col in ['prolonged_qt', 'label_prolonged_qt', 'has_prolonged_qt', 
                'snomed_prolonged_qt', 'diagnosis_prolonged_qt']:
        if col in df_analysis.columns:
            prolonged_qt_col = col
            break
    
    if prolonged_qt_col:
        # Binary gold standard from clinical diagnosis
        gs2_prolonged = df_analysis[prolonged_qt_col].values.astype(bool)
        gs2_results = analyze_with_gold_standard_binary(
            gs2_prolonged,
            {'bazett': qtc_bazett, 'fridericia': qtc_fridericia,
             'framingham': qtc_framingham, 'hodges': qtc_hodges, 'kepler': qtc_kepler},
            sex,
            gold_standard_name='SNOMED prolonged QT diagnosis',
            bootstrap_n=bootstrap_n
        )
        results['gold_standard_analysis']['snomed_diagnosis'] = gs2_results
    
    return results


def analyze_cascade_strategy(qtc_reference: np.ndarray,
                              qtc_bazett: np.ndarray,
                              qtc_kepler: np.ndarray,
                              sex: pd.Series,
                              gold_standard_name: str) -> dict:
    """
    Analyze the 3-LEVEL TRIAGE strategy: Bazett screens, Kepler confirms.
    
    Strategy:
    1. Apply Bazett to all patients
    2. If Bazett = Normal → GREEN (discharge)
    3. If Bazett = Prolonged → Apply Kepler:
       - If Kepler = Prolonged → RED (urgent referral)
       - If Kepler = Normal → YELLOW (borderline, needs clinical review)
    
    This maintains Bazett's sensitivity (no cases dismissed without Bazett approval)
    while using Kepler to stratify risk levels.
    
    Classification:
    - GREEN: Low risk, discharge (Bazett normal)
    - YELLOW: Uncertain, needs follow-up (Bazett prolonged, Kepler normal)
    - RED: High risk, urgent referral (both prolonged)
    """
    results = {
        'strategy': '3-LEVEL TRIAGE (Bazett → Kepler)',
        'gold_standard': gold_standard_name,
        'description': 'GREEN (discharge) / YELLOW (follow-up) / RED (referral)',
        'levels': {
            'GREEN': 'Bazett normal → Discharge',
            'YELLOW': 'Bazett prolonged + Kepler normal → Follow-up/Clinical review',
            'RED': 'Both prolonged → Urgent cardiology referral'
        }
    }
    
    # Classify using gold standard (truth)
    gs_prolonged = classify_qtc_binary(qtc_reference, sex)
    
    # Classify using each formula
    bazett_prolonged = classify_qtc_binary(qtc_bazett, sex)
    kepler_prolonged = classify_qtc_binary(qtc_kepler, sex)
    
    # 3-Level Triage Logic
    # GREEN: Bazett says normal (regardless of Kepler)
    # YELLOW: Bazett says prolonged BUT Kepler says normal
    # RED: Both say prolonged
    
    triage_green = ~bazett_prolonged
    triage_yellow = bazett_prolonged & ~kepler_prolonged
    triage_red = bazett_prolonged & kepler_prolonged
    
    n_total = len(gs_prolonged)
    n_truly_prolonged = int(gs_prolonged.sum())
    n_truly_normal = int((~gs_prolonged).sum())
    
    # Analyze what ends up in each category
    results['triage_distribution'] = {
        'GREEN': {
            'total': int(triage_green.sum()),
            'truly_prolonged': int((gs_prolonged & triage_green).sum()),  # FN - cases missed
            'truly_normal': int((~gs_prolonged & triage_green).sum()),    # TN - correct discharge
        },
        'YELLOW': {
            'total': int(triage_yellow.sum()),
            'truly_prolonged': int((gs_prolonged & triage_yellow).sum()),  # Need follow-up (good catch)
            'truly_normal': int((~gs_prolonged & triage_yellow).sum()),    # FP filtered by Kepler
        },
        'RED': {
            'total': int(triage_red.sum()),
            'truly_prolonged': int((gs_prolonged & triage_red).sum()),     # TP - correct urgent referral
            'truly_normal': int((~gs_prolonged & triage_red).sum()),       # FP - false urgent referral
        }
    }
    
    # Calculate percentages
    for level in ['GREEN', 'YELLOW', 'RED']:
        d = results['triage_distribution'][level]
        d['pct_of_total'] = round(d['total'] / n_total * 100, 1) if n_total > 0 else 0
        d['pct_truly_prolonged'] = round(d['truly_prolonged'] / d['total'] * 100, 1) if d['total'] > 0 else 0
    
    # Clinical metrics for each strategy
    # Strategy 1: Bazett only (everything prolonged goes to referral)
    results['bazett_only'] = compute_confusion_matrix_metrics(gs_prolonged, bazett_prolonged)
    
    # Strategy 2: Kepler only
    results['kepler_only'] = compute_confusion_matrix_metrics(gs_prolonged, kepler_prolonged)
    
    # Strategy 3: 3-Level Triage - RED only as "positive" (urgent referral)
    results['triage_red_only'] = compute_confusion_matrix_metrics(gs_prolonged, triage_red)
    
    # Strategy 4: 3-Level Triage - RED + YELLOW as "positive" (any follow-up)
    triage_red_or_yellow = triage_red | triage_yellow
    results['triage_red_yellow'] = compute_confusion_matrix_metrics(gs_prolonged, triage_red_or_yellow)
    
    # Key insight: YELLOW category analysis
    yellow_dist = results['triage_distribution']['YELLOW']
    results['yellow_analysis'] = {
        'total_in_yellow': yellow_dist['total'],
        'truly_prolonged_in_yellow': yellow_dist['truly_prolonged'],
        'truly_normal_in_yellow': yellow_dist['truly_normal'],
        'yellow_purity': round(yellow_dist['truly_prolonged'] / yellow_dist['total'] * 100, 1) if yellow_dist['total'] > 0 else 0,
        'interpretation': 'Cases where Bazett flags but Kepler does not - need clinical judgment'
    }
    
    # Comparison summary
    red = results['triage_distribution']['RED']
    yellow = results['triage_distribution']['YELLOW']
    green = results['triage_distribution']['GREEN']
    
    results['comparison'] = {
        'bazett_only': {
            'referrals': results['bazett_only']['tp'] + results['bazett_only']['fp'],
            'sensitivity': results['bazett_only']['sensitivity'],
            'specificity': results['bazett_only']['specificity'],
            'ppv': results['bazett_only']['ppv'],
            'fp': results['bazett_only']['fp'],
            'fn': results['bazett_only']['fn'],
        },
        'kepler_only': {
            'referrals': results['kepler_only']['tp'] + results['kepler_only']['fp'],
            'sensitivity': results['kepler_only']['sensitivity'],
            'specificity': results['kepler_only']['specificity'],
            'ppv': results['kepler_only']['ppv'],
            'fp': results['kepler_only']['fp'],
            'fn': results['kepler_only']['fn'],
        },
        'triage_red_only': {
            'urgent_referrals': red['total'],
            'sensitivity': results['triage_red_only']['sensitivity'],
            'specificity': results['triage_red_only']['specificity'],
            'ppv': results['triage_red_only']['ppv'],
            'fp': red['truly_normal'],
            'fn': green['truly_prolonged'],  # Only GREEN are truly dismissed
        },
        'triage_full': {
            'urgent_referrals_red': red['total'],
            'followup_yellow': yellow['total'],
            'discharged_green': green['total'],
            'cases_in_yellow': yellow['truly_prolonged'],
            'false_alarms_filtered_to_yellow': yellow['truly_normal'],
            'cases_truly_missed_green': green['truly_prolonged'],
        }
    }
    
    # Trade-off analysis
    baz = results['bazett_only']
    
    # With 3-level triage:
    # - Urgent referrals reduced from Bazett total to RED only
    # - But YELLOW cases still need some attention
    # - TRUE misses are only GREEN cases that are truly prolonged
    
    results['tradeoff'] = {
        'urgent_referrals_bazett': baz['tp'] + baz['fp'],
        'urgent_referrals_triage': red['total'],
        'urgent_referral_reduction': baz['tp'] + baz['fp'] - red['total'],
        'urgent_referral_reduction_pct': round((1 - red['total'] / (baz['tp'] + baz['fp'])) * 100, 1) if (baz['tp'] + baz['fp']) > 0 else 0,
        'yellow_cases_need_review': yellow['total'],
        'true_cases_in_yellow': yellow['truly_prolonged'],
        'false_alarms_in_yellow': yellow['truly_normal'],
        'true_misses_green': green['truly_prolonged'],
        'sensitivity_if_yellow_followed_up': results['triage_red_yellow']['sensitivity'],
        'sensitivity_if_yellow_dismissed': results['triage_red_only']['sensitivity'],
    }
    
    # Clinical workflow summary
    results['workflow_summary'] = {
        'total_patients': n_total,
        'green_discharged': green['total'],
        'green_pct': round(green['total'] / n_total * 100, 1),
        'yellow_followup': yellow['total'],
        'yellow_pct': round(yellow['total'] / n_total * 100, 1),
        'red_urgent': red['total'],
        'red_pct': round(red['total'] / n_total * 100, 1),
        'workload_reduction_vs_bazett': round((1 - (red['total'] + yellow['total']) / (baz['tp'] + baz['fp'])) * 100, 1) if (baz['tp'] + baz['fp']) > 0 else 0,
    }
    
    return results


def classify_qtc_binary(qtc_values: np.ndarray, sex: pd.Series) -> np.ndarray:
    """
    Classify QTc values as prolonged (True) or not prolonged (False).
    Vectorized implementation using normalize_sex() for consistent sex mapping.
    
    Uses sex-specific 'normal' threshold: values above threshold = prolonged.
    """
    sex_norm = normalize_sex(sex)
    
    threshold = np.where(
        sex_norm.values == 'F',
        QTC_THRESHOLDS['female']['normal'],
        QTC_THRESHOLDS['male']['normal']      # 'M' and 'unknown' both use male
    )
    
    # NaN QTc → not prolonged (False)
    valid = ~np.isnan(qtc_values)
    prolonged = np.zeros(len(qtc_values), dtype=bool)
    prolonged[valid] = qtc_values[valid] > threshold[valid]
    
    return prolonged


def compute_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute full confusion matrix metrics for binary classification.
    
    Args:
        y_true: Ground truth binary labels (True = prolonged)
        y_pred: Predicted binary labels (True = prolonged)
    
    Returns:
        Dictionary with TP, TN, FP, FN and derived metrics
    """
    # Handle NaN values
    valid_mask = ~(np.isnan(y_true.astype(float)) | np.isnan(y_pred.astype(float)))
    y_true = y_true[valid_mask].astype(bool)
    y_pred = y_pred[valid_mask].astype(bool)
    
    tp = int(np.sum(y_true & y_pred))       # True Positive: truly prolonged, predicted prolonged
    tn = int(np.sum(~y_true & ~y_pred))     # True Negative: truly normal, predicted normal
    fp = int(np.sum(~y_true & y_pred))      # False Positive: truly normal, predicted prolonged
    fn = int(np.sum(y_true & ~y_pred))      # False Negative: truly prolonged, predicted normal
    
    total = tp + tn + fp + fn
    
    # Derived metrics with safe division
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall, True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0          # Positive Predictive Value, Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0          # Negative Predictive Value
    accuracy = (tp + tn) / total if total > 0 else 0.0
    f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    # Youden's J statistic (balanced accuracy - 1)
    youden_j = sensitivity + specificity - 1
    
    # Matthews Correlation Coefficient
    mcc_denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
    
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total': total,
        'sensitivity': float(sensitivity),      # How many true prolonged are detected
        'specificity': float(specificity),      # How many true normal are correctly identified
        'ppv': float(ppv),                      # If test positive, probability truly prolonged
        'npv': float(npv),                      # If test negative, probability truly normal
        'accuracy': float(accuracy),
        'f1_score': float(f1_score),
        'youden_j': float(youden_j),
        'mcc': float(mcc),
        # Clinical interpretation
        'false_positive_rate': float(1 - specificity),  # Type I error
        'false_negative_rate': float(1 - sensitivity),  # Type II error (missed diagnoses)
    }


def bootstrap_confusion_matrix_ci(y_true: np.ndarray, y_pred: np.ndarray,
                                   n_bootstrap: int = 1000,
                                   ci_level: float = 0.95,
                                   seed: int = 42) -> dict:
    """
    Compute bootstrap confidence intervals for all confusion matrix metrics.
    
    Uses stratified resampling to preserve class balance across iterations.
    
    Args:
        y_true: Ground truth binary labels (True = prolonged)
        y_pred: Predicted binary labels (True = prolonged)
        n_bootstrap: Number of bootstrap iterations
        ci_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with point estimates and CI for each metric:
        {metric_name: {'point': float, 'ci_lower': float, 'ci_upper': float}}
    """
    rng = np.random.RandomState(seed)
    
    # Handle NaN values
    valid_mask = ~(np.isnan(y_true.astype(float)) | np.isnan(y_pred.astype(float)))
    y_true = y_true[valid_mask].astype(bool)
    y_pred = y_pred[valid_mask].astype(bool)
    
    n = len(y_true)
    if n == 0:
        return {}
    
    # Point estimates
    point_metrics = compute_confusion_matrix_metrics(
        y_true.astype(bool), y_pred.astype(bool)
    )
    
    # Metrics to bootstrap
    metric_names = ['sensitivity', 'specificity', 'ppv', 'npv', 
                    'accuracy', 'f1_score', 'youden_j', 'mcc']
    
    boot_results = {m: [] for m in metric_names}
    
    alpha = 1 - ci_level
    
    for _ in range(n_bootstrap):
        # Stratified resampling: sample within positive and negative classes
        pos_idx = np.where(y_true)[0]
        neg_idx = np.where(~y_true)[0]
        
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            # Cannot stratify — fall back to simple resampling
            idx = rng.choice(n, size=n, replace=True)
        else:
            boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            idx = np.concatenate([boot_pos, boot_neg])
        
        boot_true = y_true[idx]
        boot_pred = y_pred[idx]
        
        # Compute metrics on bootstrap sample
        tp = int(np.sum(boot_true & boot_pred))
        tn = int(np.sum(~boot_true & ~boot_pred))
        fp = int(np.sum(~boot_true & boot_pred))
        fn = int(np.sum(boot_true & ~boot_pred))
        total = tp + tn + fp + fn
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv_b = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv_b = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        acc = (tp + tn) / total if total > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        yj = sens + spec - 1
        mcc_d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc_b = (tp * tn - fp * fn) / mcc_d if mcc_d > 0 else 0.0
        
        boot_results['sensitivity'].append(sens)
        boot_results['specificity'].append(spec)
        boot_results['ppv'].append(ppv_b)
        boot_results['npv'].append(npv_b)
        boot_results['accuracy'].append(acc)
        boot_results['f1_score'].append(f1)
        boot_results['youden_j'].append(yj)
        boot_results['mcc'].append(mcc_b)
    
    # Compute CIs using percentile method
    ci_results = {}
    for m in metric_names:
        values = np.array(boot_results[m])
        ci_results[m] = {
            'point': float(point_metrics[m]),
            'ci_lower': float(np.percentile(values, alpha / 2 * 100)),
            'ci_upper': float(np.percentile(values, (1 - alpha / 2) * 100)),
            'ci_level': ci_level,
            'n_bootstrap': n_bootstrap,
        }
    
    return ci_results


def analyze_with_gold_standard(qtc_reference: np.ndarray, 
                                sex: pd.Series,
                                formula_qtc_values: dict,
                                gold_standard_name: str,
                                bootstrap_n: int = 0) -> dict:
    """
    Analyze formulas against a continuous gold standard (QTc reference).
    
    The gold standard is classified using the same sex-specific thresholds,
    then each formula is compared against this classification.
    
    Args:
        bootstrap_n: If > 0, compute bootstrap 95% CI on all metrics
    """
    results = {
        'gold_standard': gold_standard_name,
        'method': 'continuous_to_categorical',
        'formulas': {}
    }
    
    # Classify gold standard
    gs_categories = classify_qtc(qtc_reference, sex)
    gs_prolonged = (gs_categories == 'prolonged')
    
    # Count gold standard distribution
    unique, counts = np.unique(gs_categories, return_counts=True)
    results['gold_standard_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
    results['n_truly_prolonged'] = int(gs_prolonged.sum())
    results['n_truly_normal'] = int((~gs_prolonged).sum())
    
    # Analyze each formula
    for formula_name, qtc_values in formula_qtc_values.items():
        formula_categories = classify_qtc(qtc_values, sex)
        formula_prolonged = (formula_categories == 'prolonged')
        
        metrics = compute_confusion_matrix_metrics(gs_prolonged, formula_prolonged)
        results['formulas'][formula_name] = metrics
        
        # Bootstrap CI (if enabled)
        if bootstrap_n > 0:
            logging.info(f"  Bootstrap CI ({bootstrap_n} iter) for {formula_name}...")
            ci = bootstrap_confusion_matrix_ci(gs_prolonged, formula_prolonged, 
                                                n_bootstrap=bootstrap_n)
            results['formulas'][formula_name]['bootstrap_ci'] = ci
    
    # Compute ranking
    results['ranking'] = compute_formula_ranking(results['formulas'])
    
    return results


def analyze_with_gold_standard_binary(gs_prolonged: np.ndarray,
                                       formula_qtc_values: dict,
                                       sex: pd.Series,
                                       gold_standard_name: str,
                                       bootstrap_n: int = 0) -> dict:
    """
    Analyze formulas against a binary gold standard (SNOMED diagnosis).
    
    [BACKLOG] This function is ready for use with datasets containing clinical 
    QT prolongation diagnoses (e.g., SNOMED codes). Currently no dataset in the 
    pipeline provides these columns. Target columns searched:
      prolonged_qt, label_prolonged_qt, has_prolonged_qt, 
      snomed_prolonged_qt, diagnosis_prolonged_qt
    
    Args:
        bootstrap_n: If > 0, compute bootstrap 95% CI on all metrics
    """
    results = {
        'gold_standard': gold_standard_name,
        'method': 'binary_diagnosis',
        'formulas': {}
    }
    
    results['n_truly_prolonged'] = int(gs_prolonged.sum())
    results['n_truly_normal'] = int((~gs_prolonged).sum())
    
    # Analyze each formula
    for formula_name, qtc_values in formula_qtc_values.items():
        formula_categories = classify_qtc(qtc_values, sex)
        formula_prolonged = (formula_categories == 'prolonged')
        
        metrics = compute_confusion_matrix_metrics(gs_prolonged, formula_prolonged)
        results['formulas'][formula_name] = metrics
        
        # Bootstrap CI (if enabled)
        if bootstrap_n > 0:
            logging.info(f"  Bootstrap CI ({bootstrap_n} iter) for {formula_name}...")
            ci = bootstrap_confusion_matrix_ci(gs_prolonged, formula_prolonged,
                                                n_bootstrap=bootstrap_n)
            results['formulas'][formula_name]['bootstrap_ci'] = ci
    
    # Compute ranking
    results['ranking'] = compute_formula_ranking(results['formulas'])
    
    return results


def compute_formula_ranking(formula_metrics: dict) -> dict:
    """
    Rank formulas by various metrics.
    """
    rankings = {
        'by_sensitivity': [],
        'by_specificity': [],
        'by_f1_score': [],
        'by_youden_j': [],
        'by_mcc': [],
    }
    
    for metric_name in ['sensitivity', 'specificity', 'f1_score', 'youden_j', 'mcc']:
        sorted_formulas = sorted(
            formula_metrics.items(),
            key=lambda x: x[1][metric_name],
            reverse=True
        )
        rankings[f'by_{metric_name}'] = [
            {'formula': f, 'value': round(m[metric_name], 4)} 
            for f, m in sorted_formulas
        ]
    
    return rankings


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(dataset: str, results_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load QTc preparation data and merge with features (demographics, superclass).
    """
    # QTc preparation file
    qtc_file = results_dir / dataset / 'qtc' / f'{dataset}_qtc_preparation.csv'
    
    # Features file (for demographics and superclass)
    features_file = results_dir / dataset / 'preprocess' / f'{dataset}_features.csv'
    
    if not qtc_file.exists():
        logging.warning(f"QTc file not found: {qtc_file}")
        return None
    
    # Load QTc data
    logging.info(f"Loading {qtc_file}")
    df_qtc = pd.read_csv(qtc_file)
    
    # Standardize column names - map from actual file columns to expected names
    col_mapping = {
        # QT interval variations
        'QT_interval_ms': 'QT_ms',
        'qt_interval': 'QT_ms',
        'QT': 'QT_ms',
        # RR interval variations
        'RR_interval_sec': 'RR_s',
        'RR_interval_ms': 'RR_ms',
        'rr_interval': 'RR_ms',
        'RR': 'RR_s',
        # Superclass variations
        'primary_superclass': 'superclass',
        # QTc reference (gold standard)
        'QTc_reference_ms': 'QTc_reference_ms',  # Keep as-is but ensure it's recognized
    }
    df_qtc.rename(columns={k: v for k, v in col_mapping.items() if k in df_qtc.columns}, 
                  inplace=True)
    
    # Ensure RR is in seconds
    if 'RR_ms' in df_qtc.columns and 'RR_s' not in df_qtc.columns:
        df_qtc['RR_s'] = df_qtc['RR_ms'] / 1000
    elif 'RR_s' not in df_qtc.columns and 'rr_s' in df_qtc.columns:
        df_qtc['RR_s'] = df_qtc['rr_s']
    
    # Load features if available
    if features_file.exists():
        logging.info(f"Loading features: {features_file}")
        df_features = pd.read_csv(features_file)
        
        # Find common ID column
        id_cols = ['record_id', 'filename', 'study_id', 'exam_id']
        id_col = None
        for col in id_cols:
            if col in df_qtc.columns and col in df_features.columns:
                id_col = col
                break
        
        if id_col:
            # Merge on ID — select only columns that exist in features
            merge_cols = [id_col]
            for col in ['superclass', 'sex', 'age']:
                if col in df_features.columns:
                    merge_cols.append(col)
            
            if len(merge_cols) > 1:  # At least one feature column besides ID
                df = df_qtc.merge(
                    df_features[merge_cols].drop_duplicates(),
                    on=id_col,
                    how='left'
                )
            else:
                df = df_qtc
                logging.warning(f"No feature columns (superclass/sex/age) found in {features_file}")
        else:
            df = df_qtc
            logging.warning(f"No common ID column found for {dataset}")
    else:
        df = df_qtc
        logging.warning(f"Features file not found: {features_file}")
    
    logging.info(f"Loaded {len(df)} records from {dataset}")
    return df


# =============================================================================
# REPORTING
# =============================================================================

def print_clinical_summary(results: dict, dataset: str):
    """Print a clinical-friendly summary of results."""
    print(f"\n{'='*70}")
    print(f"CLINICAL RECLASSIFICATION ANALYSIS: {dataset.upper()}")
    print(f"{'='*70}")
    
    print(f"\nPopulation: {results.get('population', 'N/A')}")
    print(f"Total records: {results.get('n_total', 'N/A'):,}")
    print(f"NORM records analyzed: {results.get('n_norm', 'N/A'):,}")
    
    if 'error' in results:
        print(f"\nERROR: {results['error']}")
        return
    
    # Category distribution
    print(f"\n--- QTc Category Distribution ---")
    for formula in ['bazett', 'fridericia', 'kepler']:
        dist = results['category_distribution'].get(formula, {})
        total = sum(dist.values())
        if total > 0:
            prol = dist.get('prolonged', 0)
            print(f"{formula.capitalize():12s}: "
                  f"Normal {dist.get('normal', 0)/total*100:5.1f}% | "
                  f"Borderline {dist.get('borderline', 0)/total*100:5.1f}% | "
                  f"Prolonged {prol/total*100:5.1f}% (n={prol:,})")
    
    # HR correlation
    print(f"\n--- HR Correlation (target: 0) ---")
    for formula in ['bazett', 'fridericia', 'kepler']:
        r = results['hr_correlation'].get(formula, float('nan'))
        print(f"{formula.capitalize():12s}: r = {r:+.4f}")
    
    # Reclassification Kepler vs Bazett
    print(f"\n--- Reclassification: Bazett → Kepler ---")
    reclass = results['reclassification']['kepler_vs_bazett']
    matrix = np.array(reclass['matrix'])
    cats = reclass['categories']
    
    print(f"\n{'':15s} {'Kepler →':^45s}")
    print(f"{'Bazett ↓':15s} {'Normal':>12s} {'Borderline':>12s} {'Prolonged':>12s}")
    print("-" * 55)
    for i, cat in enumerate(cats):
        print(f"{cat.capitalize():15s} {matrix[i,0]:>12,d} {matrix[i,1]:>12,d} {matrix[i,2]:>12,d}")
    
    # Clinical impact
    impact = results['clinical_impact_summary']
    print(f"\n--- CLINICAL IMPACT (Kepler vs Bazett) ---")
    print(f"False positives AVOIDED (Bazett prolonged → Kepler normal/borderline): "
          f"{impact['false_positives_avoided']:,}")
    print(f"Potential missed diagnoses (Bazett normal/borderline → Kepler prolonged): "
          f"{impact['potential_missed_diagnoses']:,}")
    print(f"NET BENEFIT: {impact['net_benefit']:+,} patients")
    print(f"Ratio (avoided : missed): {impact['ratio_avoided_to_missed']:.1f} : 1")
    
    # =========================================================================
    # GOLD STANDARD ANALYSIS
    # =========================================================================
    if 'gold_standard_analysis' in results:
        gs_analysis = results['gold_standard_analysis']
        
        # Gold Standard 1: QTc Reference
        if 'qtc_reference' in gs_analysis:
            print(f"\n{'='*70}")
            print("GOLD STANDARD 1: QTc Reference (HR-independent polynomial regression)")
            print(f"{'='*70}")
            
            gs1 = gs_analysis['qtc_reference']
            print(f"\nTruly prolonged (by reference): {gs1['n_truly_prolonged']:,}")
            print(f"Truly normal (by reference): {gs1['n_truly_normal']:,}")
            
            print(f"\n{'Formula':<12s} {'Sens':>8s} {'Spec':>8s} {'PPV':>8s} {'NPV':>8s} "
                  f"{'F1':>8s} {'TP':>8s} {'FP':>8s} {'FN':>8s} {'TN':>8s}")
            print("-" * 90)
            
            for formula in ['bazett', 'fridericia', 'framingham', 'hodges', 'kepler']:
                if formula in gs1['formulas']:
                    m = gs1['formulas'][formula]
                    print(f"{formula.capitalize():<12s} "
                          f"{m['sensitivity']:>7.1%} {m['specificity']:>7.1%} "
                          f"{m['ppv']:>7.1%} {m['npv']:>7.1%} "
                          f"{m['f1_score']:>7.3f} "
                          f"{m['tp']:>8,d} {m['fp']:>8,d} {m['fn']:>8,d} {m['tn']:>8,d}")
            
            # Best formula by each metric
            print(f"\n--- Rankings ---")
            for metric in ['sensitivity', 'specificity', 'f1_score', 'youden_j']:
                ranking = gs1['ranking'][f'by_{metric}']
                best = ranking[0]
                print(f"Best {metric}: {best['formula'].capitalize()} ({best['value']:.4f})")
            
            # Bootstrap CI (if computed)
            has_ci = any('bootstrap_ci' in gs1['formulas'].get(f, {}) 
                        for f in ['bazett', 'kepler'])
            if has_ci:
                print(f"\n--- Bootstrap 95% Confidence Intervals ---")
                print(f"{'Formula':<12s} {'Sens 95% CI':>22s} {'Spec 95% CI':>22s} "
                      f"{'PPV 95% CI':>22s} {'F1 95% CI':>22s}")
                print("-" * 82)
                for formula in ['bazett', 'fridericia', 'framingham', 'hodges', 'kepler']:
                    m = gs1['formulas'].get(formula, {})
                    ci = m.get('bootstrap_ci', {})
                    if ci:
                        def fmt_ci(metric_name):
                            c = ci.get(metric_name, {})
                            if c:
                                return f"{c['point']:.3f} [{c['ci_lower']:.3f}-{c['ci_upper']:.3f}]"
                            return "N/A"
                        print(f"{formula.capitalize():<12s} {fmt_ci('sensitivity'):>22s} "
                              f"{fmt_ci('specificity'):>22s} {fmt_ci('ppv'):>22s} "
                              f"{fmt_ci('f1_score'):>22s}")
        
        # Gold Standard 2: SNOMED Diagnosis
        if 'snomed_diagnosis' in gs_analysis:
            print(f"\n{'='*70}")
            print("GOLD STANDARD 2: SNOMED Clinical Diagnosis")
            print(f"{'='*70}")
            
            gs2 = gs_analysis['snomed_diagnosis']
            print(f"\nClinically diagnosed prolonged QT: {gs2['n_truly_prolonged']:,}")
            print(f"No prolonged QT diagnosis: {gs2['n_truly_normal']:,}")
            
            print(f"\n{'Formula':<12s} {'Sens':>8s} {'Spec':>8s} {'PPV':>8s} {'NPV':>8s} "
                  f"{'F1':>8s} {'TP':>8s} {'FP':>8s} {'FN':>8s} {'TN':>8s}")
            print("-" * 90)
            
            for formula in ['bazett', 'fridericia', 'framingham', 'hodges', 'kepler']:
                if formula in gs2['formulas']:
                    m = gs2['formulas'][formula]
                    print(f"{formula.capitalize():<12s} "
                          f"{m['sensitivity']:>7.1%} {m['specificity']:>7.1%} "
                          f"{m['ppv']:>7.1%} {m['npv']:>7.1%} "
                          f"{m['f1_score']:>7.3f} "
                          f"{m['tp']:>8,d} {m['fp']:>8,d} {m['fn']:>8,d} {m['tn']:>8,d}")
    
    # Stratified by HR
    if 'stratified_by_hr' in results:
        print(f"\n--- Stratified by Heart Rate ---")
        for hr_cat, data in results['stratified_by_hr'].items():
            reclass = data['reclassification']
            matrix = np.array(reclass['matrix'])
            fp_avoided = matrix[2, 0] + matrix[2, 1]
            missed = matrix[0, 2] + matrix[1, 2]
            print(f"{hr_cat.capitalize():12s} (n={data['n']:>6,d}): "
                  f"FP avoided={fp_avoided:>5,d}, Missed={missed:>5,d}, "
                  f"Net={fp_avoided-missed:+6,d}")
    
    # =========================================================================
    # CASCADE STRATEGY RESULTS
    # =========================================================================
    if 'gold_standard_analysis' in results and 'cascade_strategy' in results['gold_standard_analysis']:
        cascade = results['gold_standard_analysis']['cascade_strategy']
        
        print(f"\n{'='*70}")
        print("3-LEVEL TRIAGE: Bazett → Kepler")
        print(f"{'='*70}")
        print("""
┌─────────────────────────────────────────────────────────────────────┐
│  WORKFLOW:                                                          │
│                                                                     │
│  ECG → Bazett → NORMALE? ──────────────────────→ 🟢 GREEN (Dimetti) │
│                    │                                                │
│                PROLUNGATO?                                          │
│                    │                                                │
│                    ▼                                                │
│               Kepler → NORMALE? ───────────→ 🟡 YELLOW (Follow-up)  │
│                    │                                                │
│                PROLUNGATO? ────────────────→ 🔴 RED (Cardiologo)    │
└─────────────────────────────────────────────────────────────────────┘
        """)
        
        # Triage distribution
        dist = cascade['triage_distribution']
        workflow = cascade['workflow_summary']
        
        print(f"{'Livello':<10} {'Totale':>10} {'%':>8} {'Veri Prol.':>12} {'Veri Norm.':>12} {'Purità':>10}")
        print("-"*65)
        print(f"{'🟢 GREEN':<10} {dist['GREEN']['total']:>10,} {workflow['green_pct']:>7.1f}% {dist['GREEN']['truly_prolonged']:>12,} {dist['GREEN']['truly_normal']:>12,} {'-':>10}")
        print(f"{'🟡 YELLOW':<10} {dist['YELLOW']['total']:>10,} {workflow['yellow_pct']:>7.1f}% {dist['YELLOW']['truly_prolonged']:>12,} {dist['YELLOW']['truly_normal']:>12,} {dist['YELLOW']['pct_truly_prolonged']:>9.1f}%")
        print(f"{'🔴 RED':<10} {dist['RED']['total']:>10,} {workflow['red_pct']:>7.1f}% {dist['RED']['truly_prolonged']:>12,} {dist['RED']['truly_normal']:>12,} {dist['RED']['pct_truly_prolonged']:>9.1f}%")
        
        # Comparison with single formulas
        print(f"\n--- Confronto con Formule Singole ---")
        comp = cascade['comparison']
        
        print(f"\n{'Strategia':<25} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'Referral':>10} {'FP':>8} {'FN':>8}")
        print("-"*80)
        print(f"{'Bazett solo':<25} {comp['bazett_only']['sensitivity']:>7.1%} {comp['bazett_only']['specificity']:>7.1%} "
              f"{comp['bazett_only']['ppv']:>7.1%} {comp['bazett_only']['referrals']:>10,} "
              f"{comp['bazett_only']['fp']:>8,} {comp['bazett_only']['fn']:>8,}")
        print(f"{'Kepler solo':<25} {comp['kepler_only']['sensitivity']:>7.1%} {comp['kepler_only']['specificity']:>7.1%} "
              f"{comp['kepler_only']['ppv']:>7.1%} {comp['kepler_only']['referrals']:>10,} "
              f"{comp['kepler_only']['fp']:>8,} {comp['kepler_only']['fn']:>8,}")
        print(f"{'Triage (solo RED)':<25} {comp['triage_red_only']['sensitivity']:>7.1%} {comp['triage_red_only']['specificity']:>7.1%} "
              f"{comp['triage_red_only']['ppv']:>7.1%} {comp['triage_red_only']['urgent_referrals']:>10,} "
              f"{comp['triage_red_only']['fp']:>8,} {comp['triage_red_only']['fn']:>8,}")
        
        # Trade-off analysis
        tradeoff = cascade['tradeoff']
        print(f"\n--- Analisi Trade-off (Triage vs Bazett) ---")
        print(f"Referral urgenti Bazett: {tradeoff['urgent_referrals_bazett']:,}")
        print(f"Referral urgenti Triage (RED): {tradeoff['urgent_referrals_triage']:,}")
        print(f"Riduzione referral urgenti: {tradeoff['urgent_referral_reduction']:,} ({tradeoff['urgent_referral_reduction_pct']:.1f}%)")
        print(f"\nCategoria YELLOW (follow-up):")
        print(f"  Totale casi in YELLOW: {tradeoff['yellow_cases_need_review']:,}")
        print(f"  - Veri prolungati (da seguire): {tradeoff['true_cases_in_yellow']:,}")
        print(f"  - Falsi allarmi Bazett (filtrati): {tradeoff['false_alarms_in_yellow']:,}")
        print(f"\nCasi persi (GREEN ma veri prolungati): {tradeoff['true_misses_green']:,}")
        print(f"\nSensibilità se YELLOW seguito: {tradeoff['sensitivity_if_yellow_followed_up']:.1%}")
        print(f"Sensibilità se YELLOW dimesso: {tradeoff['sensitivity_if_yellow_dismissed']:.1%}")


def generate_report(all_results: dict, output_dir: Path) -> Path:
    """Generate comprehensive JSON and summary reports."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON report
    json_file = output_dir / f'clinical_reclassification_report_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Summary CSV
    summary_data = []
    for dataset, results in all_results['datasets'].items():
        if 'error' in results:
            continue
        
        impact = results.get('clinical_impact_summary', {})
        summary_data.append({
            'dataset': dataset,
            'n_records': results.get('n_norm', 0),
            'r_bazett': results['hr_correlation'].get('bazett', float('nan')),
            'r_kepler': results['hr_correlation'].get('kepler', float('nan')),
            'prolonged_bazett': results['category_distribution'].get('bazett', {}).get('prolonged', 0),
            'prolonged_kepler': results['category_distribution'].get('kepler', {}).get('prolonged', 0),
            'false_positives_avoided': impact.get('false_positives_avoided', 0),
            'potential_missed': impact.get('potential_missed_diagnoses', 0),
            'net_benefit': impact.get('net_benefit', 0),
            'ratio_avoided_missed': impact.get('ratio_avoided_to_missed', 0),
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / f'clinical_reclassification_summary_{timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"Summary saved to {summary_file}")
    
    return json_file


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze clinical reclassification impact of QTc formulas'
    )
    parser.add_argument(
        '--results-dir', '-r',
        type=Path,
        default=Path('results'),
        help='Path to results directory'
    )
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        default=DATASETS,
        help='Datasets to analyze'
    )
    parser.add_argument(
        '--all-patients',
        action='store_true',
        help='Analyze all patients, not just NORM superclass'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=None,
        help='Output directory (default: results/clinical_analysis)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--kepler-k',
        type=float,
        default=KEPLER_K,
        help=f'Kepler formula k coefficient in ms·s (default: {KEPLER_K})'
    )
    parser.add_argument(
        '--kepler-c',
        type=float,
        default=KEPLER_C,
        help=f'Kepler formula c constant in ms (default: {KEPLER_C})'
    )
    parser.add_argument(
        '--bootstrap-n',
        type=int,
        default=BOOTSTRAP_N_DEFAULT,
        help='Number of bootstrap iterations for 95%% CI on gold standard metrics '
             f'(0=disabled, recommended: 2000; default: {BOOTSTRAP_N_DEFAULT})'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=MIN_SAMPLES_STRATIFICATION,
        help=f'Minimum samples for stratified sub-analyses (default: {MIN_SAMPLES_STRATIFICATION})'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Output directory
    output_dir = args.output_dir or args.results_dir / 'clinical_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply CLI Kepler coefficients as active values
    active_k = args.kepler_k
    active_c = args.kepler_c
    
    if active_k != KEPLER_K or active_c != KEPLER_C:
        logging.info(f"Using custom Kepler coefficients: k={active_k}, c={active_c}")
    
    # Process each dataset
    all_results = {
        'generated': datetime.now().isoformat(),
        'filter_norm': not args.all_patients,
        'kepler_formula': f'QTc = QT + {active_k}/RR + ({active_c})',
        'kepler_k': active_k,
        'kepler_c': active_c,
        'bootstrap_n': args.bootstrap_n,
        'min_samples': args.min_samples,
        'datasets': {},
        'pooled': {},
    }
    
    pooled_data = []
    
    for dataset in args.datasets:
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing {dataset}")
        logging.info(f"{'='*50}")
        
        df = load_dataset(dataset, args.results_dir)
        if df is None:
            all_results['datasets'][dataset] = {'error': 'Failed to load data'}
            continue
        
        # Run analysis
        results = analyze_clinical_impact(
            df, 
            filter_norm=not args.all_patients,
            kepler_k=active_k,
            kepler_c=active_c,
            min_samples=args.min_samples,
            bootstrap_n=args.bootstrap_n,
        )
        results['dataset'] = dataset
        all_results['datasets'][dataset] = results
        
        # Print summary
        print_clinical_summary(results, dataset)
        
        # Collect for pooled analysis
        if 'error' not in results and 'QT_ms' in df.columns and 'RR_s' in df.columns:
            df['source_dataset'] = dataset
            pooled_data.append(df)
    
    # Pooled analysis across all datasets
    if len(pooled_data) > 1:
        logging.info(f"\n{'='*50}")
        logging.info("POOLED ANALYSIS (All Datasets)")
        logging.info(f"{'='*50}")
        
        df_pooled = pd.concat(pooled_data, ignore_index=True)
        pooled_results = analyze_clinical_impact(
            df_pooled,
            filter_norm=not args.all_patients,
            kepler_k=active_k,
            kepler_c=active_c,
            min_samples=args.min_samples,
            bootstrap_n=args.bootstrap_n,
        )
        all_results['pooled'] = pooled_results
        print_clinical_summary(pooled_results, 'POOLED (All Datasets)')
    
    # Generate reports
    report_file = generate_report(all_results, output_dir)
    logging.info(f"\nFull report saved to: {report_file}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("OVERALL CLINICAL IMPACT SUMMARY")
    print(f"{'='*70}")
    
    total_fp_avoided = 0
    total_missed = 0
    total_records = 0
    
    for dataset, results in all_results['datasets'].items():
        if 'error' not in results:
            impact = results.get('clinical_impact_summary', {})
            total_fp_avoided += impact.get('false_positives_avoided', 0)
            total_missed += impact.get('potential_missed_diagnoses', 0)
            total_records += results.get('n_norm', 0)
    
    print(f"\nTotal NORM patients analyzed: {total_records:,}")
    print(f"Total false positives AVOIDED by Kepler: {total_fp_avoided:,}")
    print(f"Total potential missed diagnoses: {total_missed:,}")
    print(f"OVERALL NET BENEFIT: {total_fp_avoided - total_missed:+,} patients")
    if total_missed > 0:
        print(f"OVERALL RATIO: {total_fp_avoided / total_missed:.1f} : 1 "
              f"(for every missed diagnosis, {total_fp_avoided / total_missed:.1f} false positives avoided)")
    
    # =========================================================================
    # 3-LEVEL TRIAGE AGGREGATE SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("3-LEVEL TRIAGE AGGREGATE SUMMARY")
    print(f"{'='*70}")
    
    # Aggregate triage metrics
    triage_totals = {
        'GREEN': {'total': 0, 'truly_prolonged': 0, 'truly_normal': 0},
        'YELLOW': {'total': 0, 'truly_prolonged': 0, 'truly_normal': 0},
        'RED': {'total': 0, 'truly_prolonged': 0, 'truly_normal': 0},
    }
    bazett_totals = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    kepler_totals = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    
    has_triage = False
    for dataset, results in all_results['datasets'].items():
        if 'error' in results:
            continue
        gs_analysis = results.get('gold_standard_analysis', {})
        if 'cascade_strategy' not in gs_analysis:
            continue
        
        has_triage = True
        cascade = gs_analysis['cascade_strategy']
        
        # Aggregate triage distribution
        for level in ['GREEN', 'YELLOW', 'RED']:
            dist = cascade['triage_distribution'][level]
            triage_totals[level]['total'] += dist['total']
            triage_totals[level]['truly_prolonged'] += dist['truly_prolonged']
            triage_totals[level]['truly_normal'] += dist['truly_normal']
        
        # Aggregate Bazett and Kepler metrics
        for key in ['tp', 'fp', 'fn', 'tn']:
            bazett_totals[key] += cascade['bazett_only'].get(key, 0)
            kepler_totals[key] += cascade['kepler_only'].get(key, 0)
    
    if has_triage:
        total_patients = sum(t['total'] for t in triage_totals.values())
        total_truly_prolonged = sum(t['truly_prolonged'] for t in triage_totals.values())
        
        print(f"\nTotale pazienti: {total_patients:,}")
        print(f"Veri QT prolungati: {total_truly_prolonged:,}")
        
        print(f"\n{'Livello':<12} {'Totale':>12} {'%':>8} {'Veri Prol.':>12} {'Veri Norm.':>12} {'Purità':>10}")
        print("-"*70)
        for level, emoji in [('GREEN', '🟢'), ('YELLOW', '🟡'), ('RED', '🔴')]:
            t = triage_totals[level]
            pct = t['total'] / total_patients * 100 if total_patients > 0 else 0
            purity = t['truly_prolonged'] / t['total'] * 100 if t['total'] > 0 else 0
            print(f"{emoji} {level:<10} {t['total']:>12,} {pct:>7.1f}% {t['truly_prolonged']:>12,} {t['truly_normal']:>12,} {purity:>9.1f}%")
        
        # Calculate aggregate metrics
        def calc_sens_spec_ppv(tp, fp, fn, tn):
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            return sens, spec, ppv
        
        # Bazett metrics
        baz_sens, baz_spec, baz_ppv = calc_sens_spec_ppv(
            bazett_totals['tp'], bazett_totals['fp'], bazett_totals['fn'], bazett_totals['tn'])
        
        # Kepler metrics  
        kep_sens, kep_spec, kep_ppv = calc_sens_spec_ppv(
            kepler_totals['tp'], kepler_totals['fp'], kepler_totals['fn'], kepler_totals['tn'])
        
        # Triage RED only metrics
        red = triage_totals['RED']
        green = triage_totals['GREEN']
        red_tp = red['truly_prolonged']
        red_fp = red['truly_normal']
        red_fn = green['truly_prolonged']  # Cases dismissed in GREEN
        red_tn = green['truly_normal']
        triage_sens, triage_spec, triage_ppv = calc_sens_spec_ppv(red_tp, red_fp, red_fn, red_tn)
        
        # Triage RED+YELLOW metrics (if YELLOW is followed up)
        yellow = triage_totals['YELLOW']
        ry_tp = red['truly_prolonged'] + yellow['truly_prolonged']
        ry_fp = red['truly_normal'] + yellow['truly_normal']
        ry_fn = green['truly_prolonged']
        ry_tn = green['truly_normal']
        triage_ry_sens, _, _ = calc_sens_spec_ppv(ry_tp, ry_fp, ry_fn, ry_tn)
        
        print(f"\n--- Confronto Strategie ---")
        print(f"\n{'Strategia':<30} {'Sens':>10} {'Spec':>10} {'PPV':>10} {'Referral':>12}")
        print("-"*75)
        print(f"{'Bazett solo':<30} {baz_sens:>9.1%} {baz_spec:>9.1%} {baz_ppv:>9.1%} {bazett_totals['tp']+bazett_totals['fp']:>12,}")
        print(f"{'Kepler solo':<30} {kep_sens:>9.1%} {kep_spec:>9.1%} {kep_ppv:>9.1%} {kepler_totals['tp']+kepler_totals['fp']:>12,}")
        print(f"{'Triage (solo RED urgente)':<30} {triage_sens:>9.1%} {triage_spec:>9.1%} {triage_ppv:>9.1%} {red['total']:>12,}")
        print(f"{'Triage (RED + YELLOW f/u)':<30} {triage_ry_sens:>9.1%} {'-':>10} {'-':>10} {red['total']+yellow['total']:>12,}")
        
        # Key insights
        print(f"\n--- INSIGHT CLINICO ---")
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│  RISULTATO DEL TRIAGE A 3 LIVELLI:                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  🟢 GREEN ({triage_totals['GREEN']['total']:,} pazienti, {triage_totals['GREEN']['total']/total_patients*100:.1f}%):                                        │
│     → Dimessi in sicurezza                                                  │
│     → Solo {triage_totals['GREEN']['truly_prolonged']:,} veri prolungati persi ({triage_totals['GREEN']['truly_prolonged']/total_truly_prolonged*100:.2f}% del totale)                  │
│                                                                             │
│  🟡 YELLOW ({triage_totals['YELLOW']['total']:,} pazienti, {triage_totals['YELLOW']['total']/total_patients*100:.1f}%):                                      │
│     → Richiedono giudizio clinico / follow-up                               │
│     → {triage_totals['YELLOW']['truly_prolonged']:,} veri prolungati ({triage_totals['YELLOW']['truly_prolonged']/triage_totals['YELLOW']['total']*100:.1f}% purità) - da non perdere!              │
│     → {triage_totals['YELLOW']['truly_normal']:,} falsi allarmi Bazett filtrati                              │
│                                                                             │
│  🔴 RED ({triage_totals['RED']['total']:,} pazienti, {triage_totals['RED']['total']/total_patients*100:.1f}%):                                           │
│     → Referral urgente cardiologia                                          │
│     → {triage_totals['RED']['truly_prolonged']:,} veri prolungati ({triage_totals['RED']['truly_prolonged']/triage_totals['RED']['total']*100:.1f}% purità = PPV)                      │
│     → Solo {triage_totals['RED']['truly_normal']:,} falsi allarmi                                            │
└─────────────────────────────────────────────────────────────────────────────┘

CONFRONTO WORKLOAD:
  • Bazett solo: {bazett_totals['tp']+bazett_totals['fp']:,} referral (tutti urgenti)
  • Triage RED:  {red['total']:,} referral urgenti (-{(1-red['total']/(bazett_totals['tp']+bazett_totals['fp']))*100:.0f}%)
  • Triage YELLOW: {yellow['total']:,} follow-up (non urgenti)
  
SE YELLOW VIENE SEGUITO:
  → Sensibilità = {triage_ry_sens:.1%} (quasi come Bazett {baz_sens:.1%})
  → Ma con {(1-(red['total']+yellow['total'])/(bazett_totals['tp']+bazett_totals['fp']))*100:.0f}% meno carico urgente

SE YELLOW VIENE DIMESSO:
  → Sensibilità = {triage_sens:.1%}
  → Perdi {yellow['truly_prolonged']:,} casi in più
        """)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
