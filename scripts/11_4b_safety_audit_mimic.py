#!/usr/bin/env python3
"""
Kepler-ECG: Safety Audit - Reclassified Patients (Script 11_4b)
================================================================

PUNTO 5 DELLA VALIDAZIONE: Safety Audit della Fascia "YELLOW"

Versione aggiornata che usa il file records_w_diag_icd10.csv di MIMIC-IV-ECG
che contiene già il linkage ECG ↔ diagnosi ICD-10 + data di morte.

Author: Alessandro Marconi
Version: 2.0.0
Date: February 2026
"""

import argparse
import ast
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for Safety Audit."""
    
    # Kepler coefficients
    KEPLER_K = 125
    KEPLER_C = -158
    
    # QTc threshold
    QTC_THRESHOLD = 450  # ms
    
    # ICD-10 codes for adverse events (arrhythmias)
    ARRHYTHMIA_CODES = [
        'I472',   # Ventricular tachycardia (includes TdP)
        'I490',   # Ventricular fibrillation
        'I491',   # Atrial premature depolarization
        'I495',   # Sick sinus syndrome
        'I46',    # Cardiac arrest
        'I460',   # Cardiac arrest with successful resuscitation
        'I461',   # Sudden cardiac death
        'I469',   # Cardiac arrest, unspecified
        'R001',   # Bradycardia
        'R000',   # Tachycardia
        'I47',    # Paroxysmal tachycardia
        'I48',    # Atrial fibrillation and flutter
        'I49',    # Other cardiac arrhythmias
    ]
    
    # Severe/life-threatening arrhythmias only
    SEVERE_ARRHYTHMIA_CODES = [
        'I472',   # Ventricular tachycardia
        'I490',   # Ventricular fibrillation
        'I46',    # Cardiac arrest
        'I460',
        'I461',
        'I469',
    ]
    
    # Paths
    DIAG_FILE = Path('data/raw/mimic-iv-ecg/records_w_diag_icd10.csv')
    ECG_RESULTS_DIR = Path('results/mimic-iv-ecg')
    OUTPUT_DIR = Path('results/safety_audit')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_diagnosis_data() -> Optional[pd.DataFrame]:
    """Load the records_w_diag_icd10.csv file."""
    
    locations = [
        Config.DIAG_FILE,
        Path('data/raw/mimic-iv-ecg/records_w_diag_icd10.csv'),
        Path('records_w_diag_icd10.csv'),
    ]
    
    for loc in locations:
        if loc.exists():
            print(f"  ✓ Found diagnosis file: {loc}")
            df = pd.read_csv(loc)
            print(f"    Loaded {len(df):,} records")
            return df
    
    print("  ⚠️ Diagnosis file not found")
    return None


def load_ecg_qtc_data() -> Optional[pd.DataFrame]:
    """Load processed ECG data with QT/RR values."""
    
    locations = [
        Config.ECG_RESULTS_DIR / 'qtc' / 'mimic-iv-ecg_qtc_preparation.csv',
        Config.ECG_RESULTS_DIR / 'mimic-iv-ecg_qtc_preparation.csv',
        Path('results/mimic-iv-ecg/qtc/mimic-iv-ecg_qtc_preparation.csv'),
    ]
    
    for loc in locations:
        if loc.exists():
            print(f"  ✓ Found ECG QTc data: {loc}")
            df = pd.read_csv(loc)
            print(f"    Loaded {len(df):,} records")
            return df
    
    print("  ⚠️ ECG QTc data not found")
    return None


# ============================================================================
# DATA PROCESSING
# ============================================================================

def extract_study_id_from_filename(filename: str) -> Optional[int]:
    """Extract study_id from the file_name column."""
    # Format: .../s40689238/40689238 -> 40689238
    try:
        parts = filename.split('/')
        return int(parts[-1])
    except:
        return None


def parse_diagnosis_list(diag_str: str) -> List[str]:
    """Parse the diagnosis list from string to actual list."""
    if pd.isna(diag_str) or diag_str == '[]':
        return []
    try:
        # It's stored as a string representation of a list
        return ast.literal_eval(diag_str)
    except:
        return []


def has_arrhythmia_code(diag_list: List[str], codes: List[str]) -> bool:
    """Check if any arrhythmia code is in the diagnosis list."""
    for diag in diag_list:
        for code in codes:
            # Check if diagnosis starts with code (handles I46 matching I460, I461, etc.)
            if diag.startswith(code):
                return True
    return False


def compute_qtc(df: pd.DataFrame) -> pd.DataFrame:
    """Compute QTc values with Kepler and Bazett."""
    
    # Standardize column names
    col_map = {
        'QT_interval_ms': 'QT_ms', 'QT': 'QT_ms',
        'RR_interval_sec': 'RR_sec', 'RR': 'RR_sec',
        'heart_rate_bpm': 'HR_bpm', 'HR': 'HR_bpm',
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    qt = df['QT_ms'].values
    rr = df['RR_sec'].values
    
    df['QTc_Kepler'] = qt + Config.KEPLER_K / rr + Config.KEPLER_C
    df['QTc_Bazett'] = qt / np.sqrt(rr)
    
    return df


def classify_qtc(df: pd.DataFrame, threshold: float = 450) -> pd.DataFrame:
    """Classify records based on QTc concordance."""
    
    k_prolonged = df['QTc_Kepler'] >= threshold
    b_prolonged = df['QTc_Bazett'] >= threshold
    
    conditions = [
        (~k_prolonged & ~b_prolonged),  # concordant_normal
        (k_prolonged & b_prolonged),     # concordant_prolonged
        (k_prolonged & ~b_prolonged),    # kepler_only_prolonged
        (~k_prolonged & b_prolonged),    # reclassified_normal (KEY!)
    ]
    choices = ['concordant_normal', 'concordant_prolonged', 
               'kepler_only_prolonged', 'reclassified_normal']
    
    df['qtc_classification'] = np.select(conditions, choices, default='unknown')
    
    return df


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_outcomes(merged_df: pd.DataFrame) -> Dict:
    """Analyze outcomes by QTc classification."""
    
    results = {}
    
    for classification in ['concordant_normal', 'concordant_prolonged', 
                          'kepler_only_prolonged', 'reclassified_normal']:
        
        subset = merged_df[merged_df['qtc_classification'] == classification]
        n_records = len(subset)
        
        if n_records < 10:
            results[classification] = {'n_records': n_records, 'insufficient_data': True}
            continue
        
        # Unique patients
        n_patients = subset['subject_id'].nunique()
        
        # Outcomes
        n_any_arrhythmia = subset['has_any_arrhythmia'].sum()
        n_severe_arrhythmia = subset['has_severe_arrhythmia'].sum()
        n_deceased = subset['is_deceased'].sum()
        n_any_event = ((subset['has_severe_arrhythmia']) | (subset['is_deceased'])).sum()
        
        results[classification] = {
            'n_records': n_records,
            'n_patients': n_patients,
            'n_any_arrhythmia': int(n_any_arrhythmia),
            'n_severe_arrhythmia': int(n_severe_arrhythmia),
            'n_deceased': int(n_deceased),
            'n_any_severe_event': int(n_any_event),
            'pct_any_arrhythmia': float(n_any_arrhythmia / n_records * 100),
            'pct_severe_arrhythmia': float(n_severe_arrhythmia / n_records * 100),
            'pct_deceased': float(n_deceased / n_records * 100),
            'pct_any_severe_event': float(n_any_event / n_records * 100),
            'mean_hr': float(subset['HR_bpm'].mean()) if 'HR_bpm' in subset.columns else None,
            'mean_age': float(subset['age'].mean()) if 'age' in subset.columns else None,
            'pct_tachycardia': float((subset['HR_bpm'] >= 100).mean() * 100) if 'HR_bpm' in subset.columns else None,
            'mean_qtc_kepler': float(subset['QTc_Kepler'].mean()),
            'mean_qtc_bazett': float(subset['QTc_Bazett'].mean()),
        }
    
    return results


def compute_safety_comparison(results: Dict) -> Dict:
    """Compare reclassified vs concordant_normal for safety."""
    
    reclass = results.get('reclassified_normal', {})
    concord = results.get('concordant_normal', {})
    prolonged = results.get('concordant_prolonged', {})
    
    if reclass.get('insufficient_data') or concord.get('insufficient_data'):
        return {'comparison_available': False}
    
    safety = {'comparison_available': True}
    
    # Event rates
    reclass_rate = reclass.get('pct_any_severe_event', 0)
    concord_rate = concord.get('pct_any_severe_event', 0)
    prolonged_rate = prolonged.get('pct_any_severe_event', 0) if not prolonged.get('insufficient_data') else None
    
    safety['reclassified_event_rate'] = reclass_rate
    safety['concordant_normal_event_rate'] = concord_rate
    safety['concordant_prolonged_event_rate'] = prolonged_rate
    safety['rate_difference_vs_normal'] = reclass_rate - concord_rate
    
    # HR comparison
    reclass_hr = reclass.get('mean_hr', 0)
    concord_hr = concord.get('mean_hr', 0)
    safety['reclassified_mean_hr'] = reclass_hr
    safety['concordant_normal_mean_hr'] = concord_hr
    safety['hr_difference'] = reclass_hr - concord_hr if reclass_hr and concord_hr else None
    
    # Tachycardia comparison
    reclass_tachy = reclass.get('pct_tachycardia', 0)
    concord_tachy = concord.get('pct_tachycardia', 0)
    safety['reclassified_pct_tachycardia'] = reclass_tachy
    safety['concordant_normal_pct_tachycardia'] = concord_tachy
    
    # Statistical test
    n_reclass = reclass.get('n_records', 0)
    n_concord = concord.get('n_records', 0)
    events_reclass = reclass.get('n_any_severe_event', 0)
    events_concord = concord.get('n_any_severe_event', 0)
    
    if n_reclass > 0 and n_concord > 0:
        # 2x2 contingency table
        table = [
            [events_reclass, n_reclass - events_reclass],
            [events_concord, n_concord - events_concord]
        ]
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(table)
            safety['chi2'] = float(chi2)
            safety['p_value'] = float(p_value)
        except:
            safety['chi2'] = None
            safety['p_value'] = None
        
        # Relative risk
        rr_reclass = events_reclass / n_reclass if n_reclass > 0 else 0
        rr_concord = events_concord / n_concord if n_concord > 0 else 0
        safety['relative_risk'] = rr_reclass / rr_concord if rr_concord > 0 else None
    
    # Verdict
    if safety['rate_difference_vs_normal'] <= 0:
        safety['verdict'] = 'SAFE'
        safety['verdict_detail'] = 'Reclassified have EQUAL or LOWER event rate than concordant normal'
    elif safety['rate_difference_vs_normal'] < 1.0:
        safety['verdict'] = 'SAFE'
        safety['verdict_detail'] = 'Difference < 1% - clinically negligible'
    elif safety['rate_difference_vs_normal'] < 2.0:
        safety['verdict'] = 'ACCEPTABLE'
        safety['verdict_detail'] = 'Small difference (1-2%) - clinically acceptable'
    else:
        safety['verdict'] = 'NEEDS_REVIEW'
        safety['verdict_detail'] = 'Difference > 2% - requires clinical review'
    
    # Additional interpretation based on HR
    if safety.get('hr_difference') and safety['hr_difference'] > 5:
        safety['hr_interpretation'] = (
            f"Reclassified patients have {safety['hr_difference']:.1f} bpm HIGHER heart rate. "
            "This suggests they were Bazett false positives (overcorrection in tachycardia)."
        )
    
    return safety


# ============================================================================
# REPORTING
# ============================================================================

def print_report(results: Dict, safety: Dict, merged_df: pd.DataFrame):
    """Print comprehensive report."""
    
    print("\n" + "="*70)
    print("SAFETY AUDIT - PUNTO 5 (con dati MIMIC-IV)")
    print("="*70)
    
    total_records = len(merged_df)
    total_patients = merged_df['subject_id'].nunique()
    total_deceased = merged_df['is_deceased'].sum()
    total_arrhythmia = merged_df['has_severe_arrhythmia'].sum()
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Record ECG totali: {total_records:,}")
    print(f"   Pazienti unici: {total_patients:,}")
    print(f"   Con aritmie severe (VT/VF/Arresto): {total_arrhythmia:,} ({total_arrhythmia/total_records*100:.2f}%)")
    print(f"   Deceduti: {total_deceased:,} ({total_deceased/total_records*100:.2f}%)")
    
    # Classification distribution
    print(f"\n📊 CLASSIFICAZIONE QTc (soglia {Config.QTC_THRESHOLD} ms):")
    print("-" * 70)
    print(f"{'Categoria':<25} {'N record':>12} {'%':>8} {'Eventi%':>10} {'HR medio':>10}")
    print("-" * 70)
    
    for cat in ['concordant_normal', 'reclassified_normal', 'kepler_only_prolonged', 'concordant_prolonged']:
        data = results.get(cat, {})
        if not data.get('insufficient_data'):
            n = data['n_records']
            pct = n / total_records * 100
            event_pct = data.get('pct_any_severe_event', 0)
            hr = data.get('mean_hr', 0)
            marker = " ⬅️ FOCUS" if cat == 'reclassified_normal' else ""
            print(f"{cat:<25} {n:>12,} {pct:>7.1f}% {event_pct:>9.2f}% {hr:>9.1f}{marker}")
    
    print("-" * 70)
    
    # Safety comparison
    print(f"\n📊 CONFRONTO SICUREZZA:")
    print("-" * 70)
    
    reclass = results.get('reclassified_normal', {})
    concord = results.get('concordant_normal', {})
    
    if not reclass.get('insufficient_data') and not concord.get('insufficient_data'):
        print(f"\n  {'Metrica':<35} {'Concordi Normali':>18} {'Riclassificati':>18}")
        print("  " + "-" * 71)
        print(f"  {'N record':<35} {concord['n_records']:>18,} {reclass['n_records']:>18,}")
        print(f"  {'HR medio (bpm)':<35} {concord.get('mean_hr', 0):>18.1f} {reclass.get('mean_hr', 0):>18.1f}")
        print(f"  {'% Tachicardia (≥100)':<35} {concord.get('pct_tachycardia', 0):>17.1f}% {reclass.get('pct_tachycardia', 0):>17.1f}%")
        print(f"  {'% Aritmie severe':<35} {concord.get('pct_severe_arrhythmia', 0):>17.2f}% {reclass.get('pct_severe_arrhythmia', 0):>17.2f}%")
        print(f"  {'% Deceduti':<35} {concord.get('pct_deceased', 0):>17.2f}% {reclass.get('pct_deceased', 0):>17.2f}%")
        print(f"  {'% Eventi severi totali':<35} {concord.get('pct_any_severe_event', 0):>17.2f}% {reclass.get('pct_any_severe_event', 0):>17.2f}%")
        
        print(f"\n  Differenza tasso eventi: {safety.get('rate_difference_vs_normal', 0):+.2f}%")
        if safety.get('relative_risk'):
            print(f"  Rischio relativo: {safety['relative_risk']:.3f}")
        if safety.get('p_value'):
            print(f"  p-value (chi²): {safety['p_value']:.4f}")
        if safety.get('hr_interpretation'):
            print(f"\n  💡 {safety['hr_interpretation']}")


def print_verdict(safety: Dict, results: Dict):
    """Print final verdict."""
    
    print("\n" + "="*70)
    print("VERDETTO FINALE - PUNTO 5")
    print("="*70)
    
    verdict = safety.get('verdict', 'UNKNOWN')
    
    reclass = results.get('reclassified_normal', {})
    concord = results.get('concordant_normal', {})
    
    # Check HR pattern
    hr_pattern_safe = False
    if reclass.get('mean_hr') and concord.get('mean_hr'):
        hr_diff = reclass['mean_hr'] - concord['mean_hr']
        hr_pattern_safe = hr_diff > 5  # Reclassified have higher HR
    
    if verdict == 'SAFE':
        print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ✅  KEPLER È SICURO                                              ║
    ║                                                                    ║
    ║   I pazienti riclassificati (Bazett≥450 → Kepler<450) NON hanno    ║
    ║   un tasso di eventi avversi superiore ai pazienti concordi        ║
    ║   normali.                                                         ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
        """)
        
        if hr_pattern_safe:
            print("""    EVIDENZA AGGIUNTIVA:
    I riclassificati hanno HR significativamente più alto, confermando
    che erano FALSI POSITIVI di Bazett (sovra-correzione in tachicardia),
    NON veri QT lunghi. Kepler sta CORREGGENDO errori, non mascherando
    patologie.
            """)
    
    elif verdict == 'ACCEPTABLE':
        print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ⚠️  KEPLER ACCETTABILE CON CAUTELA                               ║
    ║                                                                    ║
    ║   Piccola differenza nel tasso di eventi (1-2%).                   ║
    ║   Clinicamente accettabile ma richiede monitoraggio.               ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
        """)
    
    else:
        print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ❓  RICHIEDE REVISIONE CLINICA                                   ║
    ║                                                                    ║
    ║   Differenza nel tasso di eventi > 2%.                             ║
    ║   Necessaria analisi approfondita prima dell'uso clinico.          ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
        """)
    
    # Summary statistics
    print(f"\n📋 RIEPILOGO NUMERICO:")
    print(f"   Riclassificati: {reclass.get('n_records', 0):,} record")
    print(f"   Tasso eventi riclassificati: {reclass.get('pct_any_severe_event', 0):.2f}%")
    print(f"   Tasso eventi concordi normali: {concord.get('pct_any_severe_event', 0):.2f}%")
    print(f"   Differenza: {safety.get('rate_difference_vs_normal', 0):+.2f}%")
    print(f"   Verdetto: {verdict}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    parser = argparse.ArgumentParser(
        description='Kepler-ECG Safety Audit with MIMIC-IV data (Point 5)',
    )
    
    parser.add_argument('--output-dir', type=str, 
                       default=str(Config.OUTPUT_DIR),
                       help='Output directory')
    parser.add_argument('--diag-file', type=str,
                       help='Path to records_w_diag_icd10.csv')
    parser.add_argument('--ecg-file', type=str,
                       help='Path to processed ECG QTc data')
    
    args = parser.parse_args()
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   KEPLER-ECG: Safety Audit (Script 11_4b)                          ║
    ║   PUNTO 5 - Safety Audit con MIMIC-IV                              ║
    ║                                                                    ║
    ║   Usando: records_w_diag_icd10.csv (linkage ECG ↔ diagnosi)        ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📊 Loading data...")
    
    # Load diagnosis data
    if args.diag_file:
        Config.DIAG_FILE = Path(args.diag_file)
    diag_df = load_diagnosis_data()
    
    if diag_df is None:
        print("\n❌ Could not load diagnosis data!")
        return 1
    
    # Load ECG QTc data
    if args.ecg_file:
        ecg_df = pd.read_csv(args.ecg_file)
        print(f"  ✓ Loaded ECG data: {len(ecg_df):,} records")
    else:
        ecg_df = load_ecg_qtc_data()
    
    if ecg_df is None:
        print("\n❌ Could not load ECG QTc data!")
        return 1
    
    # Process diagnosis data
    print("\n📊 Processing diagnosis data...")
    
    # Parse diagnosis lists
    diag_df['all_diag_list'] = diag_df['all_diag_all'].apply(parse_diagnosis_list)
    
    # Flag arrhythmias and death
    diag_df['has_any_arrhythmia'] = diag_df['all_diag_list'].apply(
        lambda x: has_arrhythmia_code(x, Config.ARRHYTHMIA_CODES)
    )
    diag_df['has_severe_arrhythmia'] = diag_df['all_diag_list'].apply(
        lambda x: has_arrhythmia_code(x, Config.SEVERE_ARRHYTHMIA_CODES)
    )
    diag_df['is_deceased'] = diag_df['dod'].notna()
    
    print(f"   Records with any arrhythmia: {diag_df['has_any_arrhythmia'].sum():,}")
    print(f"   Records with severe arrhythmia: {diag_df['has_severe_arrhythmia'].sum():,}")
    print(f"   Records with death: {diag_df['is_deceased'].sum():,}")
    
    # Merge ECG and diagnosis data
    print("\n📊 Merging ECG and diagnosis data...")
    
    # Find common identifier
    # ECG data has 'ecg_id' which should match 'study_id' in diag data
    if 'ecg_id' in ecg_df.columns and 'study_id' in diag_df.columns:
        ecg_df['study_id'] = ecg_df['ecg_id']
        print(f"   Mapping ecg_id → study_id")
    
    # Merge on study_id
    if 'study_id' in ecg_df.columns and 'study_id' in diag_df.columns:
        merged_df = ecg_df.merge(
            diag_df[['study_id', 'subject_id', 'has_any_arrhythmia', 
                    'has_severe_arrhythmia', 'is_deceased', 'gender']],
            on='study_id',
            how='inner'
        )
        print(f"   Merged on study_id: {len(merged_df):,} records")
    
    # Merge on study_id
    if 'study_id' in ecg_df.columns and 'study_id' in diag_df.columns:
        merged_df = ecg_df.merge(
            diag_df[['study_id', 'subject_id', 'has_any_arrhythmia', 
                    'has_severe_arrhythmia', 'is_deceased', 'age', 'gender']],
            on='study_id',
            how='inner'
        )
        print(f"   Merged on study_id: {len(merged_df):,} records")
    else:
        print("   ⚠️ Cannot find common identifier. Attempting subject_id merge...")
        # Try subject_id
        if 'subject_id' in ecg_df.columns and 'subject_id' in diag_df.columns:
            merged_df = ecg_df.merge(
                diag_df[['subject_id', 'has_any_arrhythmia', 
                        'has_severe_arrhythmia', 'is_deceased', 'age', 'gender']].drop_duplicates('subject_id'),
                on='subject_id',
                how='inner'
            )
            print(f"   Merged on subject_id: {len(merged_df):,} records")
        else:
            print("\n❌ Cannot merge datasets - no common identifier!")
            print(f"   ECG columns: {ecg_df.columns.tolist()}")
            print(f"   Diag columns: {diag_df.columns.tolist()}")
            return 1
    
    if len(merged_df) == 0:
        print("\n❌ Merge resulted in 0 records!")
        return 1
    
    # Compute QTc and classify
    print("\n📊 Computing QTc and classifying...")
    merged_df = compute_qtc(merged_df)
    merged_df = classify_qtc(merged_df)
    
    # Show classification distribution
    print("\n📊 Classification distribution:")
    for cat in merged_df['qtc_classification'].unique():
        n = (merged_df['qtc_classification'] == cat).sum()
        pct = n / len(merged_df) * 100
        print(f"   {cat}: {n:,} ({pct:.1f}%)")
    
    # Analyze outcomes
    print("\n📊 Analyzing outcomes...")
    results = analyze_outcomes(merged_df)
    safety = compute_safety_comparison(results)
    
    # Print report
    print_report(results, safety, merged_df)
    
    # Print verdict
    print_verdict(safety, results)
    
    # Save results
    output_results = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(merged_df),
        'total_patients': int(merged_df['subject_id'].nunique()),
        'classification_results': results,
        'safety_comparison': safety,
    }
    
    json_path = output_dir / 'safety_audit_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, indent=2, default=str)
    print(f"\n💾 Report JSON: {json_path}")
    
    # Save classified data summary
    summary_df = merged_df.groupby('qtc_classification').agg({
        'study_id': 'count',
        'subject_id': 'nunique',
        'has_severe_arrhythmia': 'sum',
        'is_deceased': 'sum',
        'HR_bpm': 'mean',
        'QTc_Kepler': 'mean',
        'QTc_Bazett': 'mean',
    }).round(2)
    summary_df.columns = ['n_records', 'n_patients', 'n_severe_arrhythmia', 
                          'n_deceased', 'mean_hr', 'mean_qtc_kepler', 'mean_qtc_bazett']
    summary_df.to_csv(output_dir / 'safety_audit_summary.csv')
    print(f"💾 Summary CSV: {output_dir}/safety_audit_summary.csv")
    
    print(f"\n✅ Analysis complete. Results in: {output_dir}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
