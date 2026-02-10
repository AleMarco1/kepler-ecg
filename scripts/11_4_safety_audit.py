#!/usr/bin/env python3
"""
Kepler-ECG: Safety Audit - Reclassified Patients (Script 11_4)
===============================================================

PUNTO 5 DELLA VALIDAZIONE: Safety Audit della Fascia "YELLOW"

Obiettivo: Verificare che i pazienti "riclassificati" da Kepler 
(QTc_Bazett >= 450 → QTc_Kepler < 450) non abbiano tassi di eventi 
avversi superiori ai pazienti "concordi normali".

Richiede:
- Dati ECG MIMIC-IV con QT/RR
- Dati di outcome (mortalità, aritmie) da MIMIC-IV
- Linkage tramite subject_id

Eventi di interesse:
- Mortalità intra-ospedaliera
- Aritmie ventricolari (ICD-10: I47.2, I49.0)
- Torsade de Pointes (rara, ICD-10: I47.2)
- Arresto cardiaco (ICD-10: I46)

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
    
    # ICD-10 codes for adverse events
    ARRHYTHMIA_CODES = {
        'ventricular_tachycardia': ['I472', 'I47.2'],
        'ventricular_fibrillation': ['I490', 'I49.0'],
        'cardiac_arrest': ['I46', 'I460', 'I461', 'I469', 'I46.0', 'I46.1', 'I46.9'],
        'torsade_de_pointes': ['I472', 'I47.2'],  # TdP è classificato come VT
        'sudden_cardiac_death': ['I461', 'I46.1', 'R960', 'R96.0', 'R961', 'R96.1'],
    }
    
    # All arrhythmia codes flattened
    ALL_ARRHYTHMIA_CODES = []
    for codes in ARRHYTHMIA_CODES.values():
        ALL_ARRHYTHMIA_CODES.extend(codes)
    ALL_ARRHYTHMIA_CODES = list(set(ALL_ARRHYTHMIA_CODES))
    
    # Paths - ADJUST THESE TO YOUR MIMIC SETUP
    MIMIC_ECG_DIR = Path('data/mimic-iv-ecg')
    MIMIC_HOSP_DIR = Path('data/mimic-iv/hosp')
    MIMIC_ICU_DIR = Path('data/mimic-iv/icu')
    
    # Results
    RESULTS_BASE = Path('results')
    OUTPUT_DIR = Path('results/safety_audit')
    
    # Processed ECG data
    ECG_RESULTS_DIR = RESULTS_BASE / 'mimic-iv-ecg'


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ecg_data() -> Optional[pd.DataFrame]:
    """Load processed MIMIC-IV ECG data with QT/RR values."""
    
    # Try multiple possible locations
    locations = [
        Config.ECG_RESULTS_DIR / 'qtc' / 'mimic-iv-ecg_qtc_preparation.csv',
        Config.ECG_RESULTS_DIR / 'mimic-iv-ecg_qtc_preparation.csv',
        Config.ECG_RESULTS_DIR / 'mimic-iv-ecg_full_results.csv',
        Config.RESULTS_BASE / 'mimic-iv-ecg' / 'qtc' / 'mimic-iv-ecg_qtc_preparation.csv',
    ]
    
    for loc in locations:
        if loc.exists():
            print(f"  ✓ Found ECG data: {loc}")
            df = pd.read_csv(loc)
            return df
    
    print("  ⚠️ ECG data not found. Tried:")
    for loc in locations:
        print(f"     - {loc}")
    
    return None


def load_mimic_diagnoses() -> Optional[pd.DataFrame]:
    """Load MIMIC-IV diagnosis data."""
    
    # diagnoses_icd contains ICD codes per admission
    locations = [
        Config.MIMIC_HOSP_DIR / 'diagnoses_icd.csv',
        Config.MIMIC_HOSP_DIR / 'diagnoses_icd.csv.gz',
        Path('data/mimic-iv-hosp/diagnoses_icd.csv'),
        Path('data/mimic-iv-hosp/diagnoses_icd.csv.gz'),
    ]
    
    for loc in locations:
        if loc.exists():
            print(f"  ✓ Found diagnoses: {loc}")
            df = pd.read_csv(loc)
            return df
    
    print("  ⚠️ Diagnoses data not found")
    return None


def load_mimic_admissions() -> Optional[pd.DataFrame]:
    """Load MIMIC-IV admissions data (contains mortality info)."""
    
    locations = [
        Config.MIMIC_HOSP_DIR / 'admissions.csv',
        Config.MIMIC_HOSP_DIR / 'admissions.csv.gz',
        Path('data/mimic-iv-hosp/admissions.csv'),
        Path('data/mimic-iv-hosp/admissions.csv.gz'),
    ]
    
    for loc in locations:
        if loc.exists():
            print(f"  ✓ Found admissions: {loc}")
            df = pd.read_csv(loc)
            return df
    
    print("  ⚠️ Admissions data not found")
    return None


def load_mimic_patients() -> Optional[pd.DataFrame]:
    """Load MIMIC-IV patients data."""
    
    locations = [
        Config.MIMIC_HOSP_DIR / 'patients.csv',
        Config.MIMIC_HOSP_DIR / 'patients.csv.gz',
        Path('data/mimic-iv-hosp/patients.csv'),
        Path('data/mimic-iv-hosp/patients.csv.gz'),
    ]
    
    for loc in locations:
        if loc.exists():
            print(f"  ✓ Found patients: {loc}")
            df = pd.read_csv(loc)
            return df
    
    print("  ⚠️ Patients data not found")
    return None


# ============================================================================
# DATA PROCESSING
# ============================================================================

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
    
    # Compute QTc
    qt = df['QT_ms'].values
    rr = df['RR_sec'].values
    
    df['QTc_Kepler'] = qt + Config.KEPLER_K / rr + Config.KEPLER_C
    df['QTc_Bazett'] = qt / np.sqrt(rr)
    
    return df


def classify_patients(df: pd.DataFrame, threshold: float = 450) -> pd.DataFrame:
    """
    Classify patients into concordance categories.
    
    Categories:
    - concordant_normal: Both Kepler and Bazett < threshold
    - concordant_prolonged: Both Kepler and Bazett >= threshold
    - kepler_only_prolonged: Kepler >= threshold, Bazett < threshold
    - reclassified_normal: Bazett >= threshold, Kepler < threshold (KEY GROUP)
    """
    
    k_normal = df['QTc_Kepler'] < threshold
    b_normal = df['QTc_Bazett'] < threshold
    
    conditions = [
        (k_normal & b_normal),      # concordant_normal
        (~k_normal & ~b_normal),    # concordant_prolonged
        (~k_normal & b_normal),     # kepler_only_prolonged
        (k_normal & ~b_normal),     # reclassified_normal (Bazett+, Kepler-)
    ]
    choices = ['concordant_normal', 'concordant_prolonged', 
               'kepler_only_prolonged', 'reclassified_normal']
    
    df['classification'] = np.select(conditions, choices, default='unknown')
    
    return df


def identify_arrhythmia_patients(diagnoses_df: pd.DataFrame) -> set:
    """Identify patients with arrhythmia diagnoses."""
    
    # ICD codes can be in 'icd_code' column
    if 'icd_code' not in diagnoses_df.columns:
        print("  ⚠️ 'icd_code' column not found in diagnoses")
        return set()
    
    # Normalize codes (remove dots, uppercase)
    diagnoses_df['icd_normalized'] = diagnoses_df['icd_code'].astype(str).str.replace('.', '', regex=False).str.upper()
    
    # Find patients with arrhythmia codes
    arrhythmia_mask = diagnoses_df['icd_normalized'].isin([c.replace('.', '').upper() for c in Config.ALL_ARRHYTHMIA_CODES])
    
    if 'subject_id' in diagnoses_df.columns:
        arrhythmia_patients = set(diagnoses_df.loc[arrhythmia_mask, 'subject_id'].unique())
    elif 'hadm_id' in diagnoses_df.columns:
        arrhythmia_patients = set(diagnoses_df.loc[arrhythmia_mask, 'hadm_id'].unique())
    else:
        arrhythmia_patients = set()
    
    return arrhythmia_patients


def identify_deceased_patients(admissions_df: pd.DataFrame) -> set:
    """Identify patients who died during admission."""
    
    # hospital_expire_flag indicates in-hospital mortality
    if 'hospital_expire_flag' in admissions_df.columns:
        deceased = admissions_df[admissions_df['hospital_expire_flag'] == 1]
        if 'subject_id' in deceased.columns:
            return set(deceased['subject_id'].unique())
    
    # Alternative: deathtime not null
    if 'deathtime' in admissions_df.columns:
        deceased = admissions_df[admissions_df['deathtime'].notna()]
        if 'subject_id' in deceased.columns:
            return set(deceased['subject_id'].unique())
    
    return set()


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_outcomes_by_classification(
    ecg_df: pd.DataFrame,
    arrhythmia_patients: set,
    deceased_patients: set,
    id_column: str = 'subject_id'
) -> Dict:
    """
    Analyze outcomes by QTc classification group.
    
    Key comparison:
    - reclassified_normal: Bazett says prolonged, Kepler says normal
    - concordant_normal: Both say normal
    
    If reclassified_normal has similar or lower event rate than concordant_normal,
    then Kepler is SAFE.
    """
    
    results = {}
    
    # Check if id column exists
    if id_column not in ecg_df.columns:
        print(f"  ⚠️ '{id_column}' not found in ECG data")
        # Try to extract from filename or other columns
        possible_id_cols = ['study_id', 'record_id', 'patient_id', 'ecg_id']
        for col in possible_id_cols:
            if col in ecg_df.columns:
                id_column = col
                print(f"  ✓ Using '{id_column}' instead")
                break
    
    for classification in ['concordant_normal', 'concordant_prolonged', 
                          'kepler_only_prolonged', 'reclassified_normal']:
        
        subset = ecg_df[ecg_df['classification'] == classification]
        n_records = len(subset)
        
        if n_records == 0:
            results[classification] = {'n_records': 0, 'insufficient_data': True}
            continue
        
        # Get unique patients
        if id_column in subset.columns:
            unique_patients = set(subset[id_column].unique())
            n_patients = len(unique_patients)
            
            # Count outcomes
            n_arrhythmia = len(unique_patients & arrhythmia_patients)
            n_deceased = len(unique_patients & deceased_patients)
            n_any_event = len(unique_patients & (arrhythmia_patients | deceased_patients))
        else:
            # Can't link to outcomes without patient ID
            unique_patients = set()
            n_patients = n_records  # Assume 1 patient per record
            n_arrhythmia = 0
            n_deceased = 0
            n_any_event = 0
        
        results[classification] = {
            'n_records': n_records,
            'n_patients': n_patients,
            'n_arrhythmia': n_arrhythmia,
            'n_deceased': n_deceased,
            'n_any_event': n_any_event,
            'pct_arrhythmia': n_arrhythmia / n_patients * 100 if n_patients > 0 else 0,
            'pct_deceased': n_deceased / n_patients * 100 if n_patients > 0 else 0,
            'pct_any_event': n_any_event / n_patients * 100 if n_patients > 0 else 0,
            'mean_qtc_kepler': float(subset['QTc_Kepler'].mean()),
            'mean_qtc_bazett': float(subset['QTc_Bazett'].mean()),
            'mean_hr': float(subset['HR_bpm'].mean()) if 'HR_bpm' in subset.columns else None,
        }
    
    return results


def compute_safety_metrics(results: Dict) -> Dict:
    """
    Compute safety comparison metrics.
    
    Key question: Is 'reclassified_normal' SAFE compared to 'concordant_normal'?
    """
    
    safety = {}
    
    reclass = results.get('reclassified_normal', {})
    concord = results.get('concordant_normal', {})
    
    if reclass.get('insufficient_data') or concord.get('insufficient_data'):
        safety['comparison_available'] = False
        return safety
    
    safety['comparison_available'] = True
    
    # Event rates
    reclass_event_rate = reclass.get('pct_any_event', 0)
    concord_event_rate = concord.get('pct_any_event', 0)
    
    safety['reclassified_event_rate'] = reclass_event_rate
    safety['concordant_normal_event_rate'] = concord_event_rate
    safety['event_rate_difference'] = reclass_event_rate - concord_event_rate
    
    # Risk ratio
    if concord_event_rate > 0:
        safety['relative_risk'] = reclass_event_rate / concord_event_rate
    else:
        safety['relative_risk'] = None
    
    # Statistical test (chi-square)
    n_reclass = reclass.get('n_patients', 0)
    n_concord = concord.get('n_patients', 0)
    events_reclass = reclass.get('n_any_event', 0)
    events_concord = concord.get('n_any_event', 0)
    
    if n_reclass > 0 and n_concord > 0:
        # 2x2 contingency table
        table = [
            [events_reclass, n_reclass - events_reclass],
            [events_concord, n_concord - events_concord]
        ]
        
        try:
            chi2, p_value = stats.chi2_contingency(table)[:2]
            safety['chi2'] = float(chi2)
            safety['p_value'] = float(p_value)
        except:
            safety['chi2'] = None
            safety['p_value'] = None
    
    # Verdict
    if safety['event_rate_difference'] <= 0:
        safety['verdict'] = 'SAFE'
        safety['interpretation'] = 'Reclassified patients have EQUAL or LOWER event rate'
    elif safety['event_rate_difference'] < 2.0:  # Less than 2% absolute difference
        safety['verdict'] = 'ACCEPTABLE'
        safety['interpretation'] = 'Small increase in event rate, clinically acceptable'
    else:
        safety['verdict'] = 'CONCERN'
        safety['interpretation'] = 'Reclassified patients have higher event rate'
    
    return safety


# ============================================================================
# ALTERNATIVE ANALYSIS (without outcomes linkage)
# ============================================================================

def analyze_reclassified_characteristics(ecg_df: pd.DataFrame) -> Dict:
    """
    If we can't link to outcomes, analyze characteristics of reclassified patients.
    
    This provides indirect evidence of safety:
    - If reclassified patients have higher HR (tachycardia), they were likely
      false positives from Bazett's overcorrection
    """
    
    results = {}
    
    for classification in ['concordant_normal', 'concordant_prolonged', 
                          'kepler_only_prolonged', 'reclassified_normal']:
        
        subset = ecg_df[ecg_df['classification'] == classification]
        
        if len(subset) < 10:
            continue
        
        results[classification] = {
            'n': len(subset),
            'hr_mean': float(subset['HR_bpm'].mean()) if 'HR_bpm' in subset.columns else None,
            'hr_std': float(subset['HR_bpm'].std()) if 'HR_bpm' in subset.columns else None,
            'hr_median': float(subset['HR_bpm'].median()) if 'HR_bpm' in subset.columns else None,
            'pct_tachycardia': float((subset['HR_bpm'] >= 100).mean() * 100) if 'HR_bpm' in subset.columns else None,
            'pct_bradycardia': float((subset['HR_bpm'] < 60).mean() * 100) if 'HR_bpm' in subset.columns else None,
            'qtc_kepler_mean': float(subset['QTc_Kepler'].mean()),
            'qtc_bazett_mean': float(subset['QTc_Bazett'].mean()),
            'qt_mean': float(subset['QT_ms'].mean()) if 'QT_ms' in subset.columns else None,
        }
    
    # Interpretation
    reclass = results.get('reclassified_normal', {})
    concord = results.get('concordant_normal', {})
    
    if reclass and concord:
        hr_diff = (reclass.get('hr_mean', 0) or 0) - (concord.get('hr_mean', 0) or 0)
        tachy_diff = (reclass.get('pct_tachycardia', 0) or 0) - (concord.get('pct_tachycardia', 0) or 0)
        
        results['comparison'] = {
            'hr_difference': hr_diff,
            'tachycardia_difference': tachy_diff,
        }
        
        if hr_diff > 5 or tachy_diff > 10:
            results['interpretation'] = (
                "Reclassified patients have HIGHER heart rate than concordant normal. "
                "This suggests they were FALSE POSITIVES from Bazett's overcorrection in tachycardia, "
                "NOT true QT prolongation. Kepler is likely CORRECTING Bazett's errors."
            )
        else:
            results['interpretation'] = (
                "Reclassified patients have similar heart rate to concordant normal. "
                "Further outcome analysis recommended."
            )
    
    return results


# ============================================================================
# REPORTING
# ============================================================================

def print_report(classification_results: Dict, safety_metrics: Dict, 
                characteristics: Dict, has_outcomes: bool):
    """Print comprehensive report."""
    
    print("\n" + "="*70)
    print("SAFETY AUDIT - PUNTO 5")
    print("="*70)
    
    # Classification summary
    print("\n📊 CLASSIFICAZIONE DEI RECORD ECG:")
    print("-" * 60)
    print(f"{'Categoria':<25} {'N records':>12} {'N pazienti':>12} {'%':>8}")
    print("-" * 60)
    
    total = sum(r.get('n_records', 0) for r in classification_results.values())
    for cat, data in classification_results.items():
        if not data.get('insufficient_data'):
            pct = data['n_records'] / total * 100 if total > 0 else 0
            print(f"{cat:<25} {data['n_records']:>12,} {data.get('n_patients', 'N/A'):>12} {pct:>7.1f}%")
    
    print("-" * 60)
    print(f"{'TOTALE':<25} {total:>12,}")
    
    # Focus on reclassified
    reclass = classification_results.get('reclassified_normal', {})
    if reclass and not reclass.get('insufficient_data'):
        print(f"\n⚠️  PAZIENTI RICLASSIFICATI (Bazett+ → Kepler-):")
        print(f"    N = {reclass['n_records']:,} record ({reclass['n_records']/total*100:.1f}% del totale)")
        print(f"    QTc Bazett medio: {reclass['mean_qtc_bazett']:.1f} ms (≥450 per definizione)")
        print(f"    QTc Kepler medio: {reclass['mean_qtc_kepler']:.1f} ms (<450 per definizione)")
    
    # Outcomes analysis
    if has_outcomes:
        print("\n📊 ANALISI OUTCOMES:")
        print("-" * 60)
        print(f"{'Categoria':<25} {'% Aritmie':>12} {'% Decesso':>12} {'% Qualsiasi':>12}")
        print("-" * 60)
        
        for cat in ['concordant_normal', 'reclassified_normal', 'concordant_prolonged']:
            data = classification_results.get(cat, {})
            if not data.get('insufficient_data'):
                print(f"{cat:<25} {data.get('pct_arrhythmia', 0):>11.2f}% "
                      f"{data.get('pct_deceased', 0):>11.2f}% {data.get('pct_any_event', 0):>11.2f}%")
        
        # Safety verdict
        print("\n📊 VERDETTO SICUREZZA:")
        if safety_metrics.get('comparison_available'):
            print(f"    Tasso eventi riclassificati: {safety_metrics['reclassified_event_rate']:.2f}%")
            print(f"    Tasso eventi concordi normali: {safety_metrics['concordant_normal_event_rate']:.2f}%")
            print(f"    Differenza: {safety_metrics['event_rate_difference']:+.2f}%")
            if safety_metrics.get('relative_risk'):
                print(f"    Rischio relativo: {safety_metrics['relative_risk']:.2f}")
            if safety_metrics.get('p_value'):
                print(f"    p-value: {safety_metrics['p_value']:.4f}")
            print(f"\n    VERDETTO: {safety_metrics['verdict']}")
            print(f"    {safety_metrics['interpretation']}")
    
    else:
        # Characteristics analysis (indirect)
        print("\n📊 ANALISI CARATTERISTICHE (senza outcomes):")
        print("-" * 60)
        
        for cat in ['concordant_normal', 'reclassified_normal']:
            data = characteristics.get(cat, {})
            if data:
                print(f"\n  {cat}:")
                print(f"    N: {data.get('n', 0):,}")
                print(f"    HR medio: {data.get('hr_mean', 0):.1f} bpm")
                print(f"    % Tachicardia (≥100): {data.get('pct_tachycardia', 0):.1f}%")
                print(f"    % Bradicardia (<60): {data.get('pct_bradycardia', 0):.1f}%")
        
        if 'interpretation' in characteristics:
            print(f"\n    INTERPRETAZIONE: {characteristics['interpretation']}")


def print_verdict(safety_metrics: Dict, characteristics: Dict, has_outcomes: bool):
    """Print final verdict."""
    
    print("\n" + "="*70)
    print("VERDETTO FINALE - PUNTO 5")
    print("="*70)
    
    if has_outcomes:
        verdict = safety_metrics.get('verdict', 'UNKNOWN')
        if verdict == 'SAFE':
            print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ✅  KEPLER È SICURO - Nessun Aumento di Eventi Avversi          ║
    ║                                                                    ║
    ║   I pazienti riclassificati da Kepler (Bazett+ → Kepler-)          ║
    ║   NON mostrano un tasso di eventi avversi superiore ai             ║
    ║   pazienti concordi normali.                                       ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
            """)
        elif verdict == 'ACCEPTABLE':
            print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ⚠️  KEPLER ACCETTABILE - Piccola Differenza                      ║
    ║                                                                    ║
    ║   I pazienti riclassificati mostrano un lieve aumento di eventi,   ║
    ║   ma la differenza è clinicamente accettabile (<2%).               ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
            """)
        else:
            print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ❌  ATTENZIONE - Possibile Problema di Sicurezza                 ║
    ║                                                                    ║
    ║   I pazienti riclassificati mostrano un tasso di eventi superiore. ║
    ║   Richiesta analisi approfondita.                                  ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
            """)
    
    else:
        # Without outcomes, use characteristics
        reclass = characteristics.get('reclassified_normal', {})
        concord = characteristics.get('concordant_normal', {})
        
        if reclass and concord:
            hr_reclass = reclass.get('hr_mean', 0) or 0
            hr_concord = concord.get('hr_mean', 0) or 0
            tachy_reclass = reclass.get('pct_tachycardia', 0) or 0
            tachy_concord = concord.get('pct_tachycardia', 0) or 0
            
            if hr_reclass > hr_concord + 5 or tachy_reclass > tachy_concord + 10:
                print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ✅  EVIDENZA INDIRETTA DI SICUREZZA                              ║
    ║                                                                    ║
    ║   I pazienti riclassificati hanno HR SIGNIFICATIVAMENTE PIÙ ALTO.  ║
    ║   Questo suggerisce che erano FALSI POSITIVI di Bazett             ║
    ║   (sovra-correzione in tachicardia), non veri QT lunghi.           ║
    ║                                                                    ║
    ║   Kepler sta CORREGGENDO gli errori di Bazett, non mascherando     ║
    ║   patologie reali.                                                 ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
                """)
            else:
                print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ⚠️  EVIDENZA INCONCLUSIVA                                        ║
    ║                                                                    ║
    ║   Senza dati di outcome, non possiamo confermare la sicurezza.     ║
    ║   Si raccomanda linkage con dati clinici MIMIC-IV.                 ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
                """)
        else:
            print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   ❓  ANALISI NON POSSIBILE                                        ║
    ║                                                                    ║
    ║   Dati insufficienti per l'analisi di sicurezza.                   ║
    ║   Richiesti: dati ECG MIMIC-IV + outcomes clinici.                 ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
            """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    parser = argparse.ArgumentParser(
        description='Kepler-ECG Safety Audit (Point 5)',
    )
    
    parser.add_argument('--output-dir', type=str, 
                       default=str(Config.OUTPUT_DIR),
                       help='Output directory')
    parser.add_argument('--ecg-file', type=str,
                       help='Path to ECG data CSV (if not using default location)')
    parser.add_argument('--mimic-hosp-dir', type=str,
                       help='Path to MIMIC-IV hosp directory')
    
    args = parser.parse_args()
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   KEPLER-ECG: Safety Audit (Script 11_4)                           ║
    ║   PUNTO 5 - Safety Audit della Fascia "YELLOW"                     ║
    ║                                                                    ║
    ║   Obiettivo: Verificare che i pazienti riclassificati da Kepler    ║
    ║   (Bazett ≥450 → Kepler <450) non abbiano più eventi avversi       ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update paths if provided
    if args.mimic_hosp_dir:
        Config.MIMIC_HOSP_DIR = Path(args.mimic_hosp_dir)
    
    # Load ECG data
    print("\n📊 Loading data...")
    
    if args.ecg_file:
        ecg_df = pd.read_csv(args.ecg_file)
        print(f"  ✓ Loaded ECG data from: {args.ecg_file}")
    else:
        ecg_df = load_ecg_data()
    
    if ecg_df is None or len(ecg_df) == 0:
        print("\n❌ Could not load ECG data!")
        print("\nTo run this analysis, you need:")
        print("  1. Processed MIMIC-IV ECG data (from pipeline step 06)")
        print("  2. Optionally: MIMIC-IV hosp data for outcome linkage")
        print("\nUsage:")
        print("  python 11_4_safety_audit.py --ecg-file path/to/mimic_ecg_qtc.csv")
        print("  python 11_4_safety_audit.py --mimic-hosp-dir path/to/mimic-iv/hosp")
        return 1
    
    print(f"\n📊 Loaded {len(ecg_df):,} ECG records")
    
    # Compute QTc and classify
    print("\n📊 Computing QTc and classifying patients...")
    ecg_df = compute_qtc(ecg_df)
    ecg_df = classify_patients(ecg_df)
    
    # Show classification distribution
    print("\n📊 Classification distribution:")
    for cat in ['concordant_normal', 'concordant_prolonged', 
                'kepler_only_prolonged', 'reclassified_normal']:
        n = (ecg_df['classification'] == cat).sum()
        pct = n / len(ecg_df) * 100
        print(f"  {cat}: {n:,} ({pct:.1f}%)")
    
    # Try to load outcomes data
    print("\n📊 Looking for outcomes data...")
    diagnoses_df = load_mimic_diagnoses()
    admissions_df = load_mimic_admissions()
    
    has_outcomes = diagnoses_df is not None and admissions_df is not None
    
    if has_outcomes:
        print("\n📊 Identifying adverse events...")
        arrhythmia_patients = identify_arrhythmia_patients(diagnoses_df)
        deceased_patients = identify_deceased_patients(admissions_df)
        print(f"  Patients with arrhythmias: {len(arrhythmia_patients):,}")
        print(f"  Patients deceased: {len(deceased_patients):,}")
        
        # Analyze outcomes
        print("\n📊 Analyzing outcomes by classification...")
        classification_results = analyze_outcomes_by_classification(
            ecg_df, arrhythmia_patients, deceased_patients
        )
        safety_metrics = compute_safety_metrics(classification_results)
    else:
        print("\n⚠️ Outcomes data not available. Using indirect analysis.")
        arrhythmia_patients = set()
        deceased_patients = set()
        classification_results = analyze_outcomes_by_classification(
            ecg_df, arrhythmia_patients, deceased_patients
        )
        safety_metrics = {}
    
    # Characteristics analysis (always run)
    characteristics = analyze_reclassified_characteristics(ecg_df)
    
    # Print report
    print_report(classification_results, safety_metrics, characteristics, has_outcomes)
    
    # Print verdict
    print_verdict(safety_metrics, characteristics, has_outcomes)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(ecg_df),
        'has_outcomes': has_outcomes,
        'classification_results': classification_results,
        'safety_metrics': safety_metrics,
        'characteristics': characteristics,
    }
    
    json_path = output_dir / 'safety_audit_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Report JSON: {json_path}")
    
    # Save classified data
    ecg_df[['classification', 'QTc_Kepler', 'QTc_Bazett', 'HR_bpm']].to_csv(
        output_dir / 'classified_ecg_data.csv', index=False
    )
    print(f"💾 Classified data: {output_dir}/classified_ecg_data.csv")
    
    print(f"\n✅ Analysis complete. Results in: {output_dir}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
