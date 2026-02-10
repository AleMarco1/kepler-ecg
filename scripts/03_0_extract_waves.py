#!/usr/bin/env python3
"""
Kepler-ECG: Wave Delineation Pipeline

Extracts PQRST fiducial points from ECG records using NeuroKit2.
Supports multiple datasets: PTB-XL, Chapman, CPSC-2018, Georgia, MIMIC-IV-ECG.

Output features:
- RR, PR, QRS, QT intervals
- P, QRS, T wave amplitudes
- QTc values (Bazett, Fridericia, Framingham, Hodges)
- Signal quality metrics

CHECKPOINT FEATURE:
- Saves progress every 5000 records to checkpoint files
- Use --resume to continue from last checkpoint if interrupted
- Without --resume, always starts fresh

Usage:
    # Process PTB-XL
    python scripts/03_0_extract_waves.py --dataset ptb-xl
    
    # Process Chapman with custom lead
    python scripts/03_0_extract_waves.py --dataset chapman --lead 1
    
    # Process MIMIC-IV-ECG (uses record_list.csv)
    python scripts/03_0_extract_waves.py --dataset mimic-iv-ecg --n_samples 10000
    
    # Resume from checkpoint after interruption
    python scripts/03_0_extract_waves.py --dataset mimic-iv-ecg --resume
    
    # Process custom path
    python scripts/03_0_extract_waves.py --input data/raw/mydata --output results/waves

Author: Kepler-ECG Project
Version: 2.5.0 - Uses process_dataset output for age/sex metadata
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ============================================================================
# CONFIGURATION
# ============================================================================

# Checkpoint settings
CHECKPOINT_INTERVAL = 5000

# Valid sampling rates (Hz)
VALID_SAMPLING_RATES = [100, 250, 500, 1000]

# Clinical interval thresholds (ms)
# These define physiologically plausible ranges for filtering outliers
INTERVAL_THRESHOLDS = {
    'PR': {'min': 80, 'max': 400},      # PR interval: 80-400 ms
    'QRS': {'min': 40, 'max': 200},     # QRS duration: 40-200 ms
    'QT': {'min': 200, 'max': 700},     # QT interval: 200-700 ms
}


# ============================================================================
# QTc Formulas
# ============================================================================

def calculate_qtc_formulas(qt_ms: float, rr_sec: float, hr_bpm: float) -> Dict[str, float]:
    """Calculate QTc using multiple correction formulas."""
    if np.isnan(qt_ms) or np.isnan(rr_sec) or rr_sec <= 0:
        return {
            'QTc_Bazett_ms': np.nan,
            'QTc_Fridericia_ms': np.nan,
            'QTc_Framingham_ms': np.nan,
            'QTc_Hodges_ms': np.nan,
        }
    
    return {
        'QTc_Bazett_ms': qt_ms / np.sqrt(rr_sec),
        'QTc_Fridericia_ms': qt_ms / np.cbrt(rr_sec),
        'QTc_Framingham_ms': qt_ms + 154 * (1 - rr_sec),
        'QTc_Hodges_ms': qt_ms + 1.75 * (hr_bpm - 60),
    }


# ============================================================================
# Wave Delineation
# ============================================================================

def delineate_ecg(
    ecg_signal: np.ndarray,
    sampling_rate: int = 500,
    lead_idx: int = 1,
) -> Dict:
    """
    Perform wave delineation using NeuroKit2.
    
    Args:
        ecg_signal: ECG signal array (n_samples, n_leads)
        sampling_rate: Sampling rate in Hz
        lead_idx: Lead index to process (0=I, 1=II, etc.)
        
    Returns:
        Dictionary with delineation results
    """
    try:
        import neurokit2 as nk
    except ImportError:
        return {'success': False, 'error': 'neurokit2 not installed'}
    
    # Extract single lead
    if ecg_signal.ndim == 2:
        lead_signal = ecg_signal[:, lead_idx]
    else:
        lead_signal = ecg_signal
    
    try:
        # Clean signal
        ecg_cleaned = nk.ecg_clean(lead_signal, sampling_rate=sampling_rate)
        
        # Find R-peaks
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        r_peaks = rpeaks['ECG_R_Peaks']
        
        if len(r_peaks) < 3:
            return {'success': False, 'error': 'Too few R-peaks'}
        
        # Delineate waves
        _, waves = nk.ecg_delineate(
            ecg_cleaned, 
            rpeaks=rpeaks,
            sampling_rate=sampling_rate,
            method='dwt'
        )
        
        # Signal quality
        quality = nk.ecg_quality(ecg_cleaned, rpeaks=r_peaks, sampling_rate=sampling_rate)
        mean_quality = np.nanmean(quality)
        
        # Extract intervals and features
        result = extract_intervals(ecg_cleaned, r_peaks, waves, sampling_rate, mean_quality)
        result['success'] = True
        
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def extract_intervals(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    waves: Dict,
    sampling_rate: int,
    quality_score: float,
) -> Dict:
    """Extract clinical intervals from delineation results."""
    ms_per_sample = 1000 / sampling_rate
    
    # Get thresholds from configuration
    pr_min, pr_max = INTERVAL_THRESHOLDS['PR']['min'], INTERVAL_THRESHOLDS['PR']['max']
    qrs_min, qrs_max = INTERVAL_THRESHOLDS['QRS']['min'], INTERVAL_THRESHOLDS['QRS']['max']
    qt_min, qt_max = INTERVAL_THRESHOLDS['QT']['min'], INTERVAL_THRESHOLDS['QT']['max']
    
    def safe_median(arr):
        valid = [x for x in arr if x is not None and not np.isnan(x)]
        return np.median(valid) if valid else np.nan
    
    # Wave positions
    p_onsets = waves.get('ECG_P_Onsets', [])
    p_peaks = waves.get('ECG_P_Peaks', [])
    q_peaks = waves.get('ECG_Q_Peaks', [])
    s_peaks = waves.get('ECG_S_Peaks', [])
    t_peaks = waves.get('ECG_T_Peaks', [])
    t_offsets = waves.get('ECG_T_Offsets', [])
    
    # RR intervals
    rr_intervals = np.diff(r_peaks) * ms_per_sample
    rr_median = np.median(rr_intervals) if len(rr_intervals) > 0 else np.nan
    hr = 60000 / rr_median if rr_median > 0 else np.nan
    
    # PR interval
    pr_intervals = []
    for i, r in enumerate(r_peaks[:-1]):
        if i < len(p_onsets) and p_onsets[i] is not None and not np.isnan(p_onsets[i]):
            pr = (r - p_onsets[i]) * ms_per_sample
            if pr_min < pr < pr_max:
                pr_intervals.append(pr)
    pr_median = np.median(pr_intervals) if pr_intervals else np.nan
    
    # QRS duration
    qrs_durations = []
    for i in range(min(len(q_peaks), len(s_peaks))):
        q, s = q_peaks[i], s_peaks[i]
        if q is not None and s is not None and not np.isnan(q) and not np.isnan(s):
            qrs = (s - q) * ms_per_sample
            if qrs_min < qrs < qrs_max:
                qrs_durations.append(qrs)
    qrs_median = np.median(qrs_durations) if qrs_durations else np.nan
    
    # QT interval
    qt_intervals = []
    for i in range(min(len(r_peaks), len(t_offsets))):
        if i < len(q_peaks) and q_peaks[i] is not None and not np.isnan(q_peaks[i]):
            q_start = q_peaks[i]
        else:
            q_start = r_peaks[i] - int(40 / ms_per_sample)
        
        t_end = t_offsets[i] if i < len(t_offsets) else None
        
        if t_end is not None and not np.isnan(t_end):
            qt = (t_end - q_start) * ms_per_sample
            if qt_min < qt < qt_max:
                qt_intervals.append(qt)
    
    qt_median = np.median(qt_intervals) if qt_intervals else np.nan
    
    # Amplitudes
    r_amplitudes = [ecg_signal[int(r)] for r in r_peaks if 0 <= int(r) < len(ecg_signal)]
    r_amp = np.median(r_amplitudes) if r_amplitudes else np.nan
    
    p_amplitudes = []
    for p in p_peaks:
        if p is not None and not np.isnan(p) and 0 <= int(p) < len(ecg_signal):
            p_amplitudes.append(ecg_signal[int(p)])
    p_amp = np.median(p_amplitudes) if p_amplitudes else np.nan
    
    t_amplitudes = []
    for t in t_peaks:
        if t is not None and not np.isnan(t) and 0 <= int(t) < len(ecg_signal):
            t_amplitudes.append(ecg_signal[int(t)])
    t_amp = np.median(t_amplitudes) if t_amplitudes else np.nan
    
    # Detection rates
    p_detected = sum(1 for p in p_peaks if p is not None and not np.isnan(p))
    t_detected = sum(1 for t in t_offsets if t is not None and not np.isnan(t))
    n_beats = len(r_peaks)
    
    return {
        'RR_interval_ms': rr_median,
        'RR_interval_sec': rr_median / 1000 if not np.isnan(rr_median) else np.nan,
        'heart_rate_bpm': hr,
        'PR_interval_ms': pr_median,
        'QRS_duration_ms': qrs_median,
        'QT_interval_ms': qt_median,
        'R_amplitude_mV': r_amp,
        'P_amplitude_mV': p_amp,
        'T_amplitude_mV': t_amp,
        'n_beats': n_beats,
        'signal_quality': quality_score,
        'p_wave_detection_rate': p_detected / n_beats if n_beats > 0 else 0,
        't_wave_detection_rate': t_detected / n_beats if n_beats > 0 else 0,
    }


# ============================================================================
# ECG Record Discovery
# ============================================================================

def find_ecg_records(data_path: Path, dataset_name: str, n_samples: Optional[int] = None) -> List[Path]:
    """
    Find ECG record paths for a dataset.
    
    For MIMIC-IV-ECG: Uses record_list.csv for fast path lookup
    For other datasets: Uses rglob to find .hea files
    
    Args:
        data_path: Path to dataset directory
        dataset_name: Name of dataset
        n_samples: Maximum number of records to return
        
    Returns:
        List of Path objects pointing to ECG records (without extension)
    """
    
    # Special handling for MIMIC-IV-ECG
    if dataset_name.lower() == 'mimic-iv-ecg':
        record_list_path = data_path / 'record_list.csv'
        
        if not record_list_path.exists():
            logger.error(f"MIMIC-IV-ECG requires record_list.csv at {record_list_path}")
            return []
        
        logger.info("Loading MIMIC-IV-ECG record list...")
        df = pd.read_csv(record_list_path)
        
        # 'path' column contains relative paths like 'files/p1000/p10000032/s40689238/40689238'
        if 'path' not in df.columns:
            logger.error("record_list.csv must have 'path' column")
            return []
        
        # Apply n_samples limit BEFORE creating paths
        if n_samples and len(df) > n_samples:
            df = df.head(n_samples)
        
        # Convert to full paths
        ecg_files = [data_path / row['path'] for _, row in df.iterrows()]
        
        logger.info(f"Found {len(ecg_files)} records in record_list.csv")
        return ecg_files
    
    # Standard handling for other datasets
    logger.info(f"Scanning for .hea files in {data_path}...")
    ecg_files = list(data_path.rglob('*.hea'))
    
    # Convert to record paths (without extension)
    ecg_files = [f.with_suffix('') for f in ecg_files]
    
    logger.info(f"Found {len(ecg_files)} ECG records")
    
    if n_samples and len(ecg_files) > n_samples:
        ecg_files = ecg_files[:n_samples]
    
    return ecg_files


def load_ecg_record(record_path: Path, sampling_rate: int = 500) -> Optional[np.ndarray]:
    """Load ECG signal from WFDB record."""
    try:
        record = wfdb.rdrecord(str(record_path))
        return record.p_signal
    except Exception as e:
        logger.debug(f"Failed to load {record_path}: {e}")
        return None


# ============================================================================
# ECG ID Extraction
# ============================================================================

def extract_ecg_id(record_path: Path, dataset_name: str) -> str:
    """
    Extract ECG ID from record path - preserves original filename.
    
    Args:
        record_path: Path to ECG record
        dataset_name: Name of dataset (unused, kept for API consistency)
        
    Returns:
        ECG ID string (original filename without extension)
        
    Examples:
        PTB-XL: "00001_hr" -> "00001_hr"
        Chapman: "JS00017" -> "JS00017"
        CPSC-2018: "A0001" -> "A0001"
        Georgia: "E00001" -> "E00001"
        MIMIC-IV: "40689238" -> "40689238"
    """
    if hasattr(record_path, 'stem'):
        return record_path.stem
    else:
        return str(record_path).split('/')[-1]


# ============================================================================
# Metadata Loading (from process_dataset output)
# ============================================================================

def load_features_metadata(dataset_name: str) -> Optional[pd.DataFrame]:
    """
    Load age/sex metadata from process_dataset output.
    
    Uses results/{dataset}/preprocess/{dataset}_features.csv which already
    has age and sex correctly extracted by 02_0_process_dataset.py.
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        DataFrame with ecg_id as index and age/sex columns, or None if not found
    """
    # Try multiple possible paths
    possible_paths = [
        Path(f"results/{dataset_name}/preprocess/{dataset_name}_features.csv"),
        Path(f"results/{dataset_name}/{dataset_name}_features.csv"),
        Path(f"results/{dataset_name}/features.csv"),
    ]
    
    features_path = None
    for p in possible_paths:
        if p.exists():
            features_path = p
            break
    
    if features_path is None:
        logger.warning(f"Features file not found. Searched paths:")
        for p in possible_paths:
            logger.warning(f"  - {p}")
        logger.warning("Run 02_0_process_dataset.py first to extract age/sex metadata")
        logger.warning("Continuing without age/sex data...")
        return None
    
    try:
        # Read CSV
        df_full = pd.read_csv(features_path)
        
        # Debug: show what pandas actually read
        logger.info(f"   CSV shape: {df_full.shape}")
        logger.info(f"   CSV columns: {list(df_full.columns[:6])}...")
        
        # Check if first row values make sense
        first_row = df_full.iloc[0]
        logger.info(f"   First row ecg_id: {repr(first_row['ecg_id'])}")
        logger.info(f"   First row age: {repr(first_row['age'])}")
        logger.info(f"   First row sex: {repr(first_row['sex'])}")
        
        # Check required columns exist
        required_cols = ['ecg_id', 'age', 'sex']
        missing_cols = [c for c in required_cols if c not in df_full.columns]
        if missing_cols:
            logger.warning(f"Missing columns in features file: {missing_cols}")
            logger.warning(f"Available columns: {list(df_full.columns[:10])}...")
            return None
        
        # Select only needed columns
        df = df_full[['ecg_id', 'age', 'sex']].copy()
        
        # Convert ecg_id to string and set as index
        df['ecg_id'] = df['ecg_id'].astype(str)
        df = df.set_index('ecg_id')
        
        logger.info(f"✅ Loaded metadata from {features_path}: {len(df)} records")
        logger.info(f"   Sample ecg_ids: {list(df.index[:3])}")
        return df
        
    except Exception as e:
        logger.warning(f"Could not load features metadata: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def get_metadata_for_record(ecg_id: str, metadata: Optional[pd.DataFrame]) -> Dict:
    """
    Get age and sex for a specific ECG record.
    
    Args:
        ecg_id: ECG identifier string
        metadata: DataFrame with ecg_id index and age/sex columns
        
    Returns:
        Dictionary with 'age' and 'sex' keys
    """
    meta = {'age': None, 'sex': None}
    
    if metadata is None:
        return meta
    
    try:
        if ecg_id in metadata.index:
            row = metadata.loc[ecg_id]
            
            # Extract age
            if 'age' in row.index and pd.notna(row['age']):
                meta['age'] = float(row['age'])
            
            # Extract sex - 0 is valid (male)!
            if 'sex' in row.index and pd.notna(row['sex']):
                meta['sex'] = int(row['sex'])
                
    except Exception as e:
        logger.debug(f"Could not get metadata for {ecg_id}: {e}")
    
    return meta


# ============================================================================
# Checkpoint Management
# ============================================================================

def get_checkpoint_paths(output_path: Path, dataset_name: str) -> Tuple[Path, Path]:
    """Get paths for checkpoint files."""
    checkpoint_dir = output_path / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    errors_checkpoint = checkpoint_dir / f'{dataset_name}_errors_checkpoint.csv'
    progress_checkpoint = checkpoint_dir / f'{dataset_name}_progress.json'
    
    return errors_checkpoint, progress_checkpoint


def load_checkpoint(output_path: Path, dataset_name: str) -> Tuple[List[Dict], List[Dict], Set[str]]:
    """
    Load checkpoint data if it exists.
    
    Returns:
        Tuple of (results_list, errors_list, processed_ids_set)
    """
    errors_ckpt, progress_ckpt = get_checkpoint_paths(output_path, dataset_name)
    
    # Also check for partial results CSV
    partial_results_path = output_path / f'{dataset_name}_wave_features_partial.csv'
    
    results = []
    errors = []
    processed_ids = set()
    
    if progress_ckpt.exists():
        try:
            with open(progress_ckpt, 'r') as f:
                progress = json.load(f)
                processed_ids = set(progress.get('processed_ids', []))
                logger.info(f"📂 Found checkpoint: {len(processed_ids)} records already processed")
        except Exception as e:
            logger.warning(f"Could not load progress checkpoint: {e}")
    
    if partial_results_path.exists() and len(processed_ids) > 0:
        try:
            df_results = pd.read_csv(partial_results_path)
            results = df_results.to_dict('records')
            logger.info(f"   Loaded {len(results)} successful results from checkpoint")
        except Exception as e:
            logger.warning(f"Could not load results checkpoint: {e}")
            results = []
    
    if errors_ckpt.exists() and len(processed_ids) > 0:
        try:
            df_errors = pd.read_csv(errors_ckpt)
            errors = df_errors.to_dict('records')
            logger.info(f"   Loaded {len(errors)} errors from checkpoint")
        except Exception as e:
            logger.warning(f"Could not load errors checkpoint: {e}")
            errors = []
    
    return results, errors, processed_ids


def save_checkpoint(
    output_path: Path, 
    dataset_name: str, 
    results: List[Dict], 
    errors: List[Dict],
    processed_ids: Set[str]
) -> None:
    """Save checkpoint data."""
    errors_ckpt, progress_ckpt = get_checkpoint_paths(output_path, dataset_name)
    partial_results_path = output_path / f'{dataset_name}_wave_features_partial.csv'
    
    try:
        # Save partial results as CSV
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(partial_results_path, index=False)
        
        # Save errors
        if errors:
            df_errors = pd.DataFrame(errors)
            df_errors.to_csv(errors_ckpt, index=False)
        
        # Save progress
        progress = {
            'processed_ids': list(processed_ids),
            'n_results': len(results),
            'n_errors': len(errors),
            'last_update': datetime.now().isoformat()
        }
        with open(progress_ckpt, 'w') as f:
            json.dump(progress, f)
        
        logger.info(f"💾 Checkpoint saved: {len(results)} results, {len(errors)} errors")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def clear_checkpoints(output_path: Path, dataset_name: str) -> None:
    """Clear checkpoint files."""
    errors_ckpt, progress_ckpt = get_checkpoint_paths(output_path, dataset_name)
    partial_results_path = output_path / f'{dataset_name}_wave_features_partial.csv'
    
    for ckpt in [errors_ckpt, progress_ckpt, partial_results_path]:
        if ckpt.exists():
            ckpt.unlink()
            logger.debug(f"Deleted checkpoint: {ckpt.name}")


# ============================================================================
# Dataset Processing
# ============================================================================

def process_dataset(
    data_path: Path,
    output_path: Path,
    dataset_name: str,
    sampling_rate: int = 500,
    lead_idx: int = 1,
    n_samples: Optional[int] = None,
    resume: bool = False,
    features_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Process all ECGs in a dataset with checkpoint support."""
    
    # Load metadata from process_dataset output (has age/sex already extracted)
    if features_file:
        # Use explicit path if provided
        features_path = Path(features_file)
        if features_path.exists():
            try:
                df_full = pd.read_csv(features_path, engine='python', encoding='utf-8')
                if 'ecg_id' in df_full.columns and 'age' in df_full.columns and 'sex' in df_full.columns:
                    df = df_full[['ecg_id', 'age', 'sex']].copy()
                    df['ecg_id'] = df['ecg_id'].astype(str)
                    df = df.set_index('ecg_id')
                    metadata = df
                    logger.info(f"✅ Loaded metadata from {features_path}: {len(df)} records")
                    logger.info(f"   Sample ecg_ids: {list(df.index[:3])}")
                else:
                    logger.warning(f"Missing required columns in {features_path}")
                    metadata = None
            except Exception as e:
                logger.warning(f"Could not load {features_path}: {e}")
                metadata = None
        else:
            logger.warning(f"Features file not found: {features_path}")
            metadata = None
    else:
        # Auto-detect path
        metadata = load_features_metadata(dataset_name)
    
    # Find ECG records
    ecg_files = find_ecg_records(data_path, dataset_name, n_samples)
    logger.info(f"Total records to process: {len(ecg_files)}")
    
    # Load checkpoint if resuming
    results = []
    errors = []
    processed_ids: Set[str] = set()
    
    if resume:
        results, errors, processed_ids = load_checkpoint(output_path, dataset_name)
        if processed_ids:
            remaining = len(ecg_files) - len(processed_ids)
            logger.info(f"🔄 Resuming: {remaining} records remaining")
    else:
        clear_checkpoints(output_path, dataset_name)
        logger.info("Starting fresh (checkpoints cleared)")
    
    # Process records
    checkpoint_counter = 0
    
    for record_path in tqdm(ecg_files, desc="Processing ECGs"):
        ecg_id = extract_ecg_id(record_path, dataset_name)
        
        # Skip if already processed
        if ecg_id in processed_ids:
            continue
        
        try:
            # Load signal
            signal = load_ecg_record(record_path, sampling_rate)
            
            if signal is None:
                errors.append({'ecg_id': ecg_id, 'error': 'Failed to load signal'})
                processed_ids.add(ecg_id)
                checkpoint_counter += 1
                continue
            
            # Delineate
            delin_result = delineate_ecg(signal, sampling_rate, lead_idx)
            
            if not delin_result.get('success', False):
                errors.append({
                    'ecg_id': ecg_id,
                    'error': delin_result.get('error', 'Unknown error')
                })
                processed_ids.add(ecg_id)
                checkpoint_counter += 1
                continue
            
            # Calculate QTc
            qtc_values = calculate_qtc_formulas(
                delin_result['QT_interval_ms'],
                delin_result['RR_interval_sec'],
                delin_result['heart_rate_bpm']
            )
            
            # Get metadata (age/sex) from process_dataset output
            meta = get_metadata_for_record(ecg_id, metadata)
            
            # Combine results
            record_result = {
                'ecg_id': ecg_id,
                'age': meta.get('age'),
                'sex': meta.get('sex'),
                **{k: v for k, v in delin_result.items() if k != 'success'},
                **qtc_values,
            }
            
            results.append(record_result)
            processed_ids.add(ecg_id)
            checkpoint_counter += 1
            
        except Exception as e:
            errors.append({'ecg_id': ecg_id, 'error': str(e)})
            processed_ids.add(ecg_id)
            checkpoint_counter += 1
        
        # Save checkpoint every CHECKPOINT_INTERVAL records
        if checkpoint_counter >= CHECKPOINT_INTERVAL:
            save_checkpoint(output_path, dataset_name, results, errors, processed_ids)
            checkpoint_counter = 0
    
    # Final checkpoint save
    if checkpoint_counter > 0:
        save_checkpoint(output_path, dataset_name, results, errors, processed_ids)
    
    # Create DataFrames
    df_results = pd.DataFrame(results)
    df_errors = pd.DataFrame(errors)
    
    # Summary
    n_total = len(ecg_files)
    n_success = len(df_results)
    n_failed = len(df_errors)
    
    summary = {
        'processing_stats': {
            'total_records': n_total,
            'successful': n_success,
            'failed': n_failed,
            'success_rate_pct': round(100 * n_success / n_total, 2) if n_total > 0 else 0,
        },
        'interval_statistics': {},
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'dataset': dataset_name,
            'sampling_rate': sampling_rate,
            'lead_index': lead_idx,
        }
    }
    
    if len(df_results) > 0:
        for col in ['QT_interval_ms', 'RR_interval_ms', 'heart_rate_bpm', 'QTc_Bazett_ms', 'QTc_Fridericia_ms']:
            if col in df_results.columns:
                summary['interval_statistics'][col] = {
                    'mean': round(df_results[col].mean(), 2),
                    'std': round(df_results[col].std(), 2),
                    'median': round(df_results[col].median(), 2),
                }
    
    # Save final outputs
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Only CSV output (no parquet, no JSON summary)
    df_results.to_csv(output_path / f'{dataset_name}_wave_features.csv', index=False)
    
    logger.info(f"✅ Saved final results to {output_path}")
    
    # Clear checkpoints after successful completion
    clear_checkpoints(output_path, dataset_name)
    logger.info("🧹 Checkpoints cleared (processing complete)")
    
    return df_results, summary


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Kepler-ECG Wave Delineation Pipeline'
    )
    
    parser.add_argument('--dataset', '-d', type=str,
                        help='Dataset name (ptb-xl, chapman, cpsc-2018, georgia, mimic-iv-ecg)')
    parser.add_argument('--input', '-i', type=str,
                        help='Custom input data path')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory')
    parser.add_argument('--sampling-rate', type=int, default=500,
                        choices=VALID_SAMPLING_RATES,
                        help=f'Sampling rate in Hz (default: 500, valid: {VALID_SAMPLING_RATES})')
    parser.add_argument('--lead', type=int, default=1,
                        help='Lead index (default: 1=II, recommended for QT measurement)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Max records to process (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint (default: start fresh)')
    parser.add_argument('--features-file', type=str, default=None,
                        help='Path to features CSV with age/sex (from process_dataset)')
    
    args = parser.parse_args()
    
    if not args.dataset and not args.input:
        parser.error("Must provide either --dataset or --input")
    
    # Determine paths
    if args.dataset:
        data_path = Path(f"data/raw/{args.dataset}")
        output_path = Path(args.output) if args.output else Path(f"results/{args.dataset}/waves")
        dataset_name = args.dataset
    else:
        data_path = Path(args.input)
        output_path = Path(args.output) if args.output else Path("results/waves")
        dataset_name = data_path.stem
    
    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        return 1
    
    print("="*60)
    print("KEPLER-ECG WAVE DELINEATION")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Input: {data_path}")
    print(f"Output: {output_path}")
    print(f"Sampling rate: {args.sampling_rate} Hz")
    print(f"Lead: {args.lead}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL} records")
    print(f"Resume from checkpoint: {args.resume}")
    if args.features_file:
        print(f"Features file: {args.features_file}")
    
    df_results, summary = process_dataset(
        data_path=data_path,
        output_path=output_path,
        dataset_name=dataset_name,
        sampling_rate=args.sampling_rate,
        lead_idx=args.lead,
        n_samples=args.n_samples,
        resume=args.resume,
        features_file=args.features_file,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    stats = summary['processing_stats']
    print(f"Total: {stats['total_records']}")
    print(f"Success: {stats['successful']} ({stats['success_rate_pct']}%)")
    print(f"Failed: {stats['failed']}")
    
    if 'interval_statistics' in summary and summary['interval_statistics']:
        print("\nInterval Statistics:")
        for col, vals in summary['interval_statistics'].items():
            print(f"  {col}: {vals['mean']:.1f} ± {vals['std']:.1f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
