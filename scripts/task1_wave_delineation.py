#!/usr/bin/env python3
"""
Kepler-ECG Phase 4.5 - Task 1: Wave Delineation Pipeline
=========================================================

Extracts PQRST fiducial points from all PTB-XL ECG records using NeuroKit2.

Author: Kepler-ECG Project
Date: 2025-12-16

Usage:
    python task1_wave_delineation.py --data_path data/raw/ptb-xl --output_path ./results/stream_c
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wave_delineation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_ptbxl_metadata(data_path: Path) -> pd.DataFrame:
    """Load PTB-XL metadata."""
    metadata_path = data_path / 'ptbxl_database.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    df = pd.read_csv(metadata_path, index_col='ecg_id')
    logger.info(f"Loaded metadata for {len(df)} ECG records")
    return df


def load_ecg_signal(data_path: Path, filename: str, sampling_rate: int = 500) -> Optional[np.ndarray]:
    """
    Load ECG signal from PTB-XL.
    
    Args:
        data_path: Path to PTB-XL data
        filename: Relative path to the record (e.g., 'records500/00000/00001_hr')
        sampling_rate: 100 or 500 Hz
        
    Returns:
        ECG signal array (n_samples, 12) or None if failed
    """
    try:
        # Construct full path (remove _lr or _hr suffix if present in filename)
        if sampling_rate == 500:
            record_path = data_path / filename.replace('_lr', '_hr')
        else:
            record_path = data_path / filename.replace('_hr', '_lr')
        
        # Remove extension if present
        record_path = str(record_path).replace('.dat', '').replace('.hea', '')
        
        # Load using wfdb
        record = wfdb.rdrecord(record_path)
        return record.p_signal
        
    except Exception as e:
        logger.debug(f"Failed to load {filename}: {e}")
        return None


def delineate_ecg_neurokit(
    ecg_signal: np.ndarray,
    sampling_rate: int = 500,
    lead_idx: int = 0  # Lead I by default
) -> Dict:
    """
    Perform wave delineation using NeuroKit2.
    
    Args:
        ecg_signal: ECG signal array (n_samples, 12)
        sampling_rate: Sampling rate in Hz
        lead_idx: Which lead to process (0=I, 1=II, etc.)
        
    Returns:
        Dictionary with delineation results
    """
    import neurokit2 as nk
    
    # Extract single lead
    lead_signal = ecg_signal[:, lead_idx]
    
    # Process ECG with NeuroKit2
    try:
        # Clean the signal
        ecg_cleaned = nk.ecg_clean(lead_signal, sampling_rate=sampling_rate)
        
        # Find R-peaks
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        r_peaks = rpeaks['ECG_R_Peaks']
        
        if len(r_peaks) < 3:
            return {'success': False, 'error': 'Too few R-peaks detected'}
        
        # Delineate waves (PQRST)
        _, waves = nk.ecg_delineate(
            ecg_cleaned, 
            rpeaks=rpeaks,
            sampling_rate=sampling_rate,
            method='dwt'  # Discrete Wavelet Transform method
        )
        
        # Calculate signal quality
        #quality = nk.ecg_quality(ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate)
        # Corretto: passa direttamente il dizionario dei picchi o gli indici corretti
        quality = nk.ecg_quality(ecg_cleaned, rpeaks=rpeaks["ECG_R_Peaks"], sampling_rate=500)
        mean_quality = np.nanmean(quality)
        
        # Extract intervals and features
        result = extract_intervals_and_features(
            ecg_cleaned, r_peaks, waves, sampling_rate, mean_quality
        )
        result['success'] = True
        
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def extract_intervals_and_features(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    waves: Dict,
    sampling_rate: int,
    quality_score: float
) -> Dict:
    """
    Extract clinical intervals and features from delineation results.
    
    Computes median values across all detected beats for robustness.
    """
    ms_per_sample = 1000 / sampling_rate
    
    # Helper to get median of valid values
    def safe_median(arr):
        valid = [x for x in arr if x is not None and not np.isnan(x)]
        return np.median(valid) if valid else np.nan
    
    def safe_diff_median(arr1, arr2):
        """Compute median of pairwise differences."""
        diffs = []
        for a, b in zip(arr1, arr2):
            if a is not None and b is not None and not np.isnan(a) and not np.isnan(b):
                diffs.append((b - a) * ms_per_sample)
        return np.median(diffs) if diffs else np.nan
    
    # Extract wave positions
    p_onsets = waves.get('ECG_P_Onsets', [])
    p_peaks = waves.get('ECG_P_Peaks', [])
    p_offsets = waves.get('ECG_P_Offsets', [])
    q_peaks = waves.get('ECG_Q_Peaks', [])
    s_peaks = waves.get('ECG_S_Peaks', [])
    t_onsets = waves.get('ECG_T_Onsets', [])
    t_peaks = waves.get('ECG_T_Peaks', [])
    t_offsets = waves.get('ECG_T_Offsets', [])
    
    # RR intervals (from R-peaks)
    rr_intervals = np.diff(r_peaks) * ms_per_sample
    rr_median = np.median(rr_intervals) if len(rr_intervals) > 0 else np.nan
    
    # Heart rate
    hr = 60000 / rr_median if rr_median > 0 else np.nan
    
    # PR interval: P_onset to R_peak
    pr_intervals = []
    for i, r in enumerate(r_peaks[:-1]):
        if i < len(p_onsets) and p_onsets[i] is not None and not np.isnan(p_onsets[i]):
            pr = (r - p_onsets[i]) * ms_per_sample
            if 80 < pr < 400:  # Physiological range
                pr_intervals.append(pr)
    pr_median = np.median(pr_intervals) if pr_intervals else np.nan
    
    # QRS duration: Q_peak to S_peak (or estimate from R-peak vicinity)
    qrs_durations = []
    for i in range(min(len(q_peaks), len(s_peaks))):
        q = q_peaks[i]
        s = s_peaks[i]
        if q is not None and s is not None and not np.isnan(q) and not np.isnan(s):
            qrs = (s - q) * ms_per_sample
            if 40 < qrs < 200:  # Physiological range
                qrs_durations.append(qrs)
    qrs_median = np.median(qrs_durations) if qrs_durations else np.nan
    
    # QT interval: Q_peak (or R-40ms) to T_offset
    qt_intervals = []
    for i in range(min(len(r_peaks), len(t_offsets))):
        # Use Q peak if available, otherwise estimate from R-peak
        if i < len(q_peaks) and q_peaks[i] is not None and not np.isnan(q_peaks[i]):
            q_start = q_peaks[i]
        else:
            # Estimate Q as R - 40ms
            q_start = r_peaks[i] - int(40 / ms_per_sample)
        
        t_end = t_offsets[i] if i < len(t_offsets) else None
        
        if t_end is not None and not np.isnan(t_end):
            qt = (t_end - q_start) * ms_per_sample
            if 200 < qt < 700:  # Physiological range
                qt_intervals.append(qt)
    
    qt_median = np.median(qt_intervals) if qt_intervals else np.nan
    
    # Amplitudes (in mV, assuming signal is already in mV)
    def get_amplitude_at_peaks(signal, peaks):
        amps = []
        for p in peaks:
            if p is not None and not np.isnan(p) and 0 <= int(p) < len(signal):
                amps.append(signal[int(p)])
        return np.median(amps) if amps else np.nan
    
    r_amplitude = get_amplitude_at_peaks(ecg_signal, r_peaks)
    p_amplitude = get_amplitude_at_peaks(ecg_signal, [p for p in p_peaks if p is not None])
    t_amplitude = get_amplitude_at_peaks(ecg_signal, [t for t in t_peaks if t is not None])
    
    # S amplitude (negative in most cases)
    s_amplitude = get_amplitude_at_peaks(ecg_signal, [s for s in s_peaks if s is not None])
    
    # Count successful detections
    n_beats = len(r_peaks)
    n_p_detected = sum(1 for p in p_peaks if p is not None and not np.isnan(p))
    n_t_detected = sum(1 for t in t_offsets if t is not None and not np.isnan(t))
    
    return {
        # Fiducial points (median positions in samples)
        'P_onset_median': safe_median(p_onsets),
        'P_peak_median': safe_median(p_peaks),
        'P_offset_median': safe_median(p_offsets),
        'Q_peak_median': safe_median(q_peaks),
        'R_peak_median': safe_median(r_peaks) if len(r_peaks) > 0 else np.nan,
        'S_peak_median': safe_median(s_peaks),
        'T_onset_median': safe_median(t_onsets),
        'T_peak_median': safe_median(t_peaks),
        'T_offset_median': safe_median(t_offsets),
        
        # Intervals (in ms)
        'RR_interval_ms': rr_median,
        'RR_interval_sec': rr_median / 1000 if not np.isnan(rr_median) else np.nan,
        'PR_interval_ms': pr_median,
        'QRS_duration_ms': qrs_median,
        'QT_interval_ms': qt_median,
        
        # Heart rate
        'heart_rate_bpm': hr,
        
        # Amplitudes (mV)
        'P_amplitude_mV': p_amplitude,
        'R_amplitude_mV': r_amplitude,
        'S_amplitude_mV': s_amplitude,
        'T_amplitude_mV': t_amplitude,
        
        # Quality metrics
        'signal_quality': quality_score,
        'n_beats_detected': n_beats,
        'n_p_waves_detected': n_p_detected,
        'n_t_waves_detected': n_t_detected,
        'p_wave_detection_rate': n_p_detected / n_beats if n_beats > 0 else 0,
        't_wave_detection_rate': n_t_detected / n_beats if n_beats > 0 else 0,
    }


def calculate_qtc_formulas(qt_ms: float, rr_sec: float, hr_bpm: float) -> Dict:
    """
    Calculate QTc using various standard formulas.
    
    Args:
        qt_ms: QT interval in milliseconds
        rr_sec: RR interval in seconds
        hr_bpm: Heart rate in beats per minute
        
    Returns:
        Dictionary with QTc values from different formulas
    """
    if np.isnan(qt_ms) or np.isnan(rr_sec) or rr_sec <= 0:
        return {
            'QTc_Bazett_ms': np.nan,
            'QTc_Fridericia_ms': np.nan,
            'QTc_Framingham_ms': np.nan,
            'QTc_Hodges_ms': np.nan,
        }
    
    # Bazett: QTc = QT / sqrt(RR)
    qtc_bazett = qt_ms / np.sqrt(rr_sec)
    
    # Fridericia: QTc = QT / cbrt(RR)
    qtc_fridericia = qt_ms / np.cbrt(rr_sec)
    
    # Framingham: QTc = QT + 0.154 * (1 - RR) * 1000
    # Note: Original formula uses RR in seconds, correction in ms
    qtc_framingham = qt_ms + 154 * (1 - rr_sec)
    
    # Hodges: QTc = QT + 1.75 * (HR - 60)
    qtc_hodges = qt_ms + 1.75 * (hr_bpm - 60) if not np.isnan(hr_bpm) else np.nan
    
    return {
        'QTc_Bazett_ms': qtc_bazett,
        'QTc_Fridericia_ms': qtc_fridericia,
        'QTc_Framingham_ms': qtc_framingham,
        'QTc_Hodges_ms': qtc_hodges,
    }


def process_all_ecgs(
    data_path: Path,
    output_path: Path,
    sampling_rate: int = 500,
    lead_idx: int = 0,
    max_records: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process all ECG records and extract wave features.
    
    Args:
        data_path: Path to PTB-XL data
        output_path: Path for output files
        sampling_rate: 100 or 500 Hz
        lead_idx: Lead index to process
        max_records: Limit for testing (None = all)
        
    Returns:
        DataFrame with all features and summary statistics
    """
    # Load metadata
    metadata = load_ptbxl_metadata(data_path)
    
    if max_records:
        metadata = metadata.head(max_records)
        logger.info(f"Processing limited to {max_records} records (testing mode)")
    
    # Determine filename column based on sampling rate
    filename_col = 'filename_hr' if sampling_rate == 500 else 'filename_lr'
    
    results = []
    errors = []
    
    logger.info(f"Starting wave delineation for {len(metadata)} ECG records...")
    
    for ecg_id, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing ECGs"):
        try:
            # Load signal
            signal = load_ecg_signal(data_path, row[filename_col], sampling_rate)
            
            if signal is None:
                errors.append({'ecg_id': ecg_id, 'error': 'Failed to load signal'})
                continue
            
            # Perform delineation
            delin_result = delineate_ecg_neurokit(signal, sampling_rate, lead_idx)
            
            if not delin_result.get('success', False):
                errors.append({
                    'ecg_id': ecg_id, 
                    'error': delin_result.get('error', 'Unknown error')
                })
                continue
            
            # Calculate QTc formulas
            qtc_values = calculate_qtc_formulas(
                delin_result['QT_interval_ms'],
                delin_result['RR_interval_sec'],
                delin_result['heart_rate_bpm']
            )
            
            # Combine results
            record_result = {
                'ecg_id': ecg_id,
                'age': row.get('age', np.nan),
                'sex': row.get('sex', np.nan),
                'scp_codes': row.get('scp_codes', ''),
                **delin_result,
                **qtc_values,
                'delineation_success': True
            }
            
            # Remove internal keys
            record_result.pop('success', None)
            record_result.pop('error', None)
            
            results.append(record_result)
            
        except Exception as e:
            errors.append({'ecg_id': ecg_id, 'error': str(e)})
            logger.debug(f"Error processing ECG {ecg_id}: {e}")
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    df_errors = pd.DataFrame(errors)
    
    # Calculate summary statistics
    n_total = len(metadata)
    n_success = len(df_results)
    n_failed = len(df_errors)
    success_rate = n_success / n_total * 100 if n_total > 0 else 0
    
    # Quality statistics
    qt_valid = df_results['QT_interval_ms'].notna().sum()
    rr_valid = df_results['RR_interval_ms'].notna().sum()
    
    summary = {
        'processing_stats': {
            'total_records': n_total,
            'successful': n_success,
            'failed': n_failed,
            'success_rate_pct': round(success_rate, 2),
            'qt_valid_count': int(qt_valid),
            'rr_valid_count': int(rr_valid),
        },
        'interval_statistics': {
            'QT_ms': {
                'mean': round(df_results['QT_interval_ms'].mean(), 2),
                'std': round(df_results['QT_interval_ms'].std(), 2),
                'median': round(df_results['QT_interval_ms'].median(), 2),
                'min': round(df_results['QT_interval_ms'].min(), 2),
                'max': round(df_results['QT_interval_ms'].max(), 2),
            },
            'RR_ms': {
                'mean': round(df_results['RR_interval_ms'].mean(), 2),
                'std': round(df_results['RR_interval_ms'].std(), 2),
                'median': round(df_results['RR_interval_ms'].median(), 2),
            },
            'HR_bpm': {
                'mean': round(df_results['heart_rate_bpm'].mean(), 2),
                'std': round(df_results['heart_rate_bpm'].std(), 2),
                'median': round(df_results['heart_rate_bpm'].median(), 2),
            },
            'QTc_Bazett_ms': {
                'mean': round(df_results['QTc_Bazett_ms'].mean(), 2),
                'std': round(df_results['QTc_Bazett_ms'].std(), 2),
            },
            'QTc_Fridericia_ms': {
                'mean': round(df_results['QTc_Fridericia_ms'].mean(), 2),
                'std': round(df_results['QTc_Fridericia_ms'].std(), 2),
            },
        },
        'quality_metrics': {
            'mean_signal_quality': round(df_results['signal_quality'].mean(), 4),
            'mean_p_detection_rate': round(df_results['p_wave_detection_rate'].mean(), 4),
            'mean_t_detection_rate': round(df_results['t_wave_detection_rate'].mean(), 4),
        },
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'sampling_rate': sampling_rate,
            'lead_index': lead_idx,
            'method': 'neurokit2_dwt',
        }
    }
    
    # Save outputs
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    output_file = output_path / 'wave_features.parquet'
    df_results.to_parquet(output_file, index=False)
    logger.info(f"Saved wave features to {output_file}")
    
    # Also save as CSV for easy inspection
    csv_file = output_path / 'wave_features.csv'
    df_results.to_csv(csv_file, index=False)
    logger.info(f"Saved wave features to {csv_file}")
    
    # Save errors log
    if len(df_errors) > 0:
        errors_file = output_path / 'delineation_errors.csv'
        df_errors.to_csv(errors_file, index=False)
        logger.info(f"Saved {len(df_errors)} errors to {errors_file}")
    
    # Save summary
    summary_file = output_path / 'wave_delineation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")
    
    return df_results, summary


def main():
    parser = argparse.ArgumentParser(
        description='Kepler-ECG Wave Delineation Pipeline'
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        required=True,
        help='Path to PTB-XL data directory'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='./results/stream_c',
        help='Output directory for results'
    )
    parser.add_argument(
        '--sampling_rate', 
        type=int, 
        default=500,
        choices=[100, 500],
        help='Sampling rate (100 or 500 Hz)'
    )
    parser.add_argument(
        '--lead', 
        type=int, 
        default=0,
        help='Lead index to process (0=I, 1=II, etc.)'
    )
    parser.add_argument(
        '--max_records', 
        type=int, 
        default=None,
        help='Maximum records to process (for testing)'
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    logger.info("=" * 60)
    logger.info("Kepler-ECG Phase 4.5 - Task 1: Wave Delineation")
    logger.info("=" * 60)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Sampling rate: {args.sampling_rate} Hz")
    logger.info(f"Lead index: {args.lead}")
    
    df_results, summary = process_all_ecgs(
        data_path=data_path,
        output_path=output_path,
        sampling_rate=args.sampling_rate,
        lead_idx=args.lead,
        max_records=args.max_records
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("WAVE DELINEATION SUMMARY")
    print("=" * 60)
    print(f"Total records processed: {summary['processing_stats']['total_records']}")
    print(f"Successful: {summary['processing_stats']['successful']}")
    print(f"Failed: {summary['processing_stats']['failed']}")
    print(f"Success rate: {summary['processing_stats']['success_rate_pct']:.2f}%")
    print()
    print("Interval Statistics:")
    print(f"  QT interval: {summary['interval_statistics']['QT_ms']['mean']:.1f} ± "
          f"{summary['interval_statistics']['QT_ms']['std']:.1f} ms")
    print(f"  RR interval: {summary['interval_statistics']['RR_ms']['mean']:.1f} ± "
          f"{summary['interval_statistics']['RR_ms']['std']:.1f} ms")
    print(f"  Heart rate: {summary['interval_statistics']['HR_bpm']['mean']:.1f} ± "
          f"{summary['interval_statistics']['HR_bpm']['std']:.1f} bpm")
    print(f"  QTc Bazett: {summary['interval_statistics']['QTc_Bazett_ms']['mean']:.1f} ± "
          f"{summary['interval_statistics']['QTc_Bazett_ms']['std']:.1f} ms")
    print(f"  QTc Fridericia: {summary['interval_statistics']['QTc_Fridericia_ms']['mean']:.1f} ± "
          f"{summary['interval_statistics']['QTc_Fridericia_ms']['std']:.1f} ms")
    print("=" * 60)
    
    return df_results, summary


if __name__ == '__main__':
    main()
