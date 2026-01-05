#!/usr/bin/env python3
"""
Kepler-ECG: Wave Delineation Pipeline

Extracts PQRST fiducial points from ECG records using NeuroKit2.
Supports multiple datasets: PTB-XL, Chapman, CPSC-2018, Georgia.

Output features:
- RR, PR, QRS, QT intervals
- P, QRS, T wave amplitudes
- QTc values (Bazett, Fridericia, Framingham, Hodges)
- Signal quality metrics

Usage:
    # Process PTB-XL
    python scripts/extract_waves.py --dataset ptb-xl
    
    # Process Chapman with custom lead
    python scripts/extract_waves.py --dataset chapman --lead 1
    
    # Process custom path
    python scripts/extract_waves.py --input data/raw/mydata --output results/waves

Author: Kepler-ECG Project
Version: 2.0.0
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    lead_idx: int = 0,
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
            if 80 < pr < 400:
                pr_intervals.append(pr)
    pr_median = np.median(pr_intervals) if pr_intervals else np.nan
    
    # QRS duration
    qrs_durations = []
    for i in range(min(len(q_peaks), len(s_peaks))):
        q, s = q_peaks[i], s_peaks[i]
        if q is not None and s is not None and not np.isnan(q) and not np.isnan(s):
            qrs = (s - q) * ms_per_sample
            if 40 < qrs < 200:
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
            if 200 < qt < 700:
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
# Dataset Processing
# ============================================================================

def load_ecg_record(record_path: Path, sampling_rate: int = 500) -> Optional[np.ndarray]:
    """Load ECG signal from WFDB record."""
    try:
        record = wfdb.rdrecord(str(record_path))
        return record.p_signal
    except Exception as e:
        logger.debug(f"Failed to load {record_path}: {e}")
        return None


def process_dataset(
    data_path: Path,
    output_path: Path,
    dataset_name: str,
    sampling_rate: int = 500,
    lead_idx: int = 0,
    max_records: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Process all ECGs in a dataset."""
    
    # Try to load metadata
    metadata = None
    metadata_candidates = [
        data_path / 'ptbxl_database.csv',
        data_path / 'metadata.csv',
        data_path / f'{dataset_name}_metadata.csv',
    ]
    
    for mp in metadata_candidates:
        if mp.exists():
            metadata = pd.read_csv(mp)
            if 'ecg_id' in metadata.columns:
                metadata = metadata.set_index('ecg_id')
            logger.info(f"Loaded metadata: {len(metadata)} records")
            break
    
    # Find ECG records
    ecg_files = list(data_path.rglob('*.hea'))
    logger.info(f"Found {len(ecg_files)} ECG records")
    
    if max_records:
        ecg_files = ecg_files[:max_records]
    
    results = []
    errors = []
    
    for hea_file in tqdm(ecg_files, desc="Processing ECGs"):
        record_path = hea_file.with_suffix('')
        ecg_id = record_path.stem
        
        try:
            # Load signal
            signal = load_ecg_record(record_path, sampling_rate)
            
            if signal is None:
                errors.append({'ecg_id': ecg_id, 'error': 'Failed to load signal'})
                continue
            
            # Delineate
            delin_result = delineate_ecg(signal, sampling_rate, lead_idx)
            
            if not delin_result.get('success', False):
                errors.append({
                    'ecg_id': ecg_id,
                    'error': delin_result.get('error', 'Unknown error')
                })
                continue
            
            # Calculate QTc
            qtc_values = calculate_qtc_formulas(
                delin_result['QT_interval_ms'],
                delin_result['RR_interval_sec'],
                delin_result['heart_rate_bpm']
            )
            
            # Get metadata if available
            meta = {}
            if metadata is not None:
                try:
                    ecg_id_num = int(ecg_id.replace('JS', '').replace('_hr', '').replace('_lr', ''))
                    if ecg_id_num in metadata.index:
                        row = metadata.loc[ecg_id_num]
                        meta = {
                            'age': row.get('age', np.nan),
                            'sex': row.get('sex', np.nan),
                            'scp_codes': row.get('scp_codes', ''),
                        }
                except:
                    pass
            
            # Combine results
            record_result = {
                'ecg_id': ecg_id,
                **meta,
                **delin_result,
                **qtc_values,
            }
            
            record_result.pop('success', None)
            record_result.pop('error', None)
            
            results.append(record_result)
            
        except Exception as e:
            errors.append({'ecg_id': ecg_id, 'error': str(e)})
    
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
    
    # Save outputs
    output_path.mkdir(parents=True, exist_ok=True)
    
    df_results.to_parquet(output_path / 'wave_features.parquet', index=False)
    df_results.to_csv(output_path / 'wave_features.csv', index=False)
    
    if len(df_errors) > 0:
        df_errors.to_csv(output_path / 'delineation_errors.csv', index=False)
    
    with open(output_path / 'wave_delineation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")
    
    return df_results, summary


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Kepler-ECG Wave Delineation Pipeline'
    )
    
    parser.add_argument('--dataset', '-d', type=str,
                        help='Dataset name (ptb-xl, chapman, cpsc-2018, georgia)')
    parser.add_argument('--input', '-i', type=str,
                        help='Custom input data path')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory')
    parser.add_argument('--sampling-rate', type=int, default=500,
                        choices=[100, 500],
                        help='Sampling rate in Hz')
    parser.add_argument('--lead', type=int, default=0,
                        help='Lead index (0=I, 1=II, etc.)')
    parser.add_argument('--max-records', type=int, default=None,
                        help='Max records to process (for testing)')
    
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
    
    df_results, summary = process_dataset(
        data_path=data_path,
        output_path=output_path,
        dataset_name=dataset_name,
        sampling_rate=args.sampling_rate,
        lead_idx=args.lead,
        max_records=args.max_records,
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
