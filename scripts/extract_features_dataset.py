#!/usr/bin/env python3
"""
Kepler-ECG: Generic Feature Extraction Script

Estrae features avanzate da qualsiasi dataset ECG preprocessato.

Questo script:
1. Carica i dati preprocessati dalla Fase 1 (*_features.csv)
2. Carica i segnali ECG grezzi dal dataset originale
3. Estrae features avanzate (morfologiche, spettrali, wavelet, compressibilità)
4. Mappa le diagnosi SCP-ECG se disponibili (PTB-XL)
5. Salva il risultato in *_features_extracted.csv

Funziona con tutti i dataset supportati da process_dataset.py:
- PTB-XL, Chapman, Georgia, CPSC, MIT-BIH, LTAF, QTDB, etc.

Usage:
    python scripts/extract_features.py --phase1 ./results/dataset/dataset_features.csv --data-dir ./data/raw/dataset

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Importa moduli Kepler-ECG
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features import (
    MorphologicalExtractor,
    IntervalCalculator, 
    SpectralAnalyzer,
    WaveletExtractor,
    CompressibilityCalculator,
    DiagnosisMapper,
)


# ============================================================================
# Configurazione
# ============================================================================

# Features da estrarre (possono essere disabilitate via CLI)
DEFAULT_FEATURES = {
    'morphological': True,
    'intervals': True,
    'spectral': True,
    'wavelet': True,
    'compressibility': True,
    'diagnosis': True,
}

# Lead preferiti per analisi (in ordine di preferenza)
PREFERRED_LEADS = ['II', 'MLII', 'I', 'V5', 'V1', 'ECG', 'ECG1']


# ============================================================================
# Dataset Detection (riusa logica da process_dataset.py)
# ============================================================================

class DatasetInfo:
    """Information about detected dataset structure."""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.name = data_dir.name
        self.format = 'unknown'
        self.record_paths: List[Path] = []
        self.sampling_rate: Optional[int] = None
        self.n_records = 0


def detect_dataset(data_dir: Path) -> DatasetInfo:
    """Auto-detect dataset structure."""
    info = DatasetInfo(data_dir)
    
    # Count files by extension
    extension_counts = defaultdict(int)
    all_files = []
    
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            ext = Path(f).suffix.lower()
            extension_counts[ext] += 1
            all_files.append(Path(root) / f)
    
    # Detect format
    if extension_counts.get('.dat', 0) > 0 and extension_counts.get('.hea', 0) > 0:
        info.format = 'wfdb'
        # Find record files
        for f in all_files:
            if f.suffix.lower() == '.hea':
                dat_file = f.with_suffix('.dat')
                if dat_file.exists():
                    info.record_paths.append(f.with_suffix(''))
    elif extension_counts.get('.mat', 0) > 0:
        info.format = 'mat'
        info.record_paths = [f for f in all_files if f.suffix.lower() == '.mat']
    elif extension_counts.get('.npy', 0) > 0:
        info.format = 'npy'
        info.record_paths = [f for f in all_files if f.suffix.lower() == '.npy']
    
    info.record_paths = sorted(info.record_paths)
    info.n_records = len(info.record_paths)
    
    # Detect sampling rate from first record
    if info.n_records > 0:
        info.sampling_rate = _detect_sampling_rate(info.record_paths[0], info.format)
    
    return info


def _detect_sampling_rate(record_path: Path, format: str) -> Optional[int]:
    """Detect sampling rate from first record."""
    try:
        if format == 'wfdb':
            import wfdb
            record = wfdb.rdheader(str(record_path))
            return int(record.fs)
        elif format == 'mat':
            import scipy.io as sio
            mat = sio.loadmat(str(record_path))
            for key in ['fs', 'Fs', 'sampling_rate', 'sfreq', 'frequency']:
                if key in mat:
                    val = mat[key]
                    if hasattr(val, 'item'):
                        return int(val.item())
                    return int(val.flatten()[0])
            return 500  # Default
        else:
            return None
    except Exception:
        return None


# ============================================================================
# Signal Loading
# ============================================================================

def load_ecg_signal(record_path: Path, format: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[List[str]]]:
    """
    Load ECG signal from any supported format.
    
    Returns
    -------
    signal : np.ndarray or None
        ECG signal (n_samples, n_leads)
    fs : int or None
        Sampling rate
    lead_names : List[str] or None
        Names of leads
    """
    try:
        if format == 'wfdb':
            import wfdb
            record = wfdb.rdrecord(str(record_path))
            return record.p_signal, int(record.fs), record.sig_name
            
        elif format == 'mat':
            import scipy.io as sio
            mat = sio.loadmat(str(record_path))
            
            # Find signal array
            signal = None
            for key in ['val', 'ECG', 'ecg', 'signal', 'data', 'X']:
                if key in mat:
                    signal = np.asarray(mat[key], dtype=float)
                    break
            
            if signal is None:
                for key, val in mat.items():
                    if not key.startswith('_') and isinstance(val, np.ndarray):
                        if val.size > 1000:
                            signal = np.asarray(val, dtype=float)
                            break
            
            if signal is None:
                return None, None, None
            
            # Ensure (n_samples, n_leads) shape
            if signal.ndim == 1:
                signal = signal.reshape(-1, 1)
            elif signal.shape[0] < signal.shape[1]:
                signal = signal.T
            
            # Get sampling rate
            fs = 500
            for key in ['fs', 'Fs', 'sampling_rate']:
                if key in mat:
                    val = mat[key]
                    fs = int(val.item() if hasattr(val, 'item') else val.flatten()[0])
                    break
            
            return signal, fs, None
            
        elif format == 'npy':
            signal = np.load(str(record_path))
            if signal.ndim == 1:
                signal = signal.reshape(-1, 1)
            return signal, 500, None  # Assume 500 Hz
            
    except Exception as e:
        return None, None, None
    
    return None, None, None


def select_lead(signal: np.ndarray, lead_names: Optional[List[str]], lead_index: int = 0) -> np.ndarray:
    """
    Select the best lead for analysis.
    
    Parameters
    ----------
    signal : np.ndarray
        Multi-lead signal (n_samples, n_leads)
    lead_names : List[str] or None
        Names of leads
    lead_index : int
        Fallback lead index if names not available
        
    Returns
    -------
    np.ndarray
        Single-lead signal (n_samples,)
    """
    if signal.ndim == 1:
        return signal
    
    # Try to find preferred lead by name
    if lead_names is not None:
        for preferred in PREFERRED_LEADS:
            for i, name in enumerate(lead_names):
                if name.upper() == preferred.upper():
                    return signal[:, i]
    
    # Fallback to specified index
    n_leads = signal.shape[1]
    if lead_index < n_leads:
        return signal[:, lead_index]
    
    return signal[:, 0]


def find_record_path(data_dir: Path, ecg_id: str, record_paths: List[Path]) -> Optional[Path]:
    """
    Find the record path for a given ecg_id.
    
    Handles various naming conventions:
    - PTB-XL: "00001_hr" -> records500/00000/00001_hr
    - Generic: "record_name" -> exact match in record_paths
    """
    # Clean up ecg_id (remove extension if present)
    ecg_id_clean = ecg_id.replace('.hea', '').replace('.dat', '').replace('.mat', '').replace('.npy', '')
    
    # Try direct match first
    for rp in record_paths:
        if rp.stem == ecg_id_clean or rp.name == ecg_id_clean:
            return rp
    
    # Try PTB-XL style (numeric ID with _hr suffix)
    if '_hr' in ecg_id_clean:
        try:
            numeric_id = int(ecg_id_clean.split('_')[0].lstrip('0') or '0')
            folder_num = (numeric_id // 1000) * 1000
            folder = f"{folder_num:05d}"
            
            # Try records500 path
            ptbxl_path = data_dir / "records500" / folder / ecg_id_clean
            if ptbxl_path.with_suffix('.hea').exists():
                return ptbxl_path
        except ValueError:
            pass
    
    # Try matching just the numeric part
    try:
        if isinstance(ecg_id_clean, str):
            numeric_part = ecg_id_clean.split('_')[0].lstrip('0') or '0'
        else:
            numeric_part = str(int(ecg_id_clean))
        
        for rp in record_paths:
            stem_numeric = rp.stem.split('_')[0].lstrip('0') or '0'
            if stem_numeric == numeric_part:
                return rp
    except (ValueError, AttributeError):
        pass
    
    return None


# ============================================================================
# RR Interval Reconstruction
# ============================================================================

def get_rr_intervals_from_phase1(row: pd.Series) -> Optional[np.ndarray]:
    """
    Reconstruct RR intervals from Phase 1 statistics.
    
    Note: This is an approximation. For precise RR features,
    it would be better to recompute R-peaks from the signal.
    """
    rr_mean = row.get('rr_mean_ms')
    rr_std = row.get('rr_std_ms')
    n_beats = row.get('n_beats')
    
    if pd.isna(rr_mean) or pd.isna(n_beats) or n_beats < 2:
        return None
    
    # Generate synthetic RR based on mean and std
    ecg_id_raw = row.get('ecg_id', 0)
    if isinstance(ecg_id_raw, str):
        try:
            seed = int(ecg_id_raw.split('_')[0].lstrip('0') or '0')
        except ValueError:
            seed = hash(ecg_id_raw) % (2**31)
    else:
        seed = int(ecg_id_raw) if not pd.isna(ecg_id_raw) else 0
    
    np.random.seed(seed % (2**31))  # Ensure seed is valid
    rr_std_safe = rr_std if not pd.isna(rr_std) and rr_std > 0 else 50
    rr_intervals = np.random.normal(rr_mean, rr_std_safe, int(n_beats))
    rr_intervals = np.clip(rr_intervals, 300, 2000)  # Physiological limits
    
    return rr_intervals


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_all_features(
    signal: Optional[np.ndarray],
    rr_intervals: Optional[np.ndarray],
    row: pd.Series,
    extractors: Dict[str, Any],
    feature_flags: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Extract all features from a single ECG.
    
    Parameters
    ----------
    signal : np.ndarray or None
        ECG signal (single lead)
    rr_intervals : np.ndarray or None
        RR intervals in ms
    row : pd.Series
        DataFrame row with metadata
    extractors : dict
        Dictionary with initialized extractors
    feature_flags : dict
        Which features to extract
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with all extracted features
    """
    features = {}
    
    # 1. Morphological Features
    if feature_flags.get('morphological', True) and signal is not None:
        try:
            morph = extractors['morphological']
            template = signal[:500] if len(signal) >= 500 else signal
            rr_mean = row.get('rr_mean_ms', None)
            morph_features = morph.extract(template, rr_mean)
            features.update({f'morph_{k}': v for k, v in morph_features.items()})
        except Exception:
            pass
    
    # 2. Interval Features
    if feature_flags.get('intervals', True) and rr_intervals is not None and len(rr_intervals) > 1:
        try:
            intervals = extractors['intervals']
            int_features = intervals.calculate(rr_intervals)
            features.update({f'interval_{k}': v for k, v in int_features.items()})
        except Exception:
            pass
    
    # 3. Spectral HRV Features
    if feature_flags.get('spectral', True) and rr_intervals is not None and len(rr_intervals) >= 10:
        try:
            spectral = extractors['spectral']
            spec_features = spectral.extract(rr_intervals)
            features.update({f'hrv_{k}': v for k, v in spec_features.items()})
        except Exception:
            pass
    
    # 4. Wavelet Features
    if feature_flags.get('wavelet', True) and signal is not None and len(signal) >= 100:
        try:
            wavelet = extractors['wavelet']
            wav_features = wavelet.extract(signal[:2500])  # Max 5 seconds
            features.update({f'wav_{k}': v for k, v in wav_features.items()})
        except Exception:
            pass
    
    # 5. Compressibility Features
    if feature_flags.get('compressibility', True):
        try:
            compressor = extractors['compressor']
            
            # Signal compressibility
            if signal is not None and len(signal) >= 100:
                comp_signal = compressor.calculate(signal[:2500], methods=['compression', 'complexity'])
                features.update({f'comp_sig_{k}': v for k, v in comp_signal.items()})
            
            # RR compressibility
            # Limit to 1000 RR intervals to avoid O(n²) complexity explosion
            # Sample Entropy with 100k+ intervals would take hours
            if rr_intervals is not None and len(rr_intervals) >= 10:
                rr_for_entropy = rr_intervals[:1000] if len(rr_intervals) > 1000 else rr_intervals
                comp_rr = compressor.calculate(rr_for_entropy, methods=['entropy', 'complexity'])
                features.update({f'comp_rr_{k}': v for k, v in comp_rr.items()})
        except Exception:
            pass
    
    # 6. Diagnosis Mapping (works for PTB-XL, returns UNKNOWN for others)
    if feature_flags.get('diagnosis', True):
        try:
            mapper = extractors['diagnosis']
            scp_codes = row.get('scp_codes', '{}')
            
            # Skip if no SCP codes
            if scp_codes and scp_codes != '{}':
                info = mapper.map_scp_codes(scp_codes)
                
                features['diag_primary_code'] = info.primary_code
                features['diag_primary_category'] = info.primary_category
                features['diag_confidence'] = info.confidence
                features['diag_is_normal'] = int(info.is_normal)
                features['diag_is_multi_label'] = int(info.is_multi_label)
                features['diag_n_diagnoses'] = info.n_diagnoses
        except Exception:
            pass
    
    return features


# ============================================================================
# Main Processing
# ============================================================================

def process_dataset(
    phase1_path: str,
    data_dir: Optional[str],
    output_path: str,
    n_samples: Optional[int] = None,
    start_idx: int = 0,
    lead_index: int = 0,
    feature_flags: Optional[Dict[str, bool]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Process any ECG dataset and extract advanced features.
    
    Parameters
    ----------
    phase1_path : str
        Path to *_features.csv from Phase 1
    data_dir : str or None
        Path to raw dataset directory (if None, uses only RR approximations)
    output_path : str
        Path for output file
    n_samples : int or None
        Number of samples to process (None = all)
    start_idx : int
        Starting index
    lead_index : int
        Fallback lead index for analysis
    feature_flags : dict or None
        Which features to extract
    verbose : bool
        Show progress bar
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all features
    """
    # Load Phase 1 data
    print(f"Loading Phase 1 features from: {phase1_path}")
    df_phase1 = pd.read_csv(phase1_path)
    print(f"  Loaded {len(df_phase1)} rows with {len(df_phase1.columns)} columns")
    
    # Filter for success and usable
    df_valid = df_phase1[
        (df_phase1['success'] == True) & 
        (df_phase1['is_usable'] == True)
    ].copy()
    print(f"Valid ECGs: {len(df_valid)}")
    
    # Limit if requested
    if n_samples is not None:
        df_valid = df_valid.iloc[start_idx:start_idx + n_samples]
        print(f"Processing limited to {len(df_valid)} samples")
    
    # Initialize extractors
    print("\nInitializing extractors...")
    extractors = {
        'morphological': MorphologicalExtractor(),
        'intervals': IntervalCalculator(),
        'spectral': SpectralAnalyzer(),
        'wavelet': WaveletExtractor(),
        'compressor': CompressibilityCalculator(),
        'diagnosis': DiagnosisMapper(),
    }
    
    # Feature flags
    if feature_flags is None:
        feature_flags = DEFAULT_FEATURES.copy()
    
    # Detect dataset if data_dir provided
    dataset_info = None
    has_signals = False
    
    if data_dir is not None and Path(data_dir).exists():
        print(f"\nDetecting dataset structure in: {data_dir}")
        dataset_info = detect_dataset(Path(data_dir))
        has_signals = dataset_info.n_records > 0
        
        if has_signals:
            print(f"  Dataset: {dataset_info.name}")
            print(f"  Format: {dataset_info.format}")
            print(f"  Records: {dataset_info.n_records}")
            print(f"  Sampling rate: {dataset_info.sampling_rate} Hz")
        else:
            print("  ⚠ No ECG records found")
    
    if not has_signals:
        print("⚠ ECG signals not available - using only RR approximations")
    
    # Process each ECG
    print(f"\nExtracting features...")
    all_features = []
    n_success = 0
    n_errors = 0
    n_no_signal = 0
    
    iterator = tqdm(df_valid.iterrows(), total=len(df_valid), disable=not verbose)
    
    for idx, row in iterator:
        ecg_id_raw = row['ecg_id']
        
        try:
            # Load signal if available
            signal = None
            if has_signals and dataset_info is not None:
                record_path = find_record_path(
                    dataset_info.data_dir, 
                    str(ecg_id_raw), 
                    dataset_info.record_paths
                )
                
                if record_path is not None:
                    signal_full, fs, lead_names = load_ecg_signal(record_path, dataset_info.format)
                    if signal_full is not None:
                        signal = select_lead(signal_full, lead_names, lead_index)
                else:
                    n_no_signal += 1
            
            # Get RR intervals
            rr_intervals = get_rr_intervals_from_phase1(row)
            
            # Extract features
            features = extract_all_features(signal, rr_intervals, row, extractors, feature_flags)
            
            # Keep original ecg_id for merge
            features['ecg_id'] = ecg_id_raw
            
            all_features.append(features)
            n_success += 1
            
        except Exception as e:
            n_errors += 1
            if verbose:
                iterator.set_postfix({'errors': n_errors, 'no_sig': n_no_signal})
    
    print(f"\nCompleted: {n_success} success, {n_errors} errors, {n_no_signal} signals not found")
    
    # Create DataFrame with new features
    df_new_features = pd.DataFrame(all_features)
    
    # Merge with Phase 1 features
    df_result = df_valid.merge(df_new_features, on='ecg_id', how='left')
    
    # Save result
    df_result.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  Rows: {len(df_result)}")
    print(f"  Columns: {len(df_result.columns)}")
    
    # Feature statistics
    feature_cols = [c for c in df_result.columns 
                   if c.startswith(('morph_', 'interval_', 'hrv_', 'wav_', 'comp_', 'diag_'))]
    print(f"  New features: {len(feature_cols)}")
    
    return df_result


def print_feature_summary(df: pd.DataFrame):
    """Print summary of extracted features."""
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    
    categories = {
        'Base (Phase 1)': [c for c in df.columns 
                          if not c.startswith(('morph_', 'interval_', 'hrv_', 'wav_', 'comp_', 'diag_'))],
        'Morphological': [c for c in df.columns if c.startswith('morph_')],
        'Intervals': [c for c in df.columns if c.startswith('interval_')],
        'HRV Spectral': [c for c in df.columns if c.startswith('hrv_')],
        'Wavelet': [c for c in df.columns if c.startswith('wav_')],
        'Compressibility': [c for c in df.columns if c.startswith('comp_')],
        'Diagnosis': [c for c in df.columns if c.startswith('diag_')],
    }
    
    total = 0
    for cat, cols in categories.items():
        if len(cols) > 0:
            print(f"\n{cat}: {len(cols)} features")
            total += len(cols)
            if len(cols) <= 10:
                for c in cols:
                    non_null = df[c].notna().sum()
                    print(f"  - {c}: {non_null}/{len(df)} non-null")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total} features")
    print(f"{'='*60}")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract advanced features from any ECG dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (PTB-XL)
  python extract_features.py --phase1 ./results/ptb-xl/ptb-xl_features.csv \\
                             --data-dir ./data/raw/ptb-xl \\
                             --output ./results/ptb-xl/ptb-xl_features_extracted.csv

  # Other datasets (MIT-BIH, LTAF, etc.)
  python extract_features.py --phase1 ./results/ltaf/ltaf_features.csv \\
                             --data-dir ./data/raw/ltaf \\
                             --output ./results/ltaf/ltaf_features_extracted.csv

  # Without raw signals (uses RR approximations only)
  python extract_features.py --phase1 ./results/dataset/dataset_features.csv \\
                             --output ./results/dataset/dataset_features_extracted.csv

  # Test on subset
  python extract_features.py --phase1 features.csv --data-dir ./data -n 100

  # Disable specific features
  python extract_features.py --phase1 features.csv --no-wavelet --no-diagnosis

  # Batch processing
  python extract_features.py --phase1 features.csv -o batch1.csv --start 0 -n 5000
  python extract_features.py --phase1 features.csv -o batch2.csv --start 5000 -n 5000
        """
    )
    
    parser.add_argument('--phase1', required=True, 
                        help='Path to *_features.csv from Phase 1 preprocessing')
    parser.add_argument('--data-dir', default=None,
                        help='Path to raw dataset directory (optional, enables signal-based features)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file path (default: input path with _v2 suffix)')
    parser.add_argument('-n', '--n-samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting index (default: 0)')
    parser.add_argument('--lead', type=int, default=0,
                        help='Fallback lead index for analysis (default: 0, tries to find Lead II first)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Disable verbose output')
    
    # Feature toggles
    parser.add_argument('--no-morphological', action='store_true',
                        help='Disable morphological features')
    parser.add_argument('--no-intervals', action='store_true',
                        help='Disable interval features')
    parser.add_argument('--no-spectral', action='store_true',
                        help='Disable spectral HRV features')
    parser.add_argument('--no-wavelet', action='store_true',
                        help='Disable wavelet features')
    parser.add_argument('--no-compressibility', action='store_true',
                        help='Disable compressibility features')
    parser.add_argument('--no-diagnosis', action='store_true',
                        help='Disable diagnosis mapping (only relevant for PTB-XL)')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        phase1_path = Path(args.phase1)
        output_path = phase1_path.parent / (phase1_path.stem.replace('_features', '_features_extracted') + '.csv')
    else:
        output_path = args.output
    
    # Build feature flags
    feature_flags = {
        'morphological': not args.no_morphological,
        'intervals': not args.no_intervals,
        'spectral': not args.no_spectral,
        'wavelet': not args.no_wavelet,
        'compressibility': not args.no_compressibility,
        'diagnosis': not args.no_diagnosis,
    }
    
    # Run
    start_time = time.time()
    
    df = process_dataset(
        phase1_path=args.phase1,
        data_dir=args.data_dir,
        output_path=str(output_path),
        n_samples=args.n_samples,
        start_idx=args.start,
        lead_index=args.lead,
        feature_flags=feature_flags,
        verbose=not args.quiet
    )
    
    elapsed = time.time() - start_time
    rate = elapsed / len(df) * 1000 if len(df) > 0 else 0
    print(f"\nTotal time: {elapsed:.1f}s ({rate:.1f}ms/ECG)")
    
    # Print summary
    print_feature_summary(df)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
