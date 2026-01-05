#!/usr/bin/env python3
"""
Kepler-ECG: Generic Feature Extraction Script (v2.0 - Multi-Dataset Support)

Estrae features avanzate da qualsiasi dataset ECG preprocessato.
Supporta PTB-XL, Chapman, CPSC-2018, Georgia e altri.

Questo script:
1. Carica i dati preprocessati dalla Fase 1 (*_features.csv)
2. Carica i segnali ECG grezzi dal dataset originale
3. Estrae features avanzate (morfologiche, spettrali, wavelet, compressibilità)
4. Mappa le diagnosi usando LabelMapper (supporta SCP e SNOMED codes)
5. Salva il risultato in *_features_extracted.csv

Usage:
    # Nuovo modo (raccomandato) - by dataset name:
    python scripts/extract_features_dataset.py --dataset ptb-xl
    python scripts/extract_features_dataset.py --dataset chapman -n 100
    
    # Legacy mode - by paths:
    python scripts/extract_features_dataset.py --phase1 ./results/ptb-xl/ptb-xl_features.csv \\
                                               --data-dir ./data/raw/ptb-xl

Author: Alessandro Marconi for Kepler-ECG Project
Version: 2.0.0 - Integrated with LabelMapper for multi-dataset support
Issued on: January 2025
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd

# Try to import tqdm, fallback to simple iteration if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Importa moduli Kepler-ECG
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features import (
    MorphologicalExtractor,
    IntervalCalculator, 
    SpectralAnalyzer,
    WaveletExtractor,
    CompressibilityCalculator,
)

# Import core modules for label mapping and dataset registry
try:
    from core.label_schema import LabelMapper, Superclass
    from core.dataset_registry import get_dataset_config, get_registry
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("⚠️  Warning: core modules not available. Diagnosis mapping will be limited.")


# ============================================================================
# Configurazione
# ============================================================================

DEFAULT_FEATURES = {
    'morphological': True,
    'intervals': True,
    'spectral': True,
    'wavelet': True,
    'compressibility': True,
    'diagnosis': True,
}

PREFERRED_LEADS = ['II', 'MLII', 'I', 'V5', 'V1', 'ECG', 'ECG1']


# ============================================================================
# Dataset Detection
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
        for f in all_files:
            if f.suffix.lower() == '.hea':
                dat_file = f.with_suffix('.dat')
                mat_file = f.with_suffix('.mat')
                if dat_file.exists() or mat_file.exists():
                    info.record_paths.append(f.with_suffix(''))
    elif extension_counts.get('.mat', 0) > 0 and extension_counts.get('.hea', 0) > 0:
        info.format = 'wfdb'
        for f in all_files:
            if f.suffix.lower() == '.hea':
                info.record_paths.append(f.with_suffix(''))
    elif extension_counts.get('.mat', 0) > 0:
        info.format = 'mat'
        info.record_paths = [f for f in all_files if f.suffix.lower() == '.mat']
    elif extension_counts.get('.npy', 0) > 0:
        info.format = 'npy'
        info.record_paths = [f for f in all_files if f.suffix.lower() == '.npy']
    
    info.record_paths = sorted(info.record_paths)
    info.n_records = len(info.record_paths)
    
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
            return 500
        else:
            return None
    except Exception:
        return None


# ============================================================================
# Signal Loading
# ============================================================================

def load_ecg_signal(record_path: Path, format: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[List[str]]]:
    """Load ECG signal from any supported format."""
    try:
        if format == 'wfdb':
            import wfdb
            record = wfdb.rdrecord(str(record_path))
            return record.p_signal, int(record.fs), record.sig_name
            
        elif format == 'mat':
            import scipy.io as sio
            mat = sio.loadmat(str(record_path))
            
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
            
            if signal.ndim == 1:
                signal = signal.reshape(-1, 1)
            elif signal.shape[0] < signal.shape[1]:
                signal = signal.T
            
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
            return signal, 500, None
            
    except Exception:
        return None, None, None
    
    return None, None, None


def select_lead(signal: np.ndarray, lead_names: Optional[List[str]], lead_index: int = 0) -> np.ndarray:
    """Select the best lead for analysis."""
    if signal.ndim == 1:
        return signal
    
    if lead_names is not None:
        for preferred in PREFERRED_LEADS:
            for i, name in enumerate(lead_names):
                if name.upper() == preferred.upper():
                    return signal[:, i]
    
    n_leads = signal.shape[1]
    if lead_index < n_leads:
        return signal[:, lead_index]
    
    return signal[:, 0]


def find_record_path(data_dir: Path, ecg_id: str, record_paths: List[Path]) -> Optional[Path]:
    """Find the record path for a given ecg_id."""
    ecg_id_clean = ecg_id.replace('.hea', '').replace('.dat', '').replace('.mat', '').replace('.npy', '')
    
    for rp in record_paths:
        if rp.stem == ecg_id_clean or rp.name == ecg_id_clean:
            return rp
    
    if '_hr' in ecg_id_clean:
        try:
            numeric_id = int(ecg_id_clean.split('_')[0].lstrip('0') or '0')
            folder_num = (numeric_id // 1000) * 1000
            folder = f"{folder_num:05d}"
            
            ptbxl_path = data_dir / "records500" / folder / ecg_id_clean
            if ptbxl_path.with_suffix('.hea').exists():
                return ptbxl_path
        except ValueError:
            pass
    
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
    """Reconstruct RR intervals from Phase 1 statistics."""
    rr_mean = row.get('rr_mean_ms')
    rr_std = row.get('rr_std_ms')
    n_beats = row.get('n_beats')
    
    if pd.isna(rr_mean) or pd.isna(n_beats) or n_beats < 2:
        return None
    
    ecg_id_raw = row.get('ecg_id', 0)
    if isinstance(ecg_id_raw, str):
        try:
            seed = int(ecg_id_raw.split('_')[0].lstrip('0') or '0')
        except ValueError:
            seed = hash(ecg_id_raw) % (2**31)
    else:
        seed = int(ecg_id_raw) if not pd.isna(ecg_id_raw) else 0
    
    np.random.seed(seed % (2**31))
    rr_std_safe = rr_std if not pd.isna(rr_std) and rr_std > 0 else 50
    rr_intervals = np.random.normal(rr_mean, rr_std_safe, int(n_beats))
    rr_intervals = np.clip(rr_intervals, 300, 2000)
    
    return rr_intervals


# ============================================================================
# Diagnosis Mapping (v2.0 - Multi-dataset support)
# ============================================================================

def extract_diagnosis_features(row: pd.Series, label_mapper: Optional['LabelMapper']) -> Dict[str, Any]:
    """
    Extract diagnosis features using LabelMapper.
    
    Supports both SCP codes (PTB-XL) and SNOMED codes (Chapman, CPSC, Georgia).
    """
    features = {}
    
    if label_mapper is None:
        return features
    
    labels = []
    
    # Try SCP codes first (PTB-XL)
    # PRIORITY 1: Use primary_superclass from process_dataset.py if available
    # This is already correctly mapped and should be trusted
    primary_superclass_from_phase1 = row.get('primary_superclass')
    if primary_superclass_from_phase1 and primary_superclass_from_phase1 != 'nan' and pd.notna(primary_superclass_from_phase1):
        features['diag_primary_category'] = primary_superclass_from_phase1
        features['diag_is_normal'] = int(primary_superclass_from_phase1 == 'NORM')
        
        # Also copy label_* columns if present
        for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP', 'OTHER']:
            label_col = f'label_{sc}'
            if label_col in row.index:
                features[f'diag_{sc}'] = row.get(label_col, 0)
        
        return features
    
    # PRIORITY 2: Try SCP codes (PTB-XL format)
    scp_codes = row.get('scp_codes', '{}')
    if scp_codes and scp_codes != '{}' and scp_codes != 'nan':
        try:
            if isinstance(scp_codes, str):
                scp_dict = json.loads(scp_codes.replace("'", '"'))
            else:
                scp_dict = scp_codes
            
            if isinstance(scp_dict, dict) and len(scp_dict) > 0:
                labels = label_mapper.map_scp_codes(scp_dict)
        except (json.JSONDecodeError, ValueError):
            pass
    
    # PRIORITY 3: Try SNOMED codes (Chapman, CPSC, Georgia)
    if not labels:
        snomed_codes = row.get('snomed_codes', '[]')
        if snomed_codes and snomed_codes != '[]' and snomed_codes != 'nan':
            try:
                if isinstance(snomed_codes, str):
                    snomed_list = json.loads(snomed_codes.replace("'", '"'))
                else:
                    snomed_list = snomed_codes
                
                if isinstance(snomed_list, list) and len(snomed_list) > 0:
                    labels = label_mapper.map_snomed_codes(snomed_list)
            except (json.JSONDecodeError, ValueError):
                pass
    
    # Extract features from labels
    if labels:
        primary_superclass = label_mapper.get_primary_superclass(labels)
        superclass_vector = label_mapper.get_superclass_vector(labels)
        
        features['diag_primary_category'] = primary_superclass.value
        features['diag_is_normal'] = int(primary_superclass == Superclass.NORM)
        features['diag_n_diagnoses'] = len(labels)
        features['diag_is_multi_label'] = int(len(set(l.superclass for l in labels)) > 1)
        
        # Primary code and description
        if labels:
            features['diag_primary_code'] = labels[0].original_code
            features['diag_primary_description'] = labels[0].original_description
        
        # Superclass binary features
        for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP', 'OTHER']:
            features[f'diag_{sc}'] = superclass_vector.get(sc, 0)
    
    return features


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_all_features(
    signal: Optional[np.ndarray],
    rr_intervals: Optional[np.ndarray],
    row: pd.Series,
    extractors: Dict[str, Any],
    feature_flags: Dict[str, bool],
    label_mapper: Optional['LabelMapper'] = None
) -> Dict[str, Any]:
    """Extract all features from a single ECG."""
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
            wav_features = wavelet.extract(signal[:2500])
            features.update({f'wav_{k}': v for k, v in wav_features.items()})
        except Exception:
            pass
    
    # 5. Compressibility Features
    if feature_flags.get('compressibility', True):
        try:
            compressor = extractors['compressor']
            
            if signal is not None and len(signal) >= 100:
                comp_signal = compressor.calculate(signal[:2500], methods=['compression', 'complexity'])
                features.update({f'comp_sig_{k}': v for k, v in comp_signal.items()})
            
            if rr_intervals is not None and len(rr_intervals) >= 10:
                rr_for_entropy = rr_intervals[:1000] if len(rr_intervals) > 1000 else rr_intervals
                comp_rr = compressor.calculate(rr_for_entropy, methods=['entropy', 'complexity'])
                features.update({f'comp_rr_{k}': v for k, v in comp_rr.items()})
        except Exception:
            pass
    
    # 6. Diagnosis Mapping (v2.0 - uses LabelMapper)
    if feature_flags.get('diagnosis', True):
        try:
            diag_features = extract_diagnosis_features(row, label_mapper)
            features.update(diag_features)
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
    dataset_name: Optional[str] = None,
    n_samples: Optional[int] = None,
    start_idx: int = 0,
    lead_index: int = 0,
    feature_flags: Optional[Dict[str, bool]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Process any ECG dataset and extract advanced features."""
    
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
    }
    
    # Initialize LabelMapper
    label_mapper = None
    if CORE_AVAILABLE:
        label_mapper = LabelMapper(dataset_name or 'unknown')
        print(f"  LabelMapper initialized for: {dataset_name or 'unknown'}")
    
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
            features = extract_all_features(
                signal, rr_intervals, row, extractors, feature_flags, label_mapper
            )
            
            features['ecg_id'] = ecg_id_raw
            all_features.append(features)
            n_success += 1
            
        except Exception as e:
            n_errors += 1
            if verbose and HAS_TQDM:
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
        description='Extract advanced features from any ECG dataset (v2.0 with multi-dataset support)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New mode (recommended) - by dataset name:
  python scripts/extract_features_dataset.py --dataset ptb-xl
  python scripts/extract_features_dataset.py --dataset chapman -n 100
  
  # Legacy mode - by paths:
  python scripts/extract_features_dataset.py --phase1 ./results/ptb-xl/ptb-xl_features.csv \\
                                             --data-dir ./data/raw/ptb-xl

  # Without raw signals (uses RR approximations only)
  python scripts/extract_features_dataset.py --phase1 ./results/dataset/dataset_features.csv

  # Disable specific features
  python scripts/extract_features_dataset.py --dataset ptb-xl --no-wavelet --no-compressibility
        """
    )
    
    # New: dataset by name
    parser.add_argument('--dataset', '-d', type=str,
                        help='Dataset name from registry (e.g., ptb-xl, chapman, cpsc-2018, georgia)')
    
    # Legacy: paths
    parser.add_argument('--phase1', type=str,
                        help='Path to *_features.csv from Phase 1 preprocessing')
    parser.add_argument('--data-dir', default=None,
                        help='Path to raw dataset directory (optional, enables signal-based features)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file path (default: input path with _extracted suffix)')
    
    # Processing options
    parser.add_argument('-n', '--n-samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting index (default: 0)')
    parser.add_argument('--lead', type=int, default=0,
                        help='Fallback lead index for analysis (default: 0)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Disable verbose output')
    
    # Feature toggles
    parser.add_argument('--no-morphological', action='store_true')
    parser.add_argument('--no-intervals', action='store_true')
    parser.add_argument('--no-spectral', action='store_true')
    parser.add_argument('--no-wavelet', action='store_true')
    parser.add_argument('--no-compressibility', action='store_true')
    parser.add_argument('--no-diagnosis', action='store_true')
    
    args = parser.parse_args()
    
    # Determine paths based on mode
    if args.dataset:
        # New mode: use registry
        if not CORE_AVAILABLE:
            print("❌ Error: core modules not available. Cannot use --dataset mode.")
            sys.exit(1)
        
        dataset_name = args.dataset
        phase1_path = args.phase1 or f"results/{dataset_name}/{dataset_name}_features.csv"
        data_dir = args.data_dir or f"data/raw/{dataset_name}"
        output_path = args.output or f"results/{dataset_name}/{dataset_name}_features_extracted.csv"
        
        print(f"Using dataset: {dataset_name}")
    else:
        # Legacy mode
        if not args.phase1:
            parser.error("Either --dataset or --phase1 is required")
        
        dataset_name = None
        phase1_path = args.phase1
        data_dir = args.data_dir
        
        if args.output is None:
            p = Path(phase1_path)
            output_path = str(p.parent / (p.stem.replace('_features', '_features_extracted') + '.csv'))
        else:
            output_path = args.output
    
    # Check phase1 file exists
    if not Path(phase1_path).exists():
        print(f"❌ Error: Phase 1 file not found: {phase1_path}")
        print(f"   Run process_dataset.py first: python scripts/process_dataset.py --dataset {dataset_name or 'your-dataset'}")
        sys.exit(1)
    
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
        phase1_path=phase1_path,
        data_dir=data_dir,
        output_path=output_path,
        dataset_name=dataset_name,
        n_samples=args.n_samples,
        start_idx=args.start,
        lead_index=args.lead,
        feature_flags=feature_flags,
        verbose=not args.quiet
    )
    
    elapsed = time.time() - start_time
    rate = elapsed / len(df) * 1000 if len(df) > 0 else 0
    print(f"\nTotal time: {elapsed:.1f}s ({rate:.1f}ms/ECG)")
    
    print_feature_summary(df)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
