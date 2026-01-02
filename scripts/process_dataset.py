"""
Kepler-ECG: Process ECG Dataset

Generic script that auto-detects dataset structure and processes ECG files.
Uses the existing PreprocessingPipeline from the project.

Works with PTB-XL, Chapman, Georgia, CPSC, MIT-BIH, LTAF, QTDB, etc.

Supports diagnosis codes from:
- CSV metadata files (PTB-XL style with scp_codes)
- WFDB header comments with SNOMED-CT codes (Chapman, PhysioNet Challenge 2020/2021)

Usage:
    python process_dataset.py --data_dir ./data/raw/my-dataset --output_dir ./results/my-dataset

    # With specific sampling rate filter (e.g., only 500Hz files):
    python process_dataset.py --data_dir ./data/raw/ptb-xl --output_dir ./results/ptb-xl --sampling_rate 500

Supported formats:
    - WFDB (.dat + .hea or .mat + .hea)
    - MAT files (.mat)
    - NumPy (.npy)

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.1.0 - Added SNOMED-CT support from WFDB headers
Issued on: December 2025
"""

# Standard library imports for system and workflow management
import argparse  # Parse command-line arguments for configurable scripts
import json      # Read and write configuration files or export metadata
import logging   # Log info/warnings/errors to console or file instead of using print()
import os        # Interface with the operating system (e.g., environment variables)
import re        # Regular expressions for parsing header files
import sys       # Access system-specific parameters and interpreter functions
import time      # Track execution time or handle timestamping for logs

# Filesystem and data structure management
from pathlib import Path       # Modern, cross-platform path handling (replaces os.path)
from typing import Dict, List, Optional, Tuple, Generator, Any  # Type hints for better IDE support and debugging
from dataclasses import dataclass, field  # Cleanly define data objects (e.g., PatientData, ModelConfig)
from collections import defaultdict       # Dictionary that provides default values for missing keys

# Data analysis and numerical computing
import numpy as np  # High-performance array and matrix operations
import pandas as pd  # Structured data manipulation (DataFrames) for metadata and CSVs

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import preprocessing pipeline
from preprocessing import (
    PreprocessingPipeline,
    PipelineConfig,
    ProcessedECG,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SNOMED-CT CODE MAPPING
# =============================================================================

# Common SNOMED-CT codes used in PhysioNet ECG datasets
# Reference: PhysioNet Challenge 2020/2021, Chapman dataset
SNOMED_CT_MAPPING = {
    # Normal
    '426783006': 'SR',       # Sinus rhythm
    '426177001': 'SB',       # Sinus bradycardia
    '427084000': 'ST',       # Sinus tachycardia
    '164889003': 'AF',       # Atrial fibrillation
    '164890007': 'AFL',      # Atrial flutter
    '713427006': 'CRBBB',    # Complete right bundle branch block
    '713426002': 'CLBBB',    # Complete left bundle branch block
    '59118001': 'RBBB',      # Right bundle branch block
    '6374002': 'BBB',        # Bundle branch block
    '445118002': 'LAFB',     # Left anterior fascicular block
    '445211001': 'LPFB',     # Left posterior fascicular block
    '270492004': 'IAVB',     # First degree AV block
    '195042002': 'IIAVB',    # Second degree AV block
    '27885002': 'IIIAVB',    # Third degree AV block (complete heart block)
    '54016002': 'IVCD',      # Intraventricular conduction delay
    '233917008': 'AVB',      # AV block
    
    # Arrhythmias
    '284470004': 'PAC',      # Premature atrial contraction
    '427172004': 'PVC',      # Premature ventricular contraction
    '17338001': 'VEB',       # Ventricular ectopic beat
    '195060002': 'VT',       # Ventricular tachycardia
    '164947007': 'VF',       # Ventricular fibrillation
    '427393009': 'SA',       # Sinus arrhythmia
    '426995002': 'SND',      # Sinus node dysfunction
    '49436004': 'WPW',       # Wolff-Parkinson-White syndrome
    '74390002': 'AVNRT',     # AV nodal reentrant tachycardia
    '233896004': 'AVRT',     # AV reentrant tachycardia
    '251168009': 'SVT',      # Supraventricular tachycardia
    '251173003': 'AT',       # Atrial tachycardia
    '63593006': 'SVAPC',     # Supraventricular premature complex
    
    # Hypertrophy
    '164873001': 'LVH',      # Left ventricular hypertrophy
    '89792004': 'RVH',       # Right ventricular hypertrophy
    '446813000': 'LAE',      # Left atrial enlargement
    '67751000119106': 'RAE', # Right atrial enlargement
    '55827005': 'LAH',       # Left atrial hypertrophy
    '67741000119109': 'RAH', # Right atrial hypertrophy
    
    # Ischemia/Infarction
    '164865005': 'MI',       # Myocardial infarction
    '164861001': 'AMI',      # Acute myocardial infarction
    '164867002': 'OMI',      # Old myocardial infarction
    '425623009': 'STEMI',    # ST elevation MI
    '428196007': 'NSTEMI',   # Non-ST elevation MI
    '164930006': 'STD',      # ST depression
    '164931005': 'STE',      # ST elevation
    '59931005': 'TWI',       # T wave inversion
    '429622005': 'TWAB',     # T wave abnormality
    '251146004': 'STTC',     # ST-T change
    '428750005': 'NST',      # Nonspecific ST change
    '164934002': 'NSSTT',    # Nonspecific ST-T abnormality
    
    # QT abnormalities
    '111975006': 'LQT',      # Long QT syndrome
    '77867006': 'SQT',       # Short QT syndrome
    
    # Other
    '164917005': 'LQRS',     # Low QRS voltage
    '251120003': 'HQRS',     # High QRS voltage
    '39732003': 'LAD',       # Left axis deviation
    '47665007': 'RAD',       # Right axis deviation
    '164909002': 'TAB',      # T wave abnormality
    '164912004': 'QTAB',     # QT abnormality
    '251200008': 'PAB',      # P wave abnormality
    '698252002': 'NORM',     # Normal ECG
    '164942001': 'ABQRS',    # Abnormal QRS
    
    # Additional codes from Chapman
    '17366009': 'APB',       # Atrial premature beat (same as PAC)
    '11157007': 'AVBL',      # AV block (low grade)
    '251164006': 'ERD',      # Early repolarization
}

# Reverse mapping for lookup
SNOMED_CT_REVERSE = {v: k for k, v in SNOMED_CT_MAPPING.items()}


def parse_snomed_from_header(header_path: Path) -> Dict[str, Any]:
    """
    Parse SNOMED-CT codes and demographics from WFDB header file.
    
    The header file contains comments with metadata like:
    #Age: 45
    #Sex: Male
    #Dx: 426783006,427084000
    
    Parameters
    ----------
    header_path : Path
        Path to .hea file.
        
    Returns
    -------
    Dict with extracted metadata:
        - age: float or None
        - sex: int (0=Male, 1=Female) or None
        - snomed_codes: List[str] of SNOMED-CT codes
        - scp_codes: Dict mapping abbreviation to 100.0 (for compatibility with PTB-XL format)
    """
    metadata = {
        'age': None,
        'sex': None,
        'snomed_codes': [],
        'scp_codes': {},
    }
    
    if not header_path.exists():
        return metadata
    
    try:
        with open(header_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip non-comment lines
            if not line.startswith('#'):
                continue
            
            # Remove the # prefix
            line = line[1:].strip()
            
            # Parse Age
            # Formats: "Age: 45", "Age:45", "Age 45"
            age_match = re.match(r'Age[:\s]+(\d+)', line, re.IGNORECASE)
            if age_match:
                try:
                    metadata['age'] = float(age_match.group(1))
                except ValueError:
                    pass
            
            # Parse Sex
            # Formats: "Sex: Male", "Sex:Female", "Sex: M", "Sex: F"
            sex_match = re.match(r'Sex[:\s]+(\w+)', line, re.IGNORECASE)
            if sex_match:
                sex_str = sex_match.group(1).lower()
                if sex_str in ['male', 'm', '0']:
                    metadata['sex'] = 0
                elif sex_str in ['female', 'f', '1']:
                    metadata['sex'] = 1
            
            # Parse Diagnosis codes
            # Formats: "Dx: 426783006,427084000", "Dx:426783006"
            dx_match = re.match(r'Dx[:\s]+(.+)', line, re.IGNORECASE)
            if dx_match:
                dx_string = dx_match.group(1)
                # Split by comma, space, or both
                codes = re.split(r'[,\s]+', dx_string)
                codes = [c.strip() for c in codes if c.strip().isdigit()]
                metadata['snomed_codes'] = codes
                
                # Convert to scp_codes format for compatibility
                for code in codes:
                    if code in SNOMED_CT_MAPPING:
                        abbrev = SNOMED_CT_MAPPING[code]
                        metadata['scp_codes'][abbrev] = 100.0
                    else:
                        # Keep unknown codes with their SNOMED ID
                        metadata['scp_codes'][f'SNOMED_{code}'] = 100.0
        
    except Exception as e:
        logger.debug(f"Could not parse header {header_path}: {e}")
    
    return metadata


# =============================================================================
# DATASET STRUCTURE DETECTOR
# =============================================================================

@dataclass
class DatasetInfo:
    """Information about detected dataset structure."""
    data_dir: Path
    name: str  # Dataset name (from directory name)
    format: str  # 'wfdb', 'mat', 'npy'
    record_paths: List[Path] = field(default_factory=list)
    metadata_path: Optional[Path] = None
    sampling_rate: Optional[int] = None
    n_records: int = 0
    structure: Dict = field(default_factory=dict)
    has_snomed_headers: bool = False  # True if headers contain SNOMED-CT codes


class DatasetDetector:
    """Auto-detect dataset structure and file locations."""
    
    # Known metadata filenames
    METADATA_FILES = [
        'ptbxl_database.csv',
        'RECORDS',
        'metadata.csv',
        'labels.csv',
        'annotations.csv',
        'reference.csv',
        'ConditionNames_SNOMED-CT.csv',
    ]
    
    def __init__(self, data_dir: str, target_sampling_rate: Optional[int] = None):
        """
        Parameters
        ----------
        data_dir : str
            Path to dataset directory.
        target_sampling_rate : int, optional
            If specified, only include records at this sampling rate.
            Useful for PTB-XL to exclude 100Hz files when you want only 500Hz.
        """
        self.data_dir = Path(data_dir)
        self.target_sampling_rate = target_sampling_rate
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
    
    def detect(self) -> DatasetInfo:
        """Detect dataset structure."""
        logger.info(f"Scanning directory: {self.data_dir}")
        
        # Dataset name from directory
        dataset_name = self.data_dir.name
        
        info = DatasetInfo(
            data_dir=self.data_dir, 
            name=dataset_name,
            format='unknown'
        )
        
        # Count files by extension
        extension_counts = defaultdict(int)
        all_files = []
        
        for root, dirs, files in os.walk(self.data_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for f in files:
                ext = Path(f).suffix.lower()
                extension_counts[ext] += 1
                all_files.append(Path(root) / f)
        
        logger.info(f"Found {len(all_files)} files")
        logger.info(f"Extensions: {dict(extension_counts)}")
        
        # Detect format
        info.format = self._detect_format(extension_counts)
        logger.info(f"Detected format: {info.format}")
        
        # Find record files (with sampling rate filtering)
        info.record_paths = self._find_records(all_files, info.format)
        info.n_records = len(info.record_paths)
        logger.info(f"Found {info.n_records} ECG records")
        
        # Find metadata file (optional)
        info.metadata_path = self._find_metadata(all_files)
        if info.metadata_path:
            logger.info(f"Found metadata: {info.metadata_path.name}")
        
        # Check if headers contain SNOMED-CT codes (Chapman/PhysioNet Challenge style)
        if info.format == 'wfdb' and info.n_records > 0:
            info.has_snomed_headers = self._check_snomed_headers(info.record_paths[:5])
            if info.has_snomed_headers:
                logger.info("Detected SNOMED-CT codes in WFDB headers")
        
        # Detect/confirm sampling rate from first record
        if info.n_records > 0:
            detected_fs = self._detect_sampling_rate(info.record_paths[0], info.format)
            info.sampling_rate = self.target_sampling_rate or detected_fs
            logger.info(f"Sampling rate: {info.sampling_rate} Hz")
        
        # Structure summary
        info.structure = {
            'format': info.format,
            'n_records': info.n_records,
            'has_metadata': info.metadata_path is not None,
            'has_snomed_headers': info.has_snomed_headers,
            'sampling_rate': info.sampling_rate,
            'extensions': dict(extension_counts),
        }
        
        return info
    
    def _detect_format(self, extension_counts: Dict[str, int]) -> str:
        """Detect format from file extensions."""
        if extension_counts.get('.dat', 0) > 0 and extension_counts.get('.hea', 0) > 0:
            return 'wfdb'
        elif extension_counts.get('.mat', 0) > 0 and extension_counts.get('.hea', 0) > 0:
            # Chapman uses .mat + .hea (MATLAB v4 format with WFDB header)
            return 'wfdb'
        elif extension_counts.get('.mat', 0) > 0:
            return 'mat'
        elif extension_counts.get('.npy', 0) > 0:
            return 'npy'
        else:
            return 'unknown'
    
    def _find_records(self, all_files: List[Path], format: str) -> List[Path]:
        """Find all ECG record files, optionally filtering by sampling rate."""
        records = []
        
        if format == 'wfdb':
            # For WFDB, we need .hea files (header)
            for f in all_files:
                if f.suffix.lower() == '.hea':
                    # Data can be in .dat or .mat file
                    dat_file = f.with_suffix('.dat')
                    mat_file = f.with_suffix('.mat')
                    
                    if dat_file.exists() or mat_file.exists():
                        record_path = f.with_suffix('')  # Path without extension
                        
                        # Filter by sampling rate if specified
                        if self.target_sampling_rate is not None:
                            try:
                                import wfdb
                                header = wfdb.rdheader(str(record_path))
                                if header.fs != self.target_sampling_rate:
                                    continue  # Skip this record
                            except Exception:
                                continue
                        
                        records.append(record_path)
        
        elif format == 'mat':
            records = [f for f in all_files if f.suffix.lower() == '.mat']
        
        elif format == 'npy':
            records = [f for f in all_files if f.suffix.lower() == '.npy']
        
        return sorted(records)
    
    def _find_metadata(self, all_files: List[Path]) -> Optional[Path]:
        """Find metadata file if exists."""
        for f in all_files:
            if f.name in self.METADATA_FILES:
                return f
            if f.name.lower() in ['metadata.csv', 'labels.csv', 'database.csv']:
                return f
        return None
    
    def _check_snomed_headers(self, sample_paths: List[Path]) -> bool:
        """Check if header files contain SNOMED-CT diagnosis codes."""
        for record_path in sample_paths:
            header_path = record_path.with_suffix('.hea')
            if header_path.exists():
                try:
                    with open(header_path, 'r') as f:
                        content = f.read()
                    # Look for Dx: followed by numeric codes
                    if re.search(r'#\s*Dx[:\s]+\d+', content, re.IGNORECASE):
                        return True
                except Exception:
                    pass
        return False
    
    def _detect_sampling_rate(self, record_path: Path, format: str) -> Optional[int]:
        """Detect sampling rate from first record."""
        try:
            if format == 'wfdb':
                import wfdb
                record = wfdb.rdheader(str(record_path))
                return record.fs
            
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
                
        except Exception as e:
            logger.warning(f"Could not detect sampling rate: {e}")
            return None


# =============================================================================
# GENERIC ECG LOADER
# =============================================================================

class GenericECGLoader:
    """Load ECG records from any supported format."""
    
    def __init__(self, dataset_info: DatasetInfo):
        self.info = dataset_info
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Optional[pd.DataFrame]:
        """Load metadata CSV if available."""
        if self.info.metadata_path is None:
            return None
        
        try:
            # Try to auto-detect index column
            df = pd.read_csv(self.info.metadata_path)
            
            # Common index columns
            for col in ['ecg_id', 'record_id', 'id', 'filename']:
                if col in df.columns:
                    df = df.set_index(col)
                    break
            
            logger.info(f"Loaded metadata: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            return None
    
    def load_record(self, record_path: Path) -> Tuple[np.ndarray, int, Dict]:
        """
        Load a single ECG record.
        
        Returns:
            signal: np.ndarray (n_samples, n_leads) or (n_samples,)
            fs: sampling rate
            metadata: dict with any available info
        """
        if self.info.format == 'wfdb':
            return self._load_wfdb(record_path)
        elif self.info.format == 'mat':
            return self._load_mat(record_path)
        elif self.info.format == 'npy':
            return self._load_npy(record_path)
        else:
            raise ValueError(f"Unsupported format: {self.info.format}")
    
    def _load_wfdb(self, record_path: Path) -> Tuple[np.ndarray, int, Dict]:
        """Load WFDB format (supports both .dat and .mat data files)."""
        import wfdb
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal  # (n_samples, n_leads)
        fs = record.fs
        
        metadata = {
            'record_name': record_path.name,
            'n_leads': record.n_sig,
            'sig_names': record.sig_name,
            'age': None,
            'sex': None,
            'scp_codes': {},
        }
        
        # First, try to get SNOMED-CT codes from header (Chapman/PhysioNet Challenge style)
        if self.info.has_snomed_headers:
            header_path = record_path.with_suffix('.hea')
            header_metadata = parse_snomed_from_header(header_path)
            
            if header_metadata['age'] is not None:
                metadata['age'] = header_metadata['age']
            if header_metadata['sex'] is not None:
                metadata['sex'] = header_metadata['sex']
            if header_metadata['scp_codes']:
                metadata['scp_codes'] = header_metadata['scp_codes']
            if header_metadata['snomed_codes']:
                metadata['snomed_codes'] = header_metadata['snomed_codes']
        
        # Then, try to get additional metadata from CSV if available (PTB-XL style)
        if self.metadata is not None:
            try:
                # Try different ways to match record to metadata
                record_id = record_path.name
                matched_row = None
                
                # Method 1: Direct match with index
                if record_id in self.metadata.index:
                    matched_row = self.metadata.loc[record_id]
                
                # Method 2: Try numeric ID (for PTB-XL style: "00001_hr" -> 1)
                if matched_row is None:
                    try:
                        numeric_id = int(record_id.split('_')[0].lstrip('0') or '0')
                        if numeric_id in self.metadata.index:
                            matched_row = self.metadata.loc[numeric_id]
                    except (ValueError, AttributeError):
                        pass
                
                # Method 3: Try matching filename columns (PTB-XL has filename_hr, filename_lr)
                if matched_row is None:
                    for fname_col in ['filename_hr', 'filename_lr', 'filename']:
                        if fname_col in self.metadata.columns:
                            # Match partial path (e.g., "records500/00000/00001_hr")
                            mask = self.metadata[fname_col].astype(str).str.contains(record_id, regex=False)
                            if mask.any():
                                matched_row = self.metadata.loc[mask].iloc[0]
                                break
                
                # Extract metadata if we found a match (CSV overrides header values if present)
                if matched_row is not None:
                    for col in ['age', 'sex', 'scp_codes', 'patient_id']:
                        if col in matched_row.index:
                            val = matched_row[col]
                            if col == 'scp_codes' and isinstance(val, str):
                                try:
                                    val = eval(val)
                                except:
                                    pass
                            # Only override if the new value is not None/empty
                            if val is not None and (not isinstance(val, dict) or val):
                                metadata[col] = val
            except Exception:
                pass
        
        return signal, int(fs), metadata
    
    def _load_mat(self, record_path: Path) -> Tuple[np.ndarray, int, Dict]:
        """Load MAT format (standalone, not WFDB)."""
        import scipy.io as sio
        mat = sio.loadmat(str(record_path))
        
        # Find signal array
        signal = None
        for key in ['val', 'ECG', 'ecg', 'signal', 'data', 'X']:
            if key in mat:
                signal = mat[key]
                break
        
        if signal is None:
            for key, val in mat.items():
                if not key.startswith('_') and isinstance(val, np.ndarray):
                    if val.size > 1000:
                        signal = val
                        break
        
        if signal is None:
            raise ValueError(f"Could not find signal in {record_path}")
        
        # Ensure correct shape: (n_samples, n_leads)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1]:
            signal = signal.T
        
        # Get sampling rate
        fs = self.info.sampling_rate or 500
        for key in ['fs', 'Fs', 'sampling_rate']:
            if key in mat:
                val = mat[key]
                fs = int(val.flatten()[0]) if hasattr(val, 'flatten') else int(val)
                break
        
        metadata = {
            'record_name': record_path.stem,
            'n_leads': signal.shape[1] if signal.ndim > 1 else 1,
        }
        
        return signal, fs, metadata
    
    def _load_npy(self, record_path: Path) -> Tuple[np.ndarray, int, Dict]:
        """Load NPY format."""
        signal = np.load(str(record_path))
        
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1]:
            signal = signal.T
        
        fs = self.info.sampling_rate or 500
        
        metadata = {
            'record_name': record_path.stem,
            'n_leads': signal.shape[1] if signal.ndim > 1 else 1,
        }
        
        return signal, fs, metadata
    
    def iter_records(self, n_samples: Optional[int] = None) -> Generator:
        """
        Iterate over records.
        
        Yields:
            record_path, signal, fs, metadata
        """
        paths = self.info.record_paths[:n_samples] if n_samples else self.info.record_paths
        
        for record_path in paths:
            try:
                signal, fs, metadata = self.load_record(record_path)
                yield record_path, signal, fs, metadata
            except Exception as e:
                logger.warning(f"Failed to load {record_path}: {e}")
                continue
    
    def __len__(self) -> int:
        return len(self.info.record_paths)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(result: ProcessedECG, metadata: Dict) -> Dict:
    """Extract features from processed ECG."""
    features = {
        'ecg_id': result.ecg_id,
        'success': result.success,
        'processing_time_ms': result.processing_time_sec * 1000,
        
        # Metadata
        'age': metadata.get('age'),
        'sex': metadata.get('sex'),
        'scp_codes': metadata.get('scp_codes', {}),
        
        # Quality
        'quality_score': None,
        'quality_level': None,
        'snr_db': None,
        'is_usable': result.is_usable,
        
        # Heart rate
        'heart_rate_bpm': result.heart_rate_bpm,
        'heart_rate_std': None,
        'n_beats': result.n_beats,
        
        # HRV metrics
        'rr_mean_ms': None,
        'rr_std_ms': None,
        'rmssd': None,
        'pnn50': None,
        
        # Detection confidence
        'detection_confidence': None,
        'ectopic_ratio': None,
    }
    
    # Add SNOMED codes if present
    if 'snomed_codes' in metadata:
        features['snomed_codes'] = metadata['snomed_codes']
    
    # Quality metrics
    if result.quality is not None:
        features['quality_score'] = result.quality.quality_score
        features['quality_level'] = result.quality.quality_level.value
        features['snr_db'] = result.quality.snr_db
    
    # Segmentation metrics
    if result.segmentation is not None:
        features['heart_rate_std'] = result.segmentation.heart_rate_std
        features['detection_confidence'] = result.segmentation.detection_confidence
        
        rr = result.segmentation.rr_intervals
        if len(rr) > 1:
            features['rr_mean_ms'] = float(np.mean(rr))
            features['rr_std_ms'] = float(np.std(rr))
            
            # RMSSD
            rr_diff = np.diff(rr)
            features['rmssd'] = float(np.sqrt(np.mean(rr_diff ** 2)))
            
            # pNN50
            features['pnn50'] = float(np.sum(np.abs(rr_diff) > 50) / len(rr_diff) * 100)
    
    # HRV metrics
    if result.hrv is not None:
        features['ectopic_ratio'] = result.hrv.ectopic_ratio
    
    return features


# =============================================================================
# MAIN PROCESSING
# =============================================================================

@dataclass
class ProcessingConfig:
    """Configuration for processing."""
    data_dir: str
    output_dir: str
    sampling_rate: Optional[int] = None  # None = auto-detect, or filter to this rate
    n_samples: Optional[int] = None
    batch_size: int = 100


def process_dataset(config: ProcessingConfig):
    """Process any ECG dataset using the project's preprocessing pipeline."""
    
    logger.info("=" * 70)
    logger.info("KEPLER-ECG: Generic Dataset Processor v1.1")
    logger.info("=" * 70)
    
    # Step 1: Detect dataset structure
    logger.info("\n[1/4] Detecting dataset structure...")
    detector = DatasetDetector(config.data_dir, target_sampling_rate=config.sampling_rate)
    dataset_info = detector.detect()
    
    if dataset_info.n_records == 0:
        logger.error("No ECG records found!")
        return None
    
    dataset_name = dataset_info.name
    logger.info(f"Dataset: {dataset_name}")
    
    # Step 2: Initialize loader and pipeline
    logger.info("\n[2/4] Initializing loader and pipeline...")
    loader = GenericECGLoader(dataset_info)
    
    pipeline_config = PipelineConfig(
        apply_filtering=True,
        assess_quality=True,
        segment_beats=True,
        preprocess_hrv=True,
        enable_cache=False,
        save_filtered_signal=False,
        save_beats=False,
    )
    pipeline = PreprocessingPipeline(pipeline_config)
    logger.info(f"Pipeline: {repr(pipeline)}")
    
    n_total = config.n_samples or len(loader)
    logger.info(f"Will process {n_total} records at {dataset_info.sampling_rate} Hz")
    
    if dataset_info.has_snomed_headers:
        logger.info("Using SNOMED-CT codes from WFDB headers")
    
    # Step 3: Process records
    logger.info("\n[3/4] Processing ECG records...")
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    failed_ids = []
    start_time = time.time()
    
    for i, (record_path, signal, fs, metadata) in enumerate(loader.iter_records(config.n_samples)):
        # Use record name as ID
        ecg_id = record_path.stem
        
        # Process with pipeline
        result = pipeline.process(signal, fs=fs, ecg_id=ecg_id)
        
        # Extract features
        features = extract_features(result, metadata)
        
        # Add record path for reference
        features['record_path'] = str(record_path.relative_to(dataset_info.data_dir))
        
        all_features.append(features)
        
        if not result.success:
            failed_ids.append(ecg_id)
        
        # Progress
        if (i + 1) % config.batch_size == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            success_rate = (i + 1 - len(failed_ids)) / (i + 1) * 100
            
            logger.info(
                f"Progress: {i+1:5d}/{n_total} "
                f"({rate:.1f} ECG/s, ETA: {eta/60:.1f}min, "
                f"success: {success_rate:.1f}%)"
            )
    
    total_time = time.time() - start_time
    
    # Step 4: Save results
    logger.info("\n[4/4] Saving results...")
    
    n_processed = len(all_features)
    n_successful = n_processed - len(failed_ids)
    n_usable = sum(1 for f in all_features if f['is_usable'])
    
    # Quality distribution
    quality_counts = {}
    for f in all_features:
        level = f['quality_level'] or 'unknown'
        quality_counts[level] = quality_counts.get(level, 0) + 1
    
    # Diagnosis distribution (if available)
    diagnosis_counts = defaultdict(int)
    for f in all_features:
        scp = f.get('scp_codes', {})
        if scp:
            for code in scp.keys():
                diagnosis_counts[code] += 1
    
    # Save CSV
    df = pd.DataFrame(all_features)
    
    # Convert dict columns to JSON strings
    for col in ['scp_codes', 'snomed_codes']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
    
    csv_path = output_dir / f'{dataset_name}_features.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")
    
    # Save JSON
    json_path = output_dir / f'{dataset_name}_features.json'
    with open(json_path, 'w') as f:
        json.dump(all_features, f, indent=2, default=str)
    logger.info(f"Saved: {json_path}")
    
    # Save summary
    summary = {
        'dataset_name': dataset_name,
        'dataset_dir': str(config.data_dir),
        'format': dataset_info.format,
        'sampling_rate': dataset_info.sampling_rate,
        'has_snomed_headers': dataset_info.has_snomed_headers,
        'n_records_found': dataset_info.n_records,
        'n_processed': n_processed,
        'n_successful': n_successful,
        'n_usable': n_usable,
        'n_failed': len(failed_ids),
        'success_rate': n_successful / n_processed if n_processed > 0 else 0,
        'usable_rate': n_usable / n_processed if n_processed > 0 else 0,
        'total_time_sec': total_time,
        'throughput_ecg_per_sec': n_processed / total_time if total_time > 0 else 0,
        'quality_distribution': quality_counts,
        'diagnosis_distribution': dict(diagnosis_counts) if diagnosis_counts else {},
        'failed_ids': failed_ids[:100],
    }
    
    summary_path = output_dir / f'{dataset_name}_processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved: {summary_path}")
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nDataset:     {dataset_name}")
    logger.info(f"Format:      {dataset_info.format}")
    logger.info(f"SNOMED-CT:   {'Yes (from headers)' if dataset_info.has_snomed_headers else 'No'}")
    logger.info(f"Processed:   {n_processed}")
    logger.info(f"Successful:  {n_successful} ({n_successful/n_processed*100:.1f}%)")
    logger.info(f"Usable:      {n_usable} ({n_usable/n_processed*100:.1f}%)")
    logger.info(f"Failed:      {len(failed_ids)}")
    
    logger.info(f"\nPerformance:")
    logger.info(f"  Time:        {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Throughput:  {n_processed/total_time:.1f} ECG/s")
    
    logger.info(f"\nQuality distribution:")
    for level in ['excellent', 'good', 'acceptable', 'poor', 'unusable', 'unknown']:
        count = quality_counts.get(level, 0)
        if count > 0:
            logger.info(f"  {level:12s}: {count:5d} ({count/n_processed*100:.1f}%)")
    
    if diagnosis_counts:
        logger.info(f"\nTop 10 diagnoses:")
        sorted_diag = sorted(diagnosis_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for code, count in sorted_diag:
            logger.info(f"  {code:12s}: {count:5d} ({count/n_processed*100:.1f}%)")
    
    logger.info(f"\nOutput: {output_dir}")
    
    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description='Process any ECG dataset with Kepler-ECG pipeline (v1.1 with SNOMED-CT support)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to save results'
    )
    parser.add_argument(
        '--sampling_rate',
        type=int,
        default=None,
        help='Target sampling rate. For PTB-XL use 500 to exclude 100Hz files.'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Number of samples to process (all if not specified)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size for progress reporting'
    )
    
    args = parser.parse_args()
    
    config = ProcessingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sampling_rate=args.sampling_rate,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )
    
    process_dataset(config)


if __name__ == "__main__":
    main()
