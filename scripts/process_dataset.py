"""
Kepler-ECG: Process ECG Dataset (v2.0 - Multi-Dataset Support)

Generic script that processes ECG files with integrated dataset registry and label schema.
Uses the existing PreprocessingPipeline from the project.

Supports: PTB-XL, Chapman, CPSC-2018, Georgia (and more via registry).

Supports diagnosis codes from:
- CSV metadata files (PTB-XL style with scp_codes)
- WFDB header comments with SNOMED-CT codes (Chapman, PhysioNet Challenge 2020/2021)

Usage:
    # New mode (recommended) - by dataset name:
    python scripts/process_dataset.py --dataset ptb-xl
    python scripts/process_dataset.py --dataset chapman
    python scripts/process_dataset.py --dataset cpsc-2018 --n_samples 100
    
    # Legacy mode - by paths:
    python scripts/process_dataset.py --data_dir ./data/raw/ptb-xl --output_dir ./results/ptb-xl

    # With specific sampling rate filter (e.g., only 500Hz files):
    python scripts/process_dataset.py --dataset ptb-xl --sampling_rate 500

Supported formats:
    - WFDB (.dat + .hea or .mat + .hea)
    - MAT files (.mat)
    - NumPy (.npy)

Author: Alessandro Marconi for Kepler-ECG Project
Version: 2.0.0 - Integrated with DatasetRegistry and LabelSchema
Issued on: January 2025
"""

# Standard library imports
import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Data analysis
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import preprocessing pipeline
from preprocessing import (
    PreprocessingPipeline,
    PipelineConfig,
    ProcessedECG,
)

# Import core modules
try:
    from core.dataset_registry import (
        get_registry,
        get_dataset_config,
        detect_dataset_from_path,
        DatasetConfig,
        LabelSource,
    )
    from core.label_schema import (
        LabelMapper,
        Superclass,
        parse_header_labels,
        SNOMED_CODE_MAPPING,
    )
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    print(f"⚠️  Warning: core modules not fully available: {e}")
    print("   Some features may be limited.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FALLBACK SNOMED-CT MAPPING (used if core module not available)
# =============================================================================

SNOMED_CT_MAPPING_FALLBACK = {
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
    '27885002': 'IIIAVB',    # Third degree AV block
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
    '17366009': 'APB',       # Atrial premature beat
    '11157007': 'AVBL',      # AV block (low grade)
    '251164006': 'ERD',      # Early repolarization
}


def get_snomed_mapping() -> Dict[str, str]:
    """Get SNOMED-CT mapping from core module or fallback."""
    if CORE_AVAILABLE:
        # Build mapping from core module
        mapping = {}
        for code, (superclass, subclass, desc) in SNOMED_CODE_MAPPING.items():
            # Use a short code derived from description or subclass
            if subclass:
                short = subclass.value.split('_')[-1][:6].upper()
            else:
                short = superclass.value
            mapping[code] = short
        return mapping
    else:
        return SNOMED_CT_MAPPING_FALLBACK


# =============================================================================
# HEADER PARSING
# =============================================================================

def parse_snomed_from_header(header_path: Path, label_mapper: Optional['LabelMapper'] = None) -> Dict[str, Any]:
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
    label_mapper : LabelMapper, optional
        If provided, uses core label mapping for richer output.
        
    Returns
    -------
    Dict with extracted metadata:
        - age: float or None
        - sex: int (0=Male, 1=Female) or None
        - snomed_codes: List[str] of SNOMED-CT codes
        - scp_codes: Dict mapping abbreviation to 100.0 (for compatibility)
        - labels: List[DiagnosticLabel] if label_mapper provided
        - primary_superclass: str if label_mapper provided
        - superclass_vector: Dict if label_mapper provided
    """
    metadata = {
        'age': None,
        'sex': None,
        'snomed_codes': [],
        'scp_codes': {},
    }
    
    if not header_path.exists():
        return metadata
    
    snomed_mapping = get_snomed_mapping()
    
    try:
        with open(header_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line.startswith('#'):
                continue
            
            line = line[1:].strip()
            
            # Parse Age
            age_match = re.match(r'Age[:\s]+(\d+)', line, re.IGNORECASE)
            if age_match:
                try:
                    metadata['age'] = float(age_match.group(1))
                except ValueError:
                    pass
            
            # Parse Sex
            sex_match = re.match(r'Sex[:\s]+(\w+)', line, re.IGNORECASE)
            if sex_match:
                sex_str = sex_match.group(1).lower()
                if sex_str in ['male', 'm', '0']:
                    metadata['sex'] = 0
                elif sex_str in ['female', 'f', '1']:
                    metadata['sex'] = 1
            
            # Parse Diagnosis codes
            dx_match = re.match(r'Dx[:\s]+(.+)', line, re.IGNORECASE)
            if dx_match:
                dx_string = dx_match.group(1)
                codes = re.split(r'[,\s]+', dx_string)
                codes = [c.strip() for c in codes if c.strip().isdigit()]
                metadata['snomed_codes'] = codes
                
                # Convert to scp_codes format for compatibility
                for code in codes:
                    if code in snomed_mapping:
                        abbrev = snomed_mapping[code]
                        metadata['scp_codes'][abbrev] = 100.0
                    else:
                        metadata['scp_codes'][f'SNOMED_{code}'] = 100.0
                
                # Use label mapper for richer output if available
                if label_mapper and CORE_AVAILABLE:
                    labels = label_mapper.map_snomed_codes(codes)
                    metadata['labels'] = labels
                    metadata['primary_superclass'] = label_mapper.get_primary_superclass(labels).value
                    metadata['superclass_vector'] = label_mapper.get_superclass_vector(labels)
        
    except Exception as e:
        logger.debug(f"Could not parse header {header_path}: {e}")
    
    return metadata


# =============================================================================
# DATASET INFO & DETECTION
# =============================================================================

@dataclass
class DatasetInfo:
    """Information about detected dataset structure."""
    data_dir: Path
    name: str
    format: str  # 'wfdb', 'mat', 'npy'
    record_paths: List[Path] = field(default_factory=list)
    metadata_path: Optional[Path] = None
    sampling_rate: Optional[int] = None
    n_records: int = 0
    structure: Dict = field(default_factory=dict)
    has_snomed_headers: bool = False
    # New fields from registry
    registry_config: Optional[DatasetConfig] = None
    label_source: Optional[LabelSource] = None
    n_leads: int = 12


class DatasetDetector:
    """Auto-detect dataset structure and file locations."""
    
    METADATA_FILES = [
        'ptbxl_database.csv',
        'RECORDS',
        'metadata.csv',
        'labels.csv',
        'annotations.csv',
        'reference.csv',
        'ConditionNames_SNOMED-CT.csv',
    ]
    
    def __init__(
        self, 
        data_dir: str, 
        target_sampling_rate: Optional[int] = None,
        dataset_name: Optional[str] = None
    ):
        """
        Parameters
        ----------
        data_dir : str
            Path to dataset directory.
        target_sampling_rate : int, optional
            If specified, only include records at this sampling rate.
        dataset_name : str, optional
            If specified, use registry config for this dataset.
        """
        self.data_dir = Path(data_dir)
        self.target_sampling_rate = target_sampling_rate
        self.dataset_name = dataset_name
        self.registry_config = None
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
        
        # Try to get registry config
        if CORE_AVAILABLE:
            if dataset_name:
                try:
                    self.registry_config = get_dataset_config(dataset_name)
                    logger.info(f"Using registry config for: {dataset_name}")
                except ValueError:
                    logger.warning(f"Dataset '{dataset_name}' not in registry, using auto-detect")
            else:
                # Try to auto-detect from path
                detected = detect_dataset_from_path(self.data_dir)
                if detected:
                    try:
                        self.registry_config = get_dataset_config(detected)
                        logger.info(f"Auto-detected dataset: {detected}")
                    except ValueError:
                        pass
    
    def detect(self) -> DatasetInfo:
        """Detect dataset structure."""
        logger.info(f"Scanning directory: {self.data_dir}")
        
        # Dataset name from registry or directory
        if self.registry_config:
            dataset_name = self.registry_config.name
        else:
            dataset_name = self.dataset_name or self.data_dir.name
        
        info = DatasetInfo(
            data_dir=self.data_dir, 
            name=dataset_name,
            format='unknown',
            registry_config=self.registry_config,
        )
        
        # Use registry config if available
        if self.registry_config:
            info.label_source = self.registry_config.label_source
            info.n_leads = self.registry_config.n_leads
            if self.target_sampling_rate is None:
                self.target_sampling_rate = self.registry_config.sampling_rate
        
        # Count files by extension
        extension_counts = defaultdict(int)
        all_files = []
        
        for root, dirs, files in os.walk(self.data_dir):
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
        
        # Find record files
        info.record_paths = self._find_records(all_files, info.format)
        info.n_records = len(info.record_paths)
        logger.info(f"Found {info.n_records} ECG records")
        
        # Find metadata file
        info.metadata_path = self._find_metadata(all_files)
        if info.metadata_path:
            logger.info(f"Found metadata: {info.metadata_path.name}")
        
        # Check for SNOMED headers
        if info.format == 'wfdb' and info.n_records > 0:
            # Use registry hint or auto-detect
            if self.registry_config and self.registry_config.label_source in [LabelSource.SNOMED, LabelSource.CHALLENGE_LABELS]:
                info.has_snomed_headers = True
                logger.info("Dataset uses SNOMED-CT codes (from registry)")
            else:
                info.has_snomed_headers = self._check_snomed_headers(info.record_paths[:5])
                if info.has_snomed_headers:
                    logger.info("Detected SNOMED-CT codes in WFDB headers")
        
        # Detect sampling rate
        if info.n_records > 0:
            detected_fs = self._detect_sampling_rate(info.record_paths[0], info.format)
            info.sampling_rate = self.target_sampling_rate or detected_fs
            logger.info(f"Sampling rate: {info.sampling_rate} Hz")
        
        # Structure summary
        info.structure = {
            'format': info.format,
            'n_records': info.n_records,
            'n_leads': info.n_leads,
            'has_metadata': info.metadata_path is not None,
            'has_snomed_headers': info.has_snomed_headers,
            'sampling_rate': info.sampling_rate,
            'extensions': dict(extension_counts),
            'from_registry': self.registry_config is not None,
        }
        
        return info
    
    def _detect_format(self, extension_counts: Dict[str, int]) -> str:
        """Detect format from file extensions."""
        if extension_counts.get('.dat', 0) > 0 and extension_counts.get('.hea', 0) > 0:
            return 'wfdb'
        elif extension_counts.get('.mat', 0) > 0 and extension_counts.get('.hea', 0) > 0:
            return 'wfdb'
        elif extension_counts.get('.mat', 0) > 0:
            return 'mat'
        elif extension_counts.get('.npy', 0) > 0:
            return 'npy'
        else:
            return 'unknown'
    
    def _find_records(self, all_files: List[Path], format: str) -> List[Path]:
        """Find all ECG record files."""
        records = []
        
        if format == 'wfdb':
            # Build a set of all file stems for fast lookup
            # This handles case-insensitivity and various naming conventions
            file_stems_by_dir = defaultdict(set)
            files_by_stem = {}
            
            for f in all_files:
                stem_lower = f.stem.lower()
                dir_path = f.parent
                file_stems_by_dir[dir_path].add(stem_lower)
                files_by_stem[(dir_path, stem_lower, f.suffix.lower())] = f
            
            # Find all .hea files that have a corresponding .dat or .mat
            for f in all_files:
                if f.suffix.lower() == '.hea':
                    dir_path = f.parent
                    stem_lower = f.stem.lower()
                    
                    # Check if .dat or .mat exists (case-insensitive)
                    has_dat = (dir_path, stem_lower, '.dat') in files_by_stem
                    has_mat = (dir_path, stem_lower, '.mat') in files_by_stem
                    
                    if has_dat or has_mat:
                        record_path = f.with_suffix('')
                        
                        if self.target_sampling_rate is not None:
                            try:
                                import wfdb
                                header = wfdb.rdheader(str(record_path))
                                if header.fs != self.target_sampling_rate:
                                    continue
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
        # First check registry config
        if self.registry_config and self.registry_config.metadata_file:
            for f in all_files:
                if f.name == self.registry_config.metadata_file:
                    return f
        
        # Fallback to known names
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
                return 500
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
        
        # Initialize label mapper if core is available
        self.label_mapper = None
        if CORE_AVAILABLE:
            self.label_mapper = LabelMapper(dataset_info.name)
    
    def _load_metadata(self) -> Optional[pd.DataFrame]:
        """Load metadata CSV if available."""
        if self.info.metadata_path is None:
            return None
        
        try:
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
        """Load a single ECG record."""
        if self.info.format == 'wfdb':
            return self._load_wfdb(record_path)
        elif self.info.format == 'mat':
            return self._load_mat(record_path)
        elif self.info.format == 'npy':
            return self._load_npy(record_path)
        else:
            raise ValueError(f"Unsupported format: {self.info.format}")
    
    def _load_wfdb(self, record_path: Path) -> Tuple[np.ndarray, int, Dict]:
        """Load WFDB format."""
        import wfdb
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal
        fs = record.fs
        
        metadata = {
            'record_name': record_path.name,
            'n_leads': record.n_sig,
            'sig_names': record.sig_name,
            'age': None,
            'sex': None,
            'scp_codes': {},
        }
        
        # Get labels from header (SNOMED-CT)
        if self.info.has_snomed_headers:
            header_path = record_path.with_suffix('.hea')
            header_metadata = parse_snomed_from_header(header_path, self.label_mapper)
            
            if header_metadata['age'] is not None:
                metadata['age'] = header_metadata['age']
            if header_metadata['sex'] is not None:
                metadata['sex'] = header_metadata['sex']
            if header_metadata['scp_codes']:
                metadata['scp_codes'] = header_metadata['scp_codes']
            if header_metadata['snomed_codes']:
                metadata['snomed_codes'] = header_metadata['snomed_codes']
            
            # Add rich label info if available
            if 'primary_superclass' in header_metadata:
                metadata['primary_superclass'] = header_metadata['primary_superclass']
            if 'superclass_vector' in header_metadata:
                metadata['superclass_vector'] = header_metadata['superclass_vector']
        
        # Get metadata from CSV (PTB-XL style)
        if self.metadata is not None:
            try:
                record_id = record_path.name
                matched_row = None
                
                if record_id in self.metadata.index:
                    matched_row = self.metadata.loc[record_id]
                
                if matched_row is None:
                    try:
                        numeric_id = int(record_id.split('_')[0].lstrip('0') or '0')
                        if numeric_id in self.metadata.index:
                            matched_row = self.metadata.loc[numeric_id]
                    except (ValueError, AttributeError):
                        pass
                
                if matched_row is None:
                    for fname_col in ['filename_hr', 'filename_lr', 'filename']:
                        if fname_col in self.metadata.columns:
                            mask = self.metadata[fname_col].astype(str).str.contains(record_id, regex=False)
                            if mask.any():
                                matched_row = self.metadata.loc[mask].iloc[0]
                                break
                
                if matched_row is not None:
                    for col in ['age', 'sex', 'scp_codes', 'patient_id']:
                        if col in matched_row.index:
                            val = matched_row[col]
                            if col == 'scp_codes' and isinstance(val, str):
                                try:
                                    val = eval(val)
                                except:
                                    pass
                            if val is not None and (not isinstance(val, dict) or val):
                                metadata[col] = val
                    
                    # Map SCP codes using label mapper
                    if self.label_mapper and metadata.get('scp_codes'):
                        labels = self.label_mapper.map_scp_codes(metadata['scp_codes'])
                        metadata['primary_superclass'] = self.label_mapper.get_primary_superclass(labels).value
                        metadata['superclass_vector'] = self.label_mapper.get_superclass_vector(labels)
                        
            except Exception:
                pass
        
        return signal, int(fs), metadata
    
    def _load_mat(self, record_path: Path) -> Tuple[np.ndarray, int, Dict]:
        """Load MAT format."""
        import scipy.io as sio
        mat = sio.loadmat(str(record_path))
        
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
        
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1]:
            signal = signal.T
        
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
        """Iterate over records."""
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
        'snomed_codes': metadata.get('snomed_codes', []),
        
        # New: Unified label info
        'primary_superclass': metadata.get('primary_superclass'),
        
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
    }
    
    # Add superclass binary columns if available
    superclass_vector = metadata.get('superclass_vector', {})
    for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP', 'OTHER']:
        features[f'label_{sc}'] = superclass_vector.get(sc, 0)
    
    # Quality metrics
    if result.quality:
        features['quality_score'] = result.quality.quality_score
        features['quality_level'] = result.quality.quality_level.value if hasattr(result.quality.quality_level, 'value') else str(result.quality.quality_level)
        features['snr_db'] = result.quality.snr_db
    
    # HRV metrics
    if result.hrv and hasattr(result.hrv, 'rr_clean'):
        rr = result.hrv.rr_clean
        if rr is not None and len(rr) > 1:
            features['rr_mean_ms'] = float(np.mean(rr))
            features['rr_std_ms'] = float(np.std(rr))
            features['heart_rate_std'] = float(np.std(60000 / rr)) if np.all(rr > 0) else None
            
            # RMSSD
            if len(rr) > 2:
                diff = np.diff(rr)
                features['rmssd'] = float(np.sqrt(np.mean(diff ** 2)))
    
    return features


# =============================================================================
# PROCESSING CONFIG
# =============================================================================

@dataclass
class ProcessingConfig:
    """Configuration for dataset processing."""
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    dataset_name: Optional[str] = None  # New: use registry
    sampling_rate: Optional[int] = None
    n_samples: Optional[int] = None
    batch_size: int = 100
    lead_idx: int = 0  # Which lead to use for single-lead analysis


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_dataset(config: ProcessingConfig) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Process an entire ECG dataset.
    
    Returns
    -------
    df : pd.DataFrame
        Features for all processed records
    summary : dict
        Processing summary statistics
    """
    logger.info("=" * 70)
    logger.info("KEPLER-ECG DATASET PROCESSOR v2.0")
    logger.info("=" * 70)
    
    # Determine data directory
    if config.dataset_name and CORE_AVAILABLE:
        # Use registry to get default paths
        try:
            registry_config = get_dataset_config(config.dataset_name)
            if config.data_dir is None:
                config.data_dir = f"data/raw/{registry_config.name}"
            if config.output_dir is None:
                config.output_dir = f"results/{registry_config.name}"
            logger.info(f"Using registry config for: {config.dataset_name}")
        except ValueError as e:
            logger.error(f"Dataset not found in registry: {e}")
            return None, None
    
    if config.data_dir is None:
        logger.error("No data directory specified. Use --data_dir or --dataset")
        return None, None
    
    # Step 1: Detect dataset structure
    logger.info("\n[1/4] Detecting dataset structure...")
    
    detector = DatasetDetector(
        config.data_dir, 
        config.sampling_rate,
        config.dataset_name
    )
    dataset_info = detector.detect()
    
    if dataset_info.n_records == 0:
        logger.error("No ECG records found!")
        return None, None
    
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
    if loader.label_mapper:
        logger.info("Using unified label schema")
    
    # Step 3: Process records
    logger.info("\n[3/4] Processing ECG records...")
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_features = []
    failed_ids = []
    start_time = time.time()
    
    for i, (record_path, signal, fs, metadata) in enumerate(loader.iter_records(config.n_samples)):
        ecg_id = record_path.stem
        
        result = pipeline.process(signal, fs=fs, ecg_id=ecg_id)
        features = extract_features(result, metadata)
        features['record_path'] = str(record_path.relative_to(dataset_info.data_dir))
        
        all_features.append(features)
        
        if not result.success:
            failed_ids.append(ecg_id)
        
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
    
    # Superclass distribution (new)
    superclass_counts = defaultdict(int)
    for f in all_features:
        primary = f.get('primary_superclass')
        if primary:
            superclass_counts[primary] += 1
    
    # Legacy diagnosis distribution
    diagnosis_counts = defaultdict(int)
    for f in all_features:
        scp = f.get('scp_codes', {})
        if scp:
            for code in scp.keys():
                diagnosis_counts[code] += 1
    
    # Save CSV
    df = pd.DataFrame(all_features)
    
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
        'n_leads': dataset_info.n_leads,
        'has_snomed_headers': dataset_info.has_snomed_headers,
        'from_registry': dataset_info.registry_config is not None,
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
        'superclass_distribution': dict(superclass_counts),
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
    logger.info(f"Leads:       {dataset_info.n_leads}")
    logger.info(f"Registry:    {'Yes' if dataset_info.registry_config else 'No (auto-detect)'}")
    logger.info(f"SNOMED-CT:   {'Yes' if dataset_info.has_snomed_headers else 'No'}")
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
    
    if superclass_counts:
        logger.info(f"\nSuperclass distribution:")
        for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP', 'OTHER']:
            count = superclass_counts.get(sc, 0)
            if count > 0:
                logger.info(f"  {sc:12s}: {count:5d} ({count/n_processed*100:.1f}%)")
    
    if diagnosis_counts:
        logger.info(f"\nTop 10 diagnoses:")
        sorted_diag = sorted(diagnosis_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for code, count in sorted_diag:
            logger.info(f"  {code:12s}: {count:5d} ({count/n_processed*100:.1f}%)")
    
    logger.info(f"\nOutput: {output_dir}")
    
    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description='Process ECG datasets with Kepler-ECG pipeline (v2.0 with registry support)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New mode (recommended) - by dataset name:
  python scripts/process_dataset.py --dataset ptb-xl
  python scripts/process_dataset.py --dataset chapman --n_samples 100
  
  # Legacy mode - by paths:
  python scripts/process_dataset.py --data_dir ./data/raw/ptb-xl --output_dir ./results/ptb-xl
  
  # With sampling rate filter:
  python scripts/process_dataset.py --dataset ptb-xl --sampling_rate 500
        """
    )
    
    # New: dataset by name
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Dataset name from registry (e.g., ptb-xl, chapman, cpsc-2018, georgia)'
    )
    
    # Legacy: paths
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to save results'
    )
    
    # Options
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
    
    # Validate arguments
    if not args.dataset and not args.data_dir:
        parser.error("Either --dataset or --data_dir is required")
    
    config = ProcessingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        sampling_rate=args.sampling_rate,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )
    
    process_dataset(config)


if __name__ == "__main__":
    main()
