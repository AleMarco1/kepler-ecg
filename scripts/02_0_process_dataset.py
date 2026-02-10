#!/usr/bin/env python3
"""
Kepler-ECG: Process ECG Dataset (v3.5 - Multi-Dataset with Parallel Processing)

Supports: PTB-XL, Chapman, CPSC-2018, Georgia, MIMIC-IV-ECG, CODE-15%.

Usage:
    python scripts/02_process_dataset.py --dataset ptb-xl
    python scripts/02_process_dataset.py --dataset mimic-iv-ecg --n_samples 100 --fast_scan

Output:
    results/[dataset]/preprocess/[dataset]_features.csv

Version: 3.5.0 - Improved logging for metadata loading, restored v3.0 index columns
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator, Any
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import PreprocessingPipeline, PipelineConfig, ProcessedECG

try:
    from core.dataset_registry import get_dataset_config, DatasetConfig, LabelSource
    from core.label_schema import LabelMapper
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

N_WORKERS = min(mp.cpu_count(), 8)

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP', 'ARR', 'OTHER']

SUPERCLASS_PRIORITY = {'MI': 6, 'STTC': 5, 'CD': 4, 'ARR': 3, 'HYP': 2, 'OTHER': 1, 'NORM': 0}

SCP_TO_SUPERCLASS = {
    'NORM': 'NORM', 'SR': 'NORM',
    'IMI': 'MI', 'AMI': 'MI', 'LMI': 'MI', 'PMI': 'MI', 'ASMI': 'MI',
    'ILMI': 'MI', 'IPMI': 'MI', 'IPLMI': 'MI', 'ALMI': 'MI',
    'NDT': 'STTC', 'NST_': 'STTC', 'DIG': 'STTC', 'LNGQT': 'STTC',
    'ISC_': 'STTC', 'ISCA': 'STTC', 'ISCI': 'STTC', 'ISCLA': 'STTC',
    'ISCAS': 'STTC', 'ISCAL': 'STTC', 'ISCIN': 'STTC', 'ISCIL': 'STTC',
    'STD_': 'STTC', 'STE_': 'STTC', 'INVT': 'STTC', 'TAB_': 'STTC',
    'APTS': 'STTC', 'INJAL': 'STTC', 'INJAS': 'STTC', 'INJIN': 'STTC',
    'INJLA': 'STTC', 'INJIL': 'STTC', 'ANEUR': 'STTC', 'EL': 'STTC',
    'LOWT': 'STTC', 'NT_': 'STTC',
    'CLBBB': 'CD', 'CRBBB': 'CD', 'ILBBB': 'CD', 'IRBBB': 'CD',
    'LAFB': 'CD', 'LPFB': 'CD', '1AVB': 'CD', '2AVB': 'CD', '3AVB': 'CD',
    'AVB': 'CD', 'WPW': 'CD', 'IVCD': 'CD',
    'LVH': 'HYP', 'RVH': 'HYP', 'LAO/LAE': 'HYP', 'RAO/RAE': 'HYP',
    'SEHYP': 'HYP', 'VCLVH': 'HYP',
    'AFIB': 'ARR', 'AFLT': 'ARR', 'STACH': 'ARR', 'SBRAD': 'ARR',
    'SARRH': 'ARR', 'SVARR': 'ARR', 'SVTAC': 'ARR', 'PSVT': 'ARR',
    'BIGU': 'ARR', 'TRIGU': 'ARR', 'PAC': 'ARR', 'PVC': 'ARR', 'PACE': 'ARR',
    'ABQRS': 'OTHER', 'QWAVE': 'OTHER', 'LVOLT': 'OTHER', 'HVOLT': 'OTHER',
    'LPR': 'OTHER', 'PRC(S)': 'OTHER',
}

SNOMED_CT_MAPPING = {
    # Original mappings
    '426783006': ('NORM', 'Sinus rhythm'),
    '426177001': ('ARR', 'Sinus bradycardia'),
    '427084000': ('ARR', 'Sinus tachycardia'),
    '698252002': ('NORM', 'Normal ECG'),
    '164889003': ('ARR', 'Atrial fibrillation'),
    '164890007': ('ARR', 'Atrial flutter'),
    '426761007': ('ARR', 'Supraventricular tachycardia'),
    '713422000': ('ARR', 'Atrial tachycardia'),
    '233896004': ('ARR', 'AV nodal reentrant tachycardia'),
    '233897008': ('ARR', 'AV reentrant tachycardia'),
    '195060002': ('ARR', 'Ventricular tachycardia'),
    '427393009': ('ARR', 'Sinus arrhythmia'),
    '284470004': ('ARR', 'Premature atrial contraction'),
    '427172004': ('ARR', 'Premature ventricular contraction'),
    '17338001': ('ARR', 'Ventricular ectopic beat'),
    '164884008': ('ARR', 'Ventricular ectopics'),
    '713427006': ('CD', 'Complete right bundle branch block'),
    '713426002': ('CD', 'Complete left bundle branch block'),
    '59118001': ('CD', 'Right bundle branch block'),
    '164909002': ('CD', 'Left bundle branch block'),
    '6374002': ('CD', 'Bundle branch block'),
    '445118002': ('CD', 'Left anterior fascicular block'),
    '270492004': ('CD', 'First degree AV block'),
    '195042002': ('CD', 'Second degree AV block'),
    '27885002': ('CD', 'Third degree AV block'),
    '74390002': ('CD', 'Wolff-Parkinson-White syndrome'),
    '164873001': ('HYP', 'Left ventricular hypertrophy'),
    '89792004': ('HYP', 'Right ventricular hypertrophy'),
    '164865005': ('MI', 'Myocardial infarction'),
    '164861001': ('MI', 'Acute myocardial infarction'),
    '164930006': ('STTC', 'ST depression'),
    '164931005': ('STTC', 'ST elevation'),
    '429622005': ('STTC', 'ST drop down'),
    '59931005': ('STTC', 'T wave inversion'),
    '164934002': ('STTC', 'T wave change'),
    '111975006': ('STTC', 'Long QT syndrome'),
    '428750005': ('STTC', 'Nonspecific ST change'),
    '164917005': ('OTHER', 'Low QRS voltage'),
    '39732003': ('OTHER', 'Left axis deviation'),
    '47665007': ('OTHER', 'Right axis deviation'),
    
    # NEW ADDITIONS (Georgia)
    '67741000119109': ('HYP', 'Left atrial enlargement'),
    '251146004': ('OTHER', 'Low QRS voltages'),
    '426434006': ('OTHER', 'Low voltage QRS complex'),
    '428417006': ('STTC', 'Early repolarization'),
    '253352002': ('OTHER', 'Clockwise rotation of heart'),
    '445211001': ('OTHER', 'Counterclockwise cardiac rotation'),
    '425623009': ('ARR', 'Paroxysmal atrial fibrillation'),
    '251266004': ('CD', 'Short PR interval'),
    '251268003': ('CD', 'Prolonged PR interval'),
}

# Text-based diagnosis mapping (for MIMIC machine reports)
TEXT_DIAGNOSIS_MAPPING_COMPLETE = {
    # Original mappings - Normal
    'sinus rhythm': 'NORM',
    'normal sinus rhythm': 'NORM',
    'normal ecg': 'NORM',
    
    # Original mappings - Arrhythmias
    'sinus rhythm with pac(s).': 'ARR',
    'sinus rhythm with pacs.': 'ARR',
    'sinus rhythm with pvcs': 'ARR',
    'sinus rhythm with pvc(s)': 'ARR',
    'atrial premature complex': 'ARR',
    'atrial premature complexes': 'ARR',
    'ventricular premature complex': 'ARR',
    'atrial fibrillation': 'ARR',
    'atrial flutter': 'ARR',
    'sinus tachycardia': 'NORM',
    'sinus bradycardia': 'NORM',
    'sinus arrhythmia': 'ARR',
    'unknown rhythm': 'ARR',
    'irregular rhythm': 'ARR',
    
    # Original mappings - STTC
    'st-t changes': 'STTC',
    'st changes': 'STTC',
    't wave changes': 'STTC',
    't wave abnormality': 'STTC',
    'st elevation': 'STTC',
    'st depression': 'STTC',
    'septal st-t changes': 'STTC',
    'nonspecific st-t': 'STTC',
    
    # Original mappings - Conduction
    'left bundle branch block': 'CD',
    'right bundle branch block': 'CD',
    'first degree av block': 'CD',
    'second degree av block': 'CD',
    'third degree av block': 'CD',
    
    # Original mappings - Hypertrophy
    'left ventricular hypertrophy': 'HYP',
    'right ventricular hypertrophy': 'HYP',
    'left atrial enlargement': 'HYP',
    'right atrial enlargement': 'HYP',
    
    # Original mappings - MI
    'myocardial infarction': 'MI',
    'anterior mi': 'MI',
    'inferior mi': 'MI',
    'lateral mi': 'MI',
    
    # Original mappings - Other
    'low qrs voltage': 'OTHER',
    'low qrs voltages': 'OTHER',
    'leftward axis': 'OTHER',
    'left axis deviation': 'OTHER',
    'right axis deviation': 'OTHER',
    
    # NEW ADDITIONS (MIMIC)
    'borderline ecg': 'NORM',
    'sinus rhythm.': 'NORM',
    'rsr\'(v1) - probable normal variant': 'NORM',
    'poor r wave progression - probable normal variant': 'NORM',
    'st elev, probable normal early repol pattern': 'NORM',
    'sinus rhythm with borderline 1st degree a-v block': 'NORM',
    'sinus rhythm with borderline 1st degree a-v block.': 'NORM',
    '- borderline first degree a-v block': 'NORM',
    'lateral st-t changes are nonspecific': 'STTC',
    'lateral st changes are nonspecific': 'STTC',
    'anterolateral st-t changes are nonspecific': 'STTC',
    'septal st changes are nonspecific': 'STTC',
    'septal st-t changes are nonspecific': 'STTC',
    'anterior st changes are nonspecific': 'STTC',
    'ant/septal and lateral st-t changes are nonspecific': 'STTC',
    'inferior/lateral st-t changes are nonspecific': 'STTC',
    'st junctional depression is nonspecific': 'STTC',
    'repol abnrm suggests ischemia, anterolateral': 'STTC',
    'repol abnrm suggests ischemia, diffuse leads': 'STTC',
    'repol abnrm suggests ischemia, lateral leads': 'STTC',
    'nonspecific t abnrm, anterolateral leads': 'STTC',
    'prolonged qt interval': 'STTC',
    'long qtc interval': 'STTC',
    'low qrs voltages in precordial leads': 'OTHER',
    'low voltage, precordial leads': 'OTHER',
    'left axis deviation - possible left anterior fascicular block': 'CD',
    'short pr interval': 'CD',
    'nonspecific intraventricular conduction delay': 'CD',
    'rbbb and lafb': 'CD',
    'probable left atrial enlargement': 'HYP',
    'unknown rhythm, irregular rate': 'ARR',
    'afib/flut and v-paced complexes': 'ARR',
    'sinus rhythm with occasional pacs': 'ARR',
    'sinus rhythm with pac(s)': 'ARR',
}


def determine_superclass_from_scp(scp_codes: Dict[str, float]) -> Optional[str]:
    """Determine primary superclass from SCP codes or text diagnoses."""
    if not scp_codes:
        return None
    active_codes = {k: v for k, v in scp_codes.items() if v > 0}
    if not active_codes:
        return None
    
    superclasses_found = set()
    
    for code in active_codes.keys():
        # Try SCP code mapping first
        if code in SCP_TO_SUPERCLASS:
            superclasses_found.add(SCP_TO_SUPERCLASS[code])
        else:
            # Try text-based mapping (for MIMIC machine reports)
            code_lower = code.lower().strip()
            
            # Check for exact match first, then partial match
            # Sort by length descending to match longer (more specific) patterns first
            matched = False
            for text_pattern in sorted(TEXT_DIAGNOSIS_MAPPING.keys(), key=len, reverse=True):
                if text_pattern in code_lower:
                    superclasses_found.add(TEXT_DIAGNOSIS_MAPPING[text_pattern])
                    matched = True
                    break
            
            # If no match found, try reverse (code in pattern)
            if not matched:
                for text_pattern, superclass in TEXT_DIAGNOSIS_MAPPING.items():
                    if code_lower in text_pattern:
                        superclasses_found.add(superclass)
                        break
    
    if not superclasses_found:
        return None  # Let caller decide
    
    return max(superclasses_found, key=lambda x: SUPERCLASS_PRIORITY.get(x, 0))


def parse_snomed_from_header(header_path: Path) -> Dict[str, Any]:
    metadata = {'age': None, 'sex': None, 'snomed_codes': [], 'scp_codes': {}, 'primary_superclass': None}
    if not header_path.exists():
        return metadata
    try:
        with open(header_path, 'r') as f:
            content = f.read()
        for line in content.split('\n'):
            line = line.strip()
            if not line.startswith('#'):
                continue
            line = line[1:].strip()
            age_match = re.match(r'Age[:\s]+(\d+)', line, re.IGNORECASE)
            if age_match:
                metadata['age'] = float(age_match.group(1))
            sex_match = re.match(r'Sex[:\s]+(\w+)', line, re.IGNORECASE)
            if sex_match:
                sex_str = sex_match.group(1).lower()
                if sex_str in ['male', 'm', '0']:
                    metadata['sex'] = 0
                elif sex_str in ['female', 'f', '1']:
                    metadata['sex'] = 1
            dx_match = re.match(r'Dx[:\s]+(.+)', line, re.IGNORECASE)
            if dx_match:
                codes = re.split(r'[,\s]+', dx_match.group(1))
                codes = [c.strip() for c in codes if c.strip().isdigit()]
                metadata['snomed_codes'] = codes
                superclasses_found = {}
                for code in codes:
                    if code in SNOMED_CT_MAPPING:
                        sc, desc = SNOMED_CT_MAPPING[code]
                        metadata['scp_codes'][desc] = 100.0
                        superclasses_found[sc] = SUPERCLASS_PRIORITY.get(sc, 0)
                if superclasses_found:
                    metadata['primary_superclass'] = max(superclasses_found.keys(), key=lambda x: superclasses_found[x])
    except Exception:
        pass
    return metadata


@dataclass
class DatasetInfo:
    data_dir: Path
    name: str
    format: str
    record_paths: List[Path] = field(default_factory=list)
    metadata_path: Optional[Path] = None
    sampling_rate: Optional[int] = None
    n_records: int = 0
    structure: Dict = field(default_factory=dict)
    has_snomed_headers: bool = False
    registry_config: Optional[Any] = None
    label_source: Optional[Any] = None
    n_leads: int = 12
    machine_measurements: Optional[pd.DataFrame] = None


class DatasetDetector:
    METADATA_FILES = ['ptbxl_database.csv', 'RECORDS', 'exams.csv', 'machine_measurements.csv', 'record_list.csv']

    def __init__(self, data_dir: str, target_sampling_rate: Optional[int] = None,
                 dataset_name: Optional[str] = None, fast_scan: bool = False, n_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.target_sampling_rate = target_sampling_rate
        self.dataset_name = dataset_name
        self.fast_scan = fast_scan
        self.n_samples = n_samples
        self.registry_config = None
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
        if CORE_AVAILABLE and dataset_name:
            try:
                self.registry_config = get_dataset_config(dataset_name)
            except ValueError:
                pass

    def detect(self) -> DatasetInfo:
        logger.info(f"Scanning directory: {self.data_dir}")
        dataset_name = self.dataset_name or self.data_dir.name
        info = DatasetInfo(data_dir=self.data_dir, name=dataset_name, format='unknown', registry_config=self.registry_config)
        
        if self.registry_config:
            info.label_source = self.registry_config.label_source
            info.n_leads = self.registry_config.n_leads
            if self.target_sampling_rate is None:
                self.target_sampling_rate = self.registry_config.sampling_rate

        extension_counts = defaultdict(int)
        all_files = []
        metadata_file = None
        early_stop_limit = (self.n_samples or 100) * 3 if self.fast_scan else None
        records_found = 0

        if self.fast_scan:
            logger.info(f"Fast scan mode: will stop after ~{early_stop_limit} records")

        for root, dirs, files in os.walk(self.data_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                ext = Path(f).suffix.lower()
                extension_counts[ext] += 1
                file_path = Path(root) / f
                # Priority: ptbxl_database.csv > exams.csv > others
                # Don't overwrite higher-priority metadata with lower-priority
                if f in self.METADATA_FILES:
                    if metadata_file is None or f == 'ptbxl_database.csv' or (f == 'exams.csv' and metadata_file.name == 'RECORDS'):
                        metadata_file = file_path
                        logger.info(f"Found metadata file: {file_path}")
                all_files.append(file_path)
                if ext in ['.hea', '.hdf5']:
                    records_found += 1
            if early_stop_limit and records_found >= early_stop_limit:
                logger.info(f"Fast scan: stopped after {records_found} records")
                break

        logger.info(f"Found {len(all_files)} files, extensions: {dict(extension_counts)}")

        info.format = self._detect_format(extension_counts, dataset_name)
        info.record_paths = self._find_records(all_files, info.format)
        info.n_records = len(info.record_paths)
        info.metadata_path = metadata_file

        if 'mimic' in dataset_name.lower():
            mm_path = self.data_dir / 'machine_measurements.csv'
            if mm_path.exists():
                logger.info("Loading MIMIC machine_measurements.csv...")
                info.machine_measurements = pd.read_csv(mm_path)
                info.machine_measurements.set_index('study_id', inplace=True)
                logger.info(f"Loaded {len(info.machine_measurements)} measurements")

        if info.format == 'wfdb' and info.n_records > 0:
            info.has_snomed_headers = self._check_snomed_headers(info.record_paths[:5])
            detected_fs = self._detect_sampling_rate(info.record_paths[0])
            info.sampling_rate = self.target_sampling_rate or detected_fs or 500
        elif info.format == 'hdf5':
            info.sampling_rate = self.target_sampling_rate or 400

        logger.info(f"Format: {info.format}, Records: {info.n_records}, Sampling rate: {info.sampling_rate} Hz")
        return info

    def _detect_format(self, ext_counts: Dict, name: str) -> str:
        if ext_counts.get('.hdf5', 0) > 0 and 'code' in name.lower():
            return 'hdf5'
        if ext_counts.get('.hea', 0) > 0:
            return 'wfdb'
        return 'unknown'

    def _find_records(self, all_files: List[Path], fmt: str) -> List[Path]:
        if fmt == 'wfdb':
            stems = {}
            for f in all_files:
                stems[(f.parent, f.stem.lower(), f.suffix.lower())] = f
            records = []
            for f in all_files:
                if f.suffix.lower() == '.hea':
                    has_data = (f.parent, f.stem.lower(), '.dat') in stems or (f.parent, f.stem.lower(), '.mat') in stems
                    if has_data:
                        records.append(f.with_suffix(''))
            return sorted(records)
        elif fmt == 'hdf5':
            return sorted([f for f in all_files if f.suffix.lower() == '.hdf5'])
        return []

    def _check_snomed_headers(self, paths: List[Path]) -> bool:
        for p in paths:
            hea = p.with_suffix('.hea')
            if hea.exists():
                try:
                    if re.search(r'#\s*Dx[:\s]+\d+', hea.read_text(), re.IGNORECASE):
                        return True
                except:
                    pass
        return False

    def _detect_sampling_rate(self, record_path: Path) -> Optional[int]:
        try:
            import wfdb
            return wfdb.rdheader(str(record_path)).fs
        except:
            return None


class GenericECGLoader:
    def __init__(self, dataset_info: DatasetInfo):
        self.info = dataset_info
        self.metadata = self._load_metadata()
        self.label_mapper = None
        if CORE_AVAILABLE:
            try:
                self.label_mapper = LabelMapper(dataset_info.name)
            except:
                pass
        self._hdf5_exam_list = None
        if self.info.format == 'hdf5':
            self._prepare_hdf5_index()

    def _load_metadata(self) -> Optional[pd.DataFrame]:
        if self.info.metadata_path is None:
            logger.warning("No metadata_path found in dataset detection")
            return None
        try:
            logger.info(f"Loading metadata from: {self.info.metadata_path}")
            df = pd.read_csv(self.info.metadata_path)
            for col in ['ecg_id', 'record_id', 'id', 'filename', 'exam_id', 'study_id']:
                if col in df.columns:
                    df = df.set_index(col)
                    logger.info(f"Loaded metadata: {len(df)} rows, index col: {col}")
                    break
            return df
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            return None

    def _prepare_hdf5_index(self):
        if self.metadata is not None and 'trace_file' in self.metadata.columns:
            df = self.metadata.reset_index()
            df = df.sort_values(['trace_file', 'exam_id'])
            df['trace_index'] = df.groupby('trace_file').cumcount()
            self._hdf5_exam_list = df

    def iter_records(self, n_samples: Optional[int] = None) -> Generator:
        if self.info.format == 'hdf5':
            yield from self._iter_hdf5(n_samples)
        else:
            paths = self.info.record_paths[:n_samples] if n_samples else self.info.record_paths
            for p in paths:
                try:
                    yield self._load_wfdb(p)
                except Exception as e:
                    logger.warning(f"Failed: {p}: {e}")

    def _load_wfdb(self, record_path: Path) -> Tuple[Path, np.ndarray, int, Dict]:
        import wfdb
        rec = wfdb.rdrecord(str(record_path))
        meta = {'age': None, 'sex': None, 'scp_codes': {}, 'snomed_codes': [], 'primary_superclass': None}
        
        if self.info.has_snomed_headers:
            hdr = parse_snomed_from_header(record_path.with_suffix('.hea'))
            meta.update({k: v for k, v in hdr.items() if v})

        if self.metadata is not None:
            self._enrich_from_csv(record_path, meta)

        if self.info.machine_measurements is not None:
            self._enrich_from_mimic(record_path, meta)

        # Chapman/Georgia pattern: if Age is present but Sex is missing, assume Male (0)
        # These datasets only annotate Female explicitly in headers
        if meta.get('age') is not None and meta.get('sex') is None:
            if self.info.name.lower() in ['chapman', 'georgia', 'cpsc-2018', 'cpsc']:
                meta['sex'] = 0  # Default to Male

        return record_path, rec.p_signal, int(rec.fs), meta

    def _enrich_from_csv(self, record_path: Path, meta: Dict):
        try:
            rec_id = record_path.name
            row = None
            
            # Try direct match
            if rec_id in self.metadata.index:
                row = self.metadata.loc[rec_id]
            
            # Try numeric match (PTB-XL style: 00001_hr -> 1)
            if row is None:
                try:
                    base_name = rec_id.split('_')[0]  # "00001_hr" -> "00001"
                    num_id = int(base_name.lstrip('0') or '0')  # "00001" -> 1
                    if num_id in self.metadata.index:
                        row = self.metadata.loc[num_id]
                except (ValueError, AttributeError):
                    pass
            
            # Try filename column match (PTB-XL has filename_hr column)
            if row is None and self.metadata is not None:
                for fname_col in ['filename_hr', 'filename_lr', 'filename']:
                    if fname_col in self.metadata.columns:
                        mask = self.metadata[fname_col].astype(str).str.contains(rec_id, regex=False, na=False)
                        if mask.any():
                            row = self.metadata.loc[mask].iloc[0]
                            break
            
            if row is not None:
                # Extract age - handle NaN properly
                if 'age' in row.index:
                    val = row['age']
                    if pd.notna(val):
                        meta['age'] = float(val)
                
                # Extract sex - IMPORTANT: 0 is valid (male)!
                if 'sex' in row.index:
                    val = row['sex']
                    if pd.notna(val):
                        meta['sex'] = int(val)
                
                # Extract scp_codes
                if 'scp_codes' in row.index:
                    val = row['scp_codes']
                    if isinstance(val, str) and val.strip():
                        try:
                            val = ast.literal_eval(val)
                        except:
                            val = {}
                    if isinstance(val, dict) and val:
                        meta['scp_codes'] = val
                
                # Determine primary superclass
                if meta.get('scp_codes'):
                    # Try using LabelMapper first (more comprehensive)
                    if CORE_AVAILABLE and hasattr(self, 'label_mapper') and self.label_mapper:
                        try:
                            labels = self.label_mapper.map_scp_codes(meta['scp_codes'])
                            meta['primary_superclass'] = self.label_mapper.get_primary_superclass(labels).value
                        except:
                            meta['primary_superclass'] = determine_superclass_from_scp(meta['scp_codes'])
                    else:
                        meta['primary_superclass'] = determine_superclass_from_scp(meta['scp_codes'])
        except Exception as e:
            logger.debug(f"Error enriching from CSV: {e}")

    def _enrich_from_mimic(self, record_path: Path, meta: Dict):
        try:
            study_id = int(record_path.stem)
            if study_id not in self.info.machine_measurements.index:
                return
            row = self.info.machine_measurements.loc[study_id]
            reports = [str(row[f'report_{i}']) for i in range(18) if f'report_{i}' in row.index and pd.notna(row[f'report_{i}'])]
            if reports:
                txt = ' '.join(reports).lower()
                for kw, sc in [('infarct', 'MI'), ('atrial fibrillation', 'ARR'), ('atrial flutter', 'ARR'),
                               ('bundle branch block', 'CD'), ('av block', 'CD'), ('hypertrophy', 'HYP'),
                               ('st elevation', 'STTC'), ('st depression', 'STTC'), ('t wave', 'STTC'),
                               ('tachycardia', 'ARR'), ('bradycardia', 'ARR'), ('normal ecg', 'NORM'),
                               ('within normal', 'NORM'), ('abnormal', 'OTHER')]:
                    if kw in txt:
                        meta['primary_superclass'] = sc
                        break
                for r in reports[:6]:
                    if r.strip():
                        meta['scp_codes'][r.strip()] = 100.0
        except:
            pass

    def _iter_hdf5(self, n_samples: Optional[int]) -> Generator:
        if self._hdf5_exam_list is None:
            return
        exams = self._hdf5_exam_list[:n_samples] if n_samples else self._hdf5_exam_list
        for tf, grp in exams.groupby('trace_file'):
            hdf_path = self.info.data_dir / tf
            if not hdf_path.exists():
                continue
            try:
                with h5py.File(hdf_path, 'r') as f:
                    traces = f['tracings']
                    for _, row in grp.iterrows():
                        idx = row['trace_index']
                        if idx >= traces.shape[0]:
                            continue
                        sig = traces[idx]
                        exam_id = row.get('exam_id', row.name)
                        sc = 'OTHER'
                        if row.get('AF'):
                            sc = 'ARR'
                        elif row.get('1dAVb') or row.get('RBBB') or row.get('LBBB'):
                            sc = 'CD'
                        elif row.get('SB') or row.get('ST'):
                            sc = 'ARR'
                        elif row.get('normal_ecg'):
                            sc = 'NORM'
                        meta = {'age': row.get('age'), 'sex': 0 if row.get('is_male') else 1,
                                'scp_codes': {}, 'primary_superclass': sc}
                        yield Path(f"code15_{exam_id}"), sig, self.info.sampling_rate, meta
            except Exception as e:
                logger.warning(f"HDF5 error {hdf_path}: {e}")

    def __len__(self):
        if self._hdf5_exam_list is not None:
            return len(self._hdf5_exam_list)
        return len(self.info.record_paths)


def extract_features(result: ProcessedECG, meta: Dict) -> Dict:
    is_usable = bool(result.is_usable and result.n_beats > 0 and result.heart_rate_bpm)
    feat = {
        'ecg_id': result.ecg_id, 'success': result.success,
        'processing_time_ms': result.processing_time_sec * 1000,
        'age': meta.get('age'), 'sex': meta.get('sex'),
        'scp_codes': meta.get('scp_codes', {}), 'snomed_codes': meta.get('snomed_codes', []),
        'primary_superclass': meta.get('primary_superclass'),
        'quality_score': None, 'quality_level': None, 'snr_db': None, 'is_usable': is_usable,
        'heart_rate_bpm': result.heart_rate_bpm, 'heart_rate_std': None, 'n_beats': result.n_beats,
        'rr_mean_ms': None, 'rr_std_ms': None, 'rmssd': None,
    }
    for sc in SUPERCLASSES:
        feat[f'label_{sc}'] = 1 if meta.get('primary_superclass') == sc else 0
    if result.quality:
        feat['quality_score'] = result.quality.quality_score
        feat['quality_level'] = getattr(result.quality.quality_level, 'value', str(result.quality.quality_level))
        feat['snr_db'] = result.quality.snr_db
    if result.hrv and hasattr(result.hrv, 'rr_clean') and result.hrv.rr_clean is not None:
        rr = result.hrv.rr_clean
        if len(rr) > 1:
            feat['rr_mean_ms'] = float(np.mean(rr))
            feat['rr_std_ms'] = float(np.std(rr))
            feat['heart_rate_std'] = float(np.std(60000 / rr)) if np.all(rr > 0) else None
            if len(rr) > 2:
                feat['rmssd'] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    return feat


def process_single(args: Tuple) -> Dict:
    path, sig, fs, meta, cfg = args
    try:
        pipe = PreprocessingPipeline(cfg)
        ecg_id = path.stem if hasattr(path, 'stem') else str(path)
        res = pipe.process(sig, fs=fs, ecg_id=ecg_id)
        feat = extract_features(res, meta)
        feat['record_path'] = str(path)
        return {'success': True, 'features': feat, 'ecg_id': ecg_id}
    except Exception as e:
        return {'success': False, 'features': None, 'ecg_id': str(path), 'error': str(e)}


@dataclass
class ProcessingConfig:
    dataset_name: str
    data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    sampling_rate: Optional[int] = None
    n_samples: Optional[int] = None
    n_workers: int = N_WORKERS
    fast_scan: bool = False
    batch_size: int = 10000  # Process in batches to manage memory
    resume: bool = False  # Resume from last checkpoint


def process_batch(batch_records: List, config: ProcessingConfig, batch_num: int) -> Tuple[List[Dict], List[str]]:
    """Process a single batch of records."""
    all_feat, failed = [], []
    
    if config.n_workers > 1 and len(batch_records) > 50:
        with ProcessPoolExecutor(max_workers=config.n_workers) as ex:
            futs = {ex.submit(process_single, r): r[0] for r in batch_records}
            for i, fut in enumerate(as_completed(futs)):
                res = fut.result()
                if res['success']:
                    all_feat.append(res['features'])
                else:
                    failed.append(res['ecg_id'])
                if (i + 1) % 500 == 0:
                    logger.info(f"  Batch {batch_num}: {i+1}/{len(batch_records)}")
    else:
        for i, r in enumerate(batch_records):
            res = process_single(r)
            if res['success']:
                all_feat.append(res['features'])
            else:
                failed.append(res['ecg_id'])
    
    return all_feat, failed


def save_batch_results(features: List[Dict], out_dir: Path, dataset_name: str, batch_num: int):
    """Save intermediate batch results."""
    if not features:
        return
    
    df = pd.DataFrame(features)
    for col in ['scp_codes', 'snomed_codes']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
    
    batch_file = out_dir / f'{dataset_name}_batch_{batch_num:04d}.csv'
    df.to_csv(batch_file, index=False)
    logger.info(f"  Saved batch {batch_num}: {len(features)} records -> {batch_file.name}")


def merge_batch_files(out_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Merge all batch files into final output."""
    batch_files = sorted(out_dir.glob(f'{dataset_name}_batch_*.csv'))
    
    if not batch_files:
        return pd.DataFrame()
    
    logger.info(f"Merging {len(batch_files)} batch files...")
    dfs = []
    for bf in batch_files:
        dfs.append(pd.read_csv(bf))
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Save merged file
    csv_path = out_dir / f'{dataset_name}_features.csv'
    df.to_csv(csv_path, index=False)
    
    # Clean up batch files
    for bf in batch_files:
        bf.unlink()
    logger.info(f"Cleaned up {len(batch_files)} batch files")
    
    return df


def get_completed_batches(out_dir: Path, dataset_name: str) -> set:
    """Get set of already completed batch numbers for resume."""
    batch_files = out_dir.glob(f'{dataset_name}_batch_*.csv')
    completed = set()
    for bf in batch_files:
        try:
            # Extract batch number from filename
            num = int(bf.stem.split('_')[-1])
            completed.add(num)
        except ValueError:
            pass
    return completed


def process_dataset(config: ProcessingConfig):
    logger.info("=" * 70)
    logger.info("KEPLER-ECG DATASET PROCESSOR v3.7 (Batched)")
    logger.info("=" * 70)

    if config.data_dir is None:
        config.data_dir = f"data/raw/{config.dataset_name}"
    if config.output_dir is None:
        config.output_dir = f"results/{config.dataset_name}"

    detector = DatasetDetector(config.data_dir, config.sampling_rate, config.dataset_name,
                               config.fast_scan, config.n_samples)
    info = detector.detect()
    if info.n_records == 0:
        logger.error("No records found!")
        return None, None

    loader = GenericECGLoader(info)
    pipe_cfg = PipelineConfig(apply_filtering=True, assess_quality=True, segment_beats=True,
                              preprocess_hrv=True, enable_cache=False, save_filtered_signal=False, save_beats=False)

    n_total = config.n_samples or len(loader)
    n_batches = (n_total + config.batch_size - 1) // config.batch_size
    
    logger.info(f"Total records: {n_total}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Number of batches: {n_batches}")
    logger.info(f"Workers: {config.n_workers}")

    out_dir = Path(config.output_dir) / 'preprocess'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for resume
    completed_batches = set()
    if config.resume:
        completed_batches = get_completed_batches(out_dir, config.dataset_name)
        if completed_batches:
            logger.info(f"Resuming: {len(completed_batches)} batches already completed")

    t0 = time.time()
    total_processed = 0
    total_failed = 0
    all_failed_ids = []

    # Process in batches using generator to avoid loading all into memory
    batch_num = 0
    batch_records = []
    records_seen = 0
    
    for record_data in loader.iter_records(config.n_samples):
        path, sig, fs, meta = record_data
        records_seen += 1
        
        # Skip records from already completed batches (for resume)
        current_batch = (records_seen - 1) // config.batch_size + 1
        if current_batch in completed_batches:
            if records_seen % config.batch_size == 0:
                logger.info(f"Skipping batch {current_batch}/{n_batches} (already completed)")
            continue
        
        batch_records.append((path, sig, fs, meta, pipe_cfg))
        
        # When batch is full, process it
        if len(batch_records) >= config.batch_size:
            batch_num += 1
            
            logger.info(f"Processing batch {batch_num}/{n_batches} ({len(batch_records)} records)...")
            
            features, failed = process_batch(batch_records, config, batch_num)
            total_processed += len(features)
            total_failed += len(failed)
            all_failed_ids.extend(failed[:10])  # Keep only first 10 per batch
            
            # Save batch results
            save_batch_results(features, out_dir, config.dataset_name, batch_num)
            
            # Clear memory
            batch_records = []
            del features, failed
            import gc
            gc.collect()
            
            elapsed = time.time() - t0
            rate = total_processed / elapsed if elapsed > 0 else 0
            remaining = n_total - records_seen
            eta = remaining / rate / 60 if rate > 0 else 0
            logger.info(f"  Progress: {records_seen}/{n_total} ({rate:.1f} ECG/s, ETA: {eta:.1f} min)")

    # Process remaining records in last batch
    if batch_records:
        batch_num += 1
        logger.info(f"Processing final batch {batch_num}/{n_batches} ({len(batch_records)} records)...")
        features, failed = process_batch(batch_records, config, batch_num)
        total_processed += len(features)
        total_failed += len(failed)
        all_failed_ids.extend(failed[:10])
        save_batch_results(features, out_dir, config.dataset_name, batch_num)

    elapsed = time.time() - t0

    # Merge all batch files
    logger.info("\nMerging batch results...")
    df = merge_batch_files(out_dir, config.dataset_name)
    
    if len(df) == 0:
        logger.error("No results to save!")
        return None, None

    # Save JSON summary (not full data - too large)
    n_usable = df['is_usable'].sum() if 'is_usable' in df.columns else 0
    sc_dist = df['primary_superclass'].value_counts(dropna=False).to_dict()
    
    summary = {
        'dataset': config.dataset_name,
        'n_processed': len(df),
        'n_usable': int(n_usable),
        'n_failed': total_failed,
        'processing_time_sec': elapsed,
        'throughput_ecg_per_sec': len(df) / elapsed if elapsed > 0 else 0,
        'superclass_distribution': {str(k): int(v) for k, v in sc_dist.items()},
        'failed_ids_sample': all_failed_ids[:100]
    }
    
    json_path = out_dir / f'{config.dataset_name}_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"COMPLETE: {len(df)} processed, {n_usable} usable, {total_failed} failed")
    logger.info(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min), Throughput: {len(df)/elapsed:.1f} ECG/s")
    logger.info(f"Superclass distribution: {dict(sc_dist)}")
    logger.info(f"Output: {out_dir / f'{config.dataset_name}_features.csv'}")

    return df, summary


def main():
    parser = argparse.ArgumentParser(description='Process ECG datasets v3.7 (Batched)')
    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--sampling_rate', type=int)
    parser.add_argument('--fast_scan', action='store_true')
    parser.add_argument('--batch_size', type=int, default=10000, 
                        help='Records per batch (default: 10000)')
    parser.add_argument('--workers', type=int, default=N_WORKERS,
                        help=f'Number of parallel workers (default: {N_WORKERS})')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    args = parser.parse_args()
    
    config = ProcessingConfig(
        args.dataset, 
        sampling_rate=args.sampling_rate,
        n_samples=args.n_samples, 
        fast_scan=args.fast_scan,
        batch_size=args.batch_size,
        n_workers=args.workers,
        resume=args.resume
    )
    process_dataset(config)


if __name__ == "__main__":
    main()
