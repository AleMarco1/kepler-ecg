"""
Dataset Registry for Kepler-ECG Multi-Dataset Support

Central configuration for all supported ECG datasets.
Each dataset has its own configuration including:
- Sampling rate
- Number of leads
- File format
- Label schema
- Metadata loader
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class FileFormat(Enum):
    """Supported ECG file formats."""
    WFDB = "wfdb"           # PhysioNet format (.dat/.hea)
    MAT = "mat"             # MATLAB format
    NPY = "npy"             # NumPy format
    CSV = "csv"             # CSV format


class LabelSource(Enum):
    """Types of label/annotation sources."""
    SCP_CODES = "scp_codes"           # PTB-XL style SCP-ECG codes
    SNOMED = "snomed"                  # SNOMED-CT codes (Chapman)
    CHALLENGE_LABELS = "challenge"     # PhysioNet Challenge format
    BEAT_ANNOTATIONS = "beat"          # Beat-by-beat annotations (MIT-BIH)
    RHYTHM_ANNOTATIONS = "rhythm"      # Rhythm annotations


@dataclass
class DatasetConfig:
    """Configuration for a single ECG dataset."""
    
    # Basic info
    name: str
    description: str
    
    # Signal properties
    sampling_rate: int
    n_leads: int
    lead_names: List[str]
    duration_seconds: Optional[float] = 10.0  # Typical record duration
    
    # File structure
    file_format: FileFormat = FileFormat.WFDB
    data_subdir: str = ""  # Subdirectory containing signal files
    metadata_file: Optional[str] = None  # CSV/JSON with metadata
    
    # Label configuration
    label_source: LabelSource = LabelSource.SCP_CODES
    label_column: Optional[str] = None  # Column name in metadata
    
    # PhysioNet URL for download
    physionet_url: Optional[str] = None
    
    # Custom loader function name (if needed)
    custom_loader: Optional[str] = None
    
    # Additional dataset-specific settings
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_leads != len(self.lead_names):
            raise ValueError(
                f"n_leads ({self.n_leads}) must match lead_names length ({len(self.lead_names)})"
            )


# Standard 12-lead names
STANDARD_12_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# =============================================================================
# Dataset Configurations
# =============================================================================

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    
    # -------------------------------------------------------------------------
    # PTB-XL - Gold Standard
    # -------------------------------------------------------------------------
    "ptb-xl": DatasetConfig(
        name="ptb-xl",
        description="PTB-XL ECG Database - Large 12-lead ECG dataset with SCP-ECG codes",
        sampling_rate=500,
        n_leads=12,
        lead_names=STANDARD_12_LEADS,
        duration_seconds=10.0,
        file_format=FileFormat.WFDB,
        data_subdir="records500",  # 500Hz version
        metadata_file="ptbxl_database.csv",
        label_source=LabelSource.SCP_CODES,
        label_column="scp_codes",
        physionet_url="https://physionet.org/files/ptb-xl/1.0.3/",
        extra_config={
            "has_100hz_version": True,
            "scp_statements_file": "scp_statements.csv",
            "age_column": "age",
            "sex_column": "sex",
            "record_id_column": "ecg_id",
            "filename_column": "filename_hr",  # High-resolution filename
        }
    ),
    
    # -------------------------------------------------------------------------
    # Chapman-Shaoxing
    # -------------------------------------------------------------------------
    "chapman": DatasetConfig(
        name="chapman",
        description="Chapman-Shaoxing 12-lead ECG Database",
        sampling_rate=500,
        n_leads=12,
        lead_names=STANDARD_12_LEADS,
        duration_seconds=10.0,
        file_format=FileFormat.WFDB,
        data_subdir="",
        metadata_file=None,  # Labels in header files
        label_source=LabelSource.SNOMED,
        label_column=None,  # Read from .hea comments
        physionet_url="https://physionet.org/files/ecg-arrhythmia/1.0.0/",
        extra_config={
            "labels_in_header": True,
            "snomed_mapping_required": True,
        }
    ),
    
    # -------------------------------------------------------------------------
    # CPSC-2018 (PhysioNet Challenge 2020)
    # -------------------------------------------------------------------------
    "cpsc-2018": DatasetConfig(
        name="cpsc-2018",
        description="China Physiological Signal Challenge 2018 - Training data",
        sampling_rate=500,
        n_leads=12,
        lead_names=STANDARD_12_LEADS,
        duration_seconds=None,  # Variable length
        file_format=FileFormat.WFDB,
        data_subdir="",
        metadata_file=None,  # Labels in header files
        label_source=LabelSource.CHALLENGE_LABELS,
        label_column=None,
        physionet_url="https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/",
        extra_config={
            "variable_length": True,
            "labels_in_header": True,
        }
    ),
    
    # -------------------------------------------------------------------------
    # Georgia (PhysioNet Challenge 2020)
    # -------------------------------------------------------------------------
    "georgia": DatasetConfig(
        name="georgia",
        description="Georgia 12-lead ECG Challenge Database",
        sampling_rate=500,
        n_leads=12,
        lead_names=STANDARD_12_LEADS,
        duration_seconds=None,  # Variable length
        file_format=FileFormat.WFDB,
        data_subdir="",
        metadata_file=None,
        label_source=LabelSource.CHALLENGE_LABELS,
        label_column=None,
        physionet_url="https://physionet.org/files/challenge-2020/1.0.2/training/georgia/",
        extra_config={
            "variable_length": True,
            "labels_in_header": True,
        }
    ),
}


# =============================================================================
# Registry Class
# =============================================================================

class DatasetRegistry:
    """
    Central registry for managing ECG dataset configurations.
    
    Usage:
        registry = DatasetRegistry()
        config = registry.get_config("ptb-xl")
        print(config.sampling_rate)  # 500
    """
    
    def __init__(self):
        self._configs = DATASET_CONFIGS.copy()
    
    @property
    def available_datasets(self) -> List[str]:
        """List all available dataset names."""
        return list(self._configs.keys())
    
    def get_config(self, dataset_name: str) -> DatasetConfig:
        """
        Get configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'ptb-xl', 'chapman')
            
        Returns:
            DatasetConfig object
            
        Raises:
            ValueError: If dataset is not found
        """
        name_lower = dataset_name.lower().strip()
        
        if name_lower not in self._configs:
            available = ", ".join(self.available_datasets)
            raise ValueError(
                f"Unknown dataset: '{dataset_name}'. "
                f"Available datasets: {available}"
            )
        
        return self._configs[name_lower]
    
    def register_dataset(self, config: DatasetConfig) -> None:
        """
        Register a new dataset configuration.
        
        Args:
            config: DatasetConfig object to register
        """
        name_lower = config.name.lower().strip()
        if name_lower in self._configs:
            logger.warning(f"Overwriting existing dataset config: {name_lower}")
        self._configs[name_lower] = config
        logger.info(f"Registered dataset: {name_lower}")
    
    def is_supported(self, dataset_name: str) -> bool:
        """Check if a dataset is supported."""
        return dataset_name.lower().strip() in self._configs
    
    def get_physionet_url(self, dataset_name: str) -> Optional[str]:
        """Get PhysioNet download URL for a dataset."""
        config = self.get_config(dataset_name)
        return config.physionet_url
    
    def get_sampling_rate(self, dataset_name: str) -> int:
        """Get sampling rate for a dataset."""
        config = self.get_config(dataset_name)
        return config.sampling_rate
    
    def get_lead_names(self, dataset_name: str) -> List[str]:
        """Get lead names for a dataset."""
        config = self.get_config(dataset_name)
        return config.lead_names
    
    def summary(self) -> str:
        """Get a summary of all registered datasets."""
        lines = ["Registered ECG Datasets:", "=" * 50]
        for name, config in self._configs.items():
            lines.append(
                f"  {name}: {config.n_leads}-lead, {config.sampling_rate}Hz, "
                f"{config.file_format.value} format"
            )
        return "\n".join(lines)


# Global registry instance
_registry = DatasetRegistry()


def get_registry() -> DatasetRegistry:
    """Get the global dataset registry instance."""
    return _registry


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Convenience function to get dataset config."""
    return _registry.get_config(dataset_name)


# =============================================================================
# Auto-detection utilities
# =============================================================================

def detect_dataset_from_path(data_path: Path) -> Optional[str]:
    """
    Attempt to auto-detect dataset type from directory structure.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        Dataset name if detected, None otherwise
    """
    data_path = Path(data_path)
    
    # Check directory name
    dir_name = data_path.name.lower()
    for dataset_name in _registry.available_datasets:
        if dataset_name in dir_name:
            logger.info(f"Auto-detected dataset from directory name: {dataset_name}")
            return dataset_name
    
    # Check for specific files
    if (data_path / "ptbxl_database.csv").exists():
        logger.info("Auto-detected PTB-XL from metadata file")
        return "ptb-xl"
    
    if (data_path / "scp_statements.csv").exists():
        logger.info("Auto-detected PTB-XL from SCP statements file")
        return "ptb-xl"
    
    # Check for RECORDS file (common in PhysioNet)
    if (data_path / "RECORDS").exists():
        # Could be any PhysioNet dataset, need more info
        logger.warning("Found RECORDS file but cannot determine specific dataset")
    
    return None


# =============================================================================
# CLI Support
# =============================================================================

if __name__ == "__main__":
    # Print summary when run directly
    print(get_registry().summary())
    print()
    print("Example usage:")
    print("  from src.core.dataset_registry import get_dataset_config")
    print("  config = get_dataset_config('ptb-xl')")
    print("  print(config.sampling_rate)  # 500")
