"""
Kepler-ECG Core Module

Central configuration and utilities for multi-dataset support.
"""

from .dataset_registry import (
    DatasetConfig,
    DatasetRegistry,
    FileFormat,
    LabelSource,
    get_registry,
    get_dataset_config,
    detect_dataset_from_path,
    STANDARD_12_LEADS,
)

from .label_schema import (
    Superclass,
    Subclass,
    DiagnosticLabel,
    LabelMapper,
    parse_header_labels,
    create_label_dataframe_columns,
)

__all__ = [
    # Dataset Registry
    "DatasetConfig",
    "DatasetRegistry", 
    "FileFormat",
    "LabelSource",
    "get_registry",
    "get_dataset_config",
    "detect_dataset_from_path",
    "STANDARD_12_LEADS",
    # Label Schema
    "Superclass",
    "Subclass",
    "DiagnosticLabel",
    "LabelMapper",
    "parse_header_labels",
    "create_label_dataframe_columns",
]
