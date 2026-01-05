"""Kepler-ECG Features Module

Modulo per l'estrazione di features avanzate da segnali ECG:
- Morfologiche (ampiezze, durate, aree)
- Spettrali (HRV frequency domain)
- Wavelet (multi-scala)
- Pipeline integrata
- CompressibilityCalculator: calcolo metriche di compressibilità

Note: Per il mapping delle diagnosi, usare src.core.label_schema.LabelMapper
che supporta sia SCP codes (PTB-XL) che SNOMED codes (Chapman, CPSC, Georgia).
"""

from .morphological import MorphologicalExtractor, IntervalCalculator
from .spectral import SpectralAnalyzer, generate_synthetic_rr_series
from .wavelet import WaveletExtractor, generate_synthetic_ecg_beat
from .compressor import CompressibilityCalculator, CompressibilityConfig
from .feature_pipeline import (
    FeaturePipeline, FeatureConfig, ProcessedECG, FeatureVector,
    create_processed_ecg_from_dict, extract_features_from_raw
)

__all__ = [
    # Estrattori
    'MorphologicalExtractor',
    'IntervalCalculator',
    'SpectralAnalyzer',
    'WaveletExtractor',
    # Pipeline
    'FeaturePipeline',
    'FeatureConfig',
    'ProcessedECG',
    'FeatureVector',
    # Helper
    'generate_synthetic_rr_series',
    'generate_synthetic_ecg_beat',
    'create_processed_ecg_from_dict',
    'extract_features_from_raw',
    # Compressor
    'CompressibilityCalculator',
    'CompressibilityConfig',
]
