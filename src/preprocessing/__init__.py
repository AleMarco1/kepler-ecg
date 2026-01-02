"""This module provides ECG signal preprocessing components:
- Filtering (baseline removal, noise filtering)
- Quality assessment
- Beat segmentation
- HRV preprocessing
- Complete pipeline integration
"""

from preprocessing.filters import (
    BaselineRemover,
    NoiseFilter,
    ECGFilter,
    FilterConfig,
    remove_baseline,
    filter_noise,
    filter_ecg,
)

from preprocessing.quality import (
    QualityAssessor,
    QualityConfig,
    QualityMetrics,
    QualityLevel,
    assess_quality,
)

from preprocessing.segmentation import (
    BeatSegmenter,
    PanTompkinsDetector,
    SegmentationConfig,
    SegmentationResult,
    detect_r_peaks,
    segment_beats,
)

from preprocessing.hrv_preprocessing import (
    HRVPreprocessor,
    HRVConfig,
    HRVData,
    EctopicMethod,
    preprocess_hrv,
    detect_ectopic_beats,
)

from preprocessing.pipeline import (
    PreprocessingPipeline,
    PipelineConfig,
    ProcessedECG,
    create_pipeline,
    process_ecg,
)

__all__ = [
    # Filters
    "BaselineRemover",
    "NoiseFilter", 
    "ECGFilter",
    "FilterConfig",
    "remove_baseline",
    "filter_noise",
    "filter_ecg",
    # Quality
    "QualityAssessor",
    "QualityConfig",
    "QualityMetrics",
    "QualityLevel",
    "assess_quality",
    # Segmentation
    "BeatSegmenter",
    "PanTompkinsDetector",
    "SegmentationConfig",
    "SegmentationResult",
    "detect_r_peaks",
    "segment_beats",
    # HRV
    "HRVPreprocessor",
    "HRVConfig",
    "HRVData",
    "EctopicMethod",
    "preprocess_hrv",
    "detect_ectopic_beats",
    # Pipeline
    "PreprocessingPipeline",
    "PipelineConfig",
    "ProcessedECG",
    "create_pipeline",
    "process_ecg",
]