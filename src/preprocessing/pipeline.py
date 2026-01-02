"""
Kepler-ECG: Complete Preprocessing Pipeline

This module integrates all preprocessing components into a unified pipeline:
- Filtering (baseline removal, noise filtering)
- Quality assessment
- Beat segmentation (R-peak detection)
- HRV preprocessing

Features:
- Caching of processed results
- Parallel processing support
- Configurable pipeline stages
- Comprehensive logging

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

# System and Utility imports
import hashlib      # Generate unique hashes for data integrity and smart caching
import json         # Handle configuration files and human-readable metadata
import logging      # Professional tracking of events, errors, and training progress
import os           # Basic OS interactions (environment variables, process IDs)
import pickle       # Serialize and save complex Python objects (like preprocessed datasets)
import time         # Benchmark execution time and handle delays
from dataclasses import dataclass, field  # Structured data containers for configs and results
from pathlib import Path                  # Modern, cross-platform filesystem path management

# Type Hinting for robust and self-documenting code
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Concurrency for high-performance data processing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Numerical computing
import numpy as np  # Core library for tensor manipulation and signal data

# Custom Preprocessing Modules (Project-specific logic)
from preprocessing.filters import ECGFilter, FilterConfig              # Signal cleaning logic
from preprocessing.quality import QualityAssessor, QualityConfig, QualityMetrics  # Signal signal-to-noise check
from preprocessing.segmentation import BeatSegmenter, SegmentationConfig, SegmentationResult # ECG beat extraction
from preprocessing.hrv_preprocessing import HRVPreprocessor, HRVConfig, HRVData  # Heart Rate Variability analysis


# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the preprocessing pipeline."""
    
    # Component configs
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    quality_config: QualityConfig = field(default_factory=QualityConfig)
    segmentation_config: SegmentationConfig = field(default_factory=SegmentationConfig)
    hrv_config: HRVConfig = field(default_factory=HRVConfig)
    
    # Pipeline options
    apply_filtering: bool = True
    assess_quality: bool = True
    segment_beats: bool = True
    preprocess_hrv: bool = True
    
    # Quality gate
    skip_low_quality: bool = False  # Skip processing if quality too low
    min_quality_score: float = 0.3
    
    # Caching
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    
    # Parallel processing
    n_jobs: int = 1  # Number of parallel jobs (-1 for all CPUs)
    use_threading: bool = False  # Use threads instead of processes
    
    # Output options
    save_filtered_signal: bool = True
    save_beats: bool = True
    save_beat_template: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for hashing/serialization."""
        return {
            'apply_filtering': self.apply_filtering,
            'assess_quality': self.assess_quality,
            'segment_beats': self.segment_beats,
            'preprocess_hrv': self.preprocess_hrv,
            'skip_low_quality': self.skip_low_quality,
            'min_quality_score': self.min_quality_score,
        }


@dataclass
class ProcessedECG:
    """
    Complete results from ECG preprocessing pipeline.
    
    Attributes
    ----------
    ecg_id : str
        Unique identifier for the ECG record.
    success : bool
        Whether preprocessing completed successfully.
    error_message : Optional[str]
        Error message if preprocessing failed.
    processing_time_sec : float
        Time taken to process in seconds.
    
    # Filtered signal
    signal_filtered : Optional[np.ndarray]
        Filtered ECG signal (if save_filtered_signal=True).
    sampling_rate : int
        Sampling rate in Hz.
    
    # Quality metrics
    quality : Optional[QualityMetrics]
        Quality assessment results.
    
    # Segmentation results
    segmentation : Optional[SegmentationResult]
        Beat segmentation results.
    
    # HRV data
    hrv : Optional[HRVData]
        HRV preprocessing results.
    
    # Metadata
    n_leads : int
        Number of leads in the signal.
    duration_sec : float
        Signal duration in seconds.
    """
    ecg_id: str
    success: bool
    error_message: Optional[str] = None
    processing_time_sec: float = 0.0
    
    # Signal
    signal_filtered: Optional[np.ndarray] = None
    sampling_rate: int = 500
    
    # Quality
    quality: Optional[QualityMetrics] = None
    
    # Segmentation
    segmentation: Optional[SegmentationResult] = None
    
    # HRV
    hrv: Optional[HRVData] = None
    
    # Metadata
    n_leads: int = 1
    duration_sec: float = 0.0
    
    @property
    def is_usable(self) -> bool:
        """Check if the processed ECG is usable for analysis."""
        if not self.success:
            return False
        if self.quality is not None and not self.quality.is_usable:
            return False
        return True
    
    @property
    def heart_rate_bpm(self) -> Optional[float]:
        """Get heart rate if available."""
        if self.segmentation is not None:
            return self.segmentation.heart_rate_bpm
        return None
    
    @property
    def n_beats(self) -> int:
        """Get number of detected beats."""
        if self.segmentation is not None:
            return self.segmentation.n_beats
        return 0
    
    def to_dict(self, include_arrays: bool = False) -> Dict:
        """
        Convert to dictionary for serialization.
        
        Parameters
        ----------
        include_arrays : bool
            Whether to include numpy arrays (can be large).
        """
        result = {
            'ecg_id': self.ecg_id,
            'success': self.success,
            'error_message': self.error_message,
            'processing_time_sec': self.processing_time_sec,
            'sampling_rate': self.sampling_rate,
            'n_leads': self.n_leads,
            'duration_sec': self.duration_sec,
            'is_usable': self.is_usable,
            'heart_rate_bpm': self.heart_rate_bpm,
            'n_beats': self.n_beats,
        }
        
        if self.quality is not None:
            result['quality'] = self.quality.to_dict()
        
        if self.segmentation is not None:
            result['segmentation'] = self.segmentation.to_dict()
        
        if self.hrv is not None:
            result['hrv'] = self.hrv.to_dict()
        
        if include_arrays and self.signal_filtered is not None:
            result['signal_filtered'] = self.signal_filtered.tolist()
        
        return result


class PreprocessingPipeline:
    """
    Complete ECG preprocessing pipeline.
    
    This class orchestrates all preprocessing steps:
    1. Filtering (baseline removal + noise filtering)
    2. Quality assessment
    3. Beat segmentation (R-peak detection)
    4. HRV preprocessing (ectopic removal, interpolation)
    
    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration. Uses defaults if not provided.
    
    Example
    -------
    >>> pipeline = PreprocessingPipeline()
    >>> result = pipeline.process(ecg_signal, fs=500, ecg_id="record_001")
    >>> if result.is_usable:
    ...     print(f"Heart rate: {result.heart_rate_bpm:.1f} bpm")
    ...     print(f"Quality: {result.quality.quality_level.value}")
    
    Notes
    -----
    - The pipeline can be configured to skip stages or stop on low quality
    - Results are cached to avoid reprocessing
    - Parallel processing is supported for batch processing
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self._init_components()
        
        # Setup cache
        self._cache: Dict[str, ProcessedECG] = {}
        if self.config.cache_dir:
            self._cache_path = Path(self.config.cache_dir)
            self._cache_path.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_path = None
        
        # Statistics
        self._stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'cached': 0,
            'total_time': 0.0,
        }
    
    def _init_components(self):
        """Initialize preprocessing components."""
        self.ecg_filter = ECGFilter(
            baseline_cutoff=self.config.filter_config.baseline_cutoff,
            lowpass_cutoff=self.config.filter_config.lowpass_cutoff,
        )
        self.quality_assessor = QualityAssessor(self.config.quality_config)
        self.beat_segmenter = BeatSegmenter(self.config.segmentation_config)
        self.hrv_preprocessor = HRVPreprocessor(self.config.hrv_config)
    
    def process(
        self,
        signal: np.ndarray,
        fs: int,
        ecg_id: str = "unknown",
        lead: int = 0,
        use_cache: bool = True
    ) -> ProcessedECG:
        """
        Process a single ECG signal through the pipeline.
        
        Parameters
        ----------
        signal : np.ndarray
            ECG signal. Can be 1D or 2D (n_samples, n_leads).
        fs : int
            Sampling frequency in Hz.
        ecg_id : str
            Unique identifier for caching.
        lead : int
            Lead to use for R-peak detection (for multi-lead).
        use_cache : bool
            Whether to use/update cache.
            
        Returns
        -------
        ProcessedECG
            Complete preprocessing results.
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(signal, fs, ecg_id)
        if use_cache and self.config.enable_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                self._stats['cached'] += 1
                return cached
        
        try:
            result = self._process_signal(signal, fs, ecg_id, lead)
            result.processing_time_sec = time.time() - start_time
            
            # Update cache
            if use_cache and self.config.enable_cache:
                self._save_to_cache(cache_key, result)
            
            # Update stats
            self._stats['processed'] += 1
            self._stats['total_time'] += result.processing_time_sec
            if result.success:
                self._stats['successful'] += 1
            else:
                self._stats['failed'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {ecg_id}: {str(e)}")
            self._stats['processed'] += 1
            self._stats['failed'] += 1
            
            return ProcessedECG(
                ecg_id=ecg_id,
                success=False,
                error_message=str(e),
                processing_time_sec=time.time() - start_time,
                sampling_rate=fs,
            )
    
    def _process_signal(
        self,
        signal: np.ndarray,
        fs: int,
        ecg_id: str,
        lead: int
    ) -> ProcessedECG:
        """Internal processing logic."""
        signal = np.asarray(signal)
        
        # Handle dimensions
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
            n_leads = 1
        else:
            n_leads = signal.shape[1]
        
        duration_sec = len(signal) / fs
        
        # Step 1: Filtering
        if self.config.apply_filtering:
            signal_filtered = self.ecg_filter(signal, fs, axis=0)
        else:
            signal_filtered = signal.copy()
        
        # Step 2: Quality Assessment
        quality = None
        if self.config.assess_quality:
            quality = self.quality_assessor(signal_filtered, fs, axis=0)
            
            # Quality gate
            if self.config.skip_low_quality:
                if quality.quality_score < self.config.min_quality_score:
                    return ProcessedECG(
                        ecg_id=ecg_id,
                        success=False,
                        error_message=f"Quality too low: {quality.quality_score:.2f}",
                        signal_filtered=signal_filtered if self.config.save_filtered_signal else None,
                        sampling_rate=fs,
                        quality=quality,
                        n_leads=n_leads,
                        duration_sec=duration_sec,
                    )
        
        # Step 3: Beat Segmentation
        segmentation = None
        if self.config.segment_beats:
            segmentation = self.beat_segmenter(signal_filtered, fs, lead=lead)
            
            # Clear beats if not saving (to reduce memory)
            if not self.config.save_beats:
                segmentation = SegmentationResult(
                    r_peaks=segmentation.r_peaks,
                    r_peak_times=segmentation.r_peak_times,
                    rr_intervals=segmentation.rr_intervals,
                    beats=np.array([]),
                    beat_template=segmentation.beat_template if self.config.save_beat_template else np.array([]),
                    heart_rate_bpm=segmentation.heart_rate_bpm,
                    heart_rate_std=segmentation.heart_rate_std,
                    n_beats=segmentation.n_beats,
                    detection_confidence=segmentation.detection_confidence,
                )
        
        # Step 4: HRV Preprocessing
        hrv = None
        if self.config.preprocess_hrv and segmentation is not None:
            if len(segmentation.rr_intervals) >= self.config.hrv_config.min_valid_rr:
                hrv = self.hrv_preprocessor(segmentation.rr_intervals)
        
        return ProcessedECG(
            ecg_id=ecg_id,
            success=True,
            signal_filtered=signal_filtered if self.config.save_filtered_signal else None,
            sampling_rate=fs,
            quality=quality,
            segmentation=segmentation,
            hrv=hrv,
            n_leads=n_leads,
            duration_sec=duration_sec,
        )
    
    def process_batch(
        self,
        signals: List[Tuple[np.ndarray, int, str]],
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ProcessedECG]:
        """
        Process multiple ECG signals.
        
        Parameters
        ----------
        signals : List[Tuple[np.ndarray, int, str]]
            List of (signal, fs, ecg_id) tuples.
        show_progress : bool
            Whether to show progress.
        progress_callback : Callable
            Optional callback for progress updates: callback(current, total).
            
        Returns
        -------
        List[ProcessedECG]
            List of processing results.
        """
        n_total = len(signals)
        results = []
        
        if self.config.n_jobs == 1:
            # Sequential processing
            for i, (signal, fs, ecg_id) in enumerate(signals):
                result = self.process(signal, fs, ecg_id)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, n_total)
                elif show_progress and (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{n_total} signals")
        
        else:
            # Parallel processing
            n_workers = self.config.n_jobs if self.config.n_jobs > 0 else os.cpu_count()
            
            Executor = ThreadPoolExecutor if self.config.use_threading else ProcessPoolExecutor
            
            with Executor(max_workers=n_workers) as executor:
                # Submit all jobs
                futures = {
                    executor.submit(self._process_single, signal, fs, ecg_id): ecg_id
                    for signal, fs, ecg_id in signals
                }
                
                # Collect results
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, n_total)
                    elif show_progress and completed % 100 == 0:
                        logger.info(f"Processed {completed}/{n_total} signals")
        
        return results
    
    def _process_single(
        self,
        signal: np.ndarray,
        fs: int,
        ecg_id: str
    ) -> ProcessedECG:
        """Process single signal (for parallel execution)."""
        return self.process(signal, fs, ecg_id, use_cache=False)
    
    def _get_cache_key(
        self,
        signal: np.ndarray,
        fs: int,
        ecg_id: str
    ) -> str:
        """Generate cache key from signal and config."""
        # Use ecg_id as primary key, with config hash
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{ecg_id}_{fs}_{config_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[ProcessedECG]:
        """Try to get result from cache."""
        # Memory cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Disk cache
        if self._cache_path is not None:
            cache_file = self._cache_path / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    self._cache[cache_key] = result
                    return result
                except Exception as e:
                    logger.warning(f"Failed to load cache {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, result: ProcessedECG):
        """Save result to cache."""
        # Memory cache
        self._cache[cache_key] = result
        
        # Disk cache
        if self._cache_path is not None:
            cache_file = self._cache_path / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def clear_cache(self):
        """Clear all cached results."""
        self._cache.clear()
        
        if self._cache_path is not None:
            for cache_file in self._cache_path.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self._stats.copy()
        if stats['processed'] > 0:
            stats['success_rate'] = stats['successful'] / stats['processed']
            stats['avg_time_sec'] = stats['total_time'] / stats['processed']
        else:
            stats['success_rate'] = 0.0
            stats['avg_time_sec'] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset processing statistics."""
        self._stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'cached': 0,
            'total_time': 0.0,
        }
    
    def __repr__(self) -> str:
        stages = []
        if self.config.apply_filtering:
            stages.append("filter")
        if self.config.assess_quality:
            stages.append("quality")
        if self.config.segment_beats:
            stages.append("segment")
        if self.config.preprocess_hrv:
            stages.append("hrv")
        
        return f"PreprocessingPipeline(stages=[{', '.join(stages)}])"


def create_pipeline(
    baseline_cutoff: float = 0.5,
    lowpass_cutoff: float = 40.0,
    notch_freq: float = 50.0,
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    n_jobs: int = 1
) -> PreprocessingPipeline:
    """
    Convenience function to create a preprocessing pipeline.
    
    Parameters
    ----------
    baseline_cutoff : float
        High-pass cutoff for baseline removal (Hz).
    lowpass_cutoff : float
        Low-pass cutoff for noise removal (Hz).
    notch_freq : float
        Powerline frequency to remove (Hz).
    enable_cache : bool
        Whether to enable result caching.
    cache_dir : str, optional
        Directory for disk cache.
    n_jobs : int
        Number of parallel jobs.
        
    Returns
    -------
    PreprocessingPipeline
        Configured preprocessing pipeline.
    """
    filter_config = FilterConfig(
        baseline_cutoff=baseline_cutoff,
        lowpass_cutoff=lowpass_cutoff,
        notch_freq=notch_freq,
    )
    
    config = PipelineConfig(
        filter_config=filter_config,
        enable_cache=enable_cache,
        cache_dir=cache_dir,
        n_jobs=n_jobs,
    )
    
    return PreprocessingPipeline(config)


def process_ecg(
    signal: np.ndarray,
    fs: int,
    ecg_id: str = "unknown"
) -> ProcessedECG:
    """
    Convenience function to process a single ECG.
    
    Parameters
    ----------
    signal : np.ndarray
        ECG signal.
    fs : int
        Sampling frequency in Hz.
    ecg_id : str
        Identifier for the ECG.
        
    Returns
    -------
    ProcessedECG
        Processing results.
    """
    pipeline = PreprocessingPipeline()
    return pipeline.process(signal, fs, ecg_id, use_cache=False)