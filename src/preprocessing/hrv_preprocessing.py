"""
Kepler-ECG: HRV Preprocessing

This module implements preprocessing specific for Heart Rate Variability analysis:
- Ectopic beat detection and removal
- RR interval interpolation for frequency analysis
- Artifact correction

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025

References:
    Task Force of ESC and NASPE (1996). Heart rate variability: standards of 
    measurement, physiological interpretation and clinical use.
    
    Malik, M. (1995). Geometrical methods for heart rate variability assessment.
"""

from dataclasses import dataclass, field  # Automate class creation and manage default values for complex types
from enum import Enum                     # Define a set of named constants (e.g., for Signal Classes or Leads)
from typing import Dict, List, Optional, Tuple, Union  # Type safety for complex data mapping and function signatures
import numpy as np                        # High-performance array processing and mathematical operations
from scipy.interpolate import interp1d    # 1D interpolation tool to resample signals to a uniform frequency


class EctopicMethod(Enum):
    """Methods for ectopic beat detection."""
    MALIK = "malik"  # Percentage change from previous
    KAMATH = "kamath"  # Based on median filter
    KARLSSON = "karlsson"  # Adaptive threshold
    ACAR = "acar"  # Quotient filter


@dataclass
class HRVConfig:
    """Configuration for HRV preprocessing."""
    
    # Ectopic detection
    ectopic_method: EctopicMethod = EctopicMethod.MALIK
    ectopic_threshold: float = 0.2  # 20% change threshold for Malik
    
    # Interpolation for frequency analysis
    interpolation_fs: float = 4.0  # Hz - standard for HRV
    interpolation_method: str = 'cubic'  # 'linear', 'cubic', 'quadratic'
    
    # RR interval limits (physiological)
    min_rr_ms: float = 300.0  # 200 bpm max
    max_rr_ms: float = 2000.0  # 30 bpm min
    
    # Quality thresholds
    max_ectopic_ratio: float = 0.2  # Max 20% ectopic beats
    min_valid_rr: int = 10  # Minimum valid RR intervals required
    
    # Correction method
    correction_method: str = 'interpolation'  # 'interpolation', 'removal', 'median'


@dataclass
class HRVData:
    """
    Container for preprocessed HRV data.
    
    Attributes
    ----------
    rr_original : np.ndarray
        Original RR intervals in milliseconds.
    rr_clean : np.ndarray
        RR intervals after ectopic removal/correction.
    rr_interpolated : np.ndarray
        Evenly sampled RR intervals for frequency analysis.
    time_original : np.ndarray
        Cumulative time of original RR intervals.
    time_interpolated : np.ndarray
        Time axis for interpolated signal.
    ectopic_mask : np.ndarray
        Boolean mask indicating ectopic beats (True = ectopic).
    ectopic_indices : np.ndarray
        Indices of detected ectopic beats.
    ectopic_ratio : float
        Fraction of beats that are ectopic.
    interpolation_fs : float
        Sampling frequency of interpolated signal.
    is_valid : bool
        Whether data is valid for HRV analysis.
    quality_issues : List[str]
        List of quality issues found.
    """
    rr_original: np.ndarray
    rr_clean: np.ndarray
    rr_interpolated: np.ndarray
    time_original: np.ndarray
    time_interpolated: np.ndarray
    ectopic_mask: np.ndarray
    ectopic_indices: np.ndarray
    ectopic_ratio: float
    interpolation_fs: float
    is_valid: bool
    quality_issues: List[str] = field(default_factory=list)
    
    @property
    def n_beats(self) -> int:
        """Number of original beats."""
        return len(self.rr_original)
    
    @property
    def n_ectopic(self) -> int:
        """Number of ectopic beats."""
        return int(np.sum(self.ectopic_mask))
    
    @property
    def duration_sec(self) -> float:
        """Total duration in seconds."""
        return np.sum(self.rr_original) / 1000.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'rr_original': self.rr_original.tolist(),
            'rr_clean': self.rr_clean.tolist(),
            'ectopic_indices': self.ectopic_indices.tolist(),
            'ectopic_ratio': self.ectopic_ratio,
            'interpolation_fs': self.interpolation_fs,
            'is_valid': self.is_valid,
            'quality_issues': self.quality_issues,
            'n_beats': self.n_beats,
            'duration_sec': self.duration_sec,
        }


class HRVPreprocessor:
    """
    Preprocessor for Heart Rate Variability analysis.
    
    This class prepares RR interval data for HRV analysis by:
    1. Detecting and handling ectopic (abnormal) beats
    2. Correcting artifacts in the RR series
    3. Interpolating to uniform sampling for frequency analysis
    
    Parameters
    ----------
    config : HRVConfig, optional
        Configuration parameters. Uses defaults if not provided.
    
    Example
    -------
    >>> preprocessor = HRVPreprocessor()
    >>> hrv_data = preprocessor(rr_intervals)
    >>> if hrv_data.is_valid:
    ...     # Use hrv_data.rr_clean for time-domain analysis
    ...     # Use hrv_data.rr_interpolated for frequency-domain analysis
    
    Notes
    -----
    - Ectopic beats include premature ventricular contractions (PVCs),
      premature atrial contractions (PACs), and missed beats
    - The Malik criterion detects beats where |RR_n - RR_{n-1}| / RR_{n-1} > threshold
    - Interpolation at 4 Hz is standard for HRV frequency analysis
    """
    
    def __init__(self, config: Optional[HRVConfig] = None):
        self.config = config or HRVConfig()
    
    def __call__(self, rr_intervals: np.ndarray) -> HRVData:
        """
        Preprocess RR intervals for HRV analysis.
        
        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in milliseconds.
            
        Returns
        -------
        HRVData
            Preprocessed HRV data ready for analysis.
        """
        rr = np.asarray(rr_intervals, dtype=float).flatten()
        quality_issues = []
        
        # Check minimum data
        if len(rr) < self.config.min_valid_rr:
            return self._create_invalid_result(
                rr, f"Insufficient data: {len(rr)} < {self.config.min_valid_rr} beats"
            )
        
        # Step 1: Filter physiologically impossible values
        rr_filtered, physio_mask = self._filter_physiological(rr)
        if np.sum(~physio_mask) > 0:
            quality_issues.append(
                f"Removed {np.sum(~physio_mask)} physiologically impossible RR intervals"
            )
        
        # Check if we have enough data after physiological filtering
        if len(rr_filtered) < self.config.min_valid_rr:
            return self._create_invalid_result(
                rr, f"Insufficient data after physiological filtering: {len(rr_filtered)} < {self.config.min_valid_rr} beats"
            )
        
        # Step 2: Detect ectopic beats
        ectopic_mask = self._detect_ectopic(rr_filtered)
        ectopic_ratio = float(np.mean(ectopic_mask)) if len(ectopic_mask) > 0 else 0.0
        
        if ectopic_ratio > self.config.max_ectopic_ratio:
            quality_issues.append(
                f"High ectopic ratio: {ectopic_ratio*100:.1f}% > {self.config.max_ectopic_ratio*100:.0f}%"
            )
        
        # Step 3: Correct ectopic beats
        rr_clean = self._correct_ectopic(rr_filtered, ectopic_mask)
        
        # Step 4: Compute time axis based on correction method
        if self.config.correction_method == 'removal' and np.any(ectopic_mask):
            # For removal method, recompute time from clean RR
            time_original = np.cumsum(rr_clean) / 1000.0
            time_original = np.insert(time_original, 0, 0)[:-1]
        else:
            time_original = np.cumsum(rr_filtered) / 1000.0
            time_original = np.insert(time_original, 0, 0)[:-1]
        
        # Step 5: Interpolate for frequency analysis
        rr_interpolated, time_interpolated = self._interpolate(
            rr_clean, time_original
        )
        
        # Step 6: Check validity
        is_valid = (
            len(rr_clean) >= self.config.min_valid_rr and
            ectopic_ratio <= self.config.max_ectopic_ratio
        )
        
        return HRVData(
            rr_original=rr_filtered,
            rr_clean=rr_clean,
            rr_interpolated=rr_interpolated,
            time_original=time_original,
            time_interpolated=time_interpolated,
            ectopic_mask=ectopic_mask,
            ectopic_indices=np.where(ectopic_mask)[0],
            ectopic_ratio=ectopic_ratio,
            interpolation_fs=self.config.interpolation_fs,
            is_valid=is_valid,
            quality_issues=quality_issues
        )
    
    def _filter_physiological(
        self, 
        rr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out physiologically impossible RR intervals.
        
        Returns filtered RR and mask of valid intervals.
        """
        valid_mask = (
            (rr >= self.config.min_rr_ms) & 
            (rr <= self.config.max_rr_ms) &
            np.isfinite(rr)
        )
        
        return rr[valid_mask], valid_mask
    
    def _detect_ectopic(self, rr: np.ndarray) -> np.ndarray:
        """
        Detect ectopic beats using configured method.
        """
        if len(rr) < 2:
            return np.zeros(len(rr), dtype=bool)
        
        method = self.config.ectopic_method
        
        if method == EctopicMethod.MALIK:
            return self._detect_ectopic_malik(rr)
        elif method == EctopicMethod.KAMATH:
            return self._detect_ectopic_kamath(rr)
        elif method == EctopicMethod.KARLSSON:
            return self._detect_ectopic_karlsson(rr)
        elif method == EctopicMethod.ACAR:
            return self._detect_ectopic_acar(rr)
        else:
            return self._detect_ectopic_malik(rr)
    
    def _detect_ectopic_malik(self, rr: np.ndarray) -> np.ndarray:
        """
        Malik criterion for ectopic detection.
        
        A beat is ectopic if |RR_n - RR_{n-1}| / RR_{n-1} > threshold
        
        This detects sudden changes in RR interval that indicate
        premature or delayed beats.
        """
        ectopic = np.zeros(len(rr), dtype=bool)
        
        if len(rr) < 2:
            return ectopic
        
        # Compute relative change
        rr_diff = np.abs(np.diff(rr))
        rr_prev = rr[:-1]
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_change = rr_diff / rr_prev
            relative_change = np.nan_to_num(relative_change, nan=0, posinf=1, neginf=1)
        
        # Mark beats exceeding threshold
        # The beat AFTER the large change is typically the ectopic
        ectopic_after = relative_change > self.config.ectopic_threshold
        ectopic[1:] = ectopic_after
        
        # Also check the beat before (the ectopic beat itself causes change)
        # A PVC has short RR before AND long RR after (compensatory pause)
        for i in range(1, len(rr) - 1):
            if relative_change[i-1] > self.config.ectopic_threshold and \
               relative_change[i] > self.config.ectopic_threshold:
                ectopic[i] = True
        
        return ectopic
    
    def _detect_ectopic_kamath(self, rr: np.ndarray) -> np.ndarray:
        """
        Kamath criterion using median filter comparison.
        
        Compares each RR to local median.
        """
        ectopic = np.zeros(len(rr), dtype=bool)
        
        if len(rr) < 5:
            return ectopic
        
        # Compute running median (window=5)
        window = 5
        for i in range(len(rr)):
            start = max(0, i - window // 2)
            end = min(len(rr), i + window // 2 + 1)
            local_median = np.median(rr[start:end])
            
            # Check if current RR deviates significantly from median
            if local_median > 0:
                deviation = abs(rr[i] - local_median) / local_median
                if deviation > self.config.ectopic_threshold:
                    ectopic[i] = True
        
        return ectopic
    
    def _detect_ectopic_karlsson(self, rr: np.ndarray) -> np.ndarray:
        """
        Karlsson adaptive threshold method.
        
        Uses adaptive threshold based on local variability.
        """
        ectopic = np.zeros(len(rr), dtype=bool)
        
        if len(rr) < 10:
            return self._detect_ectopic_malik(rr)
        
        # Compute local statistics
        window = 10
        
        for i in range(len(rr)):
            start = max(0, i - window)
            end = min(len(rr), i + window)
            
            local_vals = rr[start:end]
            local_mean = np.mean(local_vals)
            local_std = np.std(local_vals)
            
            # Adaptive threshold: mean ± 2*std
            if local_std > 0:
                threshold = 2.5 * local_std
            else:
                threshold = 0.2 * local_mean
            
            if abs(rr[i] - local_mean) > threshold:
                ectopic[i] = True
        
        return ectopic
    
    def _detect_ectopic_acar(self, rr: np.ndarray) -> np.ndarray:
        """
        Acar quotient filter method.
        
        Checks ratio of consecutive RR intervals.
        """
        ectopic = np.zeros(len(rr), dtype=bool)
        
        if len(rr) < 3:
            return ectopic
        
        # Quotient of consecutive RR intervals
        for i in range(1, len(rr)):
            quotient = rr[i] / rr[i-1] if rr[i-1] > 0 else 1.0
            
            # Ectopic if quotient is too far from 1
            if quotient < (1 - self.config.ectopic_threshold) or \
               quotient > (1 + self.config.ectopic_threshold):
                ectopic[i] = True
        
        return ectopic
    
    def _correct_ectopic(
        self, 
        rr: np.ndarray, 
        ectopic_mask: np.ndarray
    ) -> np.ndarray:
        """
        Correct ectopic beats using configured method.
        """
        if not np.any(ectopic_mask):
            return rr.copy()
        
        method = self.config.correction_method
        
        if method == 'removal':
            # Simply remove ectopic beats
            # Note: this changes array length, affecting time alignment
            return rr[~ectopic_mask].copy()
        
        elif method == 'median':
            # Replace with local median
            rr_corrected = rr.copy()
            window = 5
            
            # Get global median of non-ectopic values (with fallback)
            non_ectopic_rr = rr[~ectopic_mask]
            global_median = np.median(non_ectopic_rr) if len(non_ectopic_rr) > 0 else np.median(rr) if len(rr) > 0 else 0.0
            
            for i in np.where(ectopic_mask)[0]:
                start = max(0, i - window)
                end = min(len(rr), i + window + 1)
                
                # Get non-ectopic values in window
                local_vals = rr[start:end][~ectopic_mask[start:end]]
                
                if len(local_vals) > 0:
                    rr_corrected[i] = np.median(local_vals)
                else:
                    rr_corrected[i] = global_median
            
            return rr_corrected
        
        else:  # interpolation (default)
            return self._interpolate_ectopic(rr, ectopic_mask)
    
    def _interpolate_ectopic(
        self, 
        rr: np.ndarray, 
        ectopic_mask: np.ndarray
    ) -> np.ndarray:
        """
        Replace ectopic beats with interpolated values.
        """
        if not np.any(ectopic_mask) or np.all(ectopic_mask):
            return rr.copy()
        
        rr_corrected = rr.copy()
        
        # Get indices of valid (non-ectopic) beats
        valid_indices = np.where(~ectopic_mask)[0]
        ectopic_indices = np.where(ectopic_mask)[0]
        
        if len(valid_indices) < 2:
            return rr_corrected
        
        # Interpolate ectopic values
        try:
            f = interp1d(
                valid_indices, 
                rr[valid_indices],
                kind='linear',
                fill_value='extrapolate'
            )
            rr_corrected[ectopic_indices] = f(ectopic_indices)
        except Exception:
            # Fallback to median replacement
            non_ectopic_vals = rr[~ectopic_mask]
            median_val = np.median(non_ectopic_vals) if len(non_ectopic_vals) > 0 else np.median(rr) if len(rr) > 0 else 0.0
            rr_corrected[ectopic_indices] = median_val
        
        return rr_corrected
    
    def _interpolate(
        self, 
        rr: np.ndarray, 
        time: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate RR intervals to uniform sampling rate.
        
        This is required for frequency domain HRV analysis.
        """
        if len(rr) < 2 or len(time) < 2:
            return rr, time
        
        # Create uniform time axis
        t_start = time[0]
        t_end = time[-1]
        
        # Ensure we have valid time range
        if t_end <= t_start:
            return rr, time
        
        dt = 1.0 / self.config.interpolation_fs
        time_uniform = np.arange(t_start, t_end, dt)
        
        if len(time_uniform) < 2:
            return rr, time
        
        # Interpolate
        try:
            f = interp1d(
                time, 
                rr,
                kind=self.config.interpolation_method,
                fill_value='extrapolate',
                bounds_error=False
            )
            rr_interpolated = f(time_uniform)
            
            # Clip to physiological range
            rr_interpolated = np.clip(
                rr_interpolated,
                self.config.min_rr_ms,
                self.config.max_rr_ms
            )
            
        except Exception:
            # Fallback to linear interpolation
            rr_interpolated = np.interp(time_uniform, time, rr)
        
        return rr_interpolated, time_uniform
    
    def _create_invalid_result(self, rr: np.ndarray, reason: str) -> HRVData:
        """Create result for invalid data."""
        empty = np.array([])
        
        return HRVData(
            rr_original=rr if len(rr) > 0 else empty,
            rr_clean=empty,
            rr_interpolated=empty,
            time_original=empty,
            time_interpolated=empty,
            ectopic_mask=np.zeros(len(rr), dtype=bool) if len(rr) > 0 else np.array([], dtype=bool),
            ectopic_indices=np.array([], dtype=int),
            ectopic_ratio=0.0,
            interpolation_fs=self.config.interpolation_fs,
            is_valid=False,
            quality_issues=[reason]
        )
    
    def __repr__(self) -> str:
        return (f"HRVPreprocessor(method={self.config.ectopic_method.value}, "
                f"threshold={self.config.ectopic_threshold})")


def preprocess_hrv(
    rr_intervals: np.ndarray,
    config: Optional[HRVConfig] = None
) -> HRVData:
    """
    Convenience function for HRV preprocessing.
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in milliseconds.
    config : HRVConfig, optional
        Preprocessing configuration.
        
    Returns
    -------
    HRVData
        Preprocessed HRV data.
    """
    preprocessor = HRVPreprocessor(config)
    return preprocessor(rr_intervals)


def detect_ectopic_beats(
    rr_intervals: np.ndarray,
    method: str = 'malik',
    threshold: float = 0.2
) -> np.ndarray:
    """
    Convenience function for ectopic beat detection.
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in milliseconds.
    method : str, default='malik'
        Detection method: 'malik', 'kamath', 'karlsson', 'acar'.
    threshold : float, default=0.2
        Detection threshold (method-specific).
        
    Returns
    -------
    np.ndarray
        Boolean mask indicating ectopic beats.
    """
    method_enum = EctopicMethod(method.lower())
    config = HRVConfig(ectopic_method=method_enum, ectopic_threshold=threshold)
    preprocessor = HRVPreprocessor(config)
    
    rr = np.asarray(rr_intervals, dtype=float).flatten()
    return preprocessor._detect_ectopic(rr)