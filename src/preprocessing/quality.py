"""
Kepler-ECG: Quality Assessment for ECG Signals

This module implements quality assessment metrics for ECG signals:
- Signal-to-Noise Ratio (SNR)
- Flatline detection
- Saturation/clipping detection
- Signal amplitude assessment
- Overall quality scoring

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

from dataclasses import dataclass, field  # Decorators to automate class boilerplate and advanced field management
from enum import Enum                     # Create sets of symbolic names (enumerations) for clear labeling
from typing import Dict, List, Optional, Tuple, Union  # Advanced type hints for complex data structures
import numpy as np                        # Industry-standard library for numerical operations and array handling
from scipy import stats                   # Statistical functions for data analysis and distribution checking
from scipy.signal import find_peaks       # Algorithm to identify relative maxima within a signal (e.g., R-peaks in ECG)


class QualityLevel(Enum):
    """Quality levels for ECG signals."""
    EXCELLENT = "excellent"  # Score >= 0.9
    GOOD = "good"           # Score >= 0.7
    ACCEPTABLE = "acceptable"  # Score >= 0.5
    POOR = "poor"           # Score >= 0.3
    UNUSABLE = "unusable"   # Score < 0.3


@dataclass
class QualityMetrics:
    """
    Container for ECG quality metrics.
    
    Attributes
    ----------
    snr_db : float
        Signal-to-noise ratio in decibels.
    flatline_ratio : float
        Fraction of signal that appears flat (0-1).
    saturation_ratio : float
        Fraction of signal at saturation limits (0-1).
    amplitude_ok : bool
        Whether signal amplitude is within expected range.
    baseline_drift : float
        Measure of baseline instability (lower is better).
    high_freq_noise : float
        Measure of high-frequency noise content.
    quality_score : float
        Overall quality score (0-1, higher is better).
    quality_level : QualityLevel
        Categorical quality assessment.
    is_usable : bool
        Whether the signal is usable for analysis.
    issues : List[str]
        List of detected quality issues.
    lead_scores : Optional[Dict[int, float]]
        Per-lead quality scores for multi-lead ECG.
    """
    snr_db: float
    flatline_ratio: float
    saturation_ratio: float
    amplitude_ok: bool
    baseline_drift: float
    high_freq_noise: float
    quality_score: float
    quality_level: QualityLevel
    is_usable: bool
    issues: List[str] = field(default_factory=list)
    lead_scores: Optional[Dict[int, float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'snr_db': self.snr_db,
            'flatline_ratio': self.flatline_ratio,
            'saturation_ratio': self.saturation_ratio,
            'amplitude_ok': self.amplitude_ok,
            'baseline_drift': self.baseline_drift,
            'high_freq_noise': self.high_freq_noise,
            'quality_score': self.quality_score,
            'quality_level': self.quality_level.value,
            'is_usable': self.is_usable,
            'issues': self.issues,
            'lead_scores': self.lead_scores,
        }


@dataclass
class QualityConfig:
    """Configuration for quality assessment."""
    
    # SNR thresholds
    snr_min_db: float = 5.0  # Minimum acceptable SNR
    snr_excellent_db: float = 20.0  # SNR for excellent quality
    
    # Flatline detection
    flatline_threshold: float = 0.001  # Max variation for flatline
    flatline_min_duration_sec: float = 0.1  # Min flatline duration
    max_flatline_ratio: float = 0.1  # Max acceptable flatline fraction
    
    # Saturation detection
    saturation_percentile: float = 99.5  # Percentile for saturation detection
    max_saturation_ratio: float = 0.05  # Max acceptable saturation fraction
    
    # Amplitude
    min_amplitude_mv: float = 0.1  # Minimum expected amplitude
    max_amplitude_mv: float = 5.0  # Maximum expected amplitude
    
    # Baseline drift
    max_baseline_drift: float = 0.5  # Max acceptable baseline drift (normalized)
    
    # High-frequency noise
    max_hf_noise: float = 0.3  # Max acceptable HF noise ratio
    
    # Overall quality
    min_quality_score: float = 0.3  # Minimum score for usable signal


class QualityAssessor:
    """
    Assesses the quality of ECG signals.
    
    This class computes various quality metrics to determine whether
    an ECG signal is suitable for analysis. It detects common issues
    such as noise, flatlines, saturation, and baseline drift.
    
    Parameters
    ----------
    config : QualityConfig, optional
        Configuration for quality thresholds. If None, uses defaults.
    
    Example
    -------
    >>> assessor = QualityAssessor()
    >>> metrics = assessor(ecg_signal, fs=500)
    >>> if metrics.is_usable:
    ...     print(f"Quality: {metrics.quality_level.value}")
    ... else:
    ...     print(f"Issues: {metrics.issues}")
    
    Notes
    -----
    - SNR is estimated using the signal variance in different frequency bands
    - Flatlines are detected by looking for regions with very low variance
    - Saturation is detected by looking for values at the signal extremes
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
    
    def __call__(
        self,
        signal: np.ndarray,
        fs: int,
        axis: int = 0
    ) -> QualityMetrics:
        """
        Assess quality of ECG signal.
        
        Parameters
        ----------
        signal : np.ndarray
            ECG signal. Can be 1D (single lead) or 2D (n_samples, n_leads).
        fs : int
            Sampling frequency in Hz.
        axis : int, default=0
            Sample axis for 2D signals.
            
        Returns
        -------
        QualityMetrics
            Comprehensive quality assessment results.
        """
        # Input validation
        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal)
        
        if signal.size == 0:
            return self._create_unusable_metrics("Empty signal")
        
        # Handle 2D signals
        if signal.ndim == 2:
            return self._assess_multilead(signal, fs, axis)
        
        # Single lead assessment
        return self._assess_single_lead(signal, fs)
    
    def _assess_single_lead(self, signal: np.ndarray, fs: int) -> QualityMetrics:
        """Assess quality of a single-lead ECG."""
        issues = []
        
        # 1. Compute SNR
        snr_db = self._compute_snr(signal, fs)
        if snr_db < self.config.snr_min_db:
            issues.append(f"Low SNR: {snr_db:.1f} dB < {self.config.snr_min_db} dB")
        
        # 2. Detect flatlines
        flatline_ratio = self._detect_flatlines(signal, fs)
        if flatline_ratio > self.config.max_flatline_ratio:
            issues.append(f"Excessive flatlines: {flatline_ratio*100:.1f}%")
        
        # 3. Detect saturation
        saturation_ratio = self._detect_saturation(signal)
        if saturation_ratio > self.config.max_saturation_ratio:
            issues.append(f"Signal saturation: {saturation_ratio*100:.1f}%")
        
        # 4. Check amplitude
        amplitude_ok, amp_issue = self._check_amplitude(signal)
        if not amplitude_ok and amp_issue:
            issues.append(amp_issue)
        
        # 5. Measure baseline drift
        baseline_drift = self._measure_baseline_drift(signal, fs)
        if baseline_drift > self.config.max_baseline_drift:
            issues.append(f"Excessive baseline drift: {baseline_drift:.2f}")
        
        # 6. Measure high-frequency noise
        hf_noise = self._measure_hf_noise(signal, fs)
        if hf_noise > self.config.max_hf_noise:
            issues.append(f"High-frequency noise: {hf_noise:.2f}")
        
        # 7. Compute overall quality score
        quality_score = self._compute_quality_score(
            snr_db, flatline_ratio, saturation_ratio,
            amplitude_ok, baseline_drift, hf_noise
        )
        
        # 8. Determine quality level
        quality_level = self._score_to_level(quality_score)
        
        # 9. Determine if usable
        is_usable = quality_score >= self.config.min_quality_score
        
        return QualityMetrics(
            snr_db=snr_db,
            flatline_ratio=flatline_ratio,
            saturation_ratio=saturation_ratio,
            amplitude_ok=amplitude_ok,
            baseline_drift=baseline_drift,
            high_freq_noise=hf_noise,
            quality_score=quality_score,
            quality_level=quality_level,
            is_usable=is_usable,
            issues=issues,
            lead_scores=None
        )
    
    def _assess_multilead(
        self,
        signal: np.ndarray,
        fs: int,
        axis: int
    ) -> QualityMetrics:
        """Assess quality of multi-lead ECG."""
        # Move sample axis to position 0
        if axis != 0:
            signal = np.moveaxis(signal, axis, 0)
        
        n_leads = signal.shape[1]
        lead_metrics = []
        lead_scores = {}
        
        # Assess each lead
        for lead_idx in range(n_leads):
            lead_signal = signal[:, lead_idx]
            metrics = self._assess_single_lead(lead_signal, fs)
            lead_metrics.append(metrics)
            lead_scores[lead_idx] = metrics.quality_score
        
        # Aggregate metrics (use median/mean across leads)
        snr_db = np.median([m.snr_db for m in lead_metrics])
        flatline_ratio = np.max([m.flatline_ratio for m in lead_metrics])
        saturation_ratio = np.max([m.saturation_ratio for m in lead_metrics])
        amplitude_ok = all(m.amplitude_ok for m in lead_metrics)
        baseline_drift = np.median([m.baseline_drift for m in lead_metrics])
        hf_noise = np.median([m.high_freq_noise for m in lead_metrics])
        
        # Overall quality is the median of lead qualities
        quality_score = np.median(list(lead_scores.values()))
        quality_level = self._score_to_level(quality_score)
        is_usable = quality_score >= self.config.min_quality_score
        
        # Collect all issues
        all_issues = []
        for lead_idx, metrics in enumerate(lead_metrics):
            for issue in metrics.issues:
                all_issues.append(f"Lead {lead_idx}: {issue}")
        
        return QualityMetrics(
            snr_db=snr_db,
            flatline_ratio=flatline_ratio,
            saturation_ratio=saturation_ratio,
            amplitude_ok=amplitude_ok,
            baseline_drift=baseline_drift,
            high_freq_noise=hf_noise,
            quality_score=quality_score,
            quality_level=quality_level,
            is_usable=is_usable,
            issues=all_issues,
            lead_scores=lead_scores
        )
    
    def _compute_snr(self, signal: np.ndarray, fs: int) -> float:
        """
        Estimate Signal-to-Noise Ratio.
        
        Uses the ratio of signal power in the ECG band (0.5-40 Hz)
        to noise power outside this band.
        """
        from scipy.signal import butter, sosfiltfilt
        
        n = len(signal)
        if n < fs:  # Need at least 1 second
            return 0.0
        
        # Bandpass filter for ECG content (0.5-40 Hz)
        nyquist = fs / 2
        low = max(0.5 / nyquist, 0.01)
        high = min(40 / nyquist, 0.99)
        
        try:
            sos = butter(2, [low, high], btype='band', output='sos')
            signal_filtered = sosfiltfilt(sos, signal)
            
            # Signal power (in ECG band)
            signal_power = np.var(signal_filtered)
            
            # Noise estimate (residual after filtering)
            noise = signal - signal_filtered
            noise_power = np.var(noise)
            
            if noise_power < 1e-10:
                return 40.0  # Very low noise, cap at 40 dB
            
            snr = signal_power / noise_power
            snr_db = 10 * np.log10(snr + 1e-10)
            
            return np.clip(snr_db, -10, 40)
        
        except Exception:
            # If filtering fails, use simple variance ratio
            return 10.0  # Default moderate SNR
    
    def _detect_flatlines(self, signal: np.ndarray, fs: int) -> float:
        """
        Detect flatline segments in the signal.
        
        A flatline is a region where the signal variance is very low,
        indicating electrode disconnection or signal dropout.
        """
        window_samples = max(int(self.config.flatline_min_duration_sec * fs), 10)
        n_windows = len(signal) // window_samples
        
        if n_windows < 1:
            return 0.0
        
        flatline_count = 0
        signal_range = np.ptp(signal)
        
        if signal_range < 1e-10:
            return 1.0  # Entire signal is flat
        
        threshold = self.config.flatline_threshold * signal_range
        
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            window = signal[start:end]
            
            # Check if window is flat
            window_range = np.ptp(window)
            if window_range < threshold:
                flatline_count += 1
        
        return flatline_count / n_windows
    
    def _detect_saturation(self, signal: np.ndarray) -> float:
        """
        Detect signal saturation/clipping.
        
        Saturation occurs when the ADC reaches its limits,
        causing the signal to clip at constant values.
        """
        if len(signal) < 10:
            return 0.0
        
        # Find saturation limits
        lower_limit = np.percentile(signal, 100 - self.config.saturation_percentile)
        upper_limit = np.percentile(signal, self.config.saturation_percentile)
        
        # Allow small margin for noise
        margin = 0.01 * (upper_limit - lower_limit)
        
        # Count samples at limits
        at_lower = np.sum(signal <= lower_limit + margin)
        at_upper = np.sum(signal >= upper_limit - margin)
        
        # Also check for consecutive samples at same value (ADC clipping)
        diff = np.diff(signal)
        consecutive_same = np.sum(np.abs(diff) < 1e-10)
        
        saturation_samples = at_lower + at_upper + consecutive_same
        
        return saturation_samples / len(signal)
    
    def _check_amplitude(self, signal: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Check if signal amplitude is within expected range.
        
        ECG amplitude should typically be 0.1-5 mV.
        """
        amplitude = np.ptp(signal)
        
        if amplitude < self.config.min_amplitude_mv:
            return False, f"Amplitude too low: {amplitude:.3f} mV"
        
        if amplitude > self.config.max_amplitude_mv:
            return False, f"Amplitude too high: {amplitude:.3f} mV"
        
        return True, None
    
    def _measure_baseline_drift(self, signal: np.ndarray, fs: int) -> float:
        """
        Measure baseline drift/wander.
        
        Uses a moving average to extract the baseline and measures
        its variation relative to the signal amplitude.
        """
        # Use 1-second moving average as baseline estimate
        window = min(fs, len(signal) // 2)
        if window < 10:
            return 0.0
        
        # Simple moving average
        kernel = np.ones(window) / window
        baseline = np.convolve(signal, kernel, mode='valid')
        
        if len(baseline) < 2:
            return 0.0
        
        # Baseline variation normalized by signal range
        signal_range = np.ptp(signal)
        if signal_range < 1e-10:
            return 1.0
        
        baseline_variation = np.ptp(baseline)
        
        return baseline_variation / signal_range
    
    def _measure_hf_noise(self, signal: np.ndarray, fs: int) -> float:
        """
        Measure high-frequency noise content.
        
        Estimates the ratio of power above 40 Hz to total power.
        """
        from scipy.signal import butter, sosfiltfilt
        
        if len(signal) < fs:
            return 0.0
        
        nyquist = fs / 2
        if nyquist <= 40:
            return 0.0  # Can't measure HF noise
        
        try:
            # High-pass at 40 Hz to get HF content
            cutoff = 40 / nyquist
            if cutoff >= 0.99:
                return 0.0
            
            sos = butter(2, cutoff, btype='high', output='sos')
            hf_signal = sosfiltfilt(sos, signal)
            
            hf_power = np.var(hf_signal)
            total_power = np.var(signal)
            
            if total_power < 1e-10:
                return 0.0
            
            return hf_power / total_power
        
        except Exception:
            return 0.0
    
    def _compute_quality_score(
        self,
        snr_db: float,
        flatline_ratio: float,
        saturation_ratio: float,
        amplitude_ok: bool,
        baseline_drift: float,
        hf_noise: float
    ) -> float:
        """
        Compute overall quality score from individual metrics.
        
        Score is a weighted combination of normalized metrics.
        """
        # Normalize SNR to 0-1 (5 dB = 0, 20 dB = 1)
        snr_score = np.clip(
            (snr_db - self.config.snr_min_db) / 
            (self.config.snr_excellent_db - self.config.snr_min_db),
            0, 1
        )
        
        # Flatline score (0 flatlines = 1, max = 0)
        flatline_score = 1 - np.clip(
            flatline_ratio / self.config.max_flatline_ratio, 0, 1
        )
        
        # Saturation score
        saturation_score = 1 - np.clip(
            saturation_ratio / self.config.max_saturation_ratio, 0, 1
        )
        
        # Amplitude score
        amplitude_score = 1.0 if amplitude_ok else 0.5
        
        # Baseline drift score
        drift_score = 1 - np.clip(
            baseline_drift / self.config.max_baseline_drift, 0, 1
        )
        
        # HF noise score
        hf_score = 1 - np.clip(
            hf_noise / self.config.max_hf_noise, 0, 1
        )
        
        # Weighted combination
        weights = {
            'snr': 0.25,
            'flatline': 0.20,
            'saturation': 0.15,
            'amplitude': 0.10,
            'drift': 0.15,
            'hf_noise': 0.15
        }
        
        score = (
            weights['snr'] * snr_score +
            weights['flatline'] * flatline_score +
            weights['saturation'] * saturation_score +
            weights['amplitude'] * amplitude_score +
            weights['drift'] * drift_score +
            weights['hf_noise'] * hf_score
        )
        
        return np.clip(score, 0, 1)
    
    def _score_to_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNUSABLE
    
    def _create_unusable_metrics(self, reason: str) -> QualityMetrics:
        """Create metrics for unusable signal."""
        return QualityMetrics(
            snr_db=0.0,
            flatline_ratio=1.0,
            saturation_ratio=0.0,
            amplitude_ok=False,
            baseline_drift=1.0,
            high_freq_noise=0.0,
            quality_score=0.0,
            quality_level=QualityLevel.UNUSABLE,
            is_usable=False,
            issues=[reason],
            lead_scores=None
        )
    
    def __repr__(self) -> str:
        return f"QualityAssessor(snr_min={self.config.snr_min_db}dB)"


# Convenience function
def assess_quality(
    signal: np.ndarray,
    fs: int,
    config: Optional[QualityConfig] = None
) -> QualityMetrics:
    """
    Convenience function to assess ECG quality.
    
    Parameters
    ----------
    signal : np.ndarray
        ECG signal (1D or 2D).
    fs : int
        Sampling frequency in Hz.
    config : QualityConfig, optional
        Quality assessment configuration.
        
    Returns
    -------
    QualityMetrics
        Quality assessment results.
    """
    assessor = QualityAssessor(config)
    return assessor(signal, fs)