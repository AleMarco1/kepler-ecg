"""
Kepler-ECG: Beat Segmentation for ECG Signals

This module implements R-peak detection and beat segmentation:
- Pan-Tompkins algorithm for R-peak detection
- Beat extraction and alignment
- RR interval computation

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025

References:
    Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
    IEEE transactions on biomedical engineering, (3), 230-236.
"""

from dataclasses import dataclass, field  # Decorators to automate class boilerplate and advanced field management
from typing import Dict, List, Optional, Tuple, Union  # Advanced type hints for complex data structures
import numpy as np                        # Industry-standard library for numerical operations and array handling
from scipy.signal import find_peaks       # Algorithm to identify relative maxima within a signal (e.g., R-peaks in ECG)


@dataclass
class SegmentationConfig:
    """Configuration for beat segmentation."""
    
    # Pan-Tompkins parameters
    bandpass_low: float = 5.0  # Hz - low cutoff for bandpass
    bandpass_high: float = 15.0  # Hz - high cutoff for bandpass
    bandpass_order: int = 2
    
    # Integration window
    integration_window_sec: float = 0.150  # 150 ms window
    
    # Peak detection
    min_rr_sec: float = 0.2  # Minimum RR interval (300 bpm max)
    max_rr_sec: float = 2.0  # Maximum RR interval (30 bpm min)
    
    # Thresholding
    threshold_factor: float = 0.3  # Fraction of max for initial threshold
    
    # Beat extraction
    beat_window_before_sec: float = 0.2  # Time before R-peak
    beat_window_after_sec: float = 0.4  # Time after R-peak
    
    # Validation
    min_beats: int = 3  # Minimum beats required
    max_rr_std_ratio: float = 0.5  # Max RR variability for regular rhythm


@dataclass
class SegmentationResult:
    """
    Results from beat segmentation.
    
    Attributes
    ----------
    r_peaks : np.ndarray
        Indices of detected R-peaks in the original signal.
    r_peak_times : np.ndarray
        Times of R-peaks in seconds.
    rr_intervals : np.ndarray
        RR intervals in milliseconds.
    beats : np.ndarray
        Extracted and aligned beats, shape (n_beats, beat_length).
    beat_template : np.ndarray
        Average beat template.
    heart_rate_bpm : float
        Mean heart rate in beats per minute.
    heart_rate_std : float
        Standard deviation of heart rate.
    n_beats : int
        Number of detected beats.
    detection_confidence : float
        Confidence score for R-peak detection (0-1).
    """
    r_peaks: np.ndarray
    r_peak_times: np.ndarray
    rr_intervals: np.ndarray
    beats: np.ndarray
    beat_template: np.ndarray
    heart_rate_bpm: float
    heart_rate_std: float
    n_beats: int
    detection_confidence: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'r_peaks': self.r_peaks.tolist(),
            'r_peak_times': self.r_peak_times.tolist(),
            'rr_intervals': self.rr_intervals.tolist(),
            'heart_rate_bpm': self.heart_rate_bpm,
            'heart_rate_std': self.heart_rate_std,
            'n_beats': self.n_beats,
            'detection_confidence': self.detection_confidence,
        }


class PanTompkinsDetector:
    """
    Pan-Tompkins QRS detection algorithm.
    
    This implements the classic Pan-Tompkins algorithm for R-peak detection:
    1. Bandpass filter (5-15 Hz) to isolate QRS energy
    2. Derivative to emphasize slopes
    3. Squaring to make all values positive and emphasize large slopes
    4. Moving window integration to get QRS complex envelope
    5. Adaptive thresholding for peak detection
    
    Parameters
    ----------
    config : SegmentationConfig, optional
        Configuration parameters. Uses defaults if not provided.
    
    Example
    -------
    >>> detector = PanTompkinsDetector()
    >>> r_peaks = detector(ecg_signal, fs=500)
    """
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()
        self._filter_cache = {}
    
    def __call__(
        self,
        signal: np.ndarray,
        fs: int,
        lead: int = 0
    ) -> np.ndarray:
        """
        Detect R-peaks in ECG signal.
        
        Parameters
        ----------
        signal : np.ndarray
            ECG signal. If 2D, uses specified lead.
        fs : int
            Sampling frequency in Hz.
        lead : int, default=0
            Lead index to use for 2D signals.
            
        Returns
        -------
        np.ndarray
            Indices of detected R-peaks.
        """
        # Handle 2D signals
        if signal.ndim == 2:
            signal = signal[:, lead]
        
        signal = np.asarray(signal).flatten()
        
        if len(signal) < fs:  # Less than 1 second
            return np.array([], dtype=int)
        
        # Step 1: Bandpass filter (5-15 Hz)
        filtered = self._bandpass_filter(signal, fs)
        
        # Step 2: Derivative
        derivative = self._differentiate(filtered)
        
        # Step 3: Squaring
        squared = derivative ** 2
        
        # Step 4: Moving window integration
        integrated = self._integrate(squared, fs)
        
        # Step 5: Adaptive thresholding and peak detection
        r_peaks = self._detect_peaks(integrated, signal, fs)
        
        # Step 6: Refine R-peak locations using original signal
        r_peaks = self._refine_peaks(signal, r_peaks, fs)
        
        return r_peaks
    
    def _bandpass_filter(self, signal: np.ndarray, fs: int) -> np.ndarray:
        """Apply bandpass filter to isolate QRS energy."""
        cache_key = fs
        
        if cache_key not in self._filter_cache:
            nyquist = fs / 2
            low = self.config.bandpass_low / nyquist
            high = self.config.bandpass_high / nyquist
            
            # Ensure valid frequency range
            low = max(low, 0.01)
            high = min(high, 0.99)
            
            if low >= high:
                low = 0.01
                high = 0.99
            
            sos = butter(
                self.config.bandpass_order,
                [low, high],
                btype='band',
                output='sos'
            )
            self._filter_cache[cache_key] = sos
        
        return sosfiltfilt(self._filter_cache[cache_key], signal)
    
    def _differentiate(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute derivative using 5-point differentiation.
        
        This emphasizes the steep slopes of the QRS complex.
        """
        # 5-point derivative: [-2, -1, 0, 1, 2] / 8
        # This is less noisy than simple diff
        derivative = np.zeros_like(signal)
        derivative[2:-2] = (
            -signal[:-4] - 2*signal[1:-3] + 2*signal[3:-1] + signal[4:]
        ) / 8
        
        return derivative
    
    def _integrate(self, signal: np.ndarray, fs: int) -> np.ndarray:
        """
        Apply moving window integration.
        
        This smooths the squared derivative to get the QRS envelope.
        """
        window_size = int(self.config.integration_window_sec * fs)
        window_size = max(window_size, 1)
        
        # Moving average using cumsum for efficiency
        cumsum = np.cumsum(signal)
        cumsum = np.insert(cumsum, 0, 0)
        
        integrated = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        
        # Pad to original length
        pad_size = len(signal) - len(integrated)
        integrated = np.pad(integrated, (pad_size // 2, pad_size - pad_size // 2), mode='edge')
        
        return integrated
    
    def _detect_peaks(
        self,
        integrated: np.ndarray,
        original: np.ndarray,
        fs: int
    ) -> np.ndarray:
        """
        Detect QRS peaks using adaptive thresholding.
        """
        # Minimum distance between peaks
        min_distance = int(self.config.min_rr_sec * fs)
        
        # Initial threshold
        threshold = self.config.threshold_factor * np.max(integrated)
        
        # Find peaks above threshold
        peaks, properties = find_peaks(
            integrated,
            height=threshold,
            distance=min_distance
        )
        
        if len(peaks) == 0:
            # Try with lower threshold
            threshold = 0.1 * np.max(integrated)
            peaks, _ = find_peaks(
                integrated,
                height=threshold,
                distance=min_distance
            )
        
        return peaks
    
    def _refine_peaks(
        self,
        signal: np.ndarray,
        peaks: np.ndarray,
        fs: int
    ) -> np.ndarray:
        """
        Refine R-peak locations using the original signal.
        
        The Pan-Tompkins algorithm finds approximate QRS locations.
        We refine by finding the actual maximum in the original signal
        near each detected peak.
        """
        if len(peaks) == 0:
            return peaks
        
        # Search window: ±50ms
        search_window = int(0.05 * fs)
        refined_peaks = []
        
        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(signal), peak + search_window)
            
            # Find maximum in window (R-peak is usually positive)
            window = signal[start:end]
            
            # Check if R-peaks are positive or negative (inverted lead)
            if np.abs(np.max(window)) >= np.abs(np.min(window)):
                local_peak = start + np.argmax(window)
            else:
                local_peak = start + np.argmin(window)
            
            refined_peaks.append(local_peak)
        
        # Remove duplicates and sort
        refined_peaks = np.unique(refined_peaks)
        
        return refined_peaks
    
    def get_intermediate_signals(
        self,
        signal: np.ndarray,
        fs: int
    ) -> Dict[str, np.ndarray]:
        """
        Get intermediate signals from Pan-Tompkins algorithm.
        
        Useful for debugging and visualization.
        """
        if signal.ndim == 2:
            signal = signal[:, 0]
        
        filtered = self._bandpass_filter(signal, fs)
        derivative = self._differentiate(filtered)
        squared = derivative ** 2
        integrated = self._integrate(squared, fs)
        
        return {
            'original': signal,
            'filtered': filtered,
            'derivative': derivative,
            'squared': squared,
            'integrated': integrated,
        }


class BeatSegmenter:
    """
    Segments ECG signal into individual beats.
    
    This class combines R-peak detection with beat extraction to produce
    aligned beat waveforms suitable for feature extraction.
    
    Parameters
    ----------
    config : SegmentationConfig, optional
        Configuration parameters.
    
    Example
    -------
    >>> segmenter = BeatSegmenter()
    >>> result = segmenter(ecg_signal, fs=500)
    >>> print(f"Detected {result.n_beats} beats")
    >>> print(f"Heart rate: {result.heart_rate_bpm:.1f} bpm")
    """
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()
        self.detector = PanTompkinsDetector(self.config)
    
    def __call__(
        self,
        signal: np.ndarray,
        fs: int,
        lead: int = 0
    ) -> SegmentationResult:
        """
        Segment ECG signal into beats.
        
        Parameters
        ----------
        signal : np.ndarray
            ECG signal. Can be 1D or 2D (n_samples, n_leads).
        fs : int
            Sampling frequency in Hz.
        lead : int, default=0
            Lead to use for R-peak detection (for multi-lead).
            
        Returns
        -------
        SegmentationResult
            Complete segmentation results including R-peaks, beats, etc.
        """
        # Handle input
        if signal.ndim == 2:
            detection_signal = signal[:, lead]
        else:
            detection_signal = signal
        
        detection_signal = np.asarray(detection_signal).flatten()
        
        # Detect R-peaks
        r_peaks = self.detector(detection_signal, fs)
        
        # Handle edge case: no peaks detected
        if len(r_peaks) < self.config.min_beats:
            return self._create_empty_result(fs)
        
        # Compute R-peak times
        r_peak_times = r_peaks / fs
        
        # Compute RR intervals (in ms)
        rr_intervals = np.diff(r_peak_times) * 1000
        
        # Filter invalid RR intervals
        valid_rr = self._filter_rr_intervals(rr_intervals)
        
        # Extract beats
        beats = self._extract_beats(detection_signal, r_peaks, fs)
        
        # Compute beat template (average beat)
        if len(beats) > 0:
            beat_template = np.mean(beats, axis=0)
        else:
            beat_template = np.array([])
        
        # Compute heart rate statistics
        if len(valid_rr) > 0:
            mean_rr = np.mean(valid_rr)
            if mean_rr > 0:
                heart_rate_bpm = 60000 / mean_rr
                heart_rate_std = 60000 / mean_rr * (np.std(valid_rr) / mean_rr) if len(valid_rr) > 1 else 0.0
            else:
                heart_rate_bpm = 0.0
                heart_rate_std = 0.0
        else:
            heart_rate_bpm = 0.0
            heart_rate_std = 0.0
        
        # Compute detection confidence
        confidence = self._compute_confidence(
            detection_signal, r_peaks, rr_intervals, fs
        )
        
        return SegmentationResult(
            r_peaks=r_peaks,
            r_peak_times=r_peak_times,
            rr_intervals=rr_intervals,
            beats=beats,
            beat_template=beat_template,
            heart_rate_bpm=heart_rate_bpm,
            heart_rate_std=heart_rate_std,
            n_beats=len(r_peaks),
            detection_confidence=confidence
        )
    
    def _filter_rr_intervals(self, rr_intervals: np.ndarray) -> np.ndarray:
        """Filter out physiologically impossible RR intervals."""
        min_rr = self.config.min_rr_sec * 1000  # Convert to ms
        max_rr = self.config.max_rr_sec * 1000
        
        valid = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
        return rr_intervals[valid]
    
    def _extract_beats(
        self,
        signal: np.ndarray,
        r_peaks: np.ndarray,
        fs: int
    ) -> np.ndarray:
        """
        Extract individual beats centered on R-peaks.
        """
        before_samples = int(self.config.beat_window_before_sec * fs)
        after_samples = int(self.config.beat_window_after_sec * fs)
        beat_length = before_samples + after_samples
        
        beats = []
        
        for r_peak in r_peaks:
            start = r_peak - before_samples
            end = r_peak + after_samples
            
            # Skip beats too close to edges
            if start < 0 or end > len(signal):
                continue
            
            beat = signal[start:end]
            
            # Normalize beat (zero mean)
            beat = beat - np.mean(beat)
            
            beats.append(beat)
        
        if len(beats) == 0:
            return np.array([]).reshape(0, beat_length)
        
        return np.array(beats)
    
    def _compute_confidence(
        self,
        signal: np.ndarray,
        r_peaks: np.ndarray,
        rr_intervals: np.ndarray,
        fs: int
    ) -> float:
        """
        Compute confidence score for R-peak detection.
        
        Based on:
        - Regularity of RR intervals
        - R-peak amplitudes consistency
        - Physiological plausibility
        """
        if len(r_peaks) < 2:
            return 0.0
        
        scores = []
        
        # 1. RR interval regularity (for regular rhythms)
        if len(rr_intervals) > 1:
            rr_mean = np.mean(rr_intervals)
            if rr_mean > 0:
                rr_cv = np.std(rr_intervals) / rr_mean
                # CV < 0.1 is very regular, CV > 0.5 is irregular
                regularity_score = max(0, 1 - rr_cv / self.config.max_rr_std_ratio)
                scores.append(regularity_score)
        
        # 2. R-peak amplitude consistency
        r_amplitudes = signal[r_peaks]
        if len(r_amplitudes) > 1:
            amp_mean = np.mean(r_amplitudes)
            if abs(amp_mean) > 1e-10:
                amp_cv = np.std(r_amplitudes) / abs(amp_mean)
                amplitude_score = max(0, 1 - amp_cv)
                scores.append(amplitude_score)
        
        # 3. Physiological plausibility (reasonable heart rate)
        if len(rr_intervals) > 0:
            mean_rr = np.mean(rr_intervals)
            if mean_rr > 0:
                hr = 60000 / mean_rr  # bpm
                
                # Normal HR: 40-180 bpm
                if 40 <= hr <= 180:
                    hr_score = 1.0
                elif 30 <= hr <= 200:
                    hr_score = 0.7
                else:
                    hr_score = 0.3
                scores.append(hr_score)
        
        # 4. Detection density (expected number of beats)
        duration = len(signal) / fs
        expected_beats = duration * 1.2  # ~72 bpm
        actual_beats = len(r_peaks)
        density_ratio = actual_beats / expected_beats
        
        if 0.5 <= density_ratio <= 2.0:
            density_score = 1.0
        elif 0.3 <= density_ratio <= 3.0:
            density_score = 0.7
        else:
            density_score = 0.3
        scores.append(density_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _create_empty_result(self, fs: int) -> SegmentationResult:
        """Create result for failed segmentation."""
        beat_length = int(
            (self.config.beat_window_before_sec + 
             self.config.beat_window_after_sec) * fs
        )
        
        return SegmentationResult(
            r_peaks=np.array([], dtype=int),
            r_peak_times=np.array([]),
            rr_intervals=np.array([]),
            beats=np.array([]).reshape(0, beat_length),
            beat_template=np.zeros(beat_length),
            heart_rate_bpm=0.0,
            heart_rate_std=0.0,
            n_beats=0,
            detection_confidence=0.0
        )
    
    def __repr__(self) -> str:
        return f"BeatSegmenter(min_rr={self.config.min_rr_sec}s)"


def detect_r_peaks(
    signal: np.ndarray,
    fs: int,
    config: Optional[SegmentationConfig] = None
) -> np.ndarray:
    """
    Convenience function for R-peak detection.
    
    Parameters
    ----------
    signal : np.ndarray
        ECG signal.
    fs : int
        Sampling frequency in Hz.
    config : SegmentationConfig, optional
        Detection configuration.
        
    Returns
    -------
    np.ndarray
        Indices of detected R-peaks.
    """
    detector = PanTompkinsDetector(config)
    return detector(signal, fs)


def segment_beats(
    signal: np.ndarray,
    fs: int,
    config: Optional[SegmentationConfig] = None
) -> SegmentationResult:
    """
    Convenience function for beat segmentation.
    
    Parameters
    ----------
    signal : np.ndarray
        ECG signal.
    fs : int
        Sampling frequency in Hz.
    config : SegmentationConfig, optional
        Segmentation configuration.
        
    Returns
    -------
    SegmentationResult
        Segmentation results.
    """
    segmenter = BeatSegmenter(config)
    return segmenter(signal, fs)