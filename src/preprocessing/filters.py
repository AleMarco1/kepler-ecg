"""
Kepler-ECG: Filtering Components for ECG Preprocessing

This module implements signal filtering components for ECG preprocessing:
- BaselineRemover: Removes baseline wander using high-pass Butterworth filter
- NoiseFilter: Removes high-frequency noise and powerline interference

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

from dataclasses import dataclass            # Efficiently create data storage classes (e.g., for Signal Metadata)
from typing import Optional, Tuple, Union    # Type hinting for better code documentation and error checking
import numpy as np                           # Fundamental package for array manipulation and linear algebra
from scipy.signal import butter, filtfilt, iirnotch  # Signal processing tools for filtering and noise removal


@dataclass
class FilterConfig:
    """Configuration for ECG filters."""
    
    # Baseline removal (high-pass)
    baseline_cutoff: float = 0.5  # Hz
    baseline_order: int = 2
    
    # Noise filter (low-pass)
    lowpass_cutoff: float = 40.0  # Hz
    lowpass_order: int = 4
    
    # Powerline interference (notch)
    notch_freq: float = 50.0  # Hz (50Hz Europe, 60Hz Americas)
    notch_q: float = 30.0  # Quality factor


class BaselineRemover:
    """
    Removes baseline wander from ECG signals using a high-pass Butterworth filter.
    
    Baseline wander is a low-frequency artifact (<0.5 Hz) caused by:
    - Respiration
    - Body movement
    - Electrode impedance changes
    
    The filter uses zero-phase filtering (filtfilt) to avoid phase distortion,
    which is critical for preserving ECG morphology and timing.
    
    Parameters
    ----------
    cutoff : float, default=0.5
        Cutoff frequency in Hz. Standard value is 0.5 Hz to preserve
        the lowest frequency components of the ECG (P-wave, T-wave).
    order : int, default=2
        Filter order. Higher orders give sharper cutoff but may introduce
        ringing artifacts. Order 2 is a good compromise.
    
    Example
    -------
    >>> remover = BaselineRemover(cutoff=0.5, order=2)
    >>> clean_signal = remover(ecg_signal, sampling_rate=500)
    
    Notes
    -----
    - For 10-second ECG recordings (like PTB-XL), 0.5 Hz cutoff is appropriate
    - For Holter recordings, consider lower cutoff (0.05-0.1 Hz) to preserve
      very low frequency HRV components
    - The filter uses second-order sections (sos) for numerical stability
    """
    
    def __init__(self, cutoff: float = 0.5, order: int = 2):
        self.cutoff = cutoff
        self.order = order
        self._sos_cache = {}  # Cache filter coefficients per sampling rate
    
    def _get_sos(self, fs: int) -> np.ndarray:
        """
        Get or compute second-order sections for the filter.
        
        Uses caching to avoid recomputing coefficients for the same
        sampling rate.
        """
        if fs not in self._sos_cache:
            # Normalize cutoff frequency to Nyquist
            nyquist = fs / 2
            normalized_cutoff = self.cutoff / nyquist
            
            # Design Butterworth high-pass filter
            # Using 'sos' output for better numerical stability
            sos = butter(
                self.order, 
                normalized_cutoff, 
                btype='high', 
                output='sos'
            )
            self._sos_cache[fs] = sos
        
        return self._sos_cache[fs]
    
    def __call__(
        self, 
        signal: np.ndarray, 
        fs: int,
        axis: int = 0
    ) -> np.ndarray:
        """
        Apply baseline removal to ECG signal.
        
        Parameters
        ----------
        signal : np.ndarray
            ECG signal. Can be 1D (single lead) or 2D (n_samples, n_leads).
        fs : int
            Sampling frequency in Hz.
        axis : int, default=0
            Axis along which to filter (sample axis).
            
        Returns
        -------
        np.ndarray
            Filtered signal with baseline removed, same shape as input.
        """
        from scipy.signal import sosfiltfilt
        
        # Input validation
        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal)
        
        if signal.size == 0:
            raise ValueError("Empty signal provided")
        
        if fs <= 0:
            raise ValueError(f"Sampling rate must be positive, got {fs}")
        
        if self.cutoff >= fs / 2:
            raise ValueError(
                f"Cutoff frequency ({self.cutoff} Hz) must be less than "
                f"Nyquist frequency ({fs/2} Hz)"
            )
        
        # Get filter coefficients
        sos = self._get_sos(fs)
        
        # Apply zero-phase filtering
        # sosfiltfilt applies the filter forward and backward
        # for zero phase distortion
        filtered = sosfiltfilt(sos, signal, axis=axis)
        
        return filtered
    
    def get_frequency_response(
        self, 
        fs: int, 
        n_points: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the frequency response of the filter.
        
        Parameters
        ----------
        fs : int
            Sampling frequency in Hz.
        n_points : int
            Number of frequency points to compute.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            frequencies (Hz), magnitude response (dB)
        """
        from scipy.signal import sosfreqz
        
        sos = self._get_sos(fs)
        w, h = sosfreqz(sos, worN=n_points, fs=fs)
        
        # Convert to dB, avoiding log(0)
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
        
        return w, magnitude_db
    
    def __repr__(self) -> str:
        return f"BaselineRemover(cutoff={self.cutoff}Hz, order={self.order})"


class NoiseFilter:
    """
    Removes high-frequency noise and powerline interference from ECG signals.
    
    This filter combines two components:
    1. Low-pass Butterworth filter to remove high-frequency noise (EMG, etc.)
    2. Notch filter to remove powerline interference (50Hz or 60Hz)
    
    High-frequency noise sources include:
    - Electromyographic (EMG) noise from muscle activity
    - Electronic equipment interference
    - Motion artifacts (high-frequency component)
    
    Powerline interference:
    - 50 Hz in Europe, Asia, Africa, Australia
    - 60 Hz in Americas, parts of Asia
    
    Parameters
    ----------
    lowpass_cutoff : float, default=40.0
        Low-pass cutoff frequency in Hz. Standard ECG content is below 40 Hz.
        QRS complex has most energy below 25 Hz, but sharp peaks can have
        components up to 40 Hz.
    lowpass_order : int, default=4
        Low-pass filter order. Higher orders give sharper cutoff.
    notch_freq : float, default=50.0
        Powerline frequency to remove (50 Hz or 60 Hz).
    notch_q : float, default=30.0
        Quality factor of the notch filter. Higher Q means narrower notch.
        Q=30 gives a notch width of ~1.7 Hz at 50 Hz.
    apply_notch : bool, default=True
        Whether to apply the notch filter. Can be disabled if powerline
        interference is not present.
    
    Example
    -------
    >>> noise_filter = NoiseFilter(lowpass_cutoff=40, notch_freq=50)
    >>> clean_signal = noise_filter(ecg_signal, sampling_rate=500)
    
    Notes
    -----
    - The filter order affects both sharpness and potential ringing
    - For 500 Hz sampling rate, 40 Hz cutoff preserves all ECG information
    - The notch filter is very narrow to avoid affecting nearby ECG content
    """
    
    def __init__(
        self,
        lowpass_cutoff: float = 40.0,
        lowpass_order: int = 4,
        notch_freq: float = 50.0,
        notch_q: float = 30.0,
        apply_notch: bool = True
    ):
        self.lowpass_cutoff = lowpass_cutoff
        self.lowpass_order = lowpass_order
        self.notch_freq = notch_freq
        self.notch_q = notch_q
        self.apply_notch = apply_notch
        
        # Cache for filter coefficients
        self._lowpass_cache = {}
        self._notch_cache = {}
    
    def _get_lowpass_sos(self, fs: int) -> np.ndarray:
        """Get or compute low-pass filter coefficients."""
        if fs not in self._lowpass_cache:
            nyquist = fs / 2
            normalized_cutoff = self.lowpass_cutoff / nyquist
            
            # Ensure cutoff is valid
            if normalized_cutoff >= 1.0:
                normalized_cutoff = 0.99  # Just below Nyquist
            
            sos = butter(
                self.lowpass_order,
                normalized_cutoff,
                btype='low',
                output='sos'
            )
            self._lowpass_cache[fs] = sos
        
        return self._lowpass_cache[fs]
    
    def _get_notch_ba(self, fs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get or compute notch filter coefficients."""
        if fs not in self._notch_cache:
            # Normalize frequency
            w0 = self.notch_freq / (fs / 2)
            
            # Check if notch frequency is valid
            if w0 >= 1.0:
                # Notch frequency above Nyquist, skip notch
                self._notch_cache[fs] = None
            else:
                b, a = iirnotch(w0, self.notch_q)
                self._notch_cache[fs] = (b, a)
        
        return self._notch_cache[fs]
    
    def __call__(
        self,
        signal: np.ndarray,
        fs: int,
        axis: int = 0
    ) -> np.ndarray:
        """
        Apply noise filtering to ECG signal.
        
        Parameters
        ----------
        signal : np.ndarray
            ECG signal. Can be 1D (single lead) or 2D (n_samples, n_leads).
        fs : int
            Sampling frequency in Hz.
        axis : int, default=0
            Axis along which to filter (sample axis).
            
        Returns
        -------
        np.ndarray
            Filtered signal with noise removed, same shape as input.
        """
        from scipy.signal import sosfiltfilt
        
        # Input validation
        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal)
        
        if signal.size == 0:
            raise ValueError("Empty signal provided")
        
        if fs <= 0:
            raise ValueError(f"Sampling rate must be positive, got {fs}")
        
        filtered = signal.copy()
        
        # Apply low-pass filter
        if self.lowpass_cutoff < fs / 2:
            sos = self._get_lowpass_sos(fs)
            filtered = sosfiltfilt(sos, filtered, axis=axis)
        
        # Apply notch filter if enabled and valid
        if self.apply_notch and self.notch_freq < fs / 2:
            notch_coeffs = self._get_notch_ba(fs)
            if notch_coeffs is not None:
                b, a = notch_coeffs
                filtered = filtfilt(b, a, filtered, axis=axis)
        
        return filtered
    
    def apply_lowpass_only(
        self,
        signal: np.ndarray,
        fs: int,
        axis: int = 0
    ) -> np.ndarray:
        """Apply only the low-pass filter."""
        from scipy.signal import sosfiltfilt
        
        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal)
        
        if self.lowpass_cutoff >= fs / 2:
            return signal.copy()
        
        sos = self._get_lowpass_sos(fs)
        return sosfiltfilt(sos, signal, axis=axis)
    
    def apply_notch_only(
        self,
        signal: np.ndarray,
        fs: int,
        axis: int = 0
    ) -> np.ndarray:
        """Apply only the notch filter."""
        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal)
        
        if self.notch_freq >= fs / 2:
            return signal.copy()
        
        notch_coeffs = self._get_notch_ba(fs)
        if notch_coeffs is None:
            return signal.copy()
        
        b, a = notch_coeffs
        return filtfilt(b, a, signal, axis=axis)
    
    def get_frequency_response(
        self,
        fs: int,
        n_points: int = 2000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the frequency response of both filters.
        
        Parameters
        ----------
        fs : int
            Sampling frequency in Hz.
        n_points : int
            Number of frequency points to compute.
            
        Returns
        -------
        Tuple containing:
            - frequencies (Hz)
            - combined magnitude response (dB)
            - lowpass magnitude response (dB)
            - notch magnitude response (dB)
        """
        from scipy.signal import sosfreqz, freqz
        
        freqs = np.linspace(0, fs/2, n_points)
        
        # Low-pass response
        if self.lowpass_cutoff < fs / 2:
            sos = self._get_lowpass_sos(fs)
            _, h_lp = sosfreqz(sos, worN=n_points, fs=fs)
        else:
            h_lp = np.ones(n_points, dtype=complex)
        
        # Notch response
        if self.apply_notch and self.notch_freq < fs / 2:
            notch_coeffs = self._get_notch_ba(fs)
            if notch_coeffs is not None:
                b, a = notch_coeffs
                _, h_notch = freqz(b, a, worN=n_points, fs=fs)
            else:
                h_notch = np.ones(n_points, dtype=complex)
        else:
            h_notch = np.ones(n_points, dtype=complex)
        
        # Combined response
        h_combined = h_lp * h_notch
        
        # Convert to dB
        mag_lp = 20 * np.log10(np.abs(h_lp) + 1e-10)
        mag_notch = 20 * np.log10(np.abs(h_notch) + 1e-10)
        mag_combined = 20 * np.log10(np.abs(h_combined) + 1e-10)
        
        return freqs, mag_combined, mag_lp, mag_notch
    
    def __repr__(self) -> str:
        notch_str = f", notch={self.notch_freq}Hz" if self.apply_notch else ""
        return (f"NoiseFilter(lowpass={self.lowpass_cutoff}Hz, "
                f"order={self.lowpass_order}{notch_str})")


class ECGFilter:
    """
    Combined ECG filter that applies baseline removal and noise filtering.
    
    This class combines BaselineRemover and NoiseFilter into a single
    convenient interface for complete ECG filtering.
    
    Parameters
    ----------
    baseline_cutoff : float, default=0.5
        High-pass cutoff for baseline removal (Hz).
    baseline_order : int, default=2
        Order of the baseline removal filter.
    lowpass_cutoff : float, default=40.0
        Low-pass cutoff for noise removal (Hz).
    lowpass_order : int, default=4
        Order of the low-pass filter.
    notch_freq : float, default=50.0
        Powerline frequency to remove (Hz).
    notch_q : float, default=30.0
        Quality factor of the notch filter.
    apply_notch : bool, default=True
        Whether to apply notch filtering.
    
    Example
    -------
    >>> ecg_filter = ECGFilter()
    >>> clean_ecg = ecg_filter(raw_ecg, fs=500)
    """
    
    def __init__(
        self,
        baseline_cutoff: float = 0.5,
        baseline_order: int = 2,
        lowpass_cutoff: float = 40.0,
        lowpass_order: int = 4,
        notch_freq: float = 50.0,
        notch_q: float = 30.0,
        apply_notch: bool = True
    ):
        self.baseline_remover = BaselineRemover(
            cutoff=baseline_cutoff,
            order=baseline_order
        )
        self.noise_filter = NoiseFilter(
            lowpass_cutoff=lowpass_cutoff,
            lowpass_order=lowpass_order,
            notch_freq=notch_freq,
            notch_q=notch_q,
            apply_notch=apply_notch
        )
    
    def __call__(
        self,
        signal: np.ndarray,
        fs: int,
        axis: int = 0
    ) -> np.ndarray:
        """
        Apply complete ECG filtering.
        
        Order of operations:
        1. Baseline removal (high-pass)
        2. Noise filtering (low-pass + notch)
        
        Parameters
        ----------
        signal : np.ndarray
            Raw ECG signal.
        fs : int
            Sampling frequency in Hz.
        axis : int, default=0
            Axis along which to filter.
            
        Returns
        -------
        np.ndarray
            Filtered ECG signal.
        """
        # Apply baseline removal first
        filtered = self.baseline_remover(signal, fs, axis)
        
        # Then apply noise filtering
        filtered = self.noise_filter(filtered, fs, axis)
        
        return filtered
    
    def __repr__(self) -> str:
        return (f"ECGFilter(baseline={self.baseline_remover.cutoff}Hz, "
                f"lowpass={self.noise_filter.lowpass_cutoff}Hz, "
                f"notch={self.noise_filter.notch_freq}Hz)")


# Convenience function for quick usage
def remove_baseline(
    signal: np.ndarray, 
    fs: int, 
    cutoff: float = 0.5, 
    order: int = 2
) -> np.ndarray:
    """
    Convenience function to remove baseline from ECG signal.
    
    Parameters
    ----------
    signal : np.ndarray
        ECG signal (1D or 2D).
    fs : int
        Sampling frequency in Hz.
    cutoff : float, default=0.5
        High-pass cutoff frequency in Hz.
    order : int, default=2
        Filter order.
        
    Returns
    -------
    np.ndarray
        Signal with baseline removed.
    """
    remover = BaselineRemover(cutoff=cutoff, order=order)
    return remover(signal, fs)


def filter_noise(
    signal: np.ndarray,
    fs: int,
    lowpass_cutoff: float = 40.0,
    lowpass_order: int = 4,
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
    apply_notch: bool = True
) -> np.ndarray:
    """
    Convenience function to remove noise from ECG signal.
    
    Parameters
    ----------
    signal : np.ndarray
        ECG signal (1D or 2D).
    fs : int
        Sampling frequency in Hz.
    lowpass_cutoff : float, default=40.0
        Low-pass cutoff frequency in Hz.
    lowpass_order : int, default=4
        Low-pass filter order.
    notch_freq : float, default=50.0
        Powerline frequency to notch out.
    notch_q : float, default=30.0
        Notch filter quality factor.
    apply_notch : bool, default=True
        Whether to apply notch filter.
        
    Returns
    -------
    np.ndarray
        Signal with noise removed.
    """
    noise_filter = NoiseFilter(
        lowpass_cutoff=lowpass_cutoff,
        lowpass_order=lowpass_order,
        notch_freq=notch_freq,
        notch_q=notch_q,
        apply_notch=apply_notch
    )
    return noise_filter(signal, fs)


def filter_ecg(
    signal: np.ndarray,
    fs: int,
    baseline_cutoff: float = 0.5,
    lowpass_cutoff: float = 40.0,
    notch_freq: float = 50.0
) -> np.ndarray:
    """
    Convenience function for complete ECG filtering.
    
    Applies baseline removal, low-pass filtering, and notch filtering.
    
    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal (1D or 2D).
    fs : int
        Sampling frequency in Hz.
    baseline_cutoff : float, default=0.5
        High-pass cutoff for baseline removal (Hz).
    lowpass_cutoff : float, default=40.0
        Low-pass cutoff for noise removal (Hz).
    notch_freq : float, default=50.0
        Powerline frequency to remove (Hz).
        
    Returns
    -------
    np.ndarray
        Filtered ECG signal.
    """
    ecg_filter = ECGFilter(
        baseline_cutoff=baseline_cutoff,
        lowpass_cutoff=lowpass_cutoff,
        notch_freq=notch_freq
    )
    return ecg_filter(signal, fs)