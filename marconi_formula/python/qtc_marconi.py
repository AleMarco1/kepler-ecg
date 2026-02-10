"""
Marconi QTc Formula - Python Implementation
============================================

The Marconi formula for heart rate-corrected QT interval.

Formula: QTc = QT + 125/RR - 158

Where:
    QTc = Corrected QT interval (ms)
    QT  = Measured QT interval (ms)
    RR  = RR interval (seconds)

Reference:
    Marconi A. (2026). A Novel QTc Correction Formula Derived from 
    Symbolic Regression on 1.2 Million ECGs.

Author: Alessandro Marconi
License: MIT
Version: 1.0.0
"""

from typing import Union, Optional
import numpy as np


# Formula coefficients
K = 125  # Correction coefficient (ms·s)
C = 158  # Calibration constant (ms)


def calculate_qtc_marconi(
    qt_ms: Union[float, np.ndarray],
    rr_sec: Union[float, np.ndarray],
    validate: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate QTc using the Marconi formula.
    
    Formula: QTc = QT + 125/RR - 158
    
    Parameters
    ----------
    qt_ms : float or array-like
        QT interval in milliseconds
    rr_sec : float or array-like
        RR interval in seconds
    validate : bool, optional
        Whether to validate input ranges (default: True)
    
    Returns
    -------
    float or array-like
        Corrected QT interval (QTc) in milliseconds
    
    Raises
    ------
    ValueError
        If inputs are outside valid ranges and validate=True
    
    Examples
    --------
    >>> calculate_qtc_marconi(380, 0.8)
    378.25
    
    >>> calculate_qtc_marconi([380, 400], [0.8, 1.0])
    array([378.25, 367.  ])
    """
    qt_ms = np.asarray(qt_ms)
    rr_sec = np.asarray(rr_sec)
    
    if validate:
        if np.any((qt_ms < 200) | (qt_ms > 600)):
            raise ValueError("QT must be between 200 and 600 ms")
        if np.any((rr_sec < 0.4) | (rr_sec > 2.0)):
            raise ValueError("RR must be between 0.4 and 2.0 seconds")
    
    qtc = qt_ms + K / rr_sec - C
    
    return float(qtc) if qtc.ndim == 0 else qtc


def calculate_qtc_marconi_from_hr(
    qt_ms: Union[float, np.ndarray],
    hr_bpm: Union[float, np.ndarray],
    validate: bool = True
) -> Union[float, np.ndarray]:
    """
    Calculate QTc using heart rate instead of RR interval.
    
    Parameters
    ----------
    qt_ms : float or array-like
        QT interval in milliseconds
    hr_bpm : float or array-like
        Heart rate in beats per minute
    validate : bool, optional
        Whether to validate input ranges (default: True)
    
    Returns
    -------
    float or array-like
        Corrected QT interval (QTc) in milliseconds
    
    Examples
    --------
    >>> calculate_qtc_marconi_from_hr(380, 75)
    378.25
    """
    rr_sec = 60 / np.asarray(hr_bpm)
    return calculate_qtc_marconi(qt_ms, rr_sec, validate)


def classify_qtc(
    qtc_ms: float,
    sex: str = 'M'
) -> str:
    """
    Classify QTc value into clinical categories.
    
    Uses AHA/ACC sex-specific thresholds.
    
    Parameters
    ----------
    qtc_ms : float
        QTc interval in milliseconds
    sex : str, optional
        'M' for male, 'F' for female (default: 'M')
    
    Returns
    -------
    str
        Clinical classification: 'normal', 'borderline', 
        'prolonged', or 'high_risk'
    
    Examples
    --------
    >>> classify_qtc(420, 'M')
    'normal'
    >>> classify_qtc(465, 'F')
    'borderline'
    """
    threshold_normal = 450 if sex.upper() == 'M' else 460
    
    if qtc_ms < threshold_normal:
        return 'normal'
    elif qtc_ms < 470:
        return 'borderline'
    elif qtc_ms < 500:
        return 'prolonged'
    else:
        return 'high_risk'


def compare_with_bazett(
    qt_ms: Union[float, np.ndarray],
    rr_sec: Union[float, np.ndarray]
) -> dict:
    """
    Compare Marconi and Bazett QTc values.
    
    Parameters
    ----------
    qt_ms : float or array-like
        QT interval in milliseconds
    rr_sec : float or array-like
        RR interval in seconds
    
    Returns
    -------
    dict
        Dictionary with 'marconi', 'bazett', and 'difference' values
    
    Examples
    --------
    >>> compare_with_bazett(380, 0.8)
    {'marconi': 378.25, 'bazett': 424.85, 'difference': -46.6}
    """
    qt_ms = np.asarray(qt_ms)
    rr_sec = np.asarray(rr_sec)
    
    qtc_marconi = qt_ms + K / rr_sec - C
    qtc_bazett = qt_ms / np.sqrt(rr_sec)
    
    return {
        'marconi': float(qtc_marconi) if qtc_marconi.ndim == 0 else qtc_marconi,
        'bazett': float(qtc_bazett) if qtc_bazett.ndim == 0 else qtc_bazett,
        'difference': float(qtc_marconi - qtc_bazett) if qtc_marconi.ndim == 0 else qtc_marconi - qtc_bazett
    }


# Convenience functions for other common formulas
def calculate_qtc_bazett(qt_ms, rr_sec):
    """Bazett formula: QTc = QT / sqrt(RR)"""
    return np.asarray(qt_ms) / np.sqrt(np.asarray(rr_sec))


def calculate_qtc_fridericia(qt_ms, rr_sec):
    """Fridericia formula: QTc = QT / cbrt(RR)"""
    return np.asarray(qt_ms) / np.cbrt(np.asarray(rr_sec))


def calculate_qtc_framingham(qt_ms, rr_sec):
    """Framingham formula: QTc = QT + 154(1-RR)"""
    return np.asarray(qt_ms) + 154 * (1 - np.asarray(rr_sec))


def calculate_qtc_hodges(qt_ms, hr_bpm):
    """Hodges formula: QTc = QT + 1.75(HR-60)"""
    return np.asarray(qt_ms) + 1.75 * (np.asarray(hr_bpm) - 60)


if __name__ == '__main__':
    # Example usage
    print("Marconi QTc Formula Calculator")
    print("=" * 40)
    
    # Single value
    qt = 380  # ms
    rr = 0.8  # seconds (75 bpm)
    
    qtc = calculate_qtc_marconi(qt, rr)
    classification = classify_qtc(qtc, 'M')
    comparison = compare_with_bazett(qt, rr)
    
    print(f"\nInput: QT = {qt} ms, RR = {rr} s (HR = {60/rr:.0f} bpm)")
    print(f"\nMarconi QTc:  {qtc:.1f} ms ({classification})")
    print(f"Bazett QTc:   {comparison['bazett']:.1f} ms")
    print(f"Difference:   {comparison['difference']:.1f} ms")
    
    # Array example
    print("\n" + "=" * 40)
    print("Batch calculation example:")
    qt_values = np.array([350, 380, 400, 420])
    rr_values = np.array([1.0, 0.8, 0.6, 0.5])
    
    qtc_values = calculate_qtc_marconi(qt_values, rr_values)
    for i in range(len(qt_values)):
        hr = 60 / rr_values[i]
        print(f"  QT={qt_values[i]}ms, HR={hr:.0f}bpm → QTc={qtc_values[i]:.1f}ms")
