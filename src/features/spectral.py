"""
Kepler-ECG: Spectral Analysis for HRV Features

Task 3 della Fase 2: Analisi spettrale degli intervalli RR per estrarre
features HRV nel dominio della frequenza.

Features estratte:
- Potenza nelle bande VLF, LF, HF (ms²)
- Potenza normalizzata (n.u.)
- Rapporto LF/HF (balance simpatico-vagale)
- Frequenze di picco

Metodi implementati:
- Welch periodogram (per serie interpolate uniformemente)
- Lomb-Scargle periodogram (per serie non uniformi)

Riferimenti:
- Task Force of ESC and NASPE (1996). Heart rate variability: 
  standards of measurement, physiological interpretation and clinical use.

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

import numpy as np
from typing import Dict, Optional, Tuple, Literal
from dataclasses import dataclass
from scipy import signal
from scipy.interpolate import interp1d


@dataclass
class FrequencyBands:
    """Definizione delle bande di frequenza HRV standard."""
    # Bande standard (Hz) - Task Force 1996
    VLF_LOW: float = 0.003
    VLF_HIGH: float = 0.04
    LF_LOW: float = 0.04
    LF_HIGH: float = 0.15
    HF_LOW: float = 0.15
    HF_HIGH: float = 0.4
    
    # Ultra-low frequency (per registrazioni 24h)
    ULF_LOW: float = 0.0
    ULF_HIGH: float = 0.003


class SpectralAnalyzer:
    """
    Analizzatore spettrale per Heart Rate Variability (HRV).
    
    Estrae features nel dominio della frequenza dagli intervalli RR,
    seguendo gli standard della Task Force ESC/NASPE 1996.
    
    Parameters
    ----------
    method : str
        Metodo di stima spettrale: 'welch' o 'lomb_scargle' (default: 'welch')
    interpolation_fs : float
        Frequenza di campionamento per interpolazione RR (default: 4.0 Hz)
    welch_nperseg : int, optional
        Lunghezza segmento per Welch (default: auto basato su durata)
    welch_noverlap : int, optional
        Overlap per Welch (default: 50% di nperseg)
    detrend : bool
        Se rimuovere trend lineare prima dell'analisi (default: True)
    
    Notes
    -----
    Per registrazioni brevi (<5 min), la banda VLF non è affidabile.
    Per registrazioni <2 min, anche LF può essere poco affidabile.
    """
    
    def __init__(
        self,
        method: Literal['welch', 'lomb_scargle'] = 'welch',
        interpolation_fs: float = 4.0,
        welch_nperseg: Optional[int] = None,
        welch_noverlap: Optional[int] = None,
        detrend: bool = True
    ):
        self.method = method
        self.interpolation_fs = interpolation_fs
        self.welch_nperseg = welch_nperseg
        self.welch_noverlap = welch_noverlap
        self.detrend = detrend
        self.bands = FrequencyBands()
        
        # Durata minima raccomandata per analisi spettrale (secondi)
        self._min_duration_vlf = 300  # 5 min per VLF
        self._min_duration_lf = 120   # 2 min per LF
        self._min_duration_hf = 60    # 1 min per HF
        self._min_rr_count = 10       # Minimo numero di RR
    
    def extract(self, rr_intervals_ms: np.ndarray) -> Dict[str, float]:
        """
        Estrae tutte le features spettrali HRV.
        
        Parameters
        ----------
        rr_intervals_ms : np.ndarray
            Serie di intervalli RR in millisecondi.
            
        Returns
        -------
        Dict[str, float]
            Dizionario con tutte le features spettrali.
            NaN per features non calcolabili.
        """
        features = {}
        
        # Validazione input
        if not self._validate_input(rr_intervals_ms):
            return self._empty_features()
        
        # Converti in secondi per coerenza con frequenze in Hz
        rr_sec = np.asarray(rr_intervals_ms, dtype=np.float64) / 1000.0
        
        # Rimuovi outlier estremi (ectopici residui)
        rr_clean = self._remove_outliers(rr_sec)
        
        if len(rr_clean) < self._min_rr_count:
            return self._empty_features()
        
        # Calcola PSD
        try:
            if self.method == 'welch':
                freqs, psd = self._compute_welch_psd(rr_clean)
            else:
                freqs, psd = self._compute_lomb_scargle_psd(rr_clean)
        except Exception:
            return self._empty_features()
        
        # Estrai potenze nelle bande
        features.update(self._compute_band_powers(freqs, psd))
        
        # Calcola metriche derivate
        features.update(self._compute_derived_metrics(features))
        
        # Trova frequenze di picco
        features.update(self._find_peak_frequencies(freqs, psd))
        
        # Aggiungi metadata sulla qualità
        duration_sec = np.sum(rr_clean)
        features['hrv_spectral_duration_sec'] = duration_sec
        features['hrv_spectral_n_intervals'] = len(rr_clean)
        features['hrv_spectral_method'] = 1.0 if self.method == 'welch' else 2.0
        
        return features
    
    def _validate_input(self, rr_intervals: np.ndarray) -> bool:
        """Valida l'input degli intervalli RR."""
        if rr_intervals is None:
            return False
        
        rr = np.asarray(rr_intervals)
        
        if rr.ndim != 1:
            return False
        
        if len(rr) < self._min_rr_count:
            return False
        
        # Verifica valori ragionevoli (200-2000 ms = 30-300 bpm)
        if np.any(rr < 200) or np.any(rr > 2000):
            # Rimuoveremo outlier dopo, ma se troppi sono fuori range, invalido
            valid_ratio = np.sum((rr >= 200) & (rr <= 2000)) / len(rr)
            if valid_ratio < 0.5:
                return False
        
        return True
    
    def _remove_outliers(self, rr_sec: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        """
        Rimuove outlier dalla serie RR.
        
        Usa un filtro basato sulla differenza percentuale rispetto
        alla media mobile locale.
        
        Parameters
        ----------
        rr_sec : np.ndarray
            Intervalli RR in secondi
        threshold : float
            Soglia di deviazione percentuale (default: 20%)
        """
        if len(rr_sec) < 5:
            return rr_sec
        
        # Media mobile con finestra di 5 campioni
        kernel_size = min(5, len(rr_sec))
        rr_smooth = np.convolve(rr_sec, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Calcola deviazione percentuale
        deviation = np.abs(rr_sec - rr_smooth) / rr_smooth
        
        # Mantieni solo i punti entro la soglia
        mask = deviation < threshold
        
        # Assicurati di mantenere almeno il 50% dei punti
        if np.sum(mask) < len(rr_sec) * 0.5:
            # Usa soglia più permissiva
            threshold_90 = np.percentile(deviation, 90)
            mask = deviation <= threshold_90
        
        return rr_sec[mask]
    
    def _compute_welch_psd(self, rr_sec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola PSD con metodo Welch su RR interpolati.
        
        Returns
        -------
        freqs : np.ndarray
            Frequenze in Hz
        psd : np.ndarray
            Power Spectral Density in s²/Hz (convertito poi in ms²/Hz)
        """
        # Crea vettore tempo cumulativo
        time_sec = np.cumsum(rr_sec) - rr_sec[0]
        
        # Interpola a frequenza uniforme
        duration = time_sec[-1]
        n_samples = int(duration * self.interpolation_fs)
        
        if n_samples < 16:
            raise ValueError("Serie troppo corta per Welch")
        
        time_uniform = np.linspace(0, duration, n_samples)
        
        # Interpolazione cubica
        interp_func = interp1d(time_sec, rr_sec, kind='cubic', 
                               fill_value='extrapolate')
        rr_interp = interp_func(time_uniform)
        
        # Detrend se richiesto
        if self.detrend:
            rr_interp = signal.detrend(rr_interp, type='linear')
        
        # Rimuovi media
        rr_interp = rr_interp - np.mean(rr_interp)
        
        # Calcola nperseg ottimale se non specificato
        nperseg = self.welch_nperseg
        if nperseg is None:
            # Usa ~256 campioni o 1/4 della lunghezza, il minore
            nperseg = min(256, len(rr_interp) // 4)
            nperseg = max(16, nperseg)  # Minimo 16
        
        noverlap = self.welch_noverlap
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Welch PSD
        freqs, psd = signal.welch(
            rr_interp,
            fs=self.interpolation_fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,  # Già fatto sopra
            scaling='density'
        )
        
        # Converti da s²/Hz a ms²/Hz (moltiplica per 10^6)
        psd = psd * 1e6
        
        return freqs, psd
    
    def _compute_lomb_scargle_psd(self, rr_sec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola PSD con metodo Lomb-Scargle per serie non uniformi.
        
        Non richiede interpolazione, ideale per serie con gap.
        """
        # Crea vettore tempo cumulativo
        time_sec = np.cumsum(rr_sec) - rr_sec[0]
        
        # Detrend se richiesto
        rr_values = rr_sec.copy()
        if self.detrend:
            # Detrend lineare manuale
            coeffs = np.polyfit(time_sec, rr_values, 1)
            trend = np.polyval(coeffs, time_sec)
            rr_values = rr_values - trend
        
        # Rimuovi media
        rr_values = rr_values - np.mean(rr_values)
        
        # Definisci frequenze di interesse (0.003 - 0.5 Hz)
        # Risoluzione frequenziale basata sulla durata
        duration = time_sec[-1]
        freq_resolution = 1.0 / duration
        
        # Frequenze da analizzare
        f_min = max(0.003, freq_resolution)
        f_max = 0.5
        n_freqs = min(500, int((f_max - f_min) / freq_resolution))
        n_freqs = max(50, n_freqs)
        
        freqs = np.linspace(f_min, f_max, n_freqs)
        angular_freqs = 2 * np.pi * freqs
        
        # Lomb-Scargle periodogram
        psd = signal.lombscargle(time_sec, rr_values, angular_freqs, normalize=False)
        
        # Normalizza per ottenere PSD in s²/Hz
        # La normalizzazione di scipy lombscargle dà valori che devono essere scalati
        psd = psd * 2 / len(rr_values)
        
        # Converti da s²/Hz a ms²/Hz
        psd = psd * 1e6
        
        return freqs, psd
    
    def _compute_band_powers(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """Calcola la potenza integrata nelle bande di frequenza."""
        features = {}
        
        # VLF power (0.003 - 0.04 Hz)
        vlf_mask = (freqs >= self.bands.VLF_LOW) & (freqs < self.bands.VLF_HIGH)
        if np.any(vlf_mask):
            features['vlf_power_ms2'] = np.trapezoid(psd[vlf_mask], freqs[vlf_mask])
        else:
            features['vlf_power_ms2'] = np.nan
        
        # LF power (0.04 - 0.15 Hz)
        lf_mask = (freqs >= self.bands.LF_LOW) & (freqs < self.bands.LF_HIGH)
        if np.any(lf_mask):
            features['lf_power_ms2'] = np.trapezoid(psd[lf_mask], freqs[lf_mask])
        else:
            features['lf_power_ms2'] = np.nan
        
        # HF power (0.15 - 0.4 Hz)
        hf_mask = (freqs >= self.bands.HF_LOW) & (freqs <= self.bands.HF_HIGH)
        if np.any(hf_mask):
            features['hf_power_ms2'] = np.trapezoid(psd[hf_mask], freqs[hf_mask])
        else:
            features['hf_power_ms2'] = np.nan
        
        # Total power (VLF + LF + HF)
        total_mask = (freqs >= self.bands.VLF_LOW) & (freqs <= self.bands.HF_HIGH)
        if np.any(total_mask):
            features['total_power_ms2'] = np.trapezoid(psd[total_mask], freqs[total_mask])
        else:
            features['total_power_ms2'] = np.nan
        
        return features
    
    def _compute_derived_metrics(self, band_powers: Dict[str, float]) -> Dict[str, float]:
        """Calcola metriche derivate dalle potenze di banda."""
        features = {}
        
        lf = band_powers.get('lf_power_ms2', np.nan)
        hf = band_powers.get('hf_power_ms2', np.nan)
        vlf = band_powers.get('vlf_power_ms2', np.nan)
        
        # LF/HF ratio (balance simpatico-vagale)
        if not np.isnan(lf) and not np.isnan(hf) and hf > 0:
            features['lf_hf_ratio'] = lf / hf
        else:
            features['lf_hf_ratio'] = np.nan
        
        # Normalized units (% of LF+HF)
        lf_hf_sum = lf + hf if not (np.isnan(lf) or np.isnan(hf)) else np.nan
        
        if not np.isnan(lf_hf_sum) and lf_hf_sum > 0:
            features['lf_nu'] = (lf / lf_hf_sum) * 100
            features['hf_nu'] = (hf / lf_hf_sum) * 100
        else:
            features['lf_nu'] = np.nan
            features['hf_nu'] = np.nan
        
        # Percentuale VLF, LF, HF sul totale
        total = band_powers.get('total_power_ms2', np.nan)
        
        if not np.isnan(total) and total > 0:
            if not np.isnan(vlf):
                features['vlf_percent'] = (vlf / total) * 100
            else:
                features['vlf_percent'] = np.nan
                
            if not np.isnan(lf):
                features['lf_percent'] = (lf / total) * 100
            else:
                features['lf_percent'] = np.nan
                
            if not np.isnan(hf):
                features['hf_percent'] = (hf / total) * 100
            else:
                features['hf_percent'] = np.nan
        else:
            features['vlf_percent'] = np.nan
            features['lf_percent'] = np.nan
            features['hf_percent'] = np.nan
        
        return features
    
    def _find_peak_frequencies(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """Trova le frequenze di picco nelle bande LF e HF."""
        features = {}
        
        # LF peak frequency
        lf_mask = (freqs >= self.bands.LF_LOW) & (freqs < self.bands.LF_HIGH)
        if np.any(lf_mask):
            lf_freqs = freqs[lf_mask]
            lf_psd = psd[lf_mask]
            peak_idx = np.argmax(lf_psd)
            features['lf_peak_freq_hz'] = lf_freqs[peak_idx]
        else:
            features['lf_peak_freq_hz'] = np.nan
        
        # HF peak frequency
        hf_mask = (freqs >= self.bands.HF_LOW) & (freqs <= self.bands.HF_HIGH)
        if np.any(hf_mask):
            hf_freqs = freqs[hf_mask]
            hf_psd = psd[hf_mask]
            peak_idx = np.argmax(hf_psd)
            features['hf_peak_freq_hz'] = hf_freqs[peak_idx]
        else:
            features['hf_peak_freq_hz'] = np.nan
        
        return features
    
    def _empty_features(self) -> Dict[str, float]:
        """Restituisce dizionario con tutte le features a NaN."""
        return {
            'vlf_power_ms2': np.nan,
            'lf_power_ms2': np.nan,
            'hf_power_ms2': np.nan,
            'total_power_ms2': np.nan,
            'lf_hf_ratio': np.nan,
            'lf_nu': np.nan,
            'hf_nu': np.nan,
            'vlf_percent': np.nan,
            'lf_percent': np.nan,
            'hf_percent': np.nan,
            'lf_peak_freq_hz': np.nan,
            'hf_peak_freq_hz': np.nan,
            'hrv_spectral_duration_sec': np.nan,
            'hrv_spectral_n_intervals': 0,
            'hrv_spectral_method': np.nan,
        }
    
    def get_feature_names(self) -> list:
        """Restituisce la lista dei nomi delle features."""
        return list(self._empty_features().keys())
    
    def compute_psd(self, rr_intervals_ms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola e restituisce PSD per visualizzazione/debug.
        
        Parameters
        ----------
        rr_intervals_ms : np.ndarray
            Intervalli RR in millisecondi
            
        Returns
        -------
        freqs : np.ndarray
            Frequenze in Hz
        psd : np.ndarray
            PSD in ms²/Hz
        """
        if not self._validate_input(rr_intervals_ms):
            return np.array([]), np.array([])
        
        rr_sec = np.asarray(rr_intervals_ms, dtype=np.float64) / 1000.0
        rr_clean = self._remove_outliers(rr_sec)
        
        if self.method == 'welch':
            return self._compute_welch_psd(rr_clean)
        else:
            return self._compute_lomb_scargle_psd(rr_clean)


def generate_synthetic_rr_series(
    duration_sec: float = 300,
    mean_hr_bpm: float = 70,
    sdnn_ms: float = 50,
    lf_hf_ratio: float = 1.5,
    respiratory_rate_hz: float = 0.25,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Genera una serie sintetica di intervalli RR con componenti LF e HF.
    
    Utile per testing e validazione dello SpectralAnalyzer.
    
    Parameters
    ----------
    duration_sec : float
        Durata della serie in secondi
    mean_hr_bpm : float
        Frequenza cardiaca media in bpm
    sdnn_ms : float
        Deviazione standard degli intervalli NN in ms
    lf_hf_ratio : float
        Rapporto LF/HF desiderato
    respiratory_rate_hz : float
        Frequenza respiratoria (centro banda HF)
    random_seed : int, optional
        Seed per riproducibilità
        
    Returns
    -------
    np.ndarray
        Serie di intervalli RR in millisecondi
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Intervallo RR medio
    mean_rr_ms = 60000 / mean_hr_bpm
    
    # Stima numero di battiti
    n_beats = int(duration_sec * mean_hr_bpm / 60 * 1.1)  # +10% margine
    
    # Genera tempo uniformemente spaziato per le modulazioni
    t = np.arange(n_beats) * (mean_rr_ms / 1000)  # tempo in secondi
    
    # Componente LF (0.04-0.15 Hz, centro ~0.1 Hz)
    lf_freq = 0.1
    lf_amplitude = sdnn_ms * np.sqrt(lf_hf_ratio / (1 + lf_hf_ratio)) * 0.8
    lf_component = lf_amplitude * np.sin(2 * np.pi * lf_freq * t)
    
    # Componente HF (frequenza respiratoria)
    hf_amplitude = sdnn_ms * np.sqrt(1 / (1 + lf_hf_ratio)) * 0.8
    hf_component = hf_amplitude * np.sin(2 * np.pi * respiratory_rate_hz * t)
    
    # Componente random (VLF + rumore)
    random_component = np.random.normal(0, sdnn_ms * 0.3, n_beats)
    
    # Combina componenti
    rr_series = mean_rr_ms + lf_component + hf_component + random_component
    
    # Assicurati valori positivi e ragionevoli
    rr_series = np.clip(rr_series, 300, 2000)
    
    # Taglia alla durata desiderata
    cumsum = np.cumsum(rr_series) / 1000  # in secondi
    valid_idx = cumsum <= duration_sec
    
    return rr_series[valid_idx]
