"""
Kepler-ECG: Morphological Feature Extraction

Extracting morphological features from the ECG beat template.

Extracted Features:
- Wave amplitudes (P, Q, R, S, T, ST level)
- Wave durations and intervals (P duration, QRS duration, T duration, PR, QT, QTc)
- Areas under curves (QRS area, T area)
- Derived ratios (R/S ratio, T/R ratio)

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

import numpy as np  # Industry-standard library for numerical computing and array manipulation
from typing import Dict, Optional, Tuple, NamedTuple  # Type hints for structured returns and better code clarity
from dataclasses import dataclass  # Decorator to create concise and efficient data storage classes
from scipy.signal import find_peaks  # Robust algorithm to detect local maxima (e.g., QRS complexes in ECG)
from scipy.ndimage import gaussian_filter1d  # 1D Gaussian smoothing to reduce high-frequency noise and artifacts


@dataclass
class WavePoints:
    """Punti caratteristici delle onde ECG identificati nel beat template."""
    # Indici dei punti (in samples)
    p_onset: Optional[int] = None
    p_peak: Optional[int] = None
    p_offset: Optional[int] = None
    
    q_onset: Optional[int] = None
    q_peak: Optional[int] = None
    
    r_peak: Optional[int] = None
    
    s_peak: Optional[int] = None
    s_offset: Optional[int] = None
    
    t_onset: Optional[int] = None
    t_peak: Optional[int] = None
    t_offset: Optional[int] = None
    
    # Confidence scores
    confidence: float = 0.0


class MorphologicalExtractor:
    """
    Estrattore di features morfologiche dal beat template ECG.
    
    Il beat template è un singolo battito ECG (circa 300 samples a 500Hz)
    ottenuto come media dei battiti rilevati nella pipeline di preprocessing.
    
    Features estratte (~15):
    - Ampiezze: P, Q, R, S, T, ST level
    - Durate: P, QRS, T
    - Aree: QRS, T
    - Rapporti: R/S, T/R
    
    Parameters
    ----------
    sampling_rate : int
        Frequenza di campionamento in Hz (default: 500)
    smoothing_sigma : float
        Sigma per smoothing gaussiano nella delineazione (default: 2.0)
    """
    
    def __init__(self, sampling_rate: int = 500, smoothing_sigma: float = 2.0):
        self.sampling_rate = sampling_rate
        self.smoothing_sigma = smoothing_sigma
        
        # Intervalli tipici in ms per le onde ECG (usati come constraint)
        self._p_duration_range = (60, 120)    # ms
        self._qrs_duration_range = (60, 120)  # ms (normale), fino a 200 per BBB
        self._t_duration_range = (100, 250)   # ms
        self._pr_interval_range = (120, 200)  # ms
        self._qt_interval_range = (300, 500)  # ms
        
    def extract(self, beat_template: np.ndarray, 
                rr_mean_ms: Optional[float] = None) -> Dict[str, float]:
        """
        Estrae tutte le features morfologiche da un beat template.
        
        Parameters
        ----------
        beat_template : np.ndarray
            Beat template 1D (n_samples,). Centrato sull'R-peak.
        rr_mean_ms : float, optional
            Intervallo RR medio in ms, necessario per calcolare QTc.
            
        Returns
        -------
        Dict[str, float]
            Dizionario con tutte le features estratte. NaN per features
            non calcolabili.
        """
        features = {}
        
        # Validazione input
        if beat_template is None or len(beat_template) < 50:
            return self._empty_features()
        
        # Normalizza il beat template
        beat = self._normalize_beat(beat_template)
        
        # Delineazione delle onde
        wave_points = self._delineate_waves(beat)
        
        # Estrai ampiezze
        features.update(self._extract_amplitudes(beat, wave_points))
        
        # Estrai durate
        features.update(self._extract_durations(wave_points))
        
        # Estrai aree
        features.update(self._extract_areas(beat, wave_points))
        
        # Estrai rapporti
        features.update(self._extract_ratios(features))
        
        # Aggiungi confidence
        features['morphology_confidence'] = wave_points.confidence
        
        return features
    
    def extract_intervals(self, beat_template: np.ndarray,
                         rr_mean_ms: Optional[float] = None) -> Dict[str, float]:
        """
        Estrae gli intervalli ECG (PR, QRS, QT, QTc).
        
        Task 2: IntervalCalculator - Calcolo intervalli con correzioni QTc.
        
        Parameters
        ----------
        beat_template : np.ndarray
            Beat template 1D centrato sull'R-peak.
        rr_mean_ms : float, optional
            Intervallo RR medio in ms per correzione QT.
            
        Returns
        -------
        Dict[str, float]
            Intervalli in ms e QTc con diverse formule di correzione.
        """
        features = {}
        
        if beat_template is None or len(beat_template) < 50:
            return self._empty_interval_features()
        
        beat = self._normalize_beat(beat_template)
        wave_points = self._delineate_waves(beat)
        
        # PR interval: inizio P → inizio QRS
        if wave_points.p_onset is not None and wave_points.q_onset is not None:
            pr_samples = wave_points.q_onset - wave_points.p_onset
            features['pr_interval_ms'] = self._samples_to_ms(pr_samples)
        else:
            features['pr_interval_ms'] = np.nan
        
        # QRS interval: inizio Q → fine S
        if wave_points.q_onset is not None and wave_points.s_offset is not None:
            qrs_samples = wave_points.s_offset - wave_points.q_onset
            features['qrs_interval_ms'] = self._samples_to_ms(qrs_samples)
        else:
            features['qrs_interval_ms'] = np.nan
        
        # QT interval: inizio Q → fine T
        if wave_points.q_onset is not None and wave_points.t_offset is not None:
            qt_samples = wave_points.t_offset - wave_points.q_onset
            qt_ms = self._samples_to_ms(qt_samples)
            features['qt_interval_ms'] = qt_ms
            
            # Correzioni QTc (se RR disponibile)
            if rr_mean_ms is not None and rr_mean_ms > 0:
                rr_sec = rr_mean_ms / 1000.0
                
                # Bazett: QTc = QT / sqrt(RR)
                features['qtc_bazett_ms'] = qt_ms / np.sqrt(rr_sec)
                
                # Fridericia: QTc = QT / RR^(1/3)
                features['qtc_fridericia_ms'] = qt_ms / (rr_sec ** (1/3))
                
                # Framingham: QTc = QT + 0.154 * (1 - RR)
                features['qtc_framingham_ms'] = qt_ms + 154 * (1 - rr_sec)
                
                # Hodges: QTc = QT + 1.75 * (HR - 60)
                hr = 60000 / rr_mean_ms
                features['qtc_hodges_ms'] = qt_ms + 1.75 * (hr - 60)
            else:
                features['qtc_bazett_ms'] = np.nan
                features['qtc_fridericia_ms'] = np.nan
                features['qtc_framingham_ms'] = np.nan
                features['qtc_hodges_ms'] = np.nan
        else:
            features['qt_interval_ms'] = np.nan
            features['qtc_bazett_ms'] = np.nan
            features['qtc_fridericia_ms'] = np.nan
            features['qtc_framingham_ms'] = np.nan
            features['qtc_hodges_ms'] = np.nan
        
        return features
    
    def _normalize_beat(self, beat: np.ndarray) -> np.ndarray:
        """Normalizza il beat template per la delineazione."""
        beat = np.asarray(beat, dtype=np.float64).flatten()
        
        # Rimuovi offset (baseline)
        beat = beat - np.median(beat)
        
        return beat
    
    def _delineate_waves(self, beat: np.ndarray) -> WavePoints:
        """
        Delinea le onde PQRST nel beat template.
        
        Algoritmo:
        1. Trova R-peak (massimo assoluto nella regione centrale)
        2. Cerca Q (minimo locale prima di R)
        3. Cerca S (minimo locale dopo R)
        4. Cerca P (massimo locale prima di Q)
        5. Cerca T (massimo locale dopo S)
        6. Stima onset/offset per ogni onda
        """
        points = WavePoints()
        confidence_scores = []
        
        n = len(beat)
        beat_smooth = gaussian_filter1d(beat, self.smoothing_sigma)
        
        # 1. Trova R-peak (massimo nella regione centrale: 30%-70% del beat)
        center_start = int(n * 0.3)
        center_end = int(n * 0.7)
        r_idx = center_start + np.argmax(beat_smooth[center_start:center_end])
        points.r_peak = r_idx
        r_amplitude = beat_smooth[r_idx]
        
        # Confidence: R dovrebbe essere il punto più alto
        if r_amplitude > 0 and r_amplitude == np.max(beat_smooth[center_start:center_end]):
            confidence_scores.append(1.0)
        else:
            confidence_scores.append(0.5)
        
        # 2. Cerca Q-peak (minimo locale prima di R, nella finestra QRS)
        qrs_window_samples = int(self._qrs_duration_range[1] * self.sampling_rate / 1000)
        q_search_start = max(0, r_idx - qrs_window_samples)
        q_search_end = r_idx
        
        if q_search_end > q_search_start:
            q_region = beat_smooth[q_search_start:q_search_end]
            # Q è un minimo locale (negativo) prima di R
            q_idx_rel = np.argmin(q_region)
            q_idx = q_search_start + q_idx_rel
            
            # Q valido se è un minimo locale significativo
            if beat_smooth[q_idx] < beat_smooth[r_idx] * 0.1:  # Q negativo o molto basso
                points.q_peak = q_idx
                points.q_onset = self._find_onset(beat_smooth, q_idx, direction='left')
                confidence_scores.append(0.8)
            else:
                # Q non chiaramente visibile - usa punto di flessione
                points.q_onset = self._find_inflection_point(beat_smooth, q_search_start, r_idx, 'left')
                points.q_peak = points.q_onset
                confidence_scores.append(0.4)
        
        # 3. Cerca S-peak (minimo locale dopo R)
        s_search_start = r_idx
        s_search_end = min(n, r_idx + qrs_window_samples)
        
        if s_search_end > s_search_start:
            s_region = beat_smooth[s_search_start:s_search_end]
            s_idx_rel = np.argmin(s_region)
            s_idx = s_search_start + s_idx_rel
            
            if beat_smooth[s_idx] < beat_smooth[r_idx] * 0.1:
                points.s_peak = s_idx
                points.s_offset = self._find_offset(beat_smooth, s_idx, direction='right')
                confidence_scores.append(0.8)
            else:
                points.s_offset = self._find_inflection_point(beat_smooth, r_idx, s_search_end, 'right')
                points.s_peak = points.s_offset
                confidence_scores.append(0.4)
        
        # 4. Cerca P-wave (massimo locale prima del QRS)
        pr_window_samples = int(self._pr_interval_range[1] * self.sampling_rate / 1000)
        p_search_start = max(0, (points.q_onset or r_idx) - pr_window_samples)
        p_search_end = points.q_onset or (r_idx - qrs_window_samples // 2)
        
        if p_search_end > p_search_start + 10:
            p_region = beat_smooth[p_search_start:p_search_end]
            
            # Trova picchi positivi nella regione P
            peaks, properties = find_peaks(p_region, prominence=0.01 * r_amplitude)
            
            if len(peaks) > 0:
                # Prendi il picco più prominente
                best_peak = peaks[np.argmax(properties['prominences'])]
                p_idx = p_search_start + best_peak
                points.p_peak = p_idx
                
                # Stima onset e offset di P
                p_dur_samples = int(self._p_duration_range[1] * self.sampling_rate / 1000)
                points.p_onset = self._find_onset(beat_smooth, p_idx, direction='left', 
                                                   max_distance=p_dur_samples // 2)
                points.p_offset = self._find_offset(beat_smooth, p_idx, direction='right',
                                                    max_distance=p_dur_samples // 2)
                confidence_scores.append(0.8)
            else:
                confidence_scores.append(0.2)
        
        # 5. Cerca T-wave (massimo locale dopo il QRS)
        t_search_start = points.s_offset or (r_idx + qrs_window_samples // 2)
        t_window_samples = int(self._qt_interval_range[1] * self.sampling_rate / 1000)
        t_search_end = min(n, t_search_start + t_window_samples)
        
        if t_search_end > t_search_start + 10:
            t_region = beat_smooth[t_search_start:t_search_end]
            
            # T può essere positiva o negativa
            max_idx = np.argmax(t_region)
            min_idx = np.argmin(t_region)
            
            # Scegli il picco più prominente
            if abs(t_region[max_idx]) > abs(t_region[min_idx]):
                t_idx = t_search_start + max_idx
            else:
                t_idx = t_search_start + min_idx
            
            points.t_peak = t_idx
            
            # Stima onset e offset di T
            t_dur_samples = int(self._t_duration_range[1] * self.sampling_rate / 1000)
            points.t_onset = self._find_onset(beat_smooth, t_idx, direction='left',
                                               max_distance=t_dur_samples // 2)
            points.t_offset = self._find_offset(beat_smooth, t_idx, direction='right',
                                                max_distance=t_dur_samples // 2)
            confidence_scores.append(0.8)
        else:
            confidence_scores.append(0.2)
        
        # Calcola confidence totale
        points.confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return points
    
    def _find_onset(self, signal: np.ndarray, peak_idx: int, 
                    direction: str = 'left', max_distance: int = 50) -> int:
        """Trova l'onset di un'onda cercando il punto di minima pendenza."""
        if direction == 'left':
            start = max(0, peak_idx - max_distance)
            region = signal[start:peak_idx]
            if len(region) < 3:
                return start
            
            # Derivata
            gradient = np.gradient(region)
            # Onset: dove la derivata cambia segno o è minima
            zero_crossings = np.where(np.diff(np.sign(gradient)))[0]
            if len(zero_crossings) > 0:
                return start + zero_crossings[-1]
            else:
                return start + np.argmin(np.abs(gradient))
        else:
            return peak_idx
    
    def _find_offset(self, signal: np.ndarray, peak_idx: int,
                     direction: str = 'right', max_distance: int = 50) -> int:
        """Trova l'offset di un'onda cercando il ritorno alla baseline."""
        if direction == 'right':
            end = min(len(signal), peak_idx + max_distance)
            region = signal[peak_idx:end]
            if len(region) < 3:
                return end - 1
            
            gradient = np.gradient(region)
            zero_crossings = np.where(np.diff(np.sign(gradient)))[0]
            if len(zero_crossings) > 0:
                return peak_idx + zero_crossings[0]
            else:
                return peak_idx + np.argmin(np.abs(gradient))
        else:
            return peak_idx
    
    def _find_inflection_point(self, signal: np.ndarray, start: int, end: int, 
                               which: str = 'left') -> int:
        """Trova punto di flessione (seconda derivata = 0)."""
        region = signal[start:end]
        if len(region) < 5:
            return start if which == 'left' else end - 1
        
        second_derivative = np.gradient(np.gradient(region))
        zero_crossings = np.where(np.diff(np.sign(second_derivative)))[0]
        
        if len(zero_crossings) > 0:
            if which == 'left':
                return start + zero_crossings[-1]
            else:
                return start + zero_crossings[0]
        else:
            return start if which == 'left' else end - 1
    
    def _extract_amplitudes(self, beat: np.ndarray, 
                           wave_points: WavePoints) -> Dict[str, float]:
        """Estrae le ampiezze delle onde."""
        features = {}
        
        # Baseline: media dei primi e ultimi campioni
        baseline = (np.mean(beat[:10]) + np.mean(beat[-10:])) / 2
        
        # P amplitude
        if wave_points.p_peak is not None:
            features['p_amplitude_mv'] = beat[wave_points.p_peak] - baseline
        else:
            features['p_amplitude_mv'] = np.nan
        
        # Q amplitude (negativa)
        if wave_points.q_peak is not None:
            features['q_amplitude_mv'] = beat[wave_points.q_peak] - baseline
        else:
            features['q_amplitude_mv'] = np.nan
        
        # R amplitude
        if wave_points.r_peak is not None:
            features['r_amplitude_mv'] = beat[wave_points.r_peak] - baseline
        else:
            features['r_amplitude_mv'] = np.nan
        
        # S amplitude (negativa)
        if wave_points.s_peak is not None:
            features['s_amplitude_mv'] = beat[wave_points.s_peak] - baseline
        else:
            features['s_amplitude_mv'] = np.nan
        
        # T amplitude
        if wave_points.t_peak is not None:
            features['t_amplitude_mv'] = beat[wave_points.t_peak] - baseline
        else:
            features['t_amplitude_mv'] = np.nan
        
        # ST level (punto J + 60ms)
        if wave_points.s_offset is not None:
            j_point = wave_points.s_offset
            st_point = min(len(beat) - 1, j_point + int(0.06 * self.sampling_rate))
            features['st_level_mv'] = beat[st_point] - baseline
        else:
            features['st_level_mv'] = np.nan
        
        return features
    
    def _extract_durations(self, wave_points: WavePoints) -> Dict[str, float]:
        """Estrae le durate delle onde."""
        features = {}
        
        # P duration
        if wave_points.p_onset is not None and wave_points.p_offset is not None:
            features['p_duration_ms'] = self._samples_to_ms(
                wave_points.p_offset - wave_points.p_onset)
        else:
            features['p_duration_ms'] = np.nan
        
        # QRS duration
        if wave_points.q_onset is not None and wave_points.s_offset is not None:
            features['qrs_duration_ms'] = self._samples_to_ms(
                wave_points.s_offset - wave_points.q_onset)
        else:
            features['qrs_duration_ms'] = np.nan
        
        # T duration
        if wave_points.t_onset is not None and wave_points.t_offset is not None:
            features['t_duration_ms'] = self._samples_to_ms(
                wave_points.t_offset - wave_points.t_onset)
        else:
            features['t_duration_ms'] = np.nan
        
        return features
    
    def _extract_areas(self, beat: np.ndarray, 
                      wave_points: WavePoints) -> Dict[str, float]:
        """Estrae le aree sotto le onde (integrale)."""
        features = {}
        
        # Area QRS (valore assoluto dell'integrale)
        if wave_points.q_onset is not None and wave_points.s_offset is not None:
            qrs_region = beat[wave_points.q_onset:wave_points.s_offset + 1]
            # Integrale trapezoidale, convertito in mV·ms
            area = np.trapezoid(np.abs(qrs_region)) / self.sampling_rate * 1000
            features['qrs_area_mv_ms'] = area
        else:
            features['qrs_area_mv_ms'] = np.nan
        
        # Area T
        if wave_points.t_onset is not None and wave_points.t_offset is not None:
            t_region = beat[wave_points.t_onset:wave_points.t_offset + 1]
            area = np.trapezoid(np.abs(t_region)) / self.sampling_rate * 1000
            features['t_area_mv_ms'] = area
        else:
            features['t_area_mv_ms'] = np.nan
        
        return features
    
    def _extract_ratios(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calcola rapporti derivati dalle features."""
        ratios = {}
        
        # R/S ratio
        r_amp = features.get('r_amplitude_mv', np.nan)
        s_amp = features.get('s_amplitude_mv', np.nan)
        if not np.isnan(r_amp) and not np.isnan(s_amp) and abs(s_amp) > 0.01:
            ratios['r_s_ratio'] = abs(r_amp / s_amp)
        else:
            ratios['r_s_ratio'] = np.nan
        
        # T/R ratio
        t_amp = features.get('t_amplitude_mv', np.nan)
        if not np.isnan(t_amp) and not np.isnan(r_amp) and abs(r_amp) > 0.01:
            ratios['t_r_ratio'] = t_amp / r_amp
        else:
            ratios['t_r_ratio'] = np.nan
        
        # QRS/T area ratio
        qrs_area = features.get('qrs_area_mv_ms', np.nan)
        t_area = features.get('t_area_mv_ms', np.nan)
        if not np.isnan(qrs_area) and not np.isnan(t_area) and t_area > 0.01:
            ratios['qrs_t_area_ratio'] = qrs_area / t_area
        else:
            ratios['qrs_t_area_ratio'] = np.nan
        
        return ratios
    
    def _samples_to_ms(self, n_samples: int) -> float:
        """Converte numero di samples in millisecondi."""
        return (n_samples / self.sampling_rate) * 1000
    
    def _empty_features(self) -> Dict[str, float]:
        """Restituisce un dizionario con tutte le features a NaN."""
        return {
            'p_amplitude_mv': np.nan,
            'q_amplitude_mv': np.nan,
            'r_amplitude_mv': np.nan,
            's_amplitude_mv': np.nan,
            't_amplitude_mv': np.nan,
            'st_level_mv': np.nan,
            'p_duration_ms': np.nan,
            'qrs_duration_ms': np.nan,
            't_duration_ms': np.nan,
            'qrs_area_mv_ms': np.nan,
            't_area_mv_ms': np.nan,
            'r_s_ratio': np.nan,
            't_r_ratio': np.nan,
            'qrs_t_area_ratio': np.nan,
            'morphology_confidence': 0.0,
        }
    
    def _empty_interval_features(self) -> Dict[str, float]:
        """Restituisce un dizionario con tutte le interval features a NaN."""
        return {
            'pr_interval_ms': np.nan,
            'qrs_interval_ms': np.nan,
            'qt_interval_ms': np.nan,
            'qtc_bazett_ms': np.nan,
            'qtc_fridericia_ms': np.nan,
            'qtc_framingham_ms': np.nan,
            'qtc_hodges_ms': np.nan,
        }
    
    def get_feature_names(self) -> list:
        """Restituisce la lista dei nomi di tutte le features."""
        return list(self._empty_features().keys()) + list(self._empty_interval_features().keys())


class IntervalCalculator:
    """
    Calcolatore di intervalli ECG con correzioni QTc.
    
    Task 2: Wrapper per compatibilità - usa MorphologicalExtractor internamente.
    
    Formule QTc implementate:
    - Bazett: QTc = QT / sqrt(RR)
    - Fridericia: QTc = QT / RR^(1/3)
    - Framingham: QTc = QT + 0.154 * (1 - RR)
    - Hodges: QTc = QT + 1.75 * (HR - 60)
    """
    
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        self._extractor = MorphologicalExtractor(sampling_rate=sampling_rate)
    
    def calculate(self, beat_template: np.ndarray,
                 rr_mean_ms: Optional[float] = None) -> Dict[str, float]:
        """
        Calcola tutti gli intervalli ECG.
        
        Parameters
        ----------
        beat_template : np.ndarray
            Beat template 1D centrato sull'R-peak.
        rr_mean_ms : float, optional
            Intervallo RR medio in ms.
            
        Returns
        -------
        Dict[str, float]
            Dizionario con tutti gli intervalli in ms.
        """
        return self._extractor.extract_intervals(beat_template, rr_mean_ms)
    
    @staticmethod
    def correct_qt_bazett(qt_ms: float, rr_ms: float) -> float:
        """Corregge QT con formula di Bazett."""
        if rr_ms <= 0:
            return np.nan
        return qt_ms / np.sqrt(rr_ms / 1000)
    
    @staticmethod
    def correct_qt_fridericia(qt_ms: float, rr_ms: float) -> float:
        """Corregge QT con formula di Fridericia."""
        if rr_ms <= 0:
            return np.nan
        return qt_ms / ((rr_ms / 1000) ** (1/3))
    
    @staticmethod
    def correct_qt_framingham(qt_ms: float, rr_ms: float) -> float:
        """Corregge QT con formula di Framingham."""
        if rr_ms <= 0:
            return np.nan
        return qt_ms + 154 * (1 - rr_ms / 1000)
    
    @staticmethod
    def correct_qt_hodges(qt_ms: float, hr_bpm: float) -> float:
        """Corregge QT con formula di Hodges."""
        if hr_bpm <= 0:
            return np.nan
        return qt_ms + 1.75 * (hr_bpm - 60)