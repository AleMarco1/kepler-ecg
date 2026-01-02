"""
Kepler-ECG: Wavelet Feature Extraction

Task 4 della Fase 2: Estrazione di features wavelet multi-scala dal segnale ECG.

La trasformata wavelet permette di analizzare il segnale ECG a diverse
risoluzioni temporali-frequenziali, catturando:
- Dettagli fini (rumore, spike) alle scale basse
- Componenti QRS alle scale medie
- Onde P e T alle scale alte
- Baseline alle scale molto alte

Features estratte:
- Energia per livello di decomposizione
- Entropia della distribuzione energetica
- Statistiche dei coefficienti (media, std, skewness, kurtosis)

Riferimenti:
- Addison, P.S. (2005). Wavelet transforms and the ECG: a review.

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

import numpy as np
from typing import Dict, Optional, List, Literal, Tuple
from dataclasses import dataclass
import pywt
from scipy import stats


@dataclass
class WaveletConfig:
    """Configurazione per l'analisi wavelet."""
    wavelet: str = 'db4'           # Daubechies 4 (buono per ECG)
    max_level: Optional[int] = None  # Auto-calcolo se None
    mode: str = 'symmetric'         # Padding mode
    

class WaveletExtractor:
    """
    Estrattore di features wavelet multi-scala dal segnale ECG.
    
    Utilizza la Discrete Wavelet Transform (DWT) per decomporre il segnale
    in coefficienti di approssimazione e dettaglio a diverse scale.
    
    Per un segnale ECG a 500 Hz con wavelet db4:
    - Level 1 (cD1): ~125-250 Hz - rumore ad alta frequenza
    - Level 2 (cD2): ~62-125 Hz - componenti veloci
    - Level 3 (cD3): ~31-62 Hz - complesso QRS
    - Level 4 (cD4): ~16-31 Hz - componenti QRS
    - Level 5 (cD5): ~8-16 Hz - onde P, T
    - Level 6+ (cA): <8 Hz - baseline, respirazione
    
    Parameters
    ----------
    wavelet : str
        Nome della wavelet madre (default: 'db4')
        Opzioni comuni: 'db4', 'db6', 'sym4', 'sym6', 'coif3'
    max_level : int, optional
        Numero massimo di livelli di decomposizione.
        Se None, calcolato automaticamente.
    sampling_rate : int
        Frequenza di campionamento in Hz (default: 500)
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        max_level: Optional[int] = None,
        sampling_rate: int = 500
    ):
        self.wavelet = wavelet
        self.max_level = max_level
        self.sampling_rate = sampling_rate
        
        # Verifica che la wavelet sia valida
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Wavelet '{wavelet}' non supportata. "
                           f"Usa pywt.wavelist() per vedere le opzioni.")
        
        # Frequenze di banda approssimative per ogni livello (a 500 Hz)
        self._freq_bands = self._compute_freq_bands()
    
    def _compute_freq_bands(self) -> Dict[int, Tuple[float, float]]:
        """Calcola le bande di frequenza per ogni livello."""
        bands = {}
        nyquist = self.sampling_rate / 2
        
        for level in range(1, 10):
            high = nyquist / (2 ** level)
            low = nyquist / (2 ** (level + 1))
            bands[level] = (low, high)
        
        return bands
    
    def extract(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Estrae tutte le features wavelet dal segnale.
        
        Parameters
        ----------
        signal : np.ndarray
            Segnale ECG 1D (n_samples,) o beat template.
            
        Returns
        -------
        Dict[str, float]
            Dizionario con tutte le features wavelet.
        """
        features = {}
        
        # Validazione input
        if not self._validate_input(signal):
            return self._empty_features()
        
        # Prepara il segnale
        sig = np.asarray(signal, dtype=np.float64).flatten()
        
        # Rimuovi media (detrend semplice)
        sig = sig - np.mean(sig)
        
        # Calcola livello massimo di decomposizione
        max_level = self.max_level
        if max_level is None:
            max_level = pywt.dwt_max_level(len(sig), self.wavelet)
            max_level = min(max_level, 6)  # Limita a 6 livelli
        
        # Decomposizione wavelet
        try:
            coeffs = pywt.wavedec(sig, self.wavelet, mode='symmetric', level=max_level)
        except Exception:
            return self._empty_features()
        
        # coeffs = [cA_n, cD_n, cD_n-1, ..., cD_1]
        # cA_n = approssimazione al livello n
        # cD_i = dettaglio al livello i
        
        # Estrai energie per livello
        features.update(self._compute_level_energies(coeffs))
        
        # Estrai entropia della distribuzione energetica
        features.update(self._compute_energy_entropy(coeffs))
        
        # Estrai statistiche dei coefficienti
        features.update(self._compute_coefficient_statistics(coeffs))
        
        # Estrai features dal livello QRS (tipicamente livello 3-4)
        features.update(self._compute_qrs_level_features(coeffs))
        
        # Metadata
        features['wavelet_n_levels'] = len(coeffs) - 1
        features['wavelet_signal_length'] = len(sig)
        
        return features
    
    def _validate_input(self, signal: np.ndarray) -> bool:
        """Valida l'input del segnale."""
        if signal is None:
            return False
        
        sig = np.asarray(signal)
        
        if sig.ndim != 1:
            return False
        
        if len(sig) < 32:  # Minimo per decomposizione wavelet
            return False
        
        if np.all(sig == 0) or np.all(np.isnan(sig)):
            return False
        
        return True
    
    def _compute_level_energies(self, coeffs: List[np.ndarray]) -> Dict[str, float]:
        """
        Calcola l'energia per ogni livello di decomposizione.
        
        Energia = somma dei quadrati dei coefficienti normalizzata.
        """
        features = {}
        
        # Energia totale per normalizzazione
        total_energy = sum(np.sum(c ** 2) for c in coeffs)
        
        if total_energy == 0:
            total_energy = 1e-10
        
        # Energia approssimazione (ultimo livello)
        approx_energy = np.sum(coeffs[0] ** 2)
        features['wavelet_energy_approx'] = approx_energy / total_energy
        
        # Energia dettagli per livello (da 1 a n)
        n_detail_levels = len(coeffs) - 1
        
        for i in range(1, min(n_detail_levels + 1, 7)):  # Max 6 livelli di dettaglio
            if i <= n_detail_levels:
                # coeffs[i] è il dettaglio del livello (n_detail_levels - i + 1)
                detail_idx = n_detail_levels - i + 1
                if detail_idx < len(coeffs):
                    detail_energy = np.sum(coeffs[detail_idx] ** 2)
                    features[f'wavelet_energy_d{i}'] = detail_energy / total_energy
                else:
                    features[f'wavelet_energy_d{i}'] = np.nan
            else:
                features[f'wavelet_energy_d{i}'] = np.nan
        
        # Energia totale (non normalizzata, per riferimento)
        features['wavelet_total_energy'] = total_energy
        
        return features
    
    def _compute_energy_entropy(self, coeffs: List[np.ndarray]) -> Dict[str, float]:
        """
        Calcola l'entropia della distribuzione energetica tra i livelli.
        
        Un segnale con energia distribuita uniformemente ha alta entropia,
        mentre un segnale con energia concentrata ha bassa entropia.
        """
        features = {}
        
        # Calcola energie per livello
        energies = np.array([np.sum(c ** 2) for c in coeffs])
        total_energy = np.sum(energies)
        
        if total_energy == 0:
            features['wavelet_energy_entropy'] = np.nan
            features['wavelet_energy_concentration'] = np.nan
            return features
        
        # Normalizza per ottenere distribuzione di probabilità
        p = energies / total_energy
        p = p[p > 0]  # Rimuovi zeri per il log
        
        # Entropia di Shannon
        entropy = -np.sum(p * np.log2(p))
        
        # Normalizza rispetto all'entropia massima (log2(n_levels))
        max_entropy = np.log2(len(coeffs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        features['wavelet_energy_entropy'] = normalized_entropy
        
        # Concentrazione energetica (1 - entropia normalizzata)
        features['wavelet_energy_concentration'] = 1 - normalized_entropy
        
        return features
    
    def _compute_coefficient_statistics(self, coeffs: List[np.ndarray]) -> Dict[str, float]:
        """
        Calcola statistiche aggregate sui coefficienti wavelet.
        """
        features = {}
        
        # Concatena tutti i coefficienti di dettaglio (esclusa approssimazione)
        all_details = np.concatenate(coeffs[1:]) if len(coeffs) > 1 else np.array([])
        
        if len(all_details) == 0:
            features['wavelet_coef_mean_abs'] = np.nan
            features['wavelet_coef_std'] = np.nan
            features['wavelet_coef_skewness'] = np.nan
            features['wavelet_coef_kurtosis'] = np.nan
            features['wavelet_coef_max'] = np.nan
            features['wavelet_coef_min'] = np.nan
            return features
        
        # Media valore assoluto
        features['wavelet_coef_mean_abs'] = np.mean(np.abs(all_details))
        
        # Deviazione standard
        features['wavelet_coef_std'] = np.std(all_details)
        
        # Skewness (asimmetria)
        if len(all_details) > 2:
            features['wavelet_coef_skewness'] = stats.skew(all_details)
        else:
            features['wavelet_coef_skewness'] = np.nan
        
        # Kurtosis (curtosi)
        if len(all_details) > 3:
            features['wavelet_coef_kurtosis'] = stats.kurtosis(all_details)
        else:
            features['wavelet_coef_kurtosis'] = np.nan
        
        # Max e min
        features['wavelet_coef_max'] = np.max(all_details)
        features['wavelet_coef_min'] = np.min(all_details)
        
        return features
    
    def _compute_qrs_level_features(self, coeffs: List[np.ndarray]) -> Dict[str, float]:
        """
        Calcola features specifiche per i livelli associati al QRS.
        
        A 500 Hz, il QRS è principalmente nei livelli 3-4 (31-62 Hz).
        """
        features = {}
        
        n_detail_levels = len(coeffs) - 1
        
        # Livello QRS principale (tipicamente level 3 o 4)
        qrs_level = min(3, n_detail_levels)
        
        if qrs_level > 0 and qrs_level <= n_detail_levels:
            qrs_coeffs = coeffs[n_detail_levels - qrs_level + 1]
            
            # Energia QRS relativa
            qrs_energy = np.sum(qrs_coeffs ** 2)
            total_energy = sum(np.sum(c ** 2) for c in coeffs)
            
            if total_energy > 0:
                features['wavelet_qrs_energy_ratio'] = qrs_energy / total_energy
            else:
                features['wavelet_qrs_energy_ratio'] = np.nan
            
            # Picco coefficiente QRS
            features['wavelet_qrs_peak'] = np.max(np.abs(qrs_coeffs))
            
            # Variabilità coefficienti QRS
            features['wavelet_qrs_variability'] = np.std(qrs_coeffs)
        else:
            features['wavelet_qrs_energy_ratio'] = np.nan
            features['wavelet_qrs_peak'] = np.nan
            features['wavelet_qrs_variability'] = np.nan
        
        return features
    
    def extract_multilead(self, signals: np.ndarray) -> Dict[str, float]:
        """
        Estrae features wavelet da ECG multi-lead.
        
        Parameters
        ----------
        signals : np.ndarray
            Segnale ECG multi-lead (n_samples, n_leads).
            
        Returns
        -------
        Dict[str, float]
            Features aggregate su tutti i leads.
        """
        if signals is None or signals.ndim != 2:
            return self._empty_features()
        
        n_samples, n_leads = signals.shape
        
        # Estrai features per ogni lead
        all_features = []
        for lead_idx in range(n_leads):
            lead_features = self.extract(signals[:, lead_idx])
            all_features.append(lead_features)
        
        # Aggrega features (media e std tra leads)
        aggregated = {}
        
        feature_names = list(all_features[0].keys())
        for name in feature_names:
            values = [f[name] for f in all_features if not np.isnan(f[name])]
            
            if len(values) > 0:
                aggregated[f'{name}_mean'] = np.mean(values)
                if len(values) > 1:
                    aggregated[f'{name}_std'] = np.std(values)
                else:
                    aggregated[f'{name}_std'] = 0.0
            else:
                aggregated[f'{name}_mean'] = np.nan
                aggregated[f'{name}_std'] = np.nan
        
        return aggregated
    
    def decompose(self, signal: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        """
        Esegue la decomposizione wavelet e restituisce i coefficienti.
        
        Utile per visualizzazione e debug.
        
        Returns
        -------
        coeffs : List[np.ndarray]
            Lista dei coefficienti [cA_n, cD_n, ..., cD_1]
        names : List[str]
            Nomi dei coefficienti ['cA_n', 'cD_n', ..., 'cD_1']
        """
        if not self._validate_input(signal):
            return [], []
        
        sig = np.asarray(signal, dtype=np.float64).flatten()
        sig = sig - np.mean(sig)
        
        max_level = self.max_level
        if max_level is None:
            max_level = min(pywt.dwt_max_level(len(sig), self.wavelet), 6)
        
        coeffs = pywt.wavedec(sig, self.wavelet, mode='symmetric', level=max_level)
        
        # Genera nomi
        names = [f'cA{max_level}']
        for i in range(max_level, 0, -1):
            names.append(f'cD{i}')
        
        return coeffs, names
    
    def reconstruct_level(self, signal: np.ndarray, level: int) -> np.ndarray:
        """
        Ricostruisce il segnale usando solo un livello specifico.
        
        Utile per isolare componenti (es. solo QRS al livello 3).
        
        Parameters
        ----------
        signal : np.ndarray
            Segnale originale
        level : int
            Livello da ricostruire (1 = dettaglio più fine)
            
        Returns
        -------
        np.ndarray
            Segnale ricostruito dal livello specificato
        """
        if not self._validate_input(signal):
            return np.array([])
        
        sig = np.asarray(signal, dtype=np.float64).flatten()
        
        max_level = self.max_level
        if max_level is None:
            max_level = min(pywt.dwt_max_level(len(sig), self.wavelet), 6)
        
        coeffs = pywt.wavedec(sig, self.wavelet, mode='symmetric', level=max_level)
        
        # Azzera tutti i coefficienti tranne il livello richiesto
        n_levels = len(coeffs) - 1
        
        for i in range(len(coeffs)):
            if i == 0:
                # Approssimazione
                if level != 0:
                    coeffs[i] = np.zeros_like(coeffs[i])
            else:
                # Dettaglio: coeffs[i] corrisponde al livello (n_levels - i + 1)
                detail_level = n_levels - i + 1
                if detail_level != level:
                    coeffs[i] = np.zeros_like(coeffs[i])
        
        # Ricostruisci
        reconstructed = pywt.waverec(coeffs, self.wavelet, mode='symmetric')
        
        # Taglia alla lunghezza originale (waverec può aggiungere campioni)
        return reconstructed[:len(sig)]
    
    def _empty_features(self) -> Dict[str, float]:
        """Restituisce dizionario con tutte le features a NaN."""
        return {
            'wavelet_energy_approx': np.nan,
            'wavelet_energy_d1': np.nan,
            'wavelet_energy_d2': np.nan,
            'wavelet_energy_d3': np.nan,
            'wavelet_energy_d4': np.nan,
            'wavelet_energy_d5': np.nan,
            'wavelet_energy_d6': np.nan,
            'wavelet_total_energy': np.nan,
            'wavelet_energy_entropy': np.nan,
            'wavelet_energy_concentration': np.nan,
            'wavelet_coef_mean_abs': np.nan,
            'wavelet_coef_std': np.nan,
            'wavelet_coef_skewness': np.nan,
            'wavelet_coef_kurtosis': np.nan,
            'wavelet_coef_max': np.nan,
            'wavelet_coef_min': np.nan,
            'wavelet_qrs_energy_ratio': np.nan,
            'wavelet_qrs_peak': np.nan,
            'wavelet_qrs_variability': np.nan,
            'wavelet_n_levels': 0,
            'wavelet_signal_length': 0,
        }
    
    def get_feature_names(self) -> list:
        """Restituisce la lista dei nomi delle features."""
        return list(self._empty_features().keys())


def generate_synthetic_ecg_beat(
    sampling_rate: int = 500,
    duration_ms: int = 800,
    r_amplitude: float = 1.0,
    noise_level: float = 0.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Genera un beat ECG sintetico per testing.
    
    Parameters
    ----------
    sampling_rate : int
        Frequenza di campionamento
    duration_ms : int
        Durata del beat in ms
    r_amplitude : float
        Ampiezza dell'onda R
    noise_level : float
        Livello di rumore (0 = nessun rumore)
    random_seed : int, optional
        Seed per riproducibilità
        
    Returns
    -------
    np.ndarray
        Beat ECG sintetico
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = int(duration_ms * sampling_rate / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples)
    
    # Posizioni onde
    r_time = duration_ms / 1000 * 0.4
    
    # P wave
    p_time = r_time - 0.16
    p_wave = 0.15 * r_amplitude * np.exp(-((t - p_time) ** 2) / (2 * 0.02 ** 2))
    
    # QRS complex
    q_time = r_time - 0.04
    q_wave = -0.1 * r_amplitude * np.exp(-((t - q_time) ** 2) / (2 * 0.008 ** 2))
    r_wave = r_amplitude * np.exp(-((t - r_time) ** 2) / (2 * 0.01 ** 2))
    s_time = r_time + 0.04
    s_wave = -0.2 * r_amplitude * np.exp(-((t - s_time) ** 2) / (2 * 0.008 ** 2))
    
    # T wave
    t_time = r_time + 0.25
    t_wave = 0.3 * r_amplitude * np.exp(-((t - t_time) ** 2) / (2 * 0.04 ** 2))
    
    # Combina
    beat = p_wave + q_wave + r_wave + s_wave + t_wave
    
    # Aggiungi rumore
    if noise_level > 0:
        beat += np.random.normal(0, noise_level, n_samples)
    
    return beat
