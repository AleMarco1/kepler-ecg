"""
Kepler-ECG: Feature Pipeline

Task 5 della Fase 2: Pipeline integrata per l'estrazione di tutte le features.

Questa classe combina tutti gli estrattori di features:
- MorphologicalExtractor: features morfologiche dal beat template
- IntervalCalculator: intervalli ECG e correzioni QTc
- SpectralAnalyzer: features HRV nel dominio della frequenza
- WaveletExtractor: features wavelet multi-scala

La pipeline supporta:
- Estrazione singola da un ECG processato
- Estrazione batch con parallelizzazione opzionale
- Configurazione flessibile degli estrattori
- Gestione robusta degli errori

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings

from .morphological import MorphologicalExtractor, IntervalCalculator
from .spectral import SpectralAnalyzer
from .wavelet import WaveletExtractor


@dataclass
class FeatureConfig:
    """Configurazione per la feature pipeline."""
    
    # Parametri generali
    sampling_rate: int = 500
    
    # Morfologiche
    extract_morphological: bool = True
    morphological_smoothing: float = 2.0
    
    # Intervalli
    extract_intervals: bool = True
    
    # Spettrali HRV
    extract_spectral: bool = True
    spectral_method: str = 'welch'  # 'welch' o 'lomb_scargle'
    spectral_interpolation_fs: float = 4.0
    
    # Wavelet
    extract_wavelet: bool = True
    wavelet_name: str = 'db4'
    wavelet_max_level: Optional[int] = None
    
    # Processing
    n_jobs: int = 1  # Numero di thread per batch processing
    verbose: bool = False


@dataclass
class ProcessedECG:
    """
    Struttura dati per un ECG preprocessato.
    
    Contiene tutti i dati necessari per l'estrazione delle features.
    """
    ecg_id: int
    signal: Optional[np.ndarray] = None          # Segnale filtrato (n_samples,) o (n_samples, n_leads)
    beat_template: Optional[np.ndarray] = None   # Beat medio (n_samples,)
    rr_intervals_ms: Optional[np.ndarray] = None # Intervalli RR in ms
    r_peaks: Optional[np.ndarray] = None         # Indici R-peaks
    
    # Metadata
    age: Optional[float] = None
    sex: Optional[int] = None
    scp_codes: Optional[Dict] = None
    
    # Quality metrics dalla Fase 1
    quality_score: Optional[float] = None
    snr_db: Optional[float] = None
    heart_rate_bpm: Optional[float] = None
    
    # Features già estratte dalla Fase 1 (opzionali)
    rr_mean_ms: Optional[float] = None
    rmssd: Optional[float] = None
    pnn50: Optional[float] = None


@dataclass 
class FeatureVector:
    """Vettore di features estratte da un singolo ECG."""
    ecg_id: int
    features: Dict[str, float]
    extraction_time_ms: float
    success: bool
    error_message: Optional[str] = None


class FeaturePipeline:
    """
    Pipeline integrata per l'estrazione di features ECG.
    
    Combina tutti gli estrattori in un'unica interfaccia unificata.
    
    Parameters
    ----------
    config : FeatureConfig, optional
        Configurazione della pipeline. Se None, usa valori default.
    
    Examples
    --------
    >>> config = FeatureConfig(sampling_rate=500, extract_wavelet=True)
    >>> pipeline = FeaturePipeline(config)
    >>> 
    >>> # Estrazione singola
    >>> ecg = ProcessedECG(ecg_id=1, beat_template=beat, rr_intervals_ms=rr)
    >>> result = pipeline.extract(ecg)
    >>> 
    >>> # Estrazione batch
    >>> ecgs = [ProcessedECG(...), ProcessedECG(...), ...]
    >>> df = pipeline.extract_batch(ecgs)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        
        # Inizializza estrattori
        self._init_extractors()
        
        # Contatori per statistiche
        self._n_processed = 0
        self._n_success = 0
        self._n_failed = 0
    
    def _init_extractors(self):
        """Inizializza tutti gli estrattori di features."""
        
        # Morfologico
        if self.config.extract_morphological:
            self.morphological = MorphologicalExtractor(
                sampling_rate=self.config.sampling_rate,
                smoothing_sigma=self.config.morphological_smoothing
            )
        else:
            self.morphological = None
        
        # Intervalli
        if self.config.extract_intervals:
            self.intervals = IntervalCalculator(
                sampling_rate=self.config.sampling_rate
            )
        else:
            self.intervals = None
        
        # Spettrale
        if self.config.extract_spectral:
            self.spectral = SpectralAnalyzer(
                method=self.config.spectral_method,
                interpolation_fs=self.config.spectral_interpolation_fs
            )
        else:
            self.spectral = None
        
        # Wavelet
        if self.config.extract_wavelet:
            self.wavelet = WaveletExtractor(
                wavelet=self.config.wavelet_name,
                max_level=self.config.wavelet_max_level,
                sampling_rate=self.config.sampling_rate
            )
        else:
            self.wavelet = None
    
    def extract(self, ecg: ProcessedECG) -> FeatureVector:
        """
        Estrae tutte le features da un singolo ECG preprocessato.
        
        Parameters
        ----------
        ecg : ProcessedECG
            ECG preprocessato con beat template e/o RR intervals.
            
        Returns
        -------
        FeatureVector
            Vettore contenente tutte le features estratte.
        """
        start_time = time.time()
        features = {}
        error_message = None
        
        try:
            # Features morfologiche dal beat template
            if self.morphological is not None and ecg.beat_template is not None:
                morph_features = self.morphological.extract(ecg.beat_template)
                features.update(self._prefix_features(morph_features, 'morph'))
            
            # Intervalli ECG
            if self.intervals is not None and ecg.beat_template is not None:
                rr_mean = ecg.rr_mean_ms
                if rr_mean is None and ecg.rr_intervals_ms is not None:
                    rr_mean = np.mean(ecg.rr_intervals_ms)
                
                interval_features = self.intervals.calculate(
                    ecg.beat_template, 
                    rr_mean_ms=rr_mean
                )
                features.update(self._prefix_features(interval_features, 'interval'))
            
            # Features spettrali HRV
            if self.spectral is not None and ecg.rr_intervals_ms is not None:
                spectral_features = self.spectral.extract(ecg.rr_intervals_ms)
                features.update(self._prefix_features(spectral_features, 'hrv'))
            
            # Features wavelet
            if self.wavelet is not None:
                # Usa beat template se disponibile, altrimenti il segnale
                wavelet_input = ecg.beat_template
                if wavelet_input is None and ecg.signal is not None:
                    # Usa il primo lead se multi-lead
                    if ecg.signal.ndim == 2:
                        wavelet_input = ecg.signal[:, 0]
                    else:
                        wavelet_input = ecg.signal
                
                if wavelet_input is not None:
                    wavelet_features = self.wavelet.extract(wavelet_input)
                    features.update(self._prefix_features(wavelet_features, 'wav'))
            
            # Aggiungi metadata
            features['ecg_id'] = ecg.ecg_id
            if ecg.age is not None:
                features['age'] = ecg.age
            if ecg.sex is not None:
                features['sex'] = ecg.sex
            if ecg.quality_score is not None:
                features['quality_score'] = ecg.quality_score
            if ecg.heart_rate_bpm is not None:
                features['heart_rate_bpm'] = ecg.heart_rate_bpm
            
            success = True
            self._n_success += 1
            
        except Exception as e:
            error_message = str(e)
            success = False
            self._n_failed += 1
            
            if self.config.verbose:
                warnings.warn(f"Error extracting features for ECG {ecg.ecg_id}: {e}")
        
        self._n_processed += 1
        extraction_time = (time.time() - start_time) * 1000
        
        return FeatureVector(
            ecg_id=ecg.ecg_id,
            features=features,
            extraction_time_ms=extraction_time,
            success=success,
            error_message=error_message
        )
    
    def extract_batch(
        self, 
        ecgs: List[ProcessedECG],
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Estrae features per un batch di ECG.
        
        Parameters
        ----------
        ecgs : List[ProcessedECG]
            Lista di ECG preprocessati.
        progress_callback : callable, optional
            Funzione chiamata per ogni ECG processato: callback(current, total)
            
        Returns
        -------
        pd.DataFrame
            DataFrame con una riga per ECG e una colonna per feature.
        """
        results = []
        n_total = len(ecgs)
        
        if self.config.n_jobs == 1:
            # Processing sequenziale
            for i, ecg in enumerate(ecgs):
                result = self.extract(ecg)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, n_total)
        else:
            # Processing parallelo
            with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
                futures = {executor.submit(self.extract, ecg): ecg.ecg_id 
                          for ecg in ecgs}
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(i + 1, n_total)
        
        # Converti a DataFrame
        rows = []
        for result in results:
            row = result.features.copy()
            row['extraction_time_ms'] = result.extraction_time_ms
            row['extraction_success'] = result.success
            if result.error_message:
                row['extraction_error'] = result.error_message
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Ordina per ecg_id se presente
        if 'ecg_id' in df.columns:
            df = df.sort_values('ecg_id').reset_index(drop=True)
        
        return df
    
    def _prefix_features(self, features: Dict[str, float], prefix: str) -> Dict[str, float]:
        """Aggiunge un prefisso ai nomi delle features."""
        return {f"{prefix}_{k}": v for k, v in features.items()}
    
    def get_feature_names(self) -> List[str]:
        """
        Restituisce la lista di tutti i nomi delle features.
        
        Returns
        -------
        List[str]
            Lista dei nomi delle features che verranno estratte.
        """
        names = []
        
        if self.morphological is not None:
            morph_names = self.morphological.get_feature_names()
            names.extend([f"morph_{n}" for n in morph_names])
        
        if self.intervals is not None:
            # IntervalCalculator usa MorphologicalExtractor internamente
            interval_names = [
                'pr_interval_ms', 'qrs_interval_ms', 'qt_interval_ms',
                'qtc_bazett_ms', 'qtc_fridericia_ms', 'qtc_framingham_ms', 'qtc_hodges_ms'
            ]
            names.extend([f"interval_{n}" for n in interval_names])
        
        if self.spectral is not None:
            spectral_names = self.spectral.get_feature_names()
            names.extend([f"hrv_{n}" for n in spectral_names])
        
        if self.wavelet is not None:
            wavelet_names = self.wavelet.get_feature_names()
            names.extend([f"wav_{n}" for n in wavelet_names])
        
        # Metadata
        names.extend(['ecg_id', 'age', 'sex', 'quality_score', 'heart_rate_bpm',
                     'extraction_time_ms', 'extraction_success'])
        
        return names
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Restituisce statistiche sul processing.
        
        Returns
        -------
        Dict[str, Any]
            Statistiche: n_processed, n_success, n_failed, success_rate
        """
        success_rate = self._n_success / self._n_processed if self._n_processed > 0 else 0
        
        return {
            'n_processed': self._n_processed,
            'n_success': self._n_success,
            'n_failed': self._n_failed,
            'success_rate': success_rate,
        }
    
    def reset_statistics(self):
        """Resetta i contatori delle statistiche."""
        self._n_processed = 0
        self._n_success = 0
        self._n_failed = 0


def create_processed_ecg_from_dict(data: Dict[str, Any]) -> ProcessedECG:
    """
    Crea un ProcessedECG da un dizionario (es. da CSV o JSON).
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dizionario con i campi dell'ECG.
        
    Returns
    -------
    ProcessedECG
        Oggetto ProcessedECG.
    """
    return ProcessedECG(
        ecg_id=data.get('ecg_id', 0),
        signal=data.get('signal'),
        beat_template=data.get('beat_template'),
        rr_intervals_ms=data.get('rr_intervals_ms'),
        r_peaks=data.get('r_peaks'),
        age=data.get('age'),
        sex=data.get('sex'),
        scp_codes=data.get('scp_codes'),
        quality_score=data.get('quality_score'),
        snr_db=data.get('snr_db'),
        heart_rate_bpm=data.get('heart_rate_bpm'),
        rr_mean_ms=data.get('rr_mean_ms'),
        rmssd=data.get('rmssd'),
        pnn50=data.get('pnn50'),
    )


def extract_features_from_raw(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: int = 500,
    ecg_id: int = 0,
    config: Optional[FeatureConfig] = None
) -> FeatureVector:
    """
    Estrae features direttamente da segnale raw e R-peaks.
    
    Funzione di convenienza per casi semplici.
    
    Parameters
    ----------
    signal : np.ndarray
        Segnale ECG filtrato (n_samples,)
    r_peaks : np.ndarray
        Indici dei R-peaks
    sampling_rate : int
        Frequenza di campionamento
    ecg_id : int
        Identificativo dell'ECG
    config : FeatureConfig, optional
        Configurazione pipeline
        
    Returns
    -------
    FeatureVector
        Features estratte
    """
    # Calcola RR intervals
    if len(r_peaks) > 1:
        rr_samples = np.diff(r_peaks)
        rr_ms = rr_samples / sampling_rate * 1000
    else:
        rr_ms = None
    
    # Calcola beat template (media dei beat)
    beat_template = None
    if len(r_peaks) > 2:
        # Estrai beat centrati su R-peak
        beat_half_width = int(0.4 * sampling_rate)  # 400ms per lato
        beats = []
        
        for r_idx in r_peaks[1:-1]:  # Escludi primo e ultimo
            start = r_idx - beat_half_width
            end = r_idx + beat_half_width
            
            if start >= 0 and end < len(signal):
                beats.append(signal[start:end])
        
        if len(beats) > 0:
            # Media dei beat (template)
            min_len = min(len(b) for b in beats)
            beats_aligned = [b[:min_len] for b in beats]
            beat_template = np.mean(beats_aligned, axis=0)
    
    # Crea ProcessedECG
    ecg = ProcessedECG(
        ecg_id=ecg_id,
        signal=signal,
        beat_template=beat_template,
        rr_intervals_ms=rr_ms,
        r_peaks=r_peaks
    )
    
    # Estrai features
    if config is None:
        config = FeatureConfig(sampling_rate=sampling_rate)
    
    pipeline = FeaturePipeline(config)
    return pipeline.extract(ecg)
