"""
Kepler-ECG: Compressibility Calculator

Task 6 della Fase 2: Calcolo di metriche di compressibilità per segnali ECG.

La compressibilità è una proxy della complessità algoritmica (Kolmogorov).
Segnali più regolari/predicibili sono più comprimibili.

Metriche implementate:
- Compression ratios (gzip, bzip2, lzma)
- Sample Entropy (SampEn)
- Approximate Entropy (ApEn)  
- Permutation Entropy (PE)
- Lempel-Ziv Complexity (LZC)
- Stima complessità Kolmogorov

Riferimenti:
- Richman & Moorman (2000). Physiological time-series analysis using ApEn and SampEn.
- Bandt & Pompe (2002). Permutation entropy.
- Lempel & Ziv (1976). On the complexity of finite sequences.

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

import numpy as np
import gzip
import bz2
import lzma
from typing import Dict, Optional, List, Literal, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class CompressibilityConfig:
    """Configurazione per il calcolo della compressibilità."""
    
    # Sample/Approximate Entropy
    entropy_m: int = 2           # Embedding dimension
    entropy_r_factor: float = 0.2  # r = r_factor * std(signal)
    
    # Permutation Entropy
    perm_order: int = 3          # Ordine permutazione (3-7)
    perm_delay: int = 1          # Delay embedding
    
    # Lempel-Ziv
    lz_threshold: str = 'median'  # 'median', 'mean', o valore numerico
    
    # Quantizzazione per compressione
    quantize_bits: int = 8       # Bit per la quantizzazione


class CompressibilityCalculator:
    """
    Calcolatore di metriche di compressibilità per segnali ECG.
    
    Implementa diverse misure di complessità/regolarità:
    - Compression-based: gzip, bzip2, lzma ratios
    - Entropy-based: Sample Entropy, Approximate Entropy, Permutation Entropy
    - Complexity-based: Lempel-Ziv Complexity
    
    Parameters
    ----------
    config : CompressibilityConfig, optional
        Configurazione dei parametri. Default se None.
    
    Examples
    --------
    >>> calc = CompressibilityCalculator()
    >>> metrics = calc.calculate(ecg_signal)
    >>> print(f"Gzip ratio: {metrics['gzip_ratio']:.3f}")
    >>> print(f"Sample Entropy: {metrics['sample_entropy']:.3f}")
    """
    
    def __init__(self, config: Optional[CompressibilityConfig] = None):
        self.config = config or CompressibilityConfig()
    
    def calculate(
        self, 
        signal: np.ndarray,
        methods: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calcola tutte le metriche di compressibilità per un segnale.
        
        Parameters
        ----------
        signal : np.ndarray
            Segnale 1D (ECG, RR intervals, o qualsiasi serie temporale)
        methods : List[str], optional
            Lista di metodi da calcolare. Se None, calcola tutti.
            Opzioni: 'compression', 'entropy', 'complexity', 'all'
            
        Returns
        -------
        Dict[str, float]
            Dizionario con tutte le metriche calcolate.
        """
        if methods is None:
            methods = ['all']
        
        if 'all' in methods:
            methods = ['compression', 'entropy', 'complexity']
        
        features = {}
        
        # Validazione input
        if not self._validate_input(signal):
            return self._empty_features()
        
        sig = np.asarray(signal, dtype=np.float64).flatten()
        
        # Rimuovi NaN
        sig = sig[~np.isnan(sig)]
        if len(sig) < 10:
            return self._empty_features()
        
        # Metriche di compressione
        if 'compression' in methods:
            features.update(self._compute_compression_ratios(sig))
        
        # Metriche di entropia
        if 'entropy' in methods:
            features.update(self._compute_entropy_metrics(sig))
        
        # Metriche di complessità
        if 'complexity' in methods:
            features.update(self._compute_complexity_metrics(sig))
        
        # Stima Kolmogorov (basata su compressione)
        if 'compression' in methods:
            features['kolmogorov_estimate'] = features.get('lzma_ratio', np.nan)
        
        return features
    
    def calculate_for_rr(self, rr_intervals_ms: np.ndarray) -> Dict[str, float]:
        """
        Calcola metriche di compressibilità specifiche per serie RR.
        
        Le serie RR hanno caratteristiche particolari:
        - Valori tipicamente tra 300-1500 ms
        - La variabilità è il segnale di interesse
        
        Parameters
        ----------
        rr_intervals_ms : np.ndarray
            Serie di intervalli RR in millisecondi.
            
        Returns
        -------
        Dict[str, float]
            Metriche di compressibilità per HRV.
        """
        features = {}
        
        if not self._validate_input(rr_intervals_ms):
            return self._empty_rr_features()
        
        rr = np.asarray(rr_intervals_ms, dtype=np.float64).flatten()
        rr = rr[~np.isnan(rr)]
        
        # Filtra valori non fisiologici
        rr = rr[(rr > 200) & (rr < 2000)]
        
        if len(rr) < 10:
            return self._empty_rr_features()
        
        # Calcola su RR
        base_metrics = self.calculate(rr)
        features.update({f'rr_{k}': v for k, v in base_metrics.items()})
        
        # Calcola anche su differenze RR (variabilità beat-to-beat)
        if len(rr) > 1:
            rr_diff = np.diff(rr)
            diff_metrics = self.calculate(rr_diff, methods=['entropy', 'complexity'])
            features.update({f'rr_diff_{k}': v for k, v in diff_metrics.items()})
        
        return features
    
    def _validate_input(self, signal: np.ndarray) -> bool:
        """Valida l'input."""
        if signal is None:
            return False
        
        sig = np.asarray(signal)
        
        if sig.ndim != 1:
            return False
        
        if len(sig) < 10:
            return False
        
        if np.all(np.isnan(sig)):
            return False
        
        return True
    
    def _compute_compression_ratios(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Calcola i rapporti di compressione con diversi algoritmi.
        
        Il segnale viene quantizzato a 8 bit prima della compressione.
        """
        features = {}
        
        # Quantizza il segnale
        quantized = self._quantize_signal(signal)
        original_bytes = quantized.tobytes()
        original_size = len(original_bytes)
        
        # Gzip
        try:
            compressed = gzip.compress(original_bytes, compresslevel=9)
            features['gzip_ratio'] = len(compressed) / original_size
        except Exception:
            features['gzip_ratio'] = np.nan
        
        # Bzip2
        try:
            compressed = bz2.compress(original_bytes, compresslevel=9)
            features['bzip2_ratio'] = len(compressed) / original_size
        except Exception:
            features['bzip2_ratio'] = np.nan
        
        # LZMA (migliore compressione, proxy per Kolmogorov)
        try:
            compressed = lzma.compress(original_bytes, preset=9)
            features['lzma_ratio'] = len(compressed) / original_size
        except Exception:
            features['lzma_ratio'] = np.nan
        
        # Media dei rapporti (robustezza)
        ratios = [features['gzip_ratio'], features['bzip2_ratio'], features['lzma_ratio']]
        valid_ratios = [r for r in ratios if not np.isnan(r)]
        features['compression_ratio_mean'] = np.mean(valid_ratios) if valid_ratios else np.nan
        
        return features
    
    def _quantize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Quantizza il segnale a N bit."""
        bits = self.config.quantize_bits
        levels = 2 ** bits
        
        # Normalizza a [0, 1]
        sig_min = np.min(signal)
        sig_max = np.max(signal)
        
        if sig_max == sig_min:
            return np.zeros(len(signal), dtype=np.uint8)
        
        normalized = (signal - sig_min) / (sig_max - sig_min)
        
        # Quantizza
        quantized = np.floor(normalized * (levels - 1)).astype(np.uint8)
        
        return quantized
    
    def _compute_entropy_metrics(self, signal: np.ndarray) -> Dict[str, float]:
        """Calcola metriche basate sull'entropia."""
        features = {}
        
        # Sample Entropy
        try:
            features['sample_entropy'] = self._sample_entropy(signal)
        except Exception:
            features['sample_entropy'] = np.nan
        
        # Approximate Entropy
        try:
            features['approx_entropy'] = self._approximate_entropy(signal)
        except Exception:
            features['approx_entropy'] = np.nan
        
        # Permutation Entropy
        try:
            features['permutation_entropy'] = self._permutation_entropy(signal)
        except Exception:
            features['permutation_entropy'] = np.nan
        
        # Entropia di Shannon (sulla distribuzione)
        try:
            features['shannon_entropy'] = self._shannon_entropy(signal)
        except Exception:
            features['shannon_entropy'] = np.nan
        
        return features
    
    def _sample_entropy(self, signal: np.ndarray) -> float:
        """
        Calcola Sample Entropy (SampEn).
        
        SampEn misura la regolarità/predicibilità di una serie temporale.
        Valori bassi = più regolare, valori alti = più complesso.
        
        SampEn = -log(A/B)
        dove A = numero di template matches di lunghezza m+1
             B = numero di template matches di lunghezza m
        """
        m = self.config.entropy_m
        r = self.config.entropy_r_factor * np.std(signal)
        
        N = len(signal)
        
        if N < m + 2:
            return np.nan
        
        # Conta matches per m e m+1
        B = self._count_matches(signal, m, r)
        A = self._count_matches(signal, m + 1, r)
        
        if B == 0 or A == 0:
            return np.nan
        
        return -np.log(A / B)
    
    def _approximate_entropy(self, signal: np.ndarray) -> float:
        """
        Calcola Approximate Entropy (ApEn).
        
        Simile a SampEn ma include self-matches.
        """
        m = self.config.entropy_m
        r = self.config.entropy_r_factor * np.std(signal)
        
        N = len(signal)
        
        if N < m + 2:
            return np.nan
        
        def phi(m_val):
            templates = np.array([signal[i:i+m_val] for i in range(N - m_val + 1)])
            C = np.zeros(len(templates))
            
            for i, template in enumerate(templates):
                # Conta matches (incluso self-match)
                distances = np.max(np.abs(templates - template), axis=1)
                C[i] = np.sum(distances <= r) / (N - m_val + 1)
            
            return np.mean(np.log(C[C > 0]))
        
        return phi(m) - phi(m + 1)
    
    def _count_matches(self, signal: np.ndarray, m: int, r: float) -> int:
        """Conta il numero di template matches (per SampEn)."""
        N = len(signal)
        templates = np.array([signal[i:i+m] for i in range(N - m)])
        
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                # Distanza Chebyshev (max della differenza assoluta)
                dist = np.max(np.abs(templates[i] - templates[j]))
                if dist <= r:
                    count += 1
        
        return count
    
    def _permutation_entropy(self, signal: np.ndarray) -> float:
        """
        Calcola Permutation Entropy (PE).
        
        PE misura la complessità basata sui pattern ordinali.
        """
        order = self.config.perm_order
        delay = self.config.perm_delay
        
        N = len(signal)
        
        if N < order + (order - 1) * delay:
            return np.nan
        
        # Estrai pattern embedded
        n_patterns = N - (order - 1) * delay
        
        # Costruisci pattern e calcola permutazioni
        perm_list = []
        for i in range(n_patterns):
            # Estrai il pattern con delay
            indices = [i + j * delay for j in range(order)]
            if indices[-1] >= N:
                break
            pattern = signal[indices]
            # Converti in permutazione (tuple di ranking)
            perm = tuple(np.argsort(np.argsort(pattern)))
            perm_list.append(perm)
        
        if len(perm_list) == 0:
            return np.nan
        
        # Conta frequenze delle permutazioni uniche
        from collections import Counter
        perm_counts = Counter(perm_list)
        
        # Calcola probabilità
        total = len(perm_list)
        probs = np.array(list(perm_counts.values())) / total
        
        # Entropia di Shannon
        pe = -np.sum(probs * np.log2(probs))
        
        # Normalizza rispetto al massimo (log2(order!))
        import math
        max_entropy = np.log2(math.factorial(order))
        pe_normalized = pe / max_entropy if max_entropy > 0 else 0
        
        return pe_normalized
    
    def _shannon_entropy(self, signal: np.ndarray, n_bins: int = 50) -> float:
        """Calcola entropia di Shannon sulla distribuzione dei valori."""
        # Istogramma
        hist, _ = np.histogram(signal, bins=n_bins, density=True)
        
        # Normalizza
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        
        # Entropia
        entropy = -np.sum(hist * np.log2(hist))
        
        # Normalizza rispetto al massimo
        max_entropy = np.log2(n_bins)
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _compute_complexity_metrics(self, signal: np.ndarray) -> Dict[str, float]:
        """Calcola metriche di complessità."""
        features = {}
        
        # Lempel-Ziv Complexity
        try:
            features['lempel_ziv_complexity'] = self._lempel_ziv_complexity(signal)
        except Exception:
            features['lempel_ziv_complexity'] = np.nan
        
        # Complessità di Hjorth
        try:
            activity, mobility, complexity = self._hjorth_parameters(signal)
            features['hjorth_activity'] = activity
            features['hjorth_mobility'] = mobility
            features['hjorth_complexity'] = complexity
        except Exception:
            features['hjorth_activity'] = np.nan
            features['hjorth_mobility'] = np.nan
            features['hjorth_complexity'] = np.nan
        
        return features
    
    def _lempel_ziv_complexity(self, signal: np.ndarray) -> float:
        """
        Calcola Lempel-Ziv Complexity (LZC).
        
        Misura il numero di "pattern unici" nella sequenza binarizzata.
        """
        # Binarizza rispetto alla soglia
        if self.config.lz_threshold == 'median':
            threshold = np.median(signal)
        elif self.config.lz_threshold == 'mean':
            threshold = np.mean(signal)
        else:
            threshold = float(self.config.lz_threshold)
        
        binary = (signal >= threshold).astype(int)
        
        # Converti in stringa
        s = ''.join(map(str, binary))
        
        # Algoritmo LZ76
        n = len(s)
        i = 0
        c = 1  # Complessità
        k = 1
        l = 1
        
        while True:
            if s[i:i+l] in s[0:i+l-1]:
                l += 1
            else:
                c += 1
                i += l
                l = 1
            
            if i + l > n:
                break
        
        # Normalizza
        if n > 0:
            b = n / np.log2(n) if n > 1 else 1
            lzc = c / b
        else:
            lzc = 0
        
        return lzc
    
    def _hjorth_parameters(self, signal: np.ndarray) -> Tuple[float, float, float]:
        """
        Calcola i parametri di Hjorth.
        
        - Activity: varianza del segnale
        - Mobility: deviazione standard della derivata / std del segnale
        - Complexity: mobility della derivata / mobility del segnale
        """
        # Activity
        activity = np.var(signal)
        
        # Prima derivata
        d1 = np.diff(signal)
        
        # Seconda derivata
        d2 = np.diff(d1)
        
        # Mobility
        if activity > 0:
            mobility = np.sqrt(np.var(d1) / activity)
        else:
            mobility = 0
        
        # Complexity
        if np.var(d1) > 0:
            mobility_d1 = np.sqrt(np.var(d2) / np.var(d1))
            complexity = mobility_d1 / mobility if mobility > 0 else 0
        else:
            complexity = 0
        
        return activity, mobility, complexity
    
    def _empty_features(self) -> Dict[str, float]:
        """Restituisce features vuote."""
        return {
            'gzip_ratio': np.nan,
            'bzip2_ratio': np.nan,
            'lzma_ratio': np.nan,
            'compression_ratio_mean': np.nan,
            'sample_entropy': np.nan,
            'approx_entropy': np.nan,
            'permutation_entropy': np.nan,
            'shannon_entropy': np.nan,
            'lempel_ziv_complexity': np.nan,
            'hjorth_activity': np.nan,
            'hjorth_mobility': np.nan,
            'hjorth_complexity': np.nan,
            'kolmogorov_estimate': np.nan,
        }
    
    def _empty_rr_features(self) -> Dict[str, float]:
        """Restituisce features RR vuote."""
        base = self._empty_features()
        rr_features = {f'rr_{k}': np.nan for k in base.keys()}
        
        # Aggiungi features per diff
        diff_keys = ['sample_entropy', 'approx_entropy', 'permutation_entropy', 
                     'shannon_entropy', 'lempel_ziv_complexity', 
                     'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']
        rr_features.update({f'rr_diff_{k}': np.nan for k in diff_keys})
        
        return rr_features
    
    def get_feature_names(self) -> List[str]:
        """Restituisce i nomi delle features."""
        return list(self._empty_features().keys())
    
    def get_rr_feature_names(self) -> List[str]:
        """Restituisce i nomi delle features RR."""
        return list(self._empty_rr_features().keys())
