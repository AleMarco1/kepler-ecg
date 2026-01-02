"""
Kepler-ECG: Diagnosis Mapper

Task 7 della Fase 2: Mapping dei codici diagnostici SCP-ECG a categorie
standardizzate per l'analisi della compressibilità.

Il dataset PTB-XL usa codici SCP-ECG per le diagnosi. Questo modulo:
- Mappa i codici a 5 categorie principali (NORM, MI, STTC, CD, HYP)
- Estrae diagnosi primaria e secondarie
- Calcola statistiche di compressibilità per categoria

Riferimenti:
- Wagner et al. (2020). PTB-XL, a large publicly available electrocardiography dataset.
- SCP-ECG standard (EN 1064)

Author: Alessandro Marconi for Kepler-ECG Project
Version: 1.0.0
Issued on: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import ast


# ============================================================================
# Costanti per mapping SCP-ECG
# ============================================================================

# Categorie principali PTB-XL (superclassi)
DIAGNOSIS_CATEGORIES = {
    'NORM': 'Normal ECG',
    'MI': 'Myocardial Infarction',
    'STTC': 'ST/T Change',
    'CD': 'Conduction Disturbance',
    'HYP': 'Hypertrophy',
}

# Sottocategorie per ogni categoria principale
DIAGNOSIS_SUBCATEGORIES = {
    'NORM': ['NORM', 'SR'],
    'MI': ['IMI', 'ASMI', 'ILMI', 'AMI', 'ALMI', 'INJAS', 'LMI', 'INJAL', 
           'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI', 'INJIL'],
    'STTC': ['STTC', 'NST_', 'ISC_', 'ISCA', 'ISCI', 'ISCAL', 'ISCIN', 
             'ISCLA', 'ISCAS', 'ISCIL', 'NDT', 'DIG', 'LNGQT', 'APTS', 'EL'],
    'CD': ['LAFB', 'LPFB', 'IRBBB', 'CRBBB', 'CLBBB', 'ILBBB', 'IVCD', 
           '1AVB', '2AVB', '3AVB', 'AVB', 'WPW', 'PSVT', 'PACE'],
    'HYP': ['LVH', 'RVH', 'LAH', 'RAH', 'LAO', 'RAO', 'SEHYP', 'LVOLT'],
}

# Mapping diretto da codice SCP a categoria
SCP_TO_CATEGORY = {}
for category, codes in DIAGNOSIS_SUBCATEGORIES.items():
    for code in codes:
        SCP_TO_CATEGORY[code] = category

# Descrizioni estese dei codici più comuni
SCP_DESCRIPTIONS = {
    'NORM': 'Normal ECG',
    'SR': 'Sinus Rhythm',
    'IMI': 'Inferior Myocardial Infarction',
    'AMI': 'Anterior Myocardial Infarction',
    'LMI': 'Lateral Myocardial Infarction',
    'PMI': 'Posterior Myocardial Infarction',
    'STTC': 'ST/T Change',
    'NST_': 'Non-specific ST Change',
    'ISC_': 'Ischemia',
    'ISCAL': 'Anterolateral Ischemia',
    'ISCIN': 'Inferior Ischemia',
    'LAFB': 'Left Anterior Fascicular Block',
    'LPFB': 'Left Posterior Fascicular Block',
    'IRBBB': 'Incomplete Right Bundle Branch Block',
    'CRBBB': 'Complete Right Bundle Branch Block',
    'CLBBB': 'Complete Left Bundle Branch Block',
    'LVH': 'Left Ventricular Hypertrophy',
    'RVH': 'Right Ventricular Hypertrophy',
    'LAH': 'Left Atrial Hypertrophy',
    'RAH': 'Right Atrial Hypertrophy',
    'AVB': 'Atrioventricular Block',
    '1AVB': 'First Degree AV Block',
    '2AVB': 'Second Degree AV Block',
    '3AVB': 'Third Degree AV Block',
    'WPW': 'Wolff-Parkinson-White Syndrome',
    'AFIB': 'Atrial Fibrillation',
    'AFLT': 'Atrial Flutter',
}

# Priorità delle categorie (per determinare diagnosi primaria)
CATEGORY_PRIORITY = {
    'MI': 1,      # Infarto è più critico
    'CD': 2,      # Disturbi di conduzione
    'HYP': 3,     # Ipertrofia
    'STTC': 4,    # Modifiche ST/T
    'NORM': 5,    # Normale (priorità più bassa)
}


@dataclass
class DiagnosisInfo:
    """Informazioni strutturate su una diagnosi."""
    primary_code: str
    primary_category: str
    primary_description: str
    confidence: float
    all_codes: List[str]
    all_categories: List[str]
    category_confidences: Dict[str, float]
    is_normal: bool
    is_multi_label: bool
    n_diagnoses: int


class DiagnosisMapper:
    """
    Mapper per codici diagnostici SCP-ECG del dataset PTB-XL.
    
    Converte i codici SCP-ECG grezzi in categorie standardizzate
    per l'analisi della compressibilità.
    
    Examples
    --------
    >>> mapper = DiagnosisMapper()
    >>> 
    >>> # Da dizionario di codici
    >>> scp_codes = {'NORM': 100.0, 'SR': 0.0}
    >>> info = mapper.map_scp_codes(scp_codes)
    >>> print(info.primary_category)  # 'NORM'
    >>> 
    >>> # Da DataFrame PTB-XL
    >>> df = pd.read_csv('ptbxl_database.csv')
    >>> df_mapped = mapper.map_dataframe(df)
    """
    
    def __init__(self):
        self.categories = DIAGNOSIS_CATEGORIES
        self.subcategories = DIAGNOSIS_SUBCATEGORIES
        self.scp_to_category = SCP_TO_CATEGORY
        self.descriptions = SCP_DESCRIPTIONS
        self.category_priority = CATEGORY_PRIORITY
    
    def map_scp_codes(
        self, 
        scp_codes: Union[Dict[str, float], str],
        threshold: float = 0.0
    ) -> DiagnosisInfo:
        """
        Mappa codici SCP-ECG a categorie diagnostiche.
        
        Parameters
        ----------
        scp_codes : Dict[str, float] or str
            Dizionario {codice: likelihood} o stringa rappresentante il dict.
            Likelihood tipicamente 0-100 nel PTB-XL.
        threshold : float
            Soglia minima di likelihood per considerare una diagnosi (default: 0).
            
        Returns
        -------
        DiagnosisInfo
            Informazioni strutturate sulla diagnosi.
        """
        # Parse se stringa
        if isinstance(scp_codes, str):
            try:
                scp_codes = ast.literal_eval(scp_codes)
            except (ValueError, SyntaxError):
                return self._empty_diagnosis_info()
        
        if not isinstance(scp_codes, dict) or len(scp_codes) == 0:
            return self._empty_diagnosis_info()
        
        # Filtra per threshold
        codes_filtered = {k: v for k, v in scp_codes.items() 
                         if v > threshold and k in self.scp_to_category}
        
        if len(codes_filtered) == 0:
            # Prova senza filtro se non trova nulla
            codes_filtered = {k: v for k, v in scp_codes.items() 
                             if k in self.scp_to_category}
        
        if len(codes_filtered) == 0:
            return self._empty_diagnosis_info()
        
        # Mappa a categorie
        all_codes = list(codes_filtered.keys())
        all_categories = [self.scp_to_category.get(c, 'UNKNOWN') for c in all_codes]
        
        # Calcola confidence per categoria (media delle likelihood normalizzate)
        category_confidences = {}
        for code, likelihood in codes_filtered.items():
            cat = self.scp_to_category.get(code, 'UNKNOWN')
            if cat not in category_confidences:
                category_confidences[cat] = []
            category_confidences[cat].append(likelihood)
        
        # Media per categoria
        for cat in category_confidences:
            category_confidences[cat] = np.mean(category_confidences[cat])
        
        # Determina diagnosi primaria (priorità + confidence)
        primary_code, primary_category = self._determine_primary(
            codes_filtered, all_categories
        )
        
        primary_description = self.descriptions.get(
            primary_code, 
            self.categories.get(primary_category, 'Unknown')
        )
        
        # Confidence della diagnosi primaria
        confidence = codes_filtered.get(primary_code, 0) / 100.0
        
        return DiagnosisInfo(
            primary_code=primary_code,
            primary_category=primary_category,
            primary_description=primary_description,
            confidence=confidence,
            all_codes=all_codes,
            all_categories=list(set(all_categories)),
            category_confidences=category_confidences,
            is_normal=primary_category == 'NORM',
            is_multi_label=len(set(all_categories)) > 1,
            n_diagnoses=len(all_codes)
        )
    
    def _determine_primary(
        self, 
        codes: Dict[str, float],
        categories: List[str]
    ) -> Tuple[str, str]:
        """Determina diagnosi primaria basandosi su priorità e confidence."""
        # Trova la categoria con priorità più alta (numero più basso)
        unique_categories = set(categories)
        
        best_category = min(
            unique_categories,
            key=lambda c: self.category_priority.get(c, 99)
        )
        
        # Tra i codici di quella categoria, prendi quello con likelihood più alta
        best_code = None
        best_likelihood = -1
        
        for code, likelihood in codes.items():
            if self.scp_to_category.get(code) == best_category:
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
                    best_code = code
        
        return best_code, best_category
    
    def _empty_diagnosis_info(self) -> DiagnosisInfo:
        """Restituisce info vuota per casi non mappabili."""
        return DiagnosisInfo(
            primary_code='UNKNOWN',
            primary_category='UNKNOWN',
            primary_description='Unknown/Not Mapped',
            confidence=0.0,
            all_codes=[],
            all_categories=[],
            category_confidences={},
            is_normal=False,
            is_multi_label=False,
            n_diagnoses=0
        )
    
    def map_dataframe(
        self, 
        df: pd.DataFrame,
        scp_column: str = 'scp_codes',
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Mappa un intero DataFrame PTB-XL.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con colonna dei codici SCP.
        scp_column : str
            Nome della colonna contenente i codici SCP.
        threshold : float
            Soglia minima di likelihood.
            
        Returns
        -------
        pd.DataFrame
            DataFrame originale con colonne aggiuntive per le diagnosi.
        """
        result = df.copy()
        
        # Mappa ogni riga
        diagnoses = []
        for idx, row in df.iterrows():
            scp = row.get(scp_column, {})
            info = self.map_scp_codes(scp, threshold)
            diagnoses.append(info)
        
        # Aggiungi colonne
        result['diagnosis_primary_code'] = [d.primary_code for d in diagnoses]
        result['diagnosis_primary_category'] = [d.primary_category for d in diagnoses]
        result['diagnosis_description'] = [d.primary_description for d in diagnoses]
        result['diagnosis_confidence'] = [d.confidence for d in diagnoses]
        result['diagnosis_is_normal'] = [d.is_normal for d in diagnoses]
        result['diagnosis_is_multi_label'] = [d.is_multi_label for d in diagnoses]
        result['diagnosis_n_diagnoses'] = [d.n_diagnoses for d in diagnoses]
        result['diagnosis_all_categories'] = [','.join(d.all_categories) for d in diagnoses]
        
        return result
    
    def get_category_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Conta le occorrenze per categoria.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con colonna 'diagnosis_primary_category'.
            
        Returns
        -------
        pd.DataFrame
            Conteggi per categoria.
        """
        if 'diagnosis_primary_category' not in df.columns:
            df = self.map_dataframe(df)
        
        counts = df['diagnosis_primary_category'].value_counts()
        
        result = pd.DataFrame({
            'category': counts.index,
            'count': counts.values,
            'percentage': (counts.values / len(df) * 100).round(2)
        })
        
        # Aggiungi descrizione
        result['description'] = result['category'].map(
            lambda c: self.categories.get(c, 'Unknown')
        )
        
        return result
    
    def get_diagnosis_compressibility(
        self, 
        df: pd.DataFrame,
        compressibility_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calcola statistiche di compressibilità per categoria diagnostica.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con diagnosi e metriche di compressibilità.
        compressibility_columns : List[str], optional
            Colonne di compressibilità da analizzare.
            Se None, cerca colonne che contengono 'compress', 'entropy', 'lz'.
            
        Returns
        -------
        pd.DataFrame
            Statistiche di compressibilità per categoria.
        """
        # Assicurati che le diagnosi siano mappate
        if 'diagnosis_primary_category' not in df.columns:
            df = self.map_dataframe(df)
        
        # Identifica colonne di compressibilità
        if compressibility_columns is None:
            compressibility_columns = [
                c for c in df.columns 
                if any(term in c.lower() for term in 
                       ['compress', 'entropy', 'lempel', 'hjorth', 'kolmogorov', 'gzip', 'bzip', 'lzma'])
            ]
        
        if len(compressibility_columns) == 0:
            return pd.DataFrame()
        
        # Calcola statistiche per categoria
        results = []
        
        for category in df['diagnosis_primary_category'].unique():
            if category == 'UNKNOWN':
                continue
                
            mask = df['diagnosis_primary_category'] == category
            subset = df.loc[mask, compressibility_columns]
            
            row = {'category': category, 'n_samples': mask.sum()}
            
            for col in compressibility_columns:
                values = subset[col].dropna()
                if len(values) > 0:
                    row[f'{col}_mean'] = values.mean()
                    row[f'{col}_std'] = values.std()
                    row[f'{col}_median'] = values.median()
                else:
                    row[f'{col}_mean'] = np.nan
                    row[f'{col}_std'] = np.nan
                    row[f'{col}_median'] = np.nan
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def filter_by_category(
        self, 
        df: pd.DataFrame, 
        categories: Union[str, List[str]]
    ) -> pd.DataFrame:
        """
        Filtra DataFrame per categoria diagnostica.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con diagnosi mappate.
        categories : str or List[str]
            Categoria/e da selezionare.
            
        Returns
        -------
        pd.DataFrame
            Subset filtrato.
        """
        if 'diagnosis_primary_category' not in df.columns:
            df = self.map_dataframe(df)
        
        if isinstance(categories, str):
            categories = [categories]
        
        return df[df['diagnosis_primary_category'].isin(categories)].copy()
    
    def get_code_description(self, code: str) -> str:
        """Restituisce la descrizione di un codice SCP."""
        return self.descriptions.get(code, f'Unknown code: {code}')
    
    def get_category_for_code(self, code: str) -> str:
        """Restituisce la categoria per un codice SCP."""
        return self.scp_to_category.get(code, 'UNKNOWN')
    
    def get_all_codes_for_category(self, category: str) -> List[str]:
        """Restituisce tutti i codici SCP per una categoria."""
        return self.subcategories.get(category, [])
    
    def summary(self) -> Dict[str, Any]:
        """Restituisce un sommario del mapping."""
        return {
            'n_categories': len(self.categories),
            'categories': list(self.categories.keys()),
            'n_subcategories': sum(len(v) for v in self.subcategories.values()),
            'n_known_codes': len(self.scp_to_category),
            'n_descriptions': len(self.descriptions),
        }


def create_diagnosis_features(df: pd.DataFrame, scp_column: str = 'scp_codes') -> pd.DataFrame:
    """
    Funzione di convenienza per creare features diagnostiche.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame PTB-XL o simile.
    scp_column : str
        Nome colonna con codici SCP.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con features diagnostiche aggiunte.
    """
    mapper = DiagnosisMapper()
    return mapper.map_dataframe(df, scp_column)
