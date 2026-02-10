#!/usr/bin/env python3
"""
06_0_sr_qtc_validation.py - Validazione Statistica Formule QTc

Script per validare le formule QTc scoperte da 05_0_sr_qtc_discovery.py.
Esegue analisi statistiche complete, confronto con formule standard,
valutazione di utilità clinica, plausibilità fisiologica e interesse scientifico.

Versione: 1.0.0
Autore: Alessandro Marconi
Progetto: Kepler-ECG
Data: 28 Gennaio 2026
"""

import argparse
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

VERSION = "1.0.0"

# Datasets supportati
SUPPORTED_DATASETS = [
    "pool6",
    "ptb-xl",
    "chapman",
    "cpsc-2018",
    "georgia",
    "code-15",
    "mimic-iv-ecg",
]

# Formule standard di riferimento
STANDARD_FORMULAS = {
    "Bazett": {
        "year": 1920,
        "equation": "QT / sqrt(RR)",
        "formula": lambda qt, rr, hr: qt / np.sqrt(rr),
        "column": "QTc_Bazett_ms",
        "type": "factor",
    },
    "Fridericia": {
        "year": 1920,
        "equation": "QT / cbrt(RR)",
        "formula": lambda qt, rr, hr: qt / np.cbrt(rr),
        "column": "QTc_Fridericia_ms",
        "type": "factor",
    },
    "Framingham": {
        "year": 1992,
        "equation": "QT + 154*(1 - RR)",
        "formula": lambda qt, rr, hr: qt + 154 * (1 - rr),
        "column": "QTc_Framingham_ms",
        "type": "additive",
    },
    "Hodges": {
        "year": 1983,
        "equation": "QT + 1.75*(HR - 60)",
        "formula": lambda qt, rr, hr: qt + 1.75 * (hr - 60),
        "column": "QTc_Hodges_ms",
        "type": "additive",
    },
}

# Soglie cliniche per QTc (ms)
CLINICAL_THRESHOLDS = {
    "normal_upper_male": 450,
    "normal_upper_female": 460,
    "prolonged_moderate": 480,
    "prolonged_severe": 500,
    "short_lower": 340,
}

# Range HR per analisi stratificata
HR_BINS = [
    (30, 50, "Bradicardia severa"),
    (50, 60, "Bradicardia"),
    (60, 80, "Normale"),
    (80, 100, "Tachicardia lieve"),
    (100, 120, "Tachicardia"),
    (120, 150, "Tachicardia severa"),
]

# Soglie qualità per validazione
QUALITY_THRESHOLDS = {
    "hr_correlation_excellent": 0.02,
    "hr_correlation_good": 0.05,
    "hr_correlation_acceptable": 0.10,
    "stability_cv_max": 0.10,  # Coefficiente di variazione massimo per stabilità
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def log_info(msg: str) -> None:
    """Log informativo."""
    print(f"[INFO] {msg}")


def log_warning(msg: str) -> None:
    """Log warning."""
    print(f"[WARNING] {msg}")


def log_error(msg: str) -> None:
    """Log errore."""
    print(f"[ERROR] {msg}", file=sys.stderr)


def safe_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Calcola correlazione Pearson in modo sicuro."""
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    if mask.sum() < 10:
        return np.nan, np.nan
    try:
        r, p = stats.pearsonr(x[mask], y[mask])
        return r, p
    except Exception:
        return np.nan, np.nan


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Calcola correlazione Spearman in modo sicuro."""
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    if mask.sum() < 10:
        return np.nan, np.nan
    try:
        r, p = stats.spearmanr(x[mask], y[mask])
        return r, p
    except Exception:
        return np.nan, np.nan


def compute_metrics(
    qtc_values: np.ndarray, reference: np.ndarray, hr: np.ndarray
) -> Dict[str, float]:
    """Calcola metriche complete per una formula QTc."""
    mask = ~(
        np.isnan(qtc_values)
        | np.isnan(reference)
        | np.isnan(hr)
        | np.isinf(qtc_values)
        | np.isinf(reference)
    )
    qtc = qtc_values[mask]
    ref = reference[mask]
    hr_valid = hr[mask]

    if len(qtc) < 10:
        return {
            "n_valid": 0,
            "r_vs_hr": np.nan,
            "r_vs_hr_p": np.nan,
            "spearman_vs_hr": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "cv": np.nan,
        }

    r_hr, p_hr = safe_correlation(qtc, hr_valid)
    rho_hr, _ = safe_spearman(qtc, hr_valid)

    errors = qtc - ref
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mean_qtc = np.mean(qtc)
    std_qtc = np.std(qtc)
    cv = std_qtc / mean_qtc if mean_qtc > 0 else np.nan

    return {
        "n_valid": len(qtc),
        "r_vs_hr": r_hr,
        "r_vs_hr_p": p_hr,
        "spearman_vs_hr": rho_hr,
        "mae": mae,
        "rmse": rmse,
        "mean": mean_qtc,
        "std": std_qtc,
        "cv": cv,
    }


def parse_pysr_equation(
    equation: str, approach: str
) -> Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
    """
    Converte un'equazione PySR in una funzione Python callable.

    Args:
        equation: Stringa equazione da PySR
        approach: 'direct', 'factor', o 'additive'

    Returns:
        Funzione che accetta (QT, RR, HR) e ritorna QTc
    """
    try:
        # Normalizza l'equazione
        eq = equation.strip()

        # Sostituisci variabili PySR con nomi Python
        eq = re.sub(r"\bQT_interval_ms\b", "QT", eq)
        eq = re.sub(r"\bRR_interval_sec\b", "RR", eq)
        eq = re.sub(r"\bheart_rate_bpm\b", "HR", eq)

        # Sostituisci operatori
        eq = eq.replace("^", "**")
        eq = re.sub(r"\bsqrt\(", "np.sqrt(", eq)
        eq = re.sub(r"\bcbrt\(", "np.cbrt(", eq)
        eq = re.sub(r"\binv\(([^)]+)\)", r"(1.0/(\1))", eq)
        eq = re.sub(r"\bsquare\(([^)]+)\)", r"((\1)**2)", eq)
        eq = re.sub(r"\bcube\(([^)]+)\)", r"((\1)**3)", eq)
        eq = re.sub(r"\bexp\(", "np.exp(", eq)
        eq = re.sub(r"\blog\(", "np.log(", eq)
        eq = re.sub(r"\babs\(", "np.abs(", eq)

        # Costruisci la funzione completa basata sull'approccio
        if approach == "direct":
            full_eq = eq
        elif approach == "factor":
            # QTc = QT * f(RR), quindi f(RR) è l'equazione
            full_eq = f"QT * ({eq})"
        elif approach == "additive":
            # QTc = QT + f(RR), quindi f(RR) è l'equazione
            full_eq = f"QT + ({eq})"
        else:
            full_eq = eq

        # Crea funzione
        def formula_func(
            qt: np.ndarray, rr: np.ndarray, hr: np.ndarray, _eq=full_eq
        ) -> np.ndarray:
            QT = qt  # noqa: N806
            RR = rr  # noqa: N806
            HR = hr  # noqa: N806
            return eval(_eq)

        # Test su valori tipici
        test_qt = np.array([400.0])
        test_rr = np.array([1.0])
        test_hr = np.array([60.0])
        result = formula_func(test_qt, test_rr, test_hr)
        if np.isnan(result[0]) or np.isinf(result[0]):
            return None

        return formula_func

    except Exception as e:
        log_warning(f"Impossibile parsare equazione '{equation}': {e}")
        return None


# =============================================================================
# VALIDATION CLASSES
# =============================================================================


class FormulaValidator:
    """Classe per validare una singola formula QTc."""

    def __init__(
        self,
        name: str,
        equation: str,
        approach: str,
        formula_func: Optional[Callable] = None,
        is_standard: bool = False,
        column: Optional[str] = None,
    ):
        self.name = name
        self.equation = equation
        self.approach = approach
        self.is_standard = is_standard
        self.column = column

        if formula_func is not None:
            self.formula_func = formula_func
        else:
            self.formula_func = parse_pysr_equation(equation, approach)

        self.is_valid = self.formula_func is not None

    def compute_qtc(
        self, qt: np.ndarray, rr: np.ndarray, hr: np.ndarray
    ) -> np.ndarray:
        """Calcola QTc usando la formula."""
        if not self.is_valid:
            return np.full_like(qt, np.nan)
        try:
            return self.formula_func(qt, rr, hr)
        except Exception:
            return np.full_like(qt, np.nan)


class DatasetValidator:
    """Classe per validare formule su un singolo dataset."""

    def __init__(self, name: str, data: pd.DataFrame):
        self.name = name
        self.data = data.copy()

        # Colonne essenziali
        self.qt = data["QT_interval_ms"].values
        self.rr = data["RR_interval_sec"].values
        self.hr = data["heart_rate_bpm"].values

        # Reference se disponibile
        if "QTc_reference_ms" in data.columns:
            self.reference = data["QTc_reference_ms"].values
        else:
            # Usa Fridericia come reference se QTc_reference non disponibile
            self.reference = self.qt / np.cbrt(self.rr)

        # Metadata
        self.age = data["age"].values if "age" in data.columns else None
        self.sex = data["sex"].values if "sex" in data.columns else None

        # Diagnostica se disponibile
        self.superclass = (
            data["primary_superclass"].values
            if "primary_superclass" in data.columns
            else None
        )

        self.n_records = len(data)

    def validate_formula(self, formula: FormulaValidator) -> Dict[str, Any]:
        """Valida una formula su questo dataset."""
        if formula.column and formula.column in self.data.columns:
            qtc = self.data[formula.column].values
        else:
            qtc = formula.compute_qtc(self.qt, self.rr, self.hr)

        # Metriche globali
        metrics = compute_metrics(qtc, self.reference, self.hr)

        # Analisi stratificata per HR
        hr_analysis = self._analyze_by_hr_bins(qtc)

        # Analisi per sesso (se disponibile)
        sex_analysis = None
        if self.sex is not None:
            sex_analysis = self._analyze_by_sex(qtc)

        # Analisi per età (se disponibile)
        age_analysis = None
        if self.age is not None:
            age_analysis = self._analyze_by_age(qtc)

        # Analisi per diagnosi (se disponibile)
        diag_analysis = None
        if self.superclass is not None:
            diag_analysis = self._analyze_by_diagnosis(qtc)

        return {
            "global_metrics": metrics,
            "hr_stratified": hr_analysis,
            "sex_stratified": sex_analysis,
            "age_stratified": age_analysis,
            "diagnosis_stratified": diag_analysis,
        }

    def _analyze_by_hr_bins(self, qtc: np.ndarray) -> Dict[str, Any]:
        """Analizza QTc stratificato per range di HR."""
        results = {}
        for hr_min, hr_max, label in HR_BINS:
            mask = (self.hr >= hr_min) & (self.hr < hr_max)
            n = mask.sum()
            if n < 10:
                results[label] = {"n": n, "mean": np.nan, "std": np.nan, "r_vs_hr": np.nan}
                continue

            qtc_bin = qtc[mask]
            hr_bin = self.hr[mask]
            ref_bin = self.reference[mask]

            valid = ~(np.isnan(qtc_bin) | np.isinf(qtc_bin))
            if valid.sum() < 10:
                results[label] = {"n": n, "mean": np.nan, "std": np.nan, "r_vs_hr": np.nan}
                continue

            r_hr, _ = safe_correlation(qtc_bin[valid], hr_bin[valid])

            results[label] = {
                "n": int(valid.sum()),
                "mean": float(np.nanmean(qtc_bin[valid])),
                "std": float(np.nanstd(qtc_bin[valid])),
                "r_vs_hr": float(r_hr) if not np.isnan(r_hr) else None,
            }

        return results

    def _analyze_by_sex(self, qtc: np.ndarray) -> Dict[str, Any]:
        """Analizza QTc stratificato per sesso."""
        results = {}
        for sex_val, sex_label in [(0, "Male"), (1, "Female")]:
            mask = self.sex == sex_val
            n = mask.sum()
            if n < 10:
                results[sex_label] = {
                    "n": n,
                    "mean": np.nan,
                    "std": np.nan,
                    "r_vs_hr": np.nan,
                }
                continue

            qtc_sex = qtc[mask]
            hr_sex = self.hr[mask]

            valid = ~(np.isnan(qtc_sex) | np.isinf(qtc_sex))
            if valid.sum() < 10:
                results[sex_label] = {
                    "n": n,
                    "mean": np.nan,
                    "std": np.nan,
                    "r_vs_hr": np.nan,
                }
                continue

            r_hr, _ = safe_correlation(qtc_sex[valid], hr_sex[valid])

            results[sex_label] = {
                "n": int(valid.sum()),
                "mean": float(np.nanmean(qtc_sex[valid])),
                "std": float(np.nanstd(qtc_sex[valid])),
                "r_vs_hr": float(r_hr) if not np.isnan(r_hr) else None,
            }

        return results

    def _analyze_by_age(self, qtc: np.ndarray) -> Dict[str, Any]:
        """Analizza QTc stratificato per fasce d'età."""
        age_bins = [(0, 40, "<40"), (40, 60, "40-60"), (60, 80, "60-80"), (80, 150, ">80")]
        results = {}

        for age_min, age_max, label in age_bins:
            mask = (self.age >= age_min) & (self.age < age_max) & ~np.isnan(self.age)
            n = mask.sum()
            if n < 10:
                results[label] = {"n": n, "mean": np.nan, "std": np.nan, "r_vs_hr": np.nan}
                continue

            qtc_age = qtc[mask]
            hr_age = self.hr[mask]

            valid = ~(np.isnan(qtc_age) | np.isinf(qtc_age))
            if valid.sum() < 10:
                results[label] = {"n": n, "mean": np.nan, "std": np.nan, "r_vs_hr": np.nan}
                continue

            r_hr, _ = safe_correlation(qtc_age[valid], hr_age[valid])

            results[label] = {
                "n": int(valid.sum()),
                "mean": float(np.nanmean(qtc_age[valid])),
                "std": float(np.nanstd(qtc_age[valid])),
                "r_vs_hr": float(r_hr) if not np.isnan(r_hr) else None,
            }

        return results

    def _analyze_by_diagnosis(self, qtc: np.ndarray) -> Dict[str, Any]:
        """Analizza QTc stratificato per diagnosi."""
        results = {}
        
        # Gestisci array con mix di stringhe e NaN
        # Converti in Series pandas per usare dropna() e unique()
        superclass_series = pd.Series(self.superclass)
        unique_diag = superclass_series.dropna().unique().tolist()

        for diag in unique_diag:
            mask = self.superclass == diag
            n = mask.sum()
            if n < 10:
                continue

            qtc_diag = qtc[mask]
            hr_diag = self.hr[mask]

            valid = ~(np.isnan(qtc_diag) | np.isinf(qtc_diag))
            if valid.sum() < 10:
                continue

            r_hr, _ = safe_correlation(qtc_diag[valid], hr_diag[valid])

            results[diag] = {
                "n": int(valid.sum()),
                "mean": float(np.nanmean(qtc_diag[valid])),
                "std": float(np.nanstd(qtc_diag[valid])),
                "r_vs_hr": float(r_hr) if not np.isnan(r_hr) else None,
            }

        return results


# =============================================================================
# CLINICAL UTILITY ANALYSIS
# =============================================================================


def analyze_clinical_utility(
    data: pd.DataFrame,
    formula_qtc: np.ndarray,
    formula_name: str,
    reference_formula: str = "Bazett",
) -> Dict[str, Any]:
    """
    Analizza l'utilità clinica di una formula QTc.

    Valuta:
    - Reclassificazioni cliniche rispetto a Bazett
    - Falsi positivi/negativi per QT prolungato
    - Stabilità delle misurazioni

    Args:
        data: DataFrame con dati ECG
        formula_qtc: Array di valori QTc calcolati con la formula candidata
        formula_name: Nome della formula candidata
        reference_formula: Formula di riferimento per confronto

    Returns:
        Dizionario con analisi di utilità clinica
    """
    # Ottieni QTc di riferimento (Bazett)
    if f"QTc_{reference_formula}_ms" in data.columns:
        ref_qtc = data[f"QTc_{reference_formula}_ms"].values
    else:
        qt = data["QT_interval_ms"].values
        rr = data["RR_interval_sec"].values
        ref_qtc = STANDARD_FORMULAS[reference_formula]["formula"](qt, rr, None)

    # Maschera valori validi
    valid = (
        ~np.isnan(formula_qtc)
        & ~np.isnan(ref_qtc)
        & ~np.isinf(formula_qtc)
        & ~np.isinf(ref_qtc)
    )
    n_valid = valid.sum()

    if n_valid < 100:
        return {"error": "Insufficient valid data", "n_valid": n_valid}

    new_qtc = formula_qtc[valid]
    bazett_qtc = ref_qtc[valid]

    # Determina soglia per sesso (se disponibile)
    if "sex" in data.columns:
        sex = data["sex"].values[valid]
        threshold = np.where(
            sex == 1,
            CLINICAL_THRESHOLDS["normal_upper_female"],
            CLINICAL_THRESHOLDS["normal_upper_male"],
        )
    else:
        # Usa soglia media
        threshold = np.full(n_valid, 455)

    # Classificazioni
    bazett_prolonged = bazett_qtc > threshold
    new_prolonged = new_qtc > threshold

    # Reclassificazioni
    # Pazienti che Bazett classifica come prolungato ma la nuova formula come normale
    reclassified_to_normal = bazett_prolonged & ~new_prolonged
    # Pazienti che Bazett classifica come normale ma la nuova formula come prolungato
    reclassified_to_prolonged = ~bazett_prolonged & new_prolonged

    n_reclassified_to_normal = reclassified_to_normal.sum()
    n_reclassified_to_prolonged = reclassified_to_prolonged.sum()

    # Analisi per range HR
    hr = data["heart_rate_bpm"].values[valid]
    reclassification_by_hr = {}

    for hr_min, hr_max, label in HR_BINS:
        hr_mask = (hr >= hr_min) & (hr < hr_max)
        if hr_mask.sum() < 10:
            continue

        n_to_normal = (reclassified_to_normal & hr_mask).sum()
        n_to_prolonged = (reclassified_to_prolonged & hr_mask).sum()

        reclassification_by_hr[label] = {
            "n_total": int(hr_mask.sum()),
            "reclassified_to_normal": int(n_to_normal),
            "reclassified_to_prolonged": int(n_to_prolonged),
            "net_change": int(n_to_normal - n_to_prolonged),
        }

    # Benefit-risk ratio
    # Benefit: pazienti correttamente reclassificati come normali (evitano ansia/trattamenti)
    # Risk: pazienti con vero QT prolungato classificati come normali
    # Usiamo come proxy il rapporto reclassificati_to_normal / reclassificati_to_prolonged
    if n_reclassified_to_prolonged > 0:
        benefit_risk_ratio = n_reclassified_to_normal / n_reclassified_to_prolonged
    else:
        benefit_risk_ratio = float("inf") if n_reclassified_to_normal > 0 else 1.0

    return {
        "n_analyzed": int(n_valid),
        "bazett_prolonged_count": int(bazett_prolonged.sum()),
        "bazett_prolonged_pct": float(100 * bazett_prolonged.sum() / n_valid),
        "new_prolonged_count": int(new_prolonged.sum()),
        "new_prolonged_pct": float(100 * new_prolonged.sum() / n_valid),
        "reclassified_to_normal": int(n_reclassified_to_normal),
        "reclassified_to_prolonged": int(n_reclassified_to_prolonged),
        "net_reclassification": int(n_reclassified_to_normal - n_reclassified_to_prolonged),
        "benefit_risk_ratio": float(benefit_risk_ratio)
        if not np.isinf(benefit_risk_ratio)
        else "inf",
        "reclassification_by_hr": reclassification_by_hr,
    }


# =============================================================================
# PHYSIOLOGICAL PLAUSIBILITY ANALYSIS
# =============================================================================


def analyze_physiological_plausibility(
    data: pd.DataFrame, formula_qtc: np.ndarray, formula_name: str
) -> Dict[str, Any]:
    """
    Valuta la plausibilità fisiologica di una formula QTc.

    Criteri:
    1. Range di output ragionevoli (300-600 ms)
    2. Comportamento monotono con HR
    3. Stabilità intra-paziente (se disponibile)
    4. Coerenza con aspettative fisiologiche

    Args:
        data: DataFrame con dati ECG
        formula_qtc: Array di valori QTc
        formula_name: Nome della formula

    Returns:
        Dizionario con analisi di plausibilità
    """
    valid = ~np.isnan(formula_qtc) & ~np.isinf(formula_qtc)
    qtc = formula_qtc[valid]

    if len(qtc) < 100:
        return {"error": "Insufficient valid data", "n_valid": len(qtc)}

    # 1. Range di output
    out_of_range_low = (qtc < 250).sum()
    out_of_range_high = (qtc > 650).sum()
    pct_in_range = 100 * (1 - (out_of_range_low + out_of_range_high) / len(qtc))

    # 2. Comportamento con HR
    hr = data["heart_rate_bpm"].values[valid]
    qt = data["QT_interval_ms"].values[valid]

    # La formula ideale dovrebbe correggere il QT che decresce con HR
    # Verifichiamo che il QTc sia relativamente costante
    r_qtc_hr, _ = safe_correlation(qtc, hr)

    # Confronta con correlazione QT-HR (dovrebbe essere negativa)
    r_qt_hr, _ = safe_correlation(qt, hr)

    # Correzione efficace se r(QTc, HR) << r(QT, HR)
    correction_effectiveness = 1 - abs(r_qtc_hr) / abs(r_qt_hr) if abs(r_qt_hr) > 0.01 else 0

    # 3. Distribuzione dei valori
    mean_qtc = np.mean(qtc)
    std_qtc = np.std(qtc)
    cv_qtc = std_qtc / mean_qtc if mean_qtc > 0 else np.nan

    percentiles = {
        "p1": float(np.percentile(qtc, 1)),
        "p5": float(np.percentile(qtc, 5)),
        "p25": float(np.percentile(qtc, 25)),
        "p50": float(np.percentile(qtc, 50)),
        "p75": float(np.percentile(qtc, 75)),
        "p95": float(np.percentile(qtc, 95)),
        "p99": float(np.percentile(qtc, 99)),
    }

    # 4. Test di monotonia locale
    # Dividi HR in quartili e verifica che QTc medio sia simile
    hr_quartiles = np.percentile(hr, [25, 50, 75])
    qtc_by_hr_quartile = []
    for i in range(4):
        if i == 0:
            mask = hr <= hr_quartiles[0]
        elif i == 3:
            mask = hr > hr_quartiles[2]
        else:
            mask = (hr > hr_quartiles[i - 1]) & (hr <= hr_quartiles[i])

        if mask.sum() > 10:
            qtc_by_hr_quartile.append(float(np.mean(qtc[mask])))
        else:
            qtc_by_hr_quartile.append(np.nan)

    # Variazione massima tra quartili
    valid_quartiles = [q for q in qtc_by_hr_quartile if not np.isnan(q)]
    if len(valid_quartiles) >= 2:
        max_quartile_diff = max(valid_quartiles) - min(valid_quartiles)
    else:
        max_quartile_diff = np.nan

    # Score plausibilità (0-100)
    plausibility_score = 0

    # Range corretto (+30 punti)
    if pct_in_range >= 99:
        plausibility_score += 30
    elif pct_in_range >= 95:
        plausibility_score += 20
    elif pct_in_range >= 90:
        plausibility_score += 10

    # Bassa correlazione con HR (+40 punti)
    if abs(r_qtc_hr) < 0.02:
        plausibility_score += 40
    elif abs(r_qtc_hr) < 0.05:
        plausibility_score += 30
    elif abs(r_qtc_hr) < 0.10:
        plausibility_score += 20
    elif abs(r_qtc_hr) < 0.15:
        plausibility_score += 10

    # Bassa variabilità tra quartili (+20 punti)
    if not np.isnan(max_quartile_diff):
        if max_quartile_diff < 5:
            plausibility_score += 20
        elif max_quartile_diff < 10:
            plausibility_score += 15
        elif max_quartile_diff < 20:
            plausibility_score += 10

    # CV ragionevole (+10 punti)
    if not np.isnan(cv_qtc) and cv_qtc < 0.08:
        plausibility_score += 10
    elif not np.isnan(cv_qtc) and cv_qtc < 0.12:
        plausibility_score += 5

    return {
        "n_analyzed": int(len(qtc)),
        "output_range": {
            "min": float(np.min(qtc)),
            "max": float(np.max(qtc)),
            "out_of_range_low": int(out_of_range_low),
            "out_of_range_high": int(out_of_range_high),
            "pct_in_range": float(pct_in_range),
        },
        "hr_relationship": {
            "r_qtc_hr": float(r_qtc_hr) if not np.isnan(r_qtc_hr) else None,
            "r_qt_hr": float(r_qt_hr) if not np.isnan(r_qt_hr) else None,
            "correction_effectiveness": float(correction_effectiveness),
        },
        "distribution": {
            "mean": float(mean_qtc),
            "std": float(std_qtc),
            "cv": float(cv_qtc) if not np.isnan(cv_qtc) else None,
            "percentiles": percentiles,
        },
        "stability": {
            "qtc_by_hr_quartile": qtc_by_hr_quartile,
            "max_quartile_diff_ms": float(max_quartile_diff)
            if not np.isnan(max_quartile_diff)
            else None,
        },
        "plausibility_score": int(plausibility_score),
        "plausibility_grade": (
            "Excellent"
            if plausibility_score >= 80
            else (
                "Good"
                if plausibility_score >= 60
                else "Acceptable" if plausibility_score >= 40 else "Poor"
            )
        ),
    }


# =============================================================================
# SCIENTIFIC INTEREST ANALYSIS
# =============================================================================


def analyze_scientific_interest(
    formula: FormulaValidator,
    validation_results: Dict[str, Any],
    n_datasets: int,
) -> Dict[str, Any]:
    """
    Valuta l'interesse scientifico di una formula QTc.

    Criteri:
    1. Novità rispetto alle formule esistenti
    2. Miglioramento rispetto a Bazett/Fridericia
    3. Generalizzabilità cross-dataset
    4. Semplicità e interpretabilità
    5. Potenziale impatto clinico

    Args:
        formula: FormulaValidator della formula candidata
        validation_results: Risultati di validazione aggregati
        n_datasets: Numero di dataset su cui è stata validata

    Returns:
        Dizionario con analisi di interesse scientifico
    """
    # 1. Analisi della struttura della formula
    equation = formula.equation
    approach = formula.approach

    # Conta operatori per stimare complessità
    n_operations = len(re.findall(r"[+\-*/^]", equation))
    has_sqrt = "sqrt" in equation.lower()
    has_cbrt = "cbrt" in equation.lower()
    has_log = "log" in equation.lower()
    has_exp = "exp" in equation.lower()

    # 2. Novità: confronta con formule standard
    # Se la formula è molto simile a una standard, è meno interessante
    similarity_to_standard = "Unknown"
    if approach == "factor":
        if has_sqrt and not has_cbrt:
            similarity_to_standard = "Similar to Bazett"
        elif has_cbrt and not has_sqrt:
            similarity_to_standard = "Similar to Fridericia"
        else:
            similarity_to_standard = "Novel structure"
    elif approach == "additive":
        similarity_to_standard = "Similar to Framingham/Hodges"
    else:
        similarity_to_standard = "Direct approach (novel)"

    # 3. Calcola miglioramenti medi rispetto a standard
    improvements = {}
    if "comparison_vs_standard" in validation_results:
        comp = validation_results["comparison_vs_standard"]
        for std_name in STANDARD_FORMULAS:
            if std_name in comp:
                improvements[std_name] = comp[std_name].get("improvement_factor", 1.0)

    # 4. Generalizzabilità
    if "cross_dataset" in validation_results:
        cross = validation_results["cross_dataset"]
        r_values = [
            d.get("r_vs_hr")
            for d in cross.values()
            if isinstance(d, dict) and d.get("r_vs_hr") is not None
        ]
        if r_values:
            mean_r = np.mean([abs(r) for r in r_values])
            std_r = np.std([abs(r) for r in r_values])
            generalizability = {
                "mean_abs_r": float(mean_r),
                "std_abs_r": float(std_r),
                "cv_r": float(std_r / mean_r) if mean_r > 0 else None,
                "n_datasets": n_datasets,
            }
        else:
            generalizability = {"mean_abs_r": None, "n_datasets": n_datasets}
    else:
        generalizability = {"n_datasets": n_datasets}

    # 5. Score interesse scientifico (0-100)
    interest_score = 0

    # Miglioramento vs Bazett (+35 punti)
    if "Bazett" in improvements:
        imp = improvements["Bazett"]
        if imp > 5:
            interest_score += 35
        elif imp > 3:
            interest_score += 25
        elif imp > 2:
            interest_score += 15
        elif imp > 1.5:
            interest_score += 10

    # Generalizzabilità (+25 punti)
    if generalizability.get("mean_abs_r") is not None:
        mean_r = generalizability["mean_abs_r"]
        if mean_r < 0.03:
            interest_score += 25
        elif mean_r < 0.05:
            interest_score += 20
        elif mean_r < 0.08:
            interest_score += 15
        elif mean_r < 0.10:
            interest_score += 10

    # Novità strutturale (+20 punti)
    if similarity_to_standard == "Novel structure" or similarity_to_standard == "Direct approach (novel)":
        interest_score += 20
    elif "Similar" in similarity_to_standard:
        interest_score += 5

    # Semplicità (+20 punti)
    if n_operations <= 2:
        interest_score += 20
    elif n_operations <= 4:
        interest_score += 15
    elif n_operations <= 6:
        interest_score += 10

    # Penalità per operatori complessi
    if has_log or has_exp:
        interest_score -= 5

    interest_score = max(0, min(100, interest_score))

    return {
        "formula_structure": {
            "approach": approach,
            "n_operations": n_operations,
            "has_sqrt": has_sqrt,
            "has_cbrt": has_cbrt,
            "has_log": has_log,
            "has_exp": has_exp,
        },
        "novelty": {"similarity_to_standard": similarity_to_standard},
        "improvements_vs_standard": improvements,
        "generalizability": generalizability,
        "interest_score": int(interest_score),
        "interest_grade": (
            "High"
            if interest_score >= 70
            else "Moderate" if interest_score >= 50 else "Low" if interest_score >= 30 else "Minimal"
        ),
        "publication_potential": (
            "Strong candidate for tier-1 journal"
            if interest_score >= 75
            else (
                "Suitable for specialty journal"
                if interest_score >= 50
                else "May require additional validation" if interest_score >= 30 else "Limited publication potential"
            )
        ),
    }


# =============================================================================
# CROSS-DATASET VALIDATION
# =============================================================================


def run_cross_dataset_validation(
    formulas: List[FormulaValidator],
    datasets: Dict[str, DatasetValidator],
) -> Dict[str, Any]:
    """
    Esegue validazione cross-dataset per tutte le formule.

    Args:
        formulas: Lista di formule da validare
        datasets: Dizionario di DatasetValidator

    Returns:
        Risultati di validazione cross-dataset
    """
    results = {}

    for formula in formulas:
        formula_results = {
            "name": formula.name,
            "equation": formula.equation,
            "approach": formula.approach,
            "is_standard": formula.is_standard,
            "datasets": {},
        }

        all_r_values = []
        total_n = 0

        for ds_name, ds_validator in datasets.items():
            ds_result = ds_validator.validate_formula(formula)
            formula_results["datasets"][ds_name] = ds_result

            r_hr = ds_result["global_metrics"].get("r_vs_hr")
            n = ds_result["global_metrics"].get("n_valid", 0)

            if r_hr is not None and not np.isnan(r_hr):
                all_r_values.append({"dataset": ds_name, "r": r_hr, "n": n})
                total_n += n

        # Aggregazione cross-dataset
        if all_r_values:
            r_vals = [x["r"] for x in all_r_values]
            abs_r_vals = [abs(r) for r in r_vals]
            weights = [x["n"] for x in all_r_values]

            # Media pesata per n (metrica originale)
            weighted_r = np.average(abs_r_vals, weights=weights)
            
            # Media NON pesata (ogni dataset conta uguale)
            unweighted_mean_r = np.mean(abs_r_vals)
            
            # Mediana (robusta agli outlier)
            median_r = np.median(abs_r_vals)
            
            # Worst-case (massimo |r|)
            worst_r = max(abs_r_vals)
            
            # Best-case (minimo |r|)
            best_r = min(abs_r_vals)

            formula_results["cross_dataset_summary"] = {
                "n_datasets": len(all_r_values),
                "total_n": total_n,
                # Metriche originali
                "mean_abs_r": float(np.mean(abs_r_vals)),
                "weighted_abs_r": float(weighted_r),
                "std_r": float(np.std(r_vals)),
                "min_r": float(min(r_vals)),
                "max_r": float(max(r_vals)),
                # Nuove metriche
                "unweighted_mean_abs_r": float(unweighted_mean_r),
                "median_abs_r": float(median_r),
                "worst_abs_r": float(worst_r),
                "best_abs_r": float(best_r),
                # Soglie
                "all_below_0.05": all(r < 0.05 for r in abs_r_vals),
                "all_below_0.10": all(r < 0.10 for r in abs_r_vals),
                "pct_below_0.05": float(100 * sum(1 for r in abs_r_vals if r < 0.05) / len(abs_r_vals)),
                # Dettaglio per dataset
                "per_dataset": {x["dataset"]: {"r": x["r"], "n": x["n"]} for x in all_r_values},
            }
        else:
            formula_results["cross_dataset_summary"] = {
                "n_datasets": 0,
                "error": "No valid results across datasets",
            }

        results[formula.name] = formula_results

    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================


def generate_summary_report(
    discovered_formulas: List[FormulaValidator],
    standard_formulas: List[FormulaValidator],
    cross_validation: Dict[str, Any],
    clinical_utility: Dict[str, Any],
    plausibility: Dict[str, Any],
    scientific_interest: Dict[str, Any],
) -> Dict[str, Any]:
    """Genera report riassuntivo completo."""

    # Trova la migliore formula scoperta
    best_discovered = None
    best_weighted_r = float("inf")

    for formula in discovered_formulas:
        name = formula.name
        if name in cross_validation:
            summary = cross_validation[name].get("cross_dataset_summary", {})
            weighted_r = summary.get("weighted_abs_r", float("inf"))
            if weighted_r < best_weighted_r:
                best_weighted_r = weighted_r
                best_discovered = name

    # Confronto con standard
    standard_comparison = {}
    for std_formula in standard_formulas:
        name = std_formula.name
        if name in cross_validation:
            summary = cross_validation[name].get("cross_dataset_summary", {})
            standard_comparison[name] = {
                "weighted_abs_r": summary.get("weighted_abs_r"),
                "mean_abs_r": summary.get("mean_abs_r"),
            }

    # Ranking finale
    all_formulas_ranked = []
    for name, result in cross_validation.items():
        summary = result.get("cross_dataset_summary", {})
        if "weighted_abs_r" in summary:
            all_formulas_ranked.append({
                "name": name,
                "weighted_abs_r": summary["weighted_abs_r"],
                "unweighted_mean_abs_r": summary.get("unweighted_mean_abs_r"),
                "median_abs_r": summary.get("median_abs_r"),
                "worst_abs_r": summary.get("worst_abs_r"),
                "is_standard": result.get("is_standard", False),
            })

    # Ordina per weighted (default), ma fornisci anche ranking alternativi
    ranked_by_weighted = sorted(all_formulas_ranked, key=lambda x: x["weighted_abs_r"])
    ranked_by_unweighted = sorted(all_formulas_ranked, key=lambda x: x.get("unweighted_mean_abs_r") or 1.0)
    ranked_by_median = sorted(all_formulas_ranked, key=lambda x: x.get("median_abs_r") or 1.0)
    ranked_by_worst = sorted(all_formulas_ranked, key=lambda x: x.get("worst_abs_r") or 1.0)

    return {
        "best_discovered_formula": best_discovered,
        "best_discovered_weighted_r": best_weighted_r if best_discovered else None,
        "standard_comparison": standard_comparison,
        "ranking_by_weighted": ranked_by_weighted[:10],
        "ranking_by_unweighted": ranked_by_unweighted[:10],
        "ranking_by_median": ranked_by_median[:10],
        "ranking_by_worst_case": ranked_by_worst[:10],
        "recommendations": _generate_recommendations(
            best_discovered,
            best_weighted_r,
            standard_comparison,
            clinical_utility.get(best_discovered, {}),
            plausibility.get(best_discovered, {}),
            scientific_interest.get(best_discovered, {}),
        ),
    }


def _generate_recommendations(
    best_formula: Optional[str],
    best_r: float,
    standard_comparison: Dict[str, Any],
    clinical: Dict[str, Any],
    plausibility: Dict[str, Any],
    interest: Dict[str, Any],
) -> List[str]:
    """Genera raccomandazioni basate sui risultati."""
    recommendations = []

    if best_formula is None:
        recommendations.append(
            "Nessuna formula valida trovata. Verificare i dati di input."
        )
        return recommendations

    # Confronto con Bazett
    bazett_r = standard_comparison.get("Bazett", {}).get("weighted_abs_r", 1.0)
    if bazett_r and best_r < bazett_r:
        improvement = bazett_r / best_r if best_r > 0 else float("inf")
        recommendations.append(
            f"La formula '{best_formula}' mostra un miglioramento di {improvement:.1f}x "
            f"rispetto a Bazett in termini di HR-indipendenza."
        )

    # Plausibilità
    plaus_grade = plausibility.get("plausibility_grade", "Unknown")
    if plaus_grade in ["Excellent", "Good"]:
        recommendations.append(
            f"La formula mostra eccellente plausibilità fisiologica (Grade: {plaus_grade})."
        )
    elif plaus_grade == "Poor":
        recommendations.append(
            "ATTENZIONE: La formula mostra scarsa plausibilità fisiologica. "
            "Verificare il range di output e il comportamento alle frequenze estreme."
        )

    # Interesse scientifico
    interest_grade = interest.get("interest_grade", "Unknown")
    pub_potential = interest.get("publication_potential", "Unknown")
    if interest_grade in ["High", "Moderate"]:
        recommendations.append(f"Interesse scientifico: {interest_grade}. {pub_potential}")

    # Utilità clinica
    if "benefit_risk_ratio" in clinical:
        ratio = clinical["benefit_risk_ratio"]
        if ratio != "inf" and isinstance(ratio, (int, float)) and ratio > 2:
            recommendations.append(
                f"Rapporto beneficio/rischio clinico favorevole: {ratio:.1f}:1"
            )

    # Generalizzabilità
    if best_r < 0.03:
        recommendations.append(
            "La formula mostra eccellente generalizzabilità cross-dataset "
            "(|r| < 0.03 su tutti i dataset)."
        )
    elif best_r < 0.05:
        recommendations.append(
            "La formula mostra buona generalizzabilità cross-dataset "
            "(|r| < 0.05 su tutti i dataset)."
        )

    return recommendations


# =============================================================================
# MAIN VALIDATION PIPELINE
# =============================================================================


def load_discovered_formulas(sr_output_dir: Path) -> List[Dict[str, Any]]:
    """Carica le formule scoperte da SR."""
    formulas = []

    # Cerca file equations_unified.csv o equations_top10.csv
    unified_file = list(sr_output_dir.glob("*_equations_unified.csv"))
    top10_file = list(sr_output_dir.glob("*_equations_top10.csv"))

    if top10_file:
        df = pd.read_csv(top10_file[0])
        log_info(f"Caricato {top10_file[0].name}: {len(df)} formule")
    elif unified_file:
        df = pd.read_csv(unified_file[0])
        log_info(f"Caricato {unified_file[0].name}: {len(df)} formule")
    else:
        log_warning(f"Nessun file equazioni trovato in {sr_output_dir}")
        return formulas

    # Converti in lista di dizionari
    for _, row in df.iterrows():
        formulas.append({
            "name": f"Discovered_{row.get('approach', 'unknown')}_{len(formulas)+1}",
            "equation": row.get("equation", ""),
            "approach": row.get("approach", "unknown"),
            "complexity": row.get("complexity", 0),
            "score": row.get("score", 1.0),
            "r_vs_HR": row.get("r_vs_HR", None),
        })

    return formulas


def load_qtc_data(dataset: str, base_path: Path) -> Optional[pd.DataFrame]:
    """Carica i dati QTc preparati per un dataset."""
    # Cerca il file qtc_preparation
    qtc_file = base_path / dataset / "qtc" / f"{dataset}_qtc_preparation.csv"

    if not qtc_file.exists():
        log_warning(f"File non trovato: {qtc_file}")
        return None

    df = pd.read_csv(qtc_file)
    log_info(f"Caricato {dataset}: {len(df)} record")
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validazione statistica formule QTc scoperte",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Validazione su singolo dataset
  python 06_0_sr_qtc_validation.py --dataset ptb-xl

  # Validazione cross-dataset (tutti i dataset disponibili)
  python 06_0_sr_qtc_validation.py --cross-dataset

  # Valida TUTTE le formule scoperte (non solo top-n)
  python 06_0_sr_qtc_validation.py --cross-dataset --all

  # Specifica directory SR
  python 06_0_sr_qtc_validation.py --dataset ptb-xl --sr-dir results/ptb-xl/sr_qtc

  # Escludi alcune analisi (più veloce)
  python 06_0_sr_qtc_validation.py --cross-dataset --all --skip-clinical
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        help="Dataset da validare",
    )
    parser.add_argument(
        "--cross-dataset",
        action="store_true",
        help="Esegui validazione cross-dataset su tutti i dataset disponibili",
    )
    parser.add_argument(
        "--sr-dir",
        type=str,
        help="Directory con output di SR (default: results/{dataset}/sr_qtc)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Directory output (default: results/{dataset}/validation)",
    )
    parser.add_argument(
        "--results-base",
        type=str,
        default="results",
        help="Directory base dei risultati (default: results)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Numero di formule top da validare in dettaglio (default: 5)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Valida tutte le formule scoperte (ignora --top-n)",
    )
    parser.add_argument(
        "--skip-clinical",
        action="store_true",
        help="Salta analisi utilità clinica",
    )
    parser.add_argument(
        "--skip-plausibility",
        action="store_true",
        help="Salta analisi plausibilità fisiologica",
    )
    parser.add_argument(
        "--skip-interest",
        action="store_true",
        help="Salta analisi interesse scientifico",
    )

    args = parser.parse_args()

    # Validazione argomenti
    if not args.dataset and not args.cross_dataset:
        parser.error("Specificare --dataset o --cross-dataset")

    base_path = Path(args.results_base)

    # Header
    print("=" * 70)
    print("KEPLER-ECG: Validazione Statistica Formule QTc")
    print(f"Versione: {VERSION}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    print()

    # Determina dataset da processare
    if args.cross_dataset:
        datasets_to_process = []
        for ds in SUPPORTED_DATASETS:
            qtc_file = base_path / ds / "qtc" / f"{ds}_qtc_preparation.csv"
            if qtc_file.exists():
                datasets_to_process.append(ds)
        log_info(f"Validazione cross-dataset su {len(datasets_to_process)} dataset: {datasets_to_process}")
    else:
        datasets_to_process = [args.dataset]

    if not datasets_to_process:
        log_error("Nessun dataset disponibile per la validazione")
        sys.exit(1)

    # Carica dati per ogni dataset
    dataset_validators = {}
    for ds in datasets_to_process:
        df = load_qtc_data(ds, base_path)
        if df is not None:
            dataset_validators[ds] = DatasetValidator(ds, df)

    if not dataset_validators:
        log_error("Nessun dato caricato")
        sys.exit(1)

    # Carica formule scoperte
    # Se cross-dataset, cerca SR da tutti i dataset
    # Altrimenti usa solo il dataset specificato
    discovered_formulas_raw = []

    if args.sr_dir:
        sr_dirs = [Path(args.sr_dir)]
    elif args.cross_dataset:
        sr_dirs = [base_path / ds / "sr_qtc" for ds in datasets_to_process]
    else:
        sr_dirs = [base_path / args.dataset / "sr_qtc"]

    for sr_dir in sr_dirs:
        if sr_dir.exists():
            formulas = load_discovered_formulas(sr_dir)
            discovered_formulas_raw.extend(formulas)

    # Rimuovi duplicati per equazione
    seen_equations = set()
    unique_formulas = []
    for f in discovered_formulas_raw:
        eq = f["equation"]
        if eq not in seen_equations:
            seen_equations.add(eq)
            unique_formulas.append(f)

    log_info(f"Formule scoperte uniche: {len(unique_formulas)}")

    # Seleziona formule per validazione dettagliata
    unique_formulas.sort(key=lambda x: x.get("score", 1.0))
    
    if args.all:
        top_formulas = unique_formulas
        log_info(f"Modalità --all: validazione di tutte le {len(top_formulas)} formule")
    else:
        top_formulas = unique_formulas[: args.top_n]
        log_info(f"Selezionate top {args.top_n} formule per validazione dettagliata")

    # Crea FormulaValidator per formule scoperte
    discovered_validators = []
    for i, f in enumerate(top_formulas):
        validator = FormulaValidator(
            name=f["name"],
            equation=f["equation"],
            approach=f["approach"],
            is_standard=False,
        )
        if validator.is_valid:
            discovered_validators.append(validator)
        else:
            log_warning(f"Formula non valida: {f['equation']}")

    # Crea FormulaValidator per formule standard
    standard_validators = []
    for name, info in STANDARD_FORMULAS.items():
        validator = FormulaValidator(
            name=name,
            equation=info["equation"],
            approach=info["type"],
            formula_func=info["formula"],
            is_standard=True,
            column=info["column"],
        )
        standard_validators.append(validator)

    all_validators = discovered_validators + standard_validators

    print()
    print("-" * 70)
    n_discovered = len(discovered_validators)
    mode_str = "tutte" if args.all else f"top {args.top_n}"
    print(f"FORMULE DA VALIDARE: {len(all_validators)}")
    print(f"  - Scoperte ({mode_str}): {n_discovered}")
    print(f"  - Standard: {len(standard_validators)}")
    print("-" * 70)
    print()

    # Esegui validazione cross-dataset
    log_info("Esecuzione validazione cross-dataset...")
    cross_validation_results = run_cross_dataset_validation(
        all_validators, dataset_validators
    )

    # Analisi aggiuntive (solo per formule scoperte)
    clinical_utility_results = {}
    plausibility_results = {}
    scientific_interest_results = {}

    # Usa il dataset più grande per analisi dettagliate
    largest_ds = max(dataset_validators.keys(), key=lambda x: dataset_validators[x].n_records)
    main_data = dataset_validators[largest_ds].data

    for formula in discovered_validators:
        formula_name = formula.name

        # Calcola QTc
        qtc = formula.compute_qtc(
            main_data["QT_interval_ms"].values,
            main_data["RR_interval_sec"].values,
            main_data["heart_rate_bpm"].values,
        )

        # Utilità clinica
        if not args.skip_clinical:
            log_info(f"Analisi utilità clinica: {formula_name}")
            clinical_utility_results[formula_name] = analyze_clinical_utility(
                main_data, qtc, formula_name
            )

        # Plausibilità fisiologica
        if not args.skip_plausibility:
            log_info(f"Analisi plausibilità: {formula_name}")
            plausibility_results[formula_name] = analyze_physiological_plausibility(
                main_data, qtc, formula_name
            )

        # Interesse scientifico
        if not args.skip_interest:
            log_info(f"Analisi interesse scientifico: {formula_name}")

            # Prepara risultati validazione per questa formula
            validation_for_interest = cross_validation_results.get(formula_name, {})

            # Calcola miglioramenti vs standard
            comparison = {}
            for std_validator in standard_validators:
                std_name = std_validator.name
                std_result = cross_validation_results.get(std_name, {})
                std_r = std_result.get("cross_dataset_summary", {}).get("weighted_abs_r", 1.0)

                formula_r = validation_for_interest.get("cross_dataset_summary", {}).get(
                    "weighted_abs_r", 1.0
                )

                if std_r and formula_r and formula_r > 0:
                    comparison[std_name] = {"improvement_factor": std_r / formula_r}

            validation_for_interest["comparison_vs_standard"] = comparison

            # Cross-dataset per generalizzabilità
            cross_ds = {}
            for ds_name, ds_result in validation_for_interest.get("datasets", {}).items():
                cross_ds[ds_name] = ds_result.get("global_metrics", {})
            validation_for_interest["cross_dataset"] = cross_ds

            scientific_interest_results[formula_name] = analyze_scientific_interest(
                formula, validation_for_interest, len(dataset_validators)
            )

    # Genera report riassuntivo
    summary = generate_summary_report(
        discovered_validators,
        standard_validators,
        cross_validation_results,
        clinical_utility_results,
        plausibility_results,
        scientific_interest_results,
    )

    # Prepara output
    if args.output:
        output_dir = Path(args.output)
    elif args.cross_dataset:
        output_dir = base_path / "cross_dataset" / "validation"
    else:
        output_dir = base_path / args.dataset / "validation"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determina il prefisso per i nomi file
    if args.cross_dataset:
        file_prefix = "cross_dataset"
    else:
        file_prefix = args.dataset

    # Report completo JSON
    full_report = {
        "metadata": {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "datasets": list(dataset_validators.keys()),
            "n_records_total": sum(dv.n_records for dv in dataset_validators.values()),
            "formulas_validated": len(all_validators),
        },
        "summary": summary,
        "cross_validation": cross_validation_results,
        "clinical_utility": clinical_utility_results,
        "plausibility": plausibility_results,
        "scientific_interest": scientific_interest_results,
    }

    report_file = output_dir / f"{file_prefix}_validation_report.json"
    with open(report_file, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    log_info(f"Report salvato: {report_file}")

    # Tabella riassuntiva CSV
    summary_rows = []
    for name, result in cross_validation_results.items():
        summary_data = result.get("cross_dataset_summary", {})
        clinical = clinical_utility_results.get(name, {})
        plausibility = plausibility_results.get(name, {})
        interest = scientific_interest_results.get(name, {})

        summary_rows.append({
            "formula": name,
            "equation": result.get("equation", ""),
            "approach": result.get("approach", ""),
            "is_standard": result.get("is_standard", False),
            "n_datasets": summary_data.get("n_datasets", 0),
            "total_n": summary_data.get("total_n", 0),
            # Metriche HR-independence
            "weighted_abs_r": summary_data.get("weighted_abs_r"),
            "unweighted_mean_abs_r": summary_data.get("unweighted_mean_abs_r"),
            "median_abs_r": summary_data.get("median_abs_r"),
            "worst_abs_r": summary_data.get("worst_abs_r"),
            "best_abs_r": summary_data.get("best_abs_r"),
            "std_r": summary_data.get("std_r"),
            "pct_below_0.05": summary_data.get("pct_below_0.05"),
            "all_below_0.05": summary_data.get("all_below_0.05"),
            # Clinical utility
            "benefit_risk_ratio": clinical.get("benefit_risk_ratio"),
            "net_reclassification": clinical.get("net_reclassification"),
            # Plausibility
            "plausibility_score": plausibility.get("plausibility_score"),
            "plausibility_grade": plausibility.get("plausibility_grade"),
            # Scientific interest
            "interest_score": interest.get("interest_score"),
            "interest_grade": interest.get("interest_grade"),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("weighted_abs_r", ascending=True)

    summary_file = output_dir / f"{file_prefix}_validation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    log_info(f"Tabella riassuntiva salvata: {summary_file}")

    # Stampa risultati principali
    print()
    print("=" * 70)
    print("RISULTATI PRINCIPALI")
    print("=" * 70)

    # Ranking per metrica pesata (originale)
    print()
    print("RANKING PER WEIGHTED |r| (pesato per n records):")
    print("-" * 70)
    for i, row in enumerate(summary.get("ranking_by_weighted", [])[:10], 1):
        std_marker = " [STD]" if row["is_standard"] else ""
        print(f"  {i:2d}. {row['name']}{std_marker}: |r| = {row['weighted_abs_r']:.4f}")

    # Ranking per mediana (robusta)
    print()
    print("RANKING PER MEDIAN |r| (ogni dataset conta uguale):")
    print("-" * 70)
    for i, row in enumerate(summary.get("ranking_by_median", [])[:10], 1):
        std_marker = " [STD]" if row["is_standard"] else ""
        median_r = row.get('median_abs_r', 0) or 0
        print(f"  {i:2d}. {row['name']}{std_marker}: |r| = {median_r:.4f}")

    # Ranking per worst-case
    print()
    print("RANKING PER WORST-CASE |r| (massima robustezza):")
    print("-" * 70)
    for i, row in enumerate(summary.get("ranking_by_worst_case", [])[:10], 1):
        std_marker = " [STD]" if row["is_standard"] else ""
        worst_r = row.get('worst_abs_r', 0) or 0
        print(f"  {i:2d}. {row['name']}{std_marker}: max|r| = {worst_r:.4f}")

    print()
    print("CONFRONTO CON STANDARD:")
    print("-" * 70)
    for std_name, std_data in summary["standard_comparison"].items():
        r_val = std_data.get("weighted_abs_r", "N/A")
        if isinstance(r_val, float):
            print(f"  {std_name}: |r| = {r_val:.4f}")

    print()
    print("RACCOMANDAZIONI:")
    print("-" * 70)
    for rec in summary["recommendations"]:
        print(f"  • {rec}")

    print()
    print("=" * 70)
    print(f"Validazione completata. Output in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
