#!/usr/bin/env python3
"""
07_0_sr_qtc_cross_validation.py - Cross-Validation Formule QTc tra Dataset

Script per testare le formule QTc scoperte su un dataset (source) 
su altri dataset (target) per valutare la generalizzabilità.

Utilizzo:
  python 07_0_sr_qtc_cross_validation.py --source-dataset ptb-xl --target-datasets chapman mimic-iv-ecg
  python 07_0_sr_qtc_cross_validation.py --source-dataset chapman --target-datasets ptb-xl georgia code-15

Output in: results/{source-dataset}/cross-validation/

Versione: 1.0.0
Autore: Alessandro Marconi
Progetto: Kepler-ECG
Data: 29 Gennaio 2026
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
    "ptb-xl",
    "chapman",
    "cpsc-2018",
    "georgia",
    "code-15",
    "mimic-iv-ecg",
    "pool6",
    "pool_screening",
    "pool_clinical",
    "ludb",
    "ecg-arrhythmia"
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

# Range HR per analisi stratificata
HR_BINS = [
    (30, 50, "Bradicardia severa"),
    (50, 60, "Bradicardia"),
    (60, 80, "Normale"),
    (80, 100, "Tachicardia lieve"),
    (100, 120, "Tachicardia"),
    (120, 150, "Tachicardia severa"),
]


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
        source_dataset: Optional[str] = None,
    ):
        self.name = name
        self.equation = equation
        self.approach = approach
        self.is_standard = is_standard
        self.column = column
        self.source_dataset = source_dataset

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

        return {
            "global_metrics": metrics,
            "hr_stratified": hr_analysis,
            "sex_stratified": sex_analysis,
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


# =============================================================================
# CROSS-VALIDATION FUNCTIONS
# =============================================================================


def run_cross_validation(
    formulas: List[FormulaValidator],
    target_datasets: Dict[str, DatasetValidator],
) -> Dict[str, Any]:
    """
    Esegue cross-validation delle formule sui dataset target.

    Args:
        formulas: Lista di formule da validare (da source dataset)
        target_datasets: Dizionario di DatasetValidator per dataset target

    Returns:
        Risultati di cross-validation
    """
    results = {}

    for formula in formulas:
        formula_results = {
            "name": formula.name,
            "equation": formula.equation,
            "approach": formula.approach,
            "is_standard": formula.is_standard,
            "source_dataset": formula.source_dataset,
            "target_datasets": {},
        }

        all_r_values = []
        total_n = 0

        for ds_name, ds_validator in target_datasets.items():
            ds_result = ds_validator.validate_formula(formula)
            formula_results["target_datasets"][ds_name] = ds_result

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

            # Media pesata per n
            weighted_r = np.average(abs_r_vals, weights=weights)

            # Media NON pesata (ogni dataset conta uguale)
            unweighted_mean_r = np.mean(abs_r_vals)

            # Mediana (robusta agli outlier)
            median_r = np.median(abs_r_vals)

            # Worst-case (massimo |r|)
            worst_r = max(abs_r_vals)

            # Best-case (minimo |r|)
            best_r = min(abs_r_vals)

            formula_results["cross_validation_summary"] = {
                "n_target_datasets": len(all_r_values),
                "total_n": total_n,
                # Metriche principali
                "weighted_abs_r": float(weighted_r),
                "unweighted_mean_abs_r": float(unweighted_mean_r),
                "median_abs_r": float(median_r),
                "worst_abs_r": float(worst_r),
                "best_abs_r": float(best_r),
                "std_r": float(np.std(r_vals)),
                # Soglie
                "all_below_0.05": all(r < 0.05 for r in abs_r_vals),
                "all_below_0.10": all(r < 0.10 for r in abs_r_vals),
                "pct_below_0.05": float(100 * sum(1 for r in abs_r_vals if r < 0.05) / len(abs_r_vals)),
                # Dettaglio per dataset
                "per_dataset": {x["dataset"]: {"r": x["r"], "abs_r": abs(x["r"]), "n": x["n"]} for x in all_r_values},
            }
        else:
            formula_results["cross_validation_summary"] = {
                "n_target_datasets": 0,
                "error": "No valid results across target datasets",
            }

        results[formula.name] = formula_results

    return results


def generate_comparison_report(
    cross_validation_results: Dict[str, Any],
    source_dataset: str,
    target_datasets: List[str],
) -> Dict[str, Any]:
    """Genera report di confronto tra formule."""

    # Ranking per diverse metriche
    formulas_data = []
    for name, result in cross_validation_results.items():
        summary = result.get("cross_validation_summary", {})
        if "weighted_abs_r" in summary:
            formulas_data.append({
                "name": name,
                "equation": result.get("equation", ""),
                "approach": result.get("approach", ""),
                "is_standard": result.get("is_standard", False),
                "source_dataset": result.get("source_dataset"),
                "weighted_abs_r": summary["weighted_abs_r"],
                "unweighted_mean_abs_r": summary.get("unweighted_mean_abs_r"),
                "median_abs_r": summary.get("median_abs_r"),
                "worst_abs_r": summary.get("worst_abs_r"),
                "best_abs_r": summary.get("best_abs_r"),
                "all_below_0.05": summary.get("all_below_0.05"),
                "n_target_datasets": summary.get("n_target_datasets"),
                "total_n": summary.get("total_n"),
            })

    # Ordina per weighted_abs_r
    formulas_data.sort(key=lambda x: x["weighted_abs_r"])

    # Separa discovered vs standard
    discovered = [f for f in formulas_data if not f["is_standard"]]
    standard = [f for f in formulas_data if f["is_standard"]]

    # Calcola miglioramenti vs standard
    improvements = {}
    if discovered and standard:
        best_discovered = discovered[0]
        for std in standard:
            if best_discovered["weighted_abs_r"] > 0:
                improvement = std["weighted_abs_r"] / best_discovered["weighted_abs_r"]
                improvements[std["name"]] = {
                    "improvement_factor": float(improvement),
                    "discovered_r": best_discovered["weighted_abs_r"],
                    "standard_r": std["weighted_abs_r"],
                }

    return {
        "source_dataset": source_dataset,
        "target_datasets": target_datasets,
        "n_formulas_tested": len(formulas_data),
        "n_discovered": len(discovered),
        "n_standard": len(standard),
        "ranking_all": formulas_data[:20],  # Top 20
        "ranking_discovered": discovered[:10],
        "ranking_standard": standard,
        "best_discovered": discovered[0] if discovered else None,
        "improvements_vs_standard": improvements,
    }


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


def load_discovered_formulas(sr_output_dir: Path, source_dataset: str) -> List[Dict[str, Any]]:
    """Carica le formule scoperte da SR."""
    formulas = []

    # Cerca file equations_unified.csv o equations_top10.csv
    unified_file = list(sr_output_dir.glob("*_equations_unified*.csv"))
    top10_file = list(sr_output_dir.glob("*_equations_top10*.csv"))

    if top10_file:
        df = pd.read_csv(top10_file[0], sep=';', decimal=',')
        log_info(f"Caricato {top10_file[0].name}: {len(df)} formule")
    elif unified_file:
        df = pd.read_csv(unified_file[0], sep=';', decimal=',')
        log_info(f"Caricato {unified_file[0].name}: {len(df)} formule")
    else:
        log_warning(f"Nessun file equazioni trovato in {sr_output_dir}")
        return formulas

    # Converti in lista di dizionari
    for _, row in df.iterrows():
        formulas.append({
            "name": f"SR_{source_dataset}_{row.get('approach', 'unknown')}_{len(formulas)+1}",
            "equation": row.get("equation", ""),
            "approach": row.get("approach", "unknown"),
            "complexity": row.get("complexity", 0),
            "score": row.get("score", 1.0),
            "r_vs_HR": row.get("r_vs_HR", None),
            "source_dataset": source_dataset,
        })

    return formulas


def load_qtc_data(dataset: str, base_path: Path) -> Optional[pd.DataFrame]:
    """Carica i dati QTc preparati per un dataset."""
    qtc_file = base_path / dataset / "qtc" / f"{dataset}_qtc_preparation.csv"

    if not qtc_file.exists():
        log_warning(f"File non trovato: {qtc_file}")
        return None

    df = pd.read_csv(qtc_file, sep=';', decimal=',')
    log_info(f"Caricato {dataset}: {len(df)} record")
    return df


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-Validation formule QTc tra dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Testa formule scoperte su ptb-xl sui dataset chapman e mimic-iv-ecg
  python 07_0_sr_qtc_cross_validation.py --source-dataset ptb-xl --target-datasets chapman mimic-iv-ecg

  # Testa formule scoperte su chapman su tutti gli altri dataset
  python 07_0_sr_qtc_cross_validation.py --source-dataset chapman --target-datasets ptb-xl georgia code-15 mimic-iv-ecg cpsc-2018

  # Specifica directory output custom
  python 07_0_sr_qtc_cross_validation.py --source-dataset ptb-xl --target-datasets chapman --output results/custom
        """,
    )

    parser.add_argument(
        "--source-dataset",
        type=str,
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset da cui prendere le formule scoperte",
    )
    parser.add_argument(
        "--target-datasets",
        type=str,
        nargs="+",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset su cui testare le formule",
    )
    parser.add_argument(
        "--sr-dir",
        type=str,
        help="Directory con output di SR (default: results/{source-dataset}/sr_qtc)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Directory output (default: results/{source-dataset}/cross-validation)",
    )
    parser.add_argument(
        "--results-base",
        type=str,
        default="results",
        help="Directory base dei risultati (default: results)",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default=None,
        help="Nome custom per source (es. 'pool6_exclude_cpsc-2018' per LODO). Default: usa --source-dataset",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Numero di formule top da testare (default: 10)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Testa tutte le formule scoperte (ignora --top-n)",
    )
    parser.add_argument(
        "--include-standard",
        action="store_true",
        default=True,
        help="Includi formule standard per confronto (default: True)",
    )
    parser.add_argument(
        "--no-standard",
        action="store_true",
        help="Escludi formule standard dal confronto",
    )

    args = parser.parse_args()
    
    # Set source_name from argument or default to source_dataset
    if args.source_name is None:
        args.source_name = args.source_dataset

    # Verifica che source non sia in target
    if args.source_dataset in args.target_datasets:
        log_warning(f"Rimuovo {args.source_dataset} dai target (è il source)")
        args.target_datasets = [d for d in args.target_datasets if d != args.source_dataset]

    if not args.target_datasets:
        log_error("Nessun dataset target specificato (dopo rimozione source)")
        sys.exit(1)

    base_path = Path(args.results_base)

    # Header
    print("=" * 70)
    print("KEPLER-ECG: Cross-Validation Formule QTc")
    print(f"Versione: {VERSION}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    print()
    print(f"Source dataset: {args.source_dataset}")
    print(f"Target datasets: {', '.join(args.target_datasets)}")
    print()

    # Carica formule scoperte dal source dataset
    if args.sr_dir:
        sr_dir = Path(args.sr_dir)
    else:
        sr_dir = base_path / args.source_dataset / "sr_qtc"

    if not sr_dir.exists():
        log_error(f"Directory SR non trovata: {sr_dir}")
        sys.exit(1)

    discovered_formulas_raw = load_discovered_formulas(sr_dir, args.source_name)

    if not discovered_formulas_raw:
        log_error(f"Nessuna formula scoperta trovata in {sr_dir}")
        sys.exit(1)

    # Rimuovi duplicati per equazione
    seen_equations = set()
    unique_formulas = []
    for f in discovered_formulas_raw:
        eq = f["equation"]
        if eq not in seen_equations:
            seen_equations.add(eq)
            unique_formulas.append(f)

    log_info(f"Formule scoperte uniche: {len(unique_formulas)}")

    # Seleziona formule per cross-validation
    unique_formulas.sort(key=lambda x: x.get("score", 1.0))

    if args.all:
        top_formulas = unique_formulas
        log_info(f"Modalità --all: test di tutte le {len(top_formulas)} formule")
    else:
        top_formulas = unique_formulas[: args.top_n]
        log_info(f"Selezionate top {args.top_n} formule")

    # Crea FormulaValidator per formule scoperte
    discovered_validators = []
    for f in top_formulas:
        validator = FormulaValidator(
            name=f["name"],
            equation=f["equation"],
            approach=f["approach"],
            is_standard=False,
            source_dataset=f.get("source_dataset"),
        )
        if validator.is_valid:
            discovered_validators.append(validator)
        else:
            log_warning(f"Formula non valida: {f['equation']}")

    log_info(f"Formule scoperte valide: {len(discovered_validators)}")

    # Crea FormulaValidator per formule standard
    standard_validators = []
    if not args.no_standard:
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
        log_info(f"Formule standard incluse: {len(standard_validators)}")

    all_validators = discovered_validators + standard_validators

    # Carica dati per ogni target dataset
    target_validators = {}
    for ds in args.target_datasets:
        df = load_qtc_data(ds, base_path)
        if df is not None:
            target_validators[ds] = DatasetValidator(ds, df)

    if not target_validators:
        log_error("Nessun dataset target caricato")
        sys.exit(1)

    print()
    print("-" * 70)
    print(f"CROSS-VALIDATION SETUP:")
    print(f"  - Formule da testare: {len(all_validators)}")
    print(f"    - Scoperte: {len(discovered_validators)}")
    print(f"    - Standard: {len(standard_validators)}")
    print(f"  - Dataset target: {len(target_validators)}")
    for ds_name, ds_val in target_validators.items():
        print(f"    - {ds_name}: {ds_val.n_records} record")
    print("-" * 70)
    print()

    # Esegui cross-validation
    log_info("Esecuzione cross-validation...")
    cross_validation_results = run_cross_validation(all_validators, target_validators)

    # Genera report di confronto
    comparison_report = generate_comparison_report(
        cross_validation_results,
        args.source_dataset,
        list(target_validators.keys()),
    )

    # Prepara output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = base_path / args.source_name / "cross-validation"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Salva report completo JSON
    full_report = {
        "metadata": {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "source_dataset": args.source_dataset,
            "source_name": args.source_name,
            "target_datasets": list(target_validators.keys()),
            "n_formulas_tested": len(all_validators),
            "n_discovered": len(discovered_validators),
            "n_standard": len(standard_validators),
            "total_records_tested": sum(dv.n_records for dv in target_validators.values()),
        },
        "comparison_summary": comparison_report,
        "detailed_results": cross_validation_results,
    }

    target_str = "_".join(sorted(target_validators.keys()))
    report_file = output_dir / f"{args.source_name}_to_{target_str}_cross_validation.json"
    with open(report_file, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    log_info(f"Report JSON salvato: {report_file}")

    # Salva tabella riassuntiva CSV
    summary_rows = []
    for name, result in cross_validation_results.items():
        summary = result.get("cross_validation_summary", {})

        row = {
            "formula": name,
            "equation": result.get("equation", ""),
            "approach": result.get("approach", ""),
            "is_standard": result.get("is_standard", False),
            "source_dataset": result.get("source_dataset", ""),
            "n_target_datasets": summary.get("n_target_datasets", 0),
            "total_n": summary.get("total_n", 0),
            "weighted_abs_r": summary.get("weighted_abs_r"),
            "unweighted_mean_abs_r": summary.get("unweighted_mean_abs_r"),
            "median_abs_r": summary.get("median_abs_r"),
            "worst_abs_r": summary.get("worst_abs_r"),
            "best_abs_r": summary.get("best_abs_r"),
            "std_r": summary.get("std_r"),
            "all_below_0.05": summary.get("all_below_0.05"),
            "pct_below_0.05": summary.get("pct_below_0.05"),
        }

        # Aggiungi colonne per ogni dataset target
        per_dataset = summary.get("per_dataset", {})
        for ds_name in target_validators.keys():
            ds_data = per_dataset.get(ds_name, {})
            row[f"r_{ds_name}"] = ds_data.get("r")
            row[f"abs_r_{ds_name}"] = ds_data.get("abs_r")
            row[f"n_{ds_name}"] = ds_data.get("n")

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("weighted_abs_r", ascending=True)

    summary_file = output_dir / f"{args.source_name}_to_{target_str}_summary.csv"
    summary_df.to_csv(summary_file, index=False, sep=';', decimal=',')
    log_info(f"Tabella riassuntiva salvata: {summary_file}")

    # Stampa risultati principali
    print()
    print("=" * 70)
    print("RISULTATI CROSS-VALIDATION")
    print("=" * 70)

    # Ranking formule scoperte
    print()
    print(f"RANKING FORMULE SCOPERTE DA {args.source_dataset.upper()}:")
    print("-" * 70)
    discovered_results = [(name, res) for name, res in cross_validation_results.items()
                          if not res.get("is_standard", False)]
    discovered_results.sort(key=lambda x: x[1].get("cross_validation_summary", {}).get("weighted_abs_r", 1.0))

    for i, (name, result) in enumerate(discovered_results[:10], 1):
        summary = result.get("cross_validation_summary", {})
        weighted_r = summary.get("weighted_abs_r", np.nan)
        worst_r = summary.get("worst_abs_r", np.nan)
        eq_short = result.get("equation", "")[:50]
        print(f"  {i:2d}. {name}")
        print(f"      Eq: {eq_short}...")
        print(f"      Weighted |r|: {weighted_r:.4f}, Worst |r|: {worst_r:.4f}")

    # Confronto con standard
    print()
    print("CONFRONTO CON FORMULE STANDARD:")
    print("-" * 70)
    standard_results = [(name, res) for name, res in cross_validation_results.items()
                        if res.get("is_standard", False)]

    for name, result in standard_results:
        summary = result.get("cross_validation_summary", {})
        weighted_r = summary.get("weighted_abs_r", np.nan)
        print(f"  {name}: weighted |r| = {weighted_r:.4f}")

    # Miglioramenti
    if comparison_report.get("improvements_vs_standard"):
        print()
        print("MIGLIORAMENTI VS STANDARD:")
        print("-" * 70)
        best = comparison_report.get("best_discovered")
        if best:
            print(f"  Migliore formula scoperta: {best['name']}")
            print(f"  Weighted |r|: {best['weighted_abs_r']:.4f}")
            for std_name, imp_data in comparison_report["improvements_vs_standard"].items():
                factor = imp_data["improvement_factor"]
                print(f"  vs {std_name}: {factor:.1f}x migliore")

    # Dettaglio per dataset target
    print()
    print("DETTAGLIO PER DATASET TARGET:")
    print("-" * 70)
    for ds_name, ds_val in target_validators.items():
        print(f"\n  {ds_name.upper()} ({ds_val.n_records} record):")
        
        # Top 3 formule scoperte su questo dataset
        ds_results = []
        for name, result in cross_validation_results.items():
            if result.get("is_standard"):
                continue
            ds_detail = result.get("cross_validation_summary", {}).get("per_dataset", {}).get(ds_name, {})
            if ds_detail:
                ds_results.append((name, ds_detail.get("abs_r", 1.0)))
        
        ds_results.sort(key=lambda x: x[1])
        for name, abs_r in ds_results[:3]:
            print(f"    {name}: |r| = {abs_r:.4f}")

    print()
    print("=" * 70)
    print(f"Cross-validation completata. Output in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
