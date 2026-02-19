#!/usr/bin/env python3
"""
reproduce_results.py
====================
Standalone script that reproduces all key metrics reported in:

    Marconi A. "Derivation and Validation of an Improved Method for
    Correcting the QT Interval." Submitted to JACC, February 2026.

This script operates on the preprocessed QTc preparation files
({dataset}_qtc_preparation.csv) produced by the upstream pipeline
(scripts 02_0 through 04_0). It does NOT re-run signal processing
or symbolic regression — it validates the published formula and
reproduces every table and figure metric from the manuscript.

CRITICAL METHODOLOGY NOTES:
  - The polynomial reference standard is fitted PER-DATASET, not on the
    pooled data. Each dataset gets its own degree-6 detrending polynomial.
    If the file already contains QTc_reference_ms, that column is used.
  - |r(QTc, HR)| is computed PER-DATASET, then POPULATION-WEIGHTED
    (by dataset size) to produce the reported values in Table 2.
  - False positive rate is computed AGAINST the polynomial reference
    standard: FP = formula says prolonged AND reference says normal.
    FP Rate = FP / (FP + TN) = FP / N_reference_normal.

Usage:
    python validation/reproduce_results.py [--data-dir ./results] [--output-dir ./results/reproduce]

Requirements:
    pip install numpy pandas scipy statsmodels

Inputs (expected in {data-dir}/{dataset}/qtc/{dataset}_qtc_preparation.csv):
    - ptb-xl, chapman, cpsc-2018, georgia, mimic-iv-ecg, code-15  (derivation)
    - ludb  (external validation)

    For outcome analysis (Table 7), also expects:
    - {data-dir}/mimic-iv-ecg/qtc/mimic_outcomes_merged.csv

Outputs:
    - reproduce_table2.csv          Heart rate independence (Table 2)
    - reproduce_table3.csv          Stratified |r| by HR zone and age (Table 3)
    - reproduce_table4.csv          Misclassification by age × HR (Table 4)
    - reproduce_table5.csv          External validation LUDB (Table 5)
    - reproduce_table6.csv          Severity-stratified errors (Table 6)
    - reproduce_table7.csv          MIMIC-IV outcomes (Table 7)
    - reproduce_table8.csv          Alternative reference standards (Table 8)
    - reproduce_summary.txt         Human-readable summary of all key metrics
    - reproduce_figure4_data.csv    Data for Figure 4 (QTc vs HR scatter)

Author: Alessandro Marconi
Date: February 2026
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DERIVATION_DATASETS = [
    "ptb-xl", "chapman", "cpsc-2018", "georgia", "mimic-iv-ecg", "code-15"
]

EXTERNAL_DATASETS = ["ludb"]

ALL_DATASETS = DERIVATION_DATASETS + EXTERNAL_DATASETS

# Sex encoding: 0 = Male, 1 = Female
QTC_THRESHOLD_MALE = 450.0    # ms
QTC_THRESHOLD_FEMALE = 460.0  # ms
SEX_FEMALE = 1
SEX_MALE = 0

THRESHOLD_10MS = 10.0   # ICH E14
HR_BRADY = 60
HR_TACHY = 100
AGE_YOUNG = 40
AGE_MIDDLE = 65
N_BOOTSTRAP = 1000

# Alternative reference standards
HR_BIN_WIDTH = 5
MIN_BIN_SIZE = 50
PROLONGATION_PERCENTILE = 97.5
POLY_DEGREE_REF = 6   # Per-dataset polynomial reference
POLY_DEGREE_RES = 3   # QT-RR residuals (Reference C)


# =============================================================================
# FORMULA DEFINITIONS
# =============================================================================

def marconi(qt, rr, hr):   return qt + 125.0 / rr - 125.0
def bazett(qt, rr, hr):    return qt / np.sqrt(rr)
def fridericia(qt, rr, hr): return qt / np.cbrt(rr)
def framingham(qt, rr, hr): return qt + 154.0 * (1.0 - rr)
def hodges(qt, rr, hr):    return qt + 1.75 * (hr - 60.0)

FORMULAS = {
    "Marconi": marconi, "Bazett": bazett, "Fridericia": fridericia,
    "Framingham": framingham, "Hodges": hodges,
}


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_dataset(data_dir: Path, name: str) -> Optional[pd.DataFrame]:
    """Load a dataset's QTc preparation CSV."""
    candidates = [
        data_dir / name / "qtc" / f"{name}_qtc_preparation.csv",
        data_dir / name / f"{name}_qtc_preparation.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path, sep=";", decimal=",")
            except Exception:
                df = pd.read_csv(path)
            return standardize_columns(df, name)
    print(f"  [SKIP] {name}: file not found")
    return None


def standardize_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Ensure consistent column names across datasets."""
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("qt_interval_ms", "qt_ms", "qt"):             col_map[c] = "QT_ms"
        elif cl in ("rr_interval_ms", "rr_ms"):                  col_map[c] = "RR_ms"
        elif cl in ("rr_interval_s", "rr_s", "rr"):              col_map[c] = "RR_s"
        elif cl in ("heart_rate", "hr", "heart_rate_bpm", "hr_bpm"): col_map[c] = "HR"
        elif cl == "sex":                                         col_map[c] = "sex"
        elif cl == "age":                                         col_map[c] = "age"
        elif cl in ("qtc_reference_ms", "qtc_ref_ms"):            col_map[c] = "QTc_reference_ms"
        elif cl in ("primary_superclass", "superclass"):          col_map[c] = "superclass"
    df = df.rename(columns=col_map)

    for col in ["QT_ms", "RR_s", "RR_ms", "HR", "age", "sex", "QTc_reference_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "RR_s" not in df.columns and "RR_ms" in df.columns:
        df["RR_s"] = df["RR_ms"] / 1000.0
    if "HR" not in df.columns and "RR_s" in df.columns:
        df["HR"] = 60.0 / df["RR_s"]
    if "RR_s" not in df.columns and "HR" in df.columns:
        df["RR_s"] = 60.0 / df["HR"]
    if "RR_ms" not in df.columns and "RR_s" in df.columns:
        df["RR_ms"] = df["RR_s"] * 1000.0

    df["source_dataset"] = dataset_name
    return df


def get_threshold(sex_value):
    return QTC_THRESHOLD_FEMALE if sex_value == SEX_FEMALE else QTC_THRESHOLD_MALE


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute QTc for all formulas + PER-DATASET polynomial reference.
    Uses QTc_reference_ms from file if present, otherwise fits degree-6 poly.
    """
    qt, rr, hr = df["QT_ms"].values, df["RR_s"].values, df["HR"].values

    for name, func in FORMULAS.items():
        df[f"QTc_{name}"] = func(qt, rr, hr)

    df["threshold"] = df["sex"].apply(get_threshold)
    for name in FORMULAS:
        df[f"prolonged_{name}"] = df[f"QTc_{name}"] >= df["threshold"]

    # Per-dataset polynomial reference standard
    if "QTc_reference_ms" in df.columns and df["QTc_reference_ms"].notna().sum() > 100:
        df["QTc_reference"] = df["QTc_reference_ms"]
    else:
        coeffs = np.polyfit(rr, qt, POLY_DEGREE_REF)
        df["QTc_reference"] = qt - np.polyval(coeffs, rr) + np.mean(qt)

    df["prolonged_reference"] = df["QTc_reference"] >= df["threshold"]
    return df


# =============================================================================
# CORE METRICS
# =============================================================================

def pearson_abs_r(qtc, hr):
    """Absolute Pearson |r| between QTc and HR."""
    mask = np.isfinite(qtc) & np.isfinite(hr)
    if mask.sum() < 10:
        return np.nan
    return abs(stats.pearsonr(qtc[mask], hr[mask])[0])


def population_weighted_r(datasets, formula_name, ds_names):
    """Population-weighted |r(QTc, HR)| across datasets."""
    total_n, weighted_sum = 0, 0.0
    for ds in ds_names:
        if ds not in datasets:
            continue
        df = datasets[ds]
        n = len(df)
        r = pearson_abs_r(df[f"QTc_{formula_name}"].values, df["HR"].values)
        if not np.isnan(r):
            weighted_sum += r * n
            total_n += n
    return weighted_sum / total_n if total_n > 0 else np.nan


def bootstrap_weighted_r(datasets, formula_name, ds_names, n_boot=N_BOOTSTRAP):
    """Bootstrap 95% CI for population-weighted |r|."""
    rng = np.random.default_rng(42)
    # Pre-extract per-dataset arrays
    ds_data = {}
    for ds in ds_names:
        if ds not in datasets:
            continue
        df = datasets[ds]
        qtc, hr = df[f"QTc_{formula_name}"].values, df["HR"].values
        mask = np.isfinite(qtc) & np.isfinite(hr)
        ds_data[ds] = (qtc[mask], hr[mask])

    boot_rs = np.empty(n_boot)
    for i in range(n_boot):
        wsum, tn = 0.0, 0
        for ds, (qtc, hr) in ds_data.items():
            n = len(qtc)
            idx = rng.integers(0, n, size=n)
            r = abs(stats.pearsonr(qtc[idx], hr[idx])[0])
            wsum += r * n
            tn += n
        boot_rs[i] = wsum / tn if tn > 0 else np.nan
    return (np.nanpercentile(boot_rs, 2.5), np.nanpercentile(boot_rs, 97.5))


def bootstrap_abs_r(qtc, hr, n_boot=N_BOOTSTRAP):
    """Bootstrap 95% CI for single-dataset |r|."""
    mask = np.isfinite(qtc) & np.isfinite(hr)
    qtc_c, hr_c = qtc[mask], hr[mask]
    n = len(qtc_c)
    if n < 30:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rs[i] = abs(stats.pearsonr(qtc_c[idx], hr_c[idx])[0])
    return (np.percentile(rs, 2.5), np.percentile(rs, 97.5))


def classification_metrics(y_true, y_pred):
    """TP, FP, TN, FN, sensitivity, specificity, PPV, NPV, accuracy."""
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else np.nan
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Sensitivity": sens, "Specificity": spec,
            "PPV": ppv, "NPV": npv, "Accuracy": acc, "FP_rate": fp_rate}


# =============================================================================
# TABLE 2: HEART RATE INDEPENDENCE (POPULATION-WEIGHTED)
# =============================================================================

def reproduce_table2(datasets, pooled):
    """
    Table 2: Population-weighted |r(QTc, HR)| for each formula.
    FP rate = FP / (FP + TN) against per-dataset polynomial reference.
    """
    print("\n" + "=" * 70)
    print("TABLE 2: Heart Rate Independence of QTc Correction Formulas")
    print("=" * 70)

    rows = []
    for name in FORMULAS:
        r_w = population_weighted_r(datasets, name, DERIVATION_DATASETS)
        ci_lo, ci_hi = bootstrap_weighted_r(datasets, name, DERIVATION_DATASETS)

        # FP rate against per-dataset polynomial reference (pooled)
        ref = pooled["prolonged_reference"].values
        pred = pooled[f"prolonged_{name}"].values
        fp = int(np.sum(~ref & pred))
        n_ref_normal = int(np.sum(~ref))
        fp_rate = fp / n_ref_normal * 100 if n_ref_normal > 0 else 0.0

        rows.append({
            "Formula": name,
            "|r(QTc,HR)|": round(r_w, 3),
            "95% CI low": round(ci_lo, 3),
            "95% CI high": round(ci_hi, 3),
            "FP Rate %": round(fp_rate, 2),
        })

    t = pd.DataFrame(rows).sort_values("|r(QTc,HR)|")
    t["Rank"] = range(1, len(t) + 1)
    baz_r = t.loc[t["Formula"] == "Bazett", "|r(QTc,HR)|"].values[0]
    t["vs Bazett"] = t["|r(QTc,HR)|"].apply(
        lambda x: f"{baz_r/x:.1f}x better" if x < baz_r else
                  (f"{x/baz_r:.1f}x worse" if x > baz_r else "---"))

    print(t.to_string(index=False))
    print(f"\nPooled N = {len(pooled):,}")

    print("\nPer-dataset |r| breakdown:")
    for ds in DERIVATION_DATASETS:
        if ds not in datasets:
            continue
        df = datasets[ds]
        r_m = pearson_abs_r(df["QTc_Marconi"].values, df["HR"].values)
        r_b = pearson_abs_r(df["QTc_Bazett"].values, df["HR"].values)
        print(f"  {ds:15s} (N={len(df):>7,}): Marconi={r_m:.4f}  Bazett={r_b:.4f}")

    return t


# =============================================================================
# TABLE 3: STRATIFIED PERFORMANCE (POPULATION-WEIGHTED)
# =============================================================================

def reproduce_table3(datasets, pooled):
    """Table 3: Population-weighted |r| by HR zone and age group."""
    print("\n" + "=" * 70)
    print("TABLE 3: Stratified Performance by Heart Rate Zone and Age Group")
    print("=" * 70)

    strata_defs = {
        "Bradycardia (<60)":  lambda df: df["HR"] < HR_BRADY,
        "Normal (60-100)":    lambda df: (df["HR"] >= HR_BRADY) & (df["HR"] <= HR_TACHY),
        "Tachycardia (>100)": lambda df: df["HR"] > HR_TACHY,
        "Age <40":            lambda df: df["age"] < AGE_YOUNG,
        "Age 40-65":          lambda df: (df["age"] >= AGE_YOUNG) & (df["age"] < AGE_MIDDLE),
        "Age >65":            lambda df: df["age"] >= AGE_MIDDLE,
    }

    rows = []
    for sname, mask_fn in strata_defs.items():
        total_n = 0
        f_wsum = {f: 0.0 for f in FORMULAS}

        for ds in DERIVATION_DATASETS:
            if ds not in datasets:
                continue
            df = datasets[ds]
            mask = mask_fn(df)
            sub = df[mask]
            n = len(sub)
            if n < 30:
                continue
            total_n += n
            hr_sub = sub["HR"].values
            for fname in FORMULAS:
                r = pearson_abs_r(sub[f"QTc_{fname}"].values, hr_sub)
                if not np.isnan(r):
                    f_wsum[fname] += r * n

        if total_n < 30:
            continue

        row = {"Stratum": sname, "N": total_n}
        for fname in FORMULAS:
            row[f"|r| {fname}"] = round(f_wsum[fname] / total_n, 3)
        r_m, r_b = row["|r| Marconi"], row["|r| Bazett"]
        row["Improvement"] = f"{r_b/r_m:.1f}x" if r_m > 0 else "—"
        rows.append(row)

    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    return t


# =============================================================================
# TABLE 4: MISCLASSIFICATION BY AGE x HR
# =============================================================================

def reproduce_table4(pooled):
    """Table 4: FP%, FN%, MC% across age × HR vs per-dataset reference."""
    print("\n" + "=" * 70)
    print("TABLE 4: Misclassification Analysis by Age and Heart Rate Strata")
    print("=" * 70)

    hr, age, ref = pooled["HR"].values, pooled["age"].values, pooled["prolonged_reference"].values

    hr_zones = {"Bradycardia": hr < HR_BRADY, "Normal": (hr >= HR_BRADY) & (hr <= HR_TACHY),
                "Tachycardia": hr > HR_TACHY}
    age_groups = {"<40": age < AGE_YOUNG, "40-65": (age >= AGE_YOUNG) & (age < AGE_MIDDLE),
                  ">65": age >= AGE_MIDDLE}

    rows = []
    for aname, amask in age_groups.items():
        for hname, hmask in hr_zones.items():
            mask = amask & hmask
            n = mask.sum()
            if n < 10:
                continue
            ref_sub = ref[mask]
            for fname in FORMULAS:
                pred = pooled[f"prolonged_{fname}"].values[mask]
                fp = int(np.sum(~ref_sub & pred))
                fn = int(np.sum(ref_sub & ~pred))
                rows.append({"Age": aname, "HR Zone": hname, "N": n, "Formula": fname,
                             "FP%": round(fp / n * 100, 2), "FN%": round(fn / n * 100, 2),
                             "MC%": round((fp + fn) / n * 100, 2)})

    t = pd.DataFrame(rows)
    for fname in ["Marconi", "Bazett"]:
        print(f"\n--- {fname} ---")
        print(t[t["Formula"] == fname].pivot_table(values="MC%", index="Age", columns="HR Zone").to_string())
    return t


# =============================================================================
# TABLE 5: EXTERNAL VALIDATION (LUDB)
# =============================================================================

def reproduce_table5(ludb):
    """Table 5: LUDB gold standard validation."""
    print("\n" + "=" * 70)
    print("TABLE 5: External Validation — LUDB Gold Standard")
    print("=" * 70)

    hr, ref = ludb["HR"].values, ludb["prolonged_reference"].values
    rows = []
    for name in FORMULAS:
        qtc = ludb[f"QTc_{name}"].values
        r_abs = pearson_abs_r(qtc, hr)
        ci_lo, ci_hi = bootstrap_abs_r(qtc, hr)
        cm = classification_metrics(ref, ludb[f"prolonged_{name}"].values)
        rows.append({
            "Formula": name, "|r(QTc,HR)|": round(r_abs, 3),
            "95% CI low": round(ci_lo, 3), "95% CI high": round(ci_hi, 3),
            "Sensitivity %": round(cm["Sensitivity"] * 100, 1) if not np.isnan(cm["Sensitivity"]) else "N/A",
            "Specificity %": round(cm["Specificity"] * 100, 1) if not np.isnan(cm["Specificity"]) else "N/A",
            "FP": cm["FP"], "FN": cm["FN"],
            "FP Rate %": round(cm["FP_rate"] * 100, 1) if not np.isnan(cm["FP_rate"]) else "N/A",
        })

    t = pd.DataFrame(rows).sort_values("|r(QTc,HR)|")
    t["Rank"] = range(1, len(t) + 1)
    baz_r = t.loc[t["Formula"] == "Bazett", "|r(QTc,HR)|"].values[0]
    t["vs Bazett"] = t["|r(QTc,HR)|"].apply(
        lambda x: f"{baz_r/x:.1f}x better" if x < baz_r else
                  (f"{x/baz_r:.1f}x worse" if x > baz_r else "---"))
    print(t.to_string(index=False))
    print(f"\nLUDB N = {len(ludb)}")
    return t


# =============================================================================
# TABLE 6: SEVERITY-STRATIFIED ERRORS
# =============================================================================

def reproduce_table6(pooled):
    """Table 6: FP/FN stratified by distance from threshold."""
    print("\n" + "=" * 70)
    print("TABLE 6: Severity-Stratified Misclassification Analysis")
    print("=" * 70)

    ref, threshold = pooled["prolonged_reference"].values, pooled["threshold"].values
    rows = []
    for fname in FORMULAS:
        qtc = pooled[f"QTc_{fname}"].values
        pred = pooled[f"prolonged_{fname}"].values
        qtc_std = np.nanstd(qtc)
        dist = np.abs(qtc - threshold)

        fp_mask = ~ref & pred
        fn_mask = ref & ~pred

        rows.append({
            "Formula": fname,
            "FP ≤10ms": int(np.sum(fp_mask & (dist <= THRESHOLD_10MS))),
            "FP >10ms": int(np.sum(fp_mask & (dist > THRESHOLD_10MS))),
            "FP >1SD":  int(np.sum(fp_mask & (dist > qtc_std))),
            "FP Total": int(np.sum(fp_mask)),
            "FN ≤10ms": int(np.sum(fn_mask & (dist <= THRESHOLD_10MS))),
            "FN >10ms": int(np.sum(fn_mask & (dist > THRESHOLD_10MS))),
            "FN >1SD":  int(np.sum(fn_mask & (dist > qtc_std))),
            "FN Total": int(np.sum(fn_mask)),
            "Significant (>10ms)": int(np.sum(fp_mask & (dist > THRESHOLD_10MS))) +
                                   int(np.sum(fn_mask & (dist > THRESHOLD_10MS))),
        })

    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    m_sig = t.loc[t["Formula"] == "Marconi", "Significant (>10ms)"].values[0]
    b_sig = t.loc[t["Formula"] == "Bazett", "Significant (>10ms)"].values[0]
    if m_sig > 0:
        print(f"\nClinically significant: Marconi {m_sig:,} vs Bazett {b_sig:,} → {b_sig/m_sig:.1f}x reduction")
    return t


# =============================================================================
# TABLE 7: MIMIC-IV HOSPITAL OUTCOMES
# =============================================================================

def reproduce_table7(data_dir):
    """Table 7: Clinical outcomes by Bazett-Marconi concordance."""
    print("\n" + "=" * 70)
    print("TABLE 7: Clinical Outcomes by QTc Classification Concordance")
    print("=" * 70)

    candidates = [
        data_dir / "mimic-iv-ecg" / "qtc" / "mimic_outcomes_merged.csv",
        data_dir / "mimic-iv-ecg" / "qtc" / "mimic-iv-ecg_qtc_preparation_with_outcomes.csv",
    ]
    df = None
    for path in candidates:
        if path.exists():
            try: df = pd.read_csv(path, sep=";", decimal=",")
            except: df = pd.read_csv(path)
            break
    if df is None:
        print("  [SKIP] MIMIC outcome data not found.")
        return None

    df = standardize_columns(df, "mimic-iv-ecg")
    df = prepare_dataset(df)

    baz, mar = df["prolonged_Bazett"].values, df["prolonged_Marconi"].values
    df["group"] = "Other"
    df.loc[~baz & ~mar, "group"] = "Concordant Normal"
    df.loc[baz & ~mar, "group"] = "Reclassified"
    df.loc[baz & mar, "group"] = "Concordant Prolonged"

    print(f"\n  Concordant Normal:    {(df['group'] == 'Concordant Normal').sum():>8,}")
    print(f"  Reclassified:         {(df['group'] == 'Reclassified').sum():>8,}")
    print(f"  Concordant Prolonged: {(df['group'] == 'Concordant Prolonged').sum():>8,}")

    # Find ICD column
    icd_col = None
    for c in df.columns:
        if "diag" in c.lower() and ("all" in c.lower() or "hosp" in c.lower()):
            icd_col = c; break
    if icd_col is None:
        print("  [SKIP] No ICD-10 column found."); return None

    def has_icd(row, prefixes):
        val = str(row.get(icd_col, ""))
        return any(p in val for p in prefixes)

    df["ventricular_arrhythmia"] = df.apply(
        lambda r: has_icd(r, ["I470", "I472", "I490", "I4901", "I4902"]), axis=1)
    df["cardiac_arrest"] = df.apply(lambda r: has_icd(r, ["I46"]), axis=1)

    for days, col in [(30, "mortality_30d"), (90, "mortality_90d")]:
        if "dod" in df.columns and "ecg_time" in df.columns:
            dod = pd.to_datetime(df["dod"], errors="coerce")
            ecg = pd.to_datetime(df["ecg_time"], errors="coerce")
            delta = (dod - ecg).dt.days
            df[col] = (delta >= 0) & (delta <= days)
        elif col not in df.columns:
            df[col] = np.nan

    if "mortality_inhospital" not in df.columns:
        df["mortality_inhospital"] = pd.to_datetime(df.get("dod"), errors="coerce").notna() \
            if "dod" in df.columns else np.nan

    outcomes = {"Ventricular Arrhythmia": "ventricular_arrhythmia",
                "Cardiac Arrest": "cardiac_arrest",
                "30-Day Mortality": "mortality_30d",
                "90-Day Mortality": "mortality_90d",
                "In-Hospital Mortality": "mortality_inhospital"}

    cn = df[df["group"] == "Concordant Normal"]
    rec = df[df["group"] == "Reclassified"]
    rows = []
    for oname, col in outcomes.items():
        if col not in df.columns or df[col].isna().all(): continue
        cn_r = cn[col].mean() if len(cn) > 0 else np.nan
        re_r = rec[col].mean() if len(rec) > 0 else np.nan
        urr = re_r / cn_r if cn_r > 0 else np.nan
        aor, cl, ch, pv = _adjusted_or(df[df["group"].isin(["Concordant Normal", "Reclassified"])], col)
        rows.append({
            "Outcome": oname,
            "CN Rate": f"{cn_r*100:.2f}%" if not np.isnan(cn_r) else "N/A",
            "Reclass Rate": f"{re_r*100:.2f}%" if not np.isnan(re_r) else "N/A",
            "Unadj RR": round(urr, 2) if not np.isnan(urr) else "N/A",
            "Adj OR": round(aor, 2) if not np.isnan(aor) else "N/A",
            "95% CI": f"({cl:.2f}-{ch:.2f})" if not np.isnan(cl) else "N/A",
            "p-value": f"{pv:.3f}" if not np.isnan(pv) else "N/A",
        })

    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    return t


def _adjusted_or(df_sub, outcome_col):
    """Logistic regression adjusted for age, sex, HR."""
    try:
        import statsmodels.api as sm
        df_c = df_sub[["group", outcome_col, "age", "sex", "HR"]].dropna()
        if len(df_c) < 100: return (np.nan,) * 4
        df_c["is_reclass"] = (df_c["group"] == "Reclassified").astype(int)
        X = sm.add_constant(df_c[["is_reclass", "age", "sex", "HR"]])
        model = sm.Logit(df_c[outcome_col].astype(int), X).fit(disp=False, maxiter=100)
        c = model.params["is_reclass"]
        ci = model.conf_int().loc["is_reclass"]
        return (np.exp(c), np.exp(ci[0]), np.exp(ci[1]), model.pvalues["is_reclass"])
    except Exception as e:
        print(f"    [WARN] Logistic regression failed: {e}")
        return (np.nan,) * 4


# =============================================================================
# TABLE 8: ALTERNATIVE REFERENCE STANDARDS
# =============================================================================

def reproduce_table8(datasets):
    """Table 8: Accuracy across polynomial, HR-bin, QT-RR residual references."""
    print("\n" + "=" * 70)
    print("TABLE 8: Diagnostic Accuracy Across Reference Standards")
    print("=" * 70)

    rows = []
    for ds in DERIVATION_DATASETS:
        if ds not in datasets: continue
        df = datasets[ds]
        ref_a = df["prolonged_reference"].values
        ref_b = _hr_bin_reference(df)
        ref_c = _qt_rr_residual_reference(df)

        for rname, rlabels in [("Polynomial", ref_a), ("HR-Bin Percentile", ref_b),
                                ("QT-RR Residuals", ref_c)]:
            if rlabels is None: continue
            for fname in FORMULAS:
                cm = classification_metrics(rlabels, df[f"prolonged_{fname}"].values)
                rows.append({
                    "Dataset": ds, "Reference": rname, "Formula": fname,
                    "FP%": round(cm["FP_rate"] * 100, 1) if not np.isnan(cm["FP_rate"]) else np.nan,
                    "Sensitivity%": round(cm["Sensitivity"] * 100, 1) if not np.isnan(cm["Sensitivity"]) else np.nan,
                })

    t = pd.DataFrame(rows)
    ds_sizes = {ds: len(datasets[ds]) for ds in DERIVATION_DATASETS if ds in datasets}
    for ref in ["Polynomial", "HR-Bin Percentile", "QT-RR Residuals"]:
        sub = t[t["Reference"] == ref]
        if len(sub) == 0: continue
        print(f"\n--- {ref} (population-weighted) ---")
        for fname in FORMULAS:
            fsub = sub[sub["Formula"] == fname]
            wfp, wsens, tn = 0.0, 0.0, 0
            for _, row in fsub.iterrows():
                n = ds_sizes.get(row["Dataset"], 0)
                if n > 0 and not np.isnan(row["FP%"]):
                    wfp += row["FP%"] * n; wsens += row["Sensitivity%"] * n; tn += n
            if tn > 0:
                print(f"  {fname:12s}: FP% = {wfp/tn:.2f}, Sens% = {wsens/tn:.1f}")
    return t


def _hr_bin_reference(df):
    """Reference B: HR-binned QT percentile."""
    qt, hr, sex = df["QT_ms"].values, df["HR"].values, df["sex"].values
    prolonged = np.zeros(len(df), dtype=bool)
    for sv in [SEX_MALE, SEX_FEMALE]:
        sm = sex == sv
        if sm.sum() < MIN_BIN_SIZE: continue
        hmin = int(np.floor(hr[sm].min() / HR_BIN_WIDTH) * HR_BIN_WIDTH)
        hmax = int(np.ceil(hr[sm].max() / HR_BIN_WIDTH) * HR_BIN_WIDTH)
        for bs in range(hmin, hmax, HR_BIN_WIDTH):
            bm = sm & (hr >= bs) & (hr < bs + HR_BIN_WIDTH)
            if bm.sum() < MIN_BIN_SIZE: continue
            prolonged[bm] = qt[bm] >= np.percentile(qt[bm], PROLONGATION_PERCENTILE)
    return prolonged


def _qt_rr_residual_reference(df):
    """Reference C: QT~RR polynomial regression residuals."""
    qt, rr, sex = df["QT_ms"].values, df["RR_s"].values, df["sex"].values
    prolonged = np.zeros(len(df), dtype=bool)
    for sv in [SEX_MALE, SEX_FEMALE]:
        sm = sex == sv
        if sm.sum() < 100: continue
        coeffs = np.polyfit(rr[sm], qt[sm], POLY_DEGREE_RES)
        res = qt[sm] - np.polyval(coeffs, rr[sm])
        prolonged[sm] = res >= np.percentile(res, PROLONGATION_PERCENTILE)
    return prolonged


# =============================================================================
# FIGURE 4 DATA & SUMMARY
# =============================================================================

def export_figure4_data(pooled, output_dir):
    n_sample = min(50000, len(pooled))
    idx = np.random.default_rng(42).choice(len(pooled), size=n_sample, replace=False)
    cols = ["HR", "sex", "threshold", "source_dataset"] + [f"QTc_{f}" for f in FORMULAS]
    pooled.iloc[idx][cols].to_csv(output_dir / "reproduce_figure4_data.csv", index=False)
    print(f"  Figure 4 data: {n_sample:,} records saved")


def write_summary(datasets, pooled, ludb, output_dir):
    """Human-readable summary of key metrics."""
    lines = ["=" * 70, "REPRODUCE RESULTS — KEY METRICS SUMMARY",
             "Marconi QTc = QT + 125/RR − 125", "=" * 70, "",
             f"Pooled derivation: N = {len(pooled):,}"]
    if ludb is not None:
        lines.append(f"LUDB external:     N = {len(ludb):,}")

    r_m = population_weighted_r(datasets, "Marconi", DERIVATION_DATASETS)
    r_b = population_weighted_r(datasets, "Bazett", DERIVATION_DATASETS)
    r_f = population_weighted_r(datasets, "Fridericia", DERIVATION_DATASETS)

    lines += ["", "--- PRIMARY ENDPOINT (population-weighted |r|) ---",
              f"Marconi    |r| = {r_m:.3f}",
              f"Bazett     |r| = {r_b:.3f}",
              f"Improvement: {r_b/r_m:.1f}x" if r_m > 0 else "",
              f"Fridericia |r| = {r_f:.3f} ({r_f/r_b:.1f}x worse than Bazett)" if r_b > 0 else ""]

    ref = pooled["prolonged_reference"].values
    fp_m = int(np.sum(~ref & pooled["prolonged_Marconi"].values))
    fp_b = int(np.sum(~ref & pooled["prolonged_Bazett"].values))
    n_norm = int(np.sum(~ref))

    lines += ["", "--- FALSE POSITIVES (vs polynomial reference) ---",
              f"Bazett  FP: {fp_b:,} ({fp_b/n_norm*100:.2f}% of reference-normals)",
              f"Marconi FP: {fp_m:,} ({fp_m/n_norm*100:.2f}% of reference-normals)",
              f"Reduction: {fp_b/fp_m:.1f}x" if fp_m > 0 else "",
              f"NNT: {len(pooled) / (fp_b - fp_m):.1f}" if fp_b > fp_m else ""]

    for fname in ["Marconi", "Bazett"]:
        cm = classification_metrics(ref, pooled[f"prolonged_{fname}"].values)
        lines += [f"\n{fname} vs Polynomial Reference:",
                  f"  Sensitivity: {cm['Sensitivity']*100:.1f}%",
                  f"  Specificity: {cm['Specificity']*100:.1f}%",
                  f"  PPV: {cm['PPV']*100:.1f}%",
                  f"  NPV: {cm['NPV']*100:.1f}%",
                  f"  Accuracy: {cm['Accuracy']*100:.1f}%"]

    if ludb is not None:
        r_ml = pearson_abs_r(ludb["QTc_Marconi"].values, ludb["HR"].values)
        r_bl = pearson_abs_r(ludb["QTc_Bazett"].values, ludb["HR"].values)
        cm_l = classification_metrics(ludb["prolonged_reference"].values,
                                      ludb["prolonged_Marconi"].values)
        lines += ["", "--- EXTERNAL VALIDATION (LUDB) ---",
                  f"Marconi |r| = {r_ml:.3f}",
                  f"Bazett  |r| = {r_bl:.3f}",
                  f"Sensitivity: {cm_l['Sensitivity']*100:.1f}%",
                  f"Specificity: {cm_l['Specificity']*100:.1f}%"]

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(output_dir / "reproduce_summary.txt", "w") as f:
        f.write(summary)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reproduce Marconi QTc paper results.")
    parser.add_argument("--data-dir", type=str, default="./results")
    parser.add_argument("--output-dir", type=str, default="./results/reproduce")
    parser.add_argument("--skip-outcomes", action="store_true")
    parser.add_argument("--skip-table8", action="store_true")
    args = parser.parse_args()

    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("KEPLER-ECG: Reproduce Paper Results")
    print("Marconi QTc = QT + 125/RR − 125")
    print("=" * 70)

    # ── Load datasets ────────────────────────────────────────────────────
    print("\n[1/9] Loading datasets...")
    datasets = {}
    for name in ALL_DATASETS:
        df = load_dataset(data_dir, name)
        if df is not None:
            datasets[name] = df
            print(f"  ✓ {name:15s} {len(df):>10,} records")

    if not datasets:
        print("\nERROR: No datasets loaded. Check --data-dir."); sys.exit(1)

    # ── Prepare EACH dataset independently (per-dataset polynomial ref) ──
    print("\n[2/9] Per-dataset QTc computation and polynomial reference...")
    for name in datasets:
        datasets[name] = prepare_dataset(datasets[name])
        r = pearson_abs_r(datasets[name]["QTc_Marconi"].values, datasets[name]["HR"].values)
        print(f"  ✓ {name:15s} Marconi |r| = {r:.4f}")

    # ── Pool (per-dataset references are preserved in each row) ──────────
    derivation_dfs = [datasets[k] for k in DERIVATION_DATASETS if k in datasets]
    if not derivation_dfs:
        print("\nERROR: No derivation datasets."); sys.exit(1)

    pooled = pd.concat(derivation_dfs, ignore_index=True)
    ludb = datasets.get("ludb")
    print(f"\n  Pooled derivation: {len(pooled):,} from {len(derivation_dfs)} datasets")

    # ── Reproduce tables ─────────────────────────────────────────────────
    print("\n[3/9] Table 2: Heart Rate Independence...")
    reproduce_table2(datasets, pooled).to_csv(output_dir / "reproduce_table2.csv", index=False)

    print("\n[4/9] Table 3: Stratified Performance...")
    reproduce_table3(datasets, pooled).to_csv(output_dir / "reproduce_table3.csv", index=False)

    print("\n[5/9] Table 4: Misclassification by Age × HR...")
    reproduce_table4(pooled).to_csv(output_dir / "reproduce_table4.csv", index=False)

    if ludb is not None:
        print("\n[6/9] Table 5: External Validation (LUDB)...")
        reproduce_table5(ludb).to_csv(output_dir / "reproduce_table5.csv", index=False)
    else:
        print("\n[6/9] Table 5: SKIPPED (LUDB not loaded)")

    print("\n[7/9] Table 6: Severity-Stratified Errors...")
    reproduce_table6(pooled).to_csv(output_dir / "reproduce_table6.csv", index=False)

    if not args.skip_outcomes:
        print("\n[8/9] Table 7: MIMIC-IV Outcomes...")
        t7 = reproduce_table7(data_dir)
        if t7 is not None:
            t7.to_csv(output_dir / "reproduce_table7.csv", index=False)
    else:
        print("\n[8/9] Table 7: SKIPPED")

    if not args.skip_table8:
        print("\n[9/9] Table 8: Alternative Reference Standards...")
        reproduce_table8(datasets).to_csv(output_dir / "reproduce_table8.csv", index=False)
    else:
        print("\n[9/9] Table 8: SKIPPED")

    export_figure4_data(pooled, output_dir)
    write_summary(datasets, pooled, ludb, output_dir)

    print("\n" + "=" * 70)
    print(f"All outputs saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
