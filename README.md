# Kepler-ECG 🫀

**Deriving Interpretable Cardiac Equations from ECG Data using Symbolic Regression**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## The Marconi Formula

The first major output of this project is a novel QTc correction formula derived through symbolic regression on **884,126 ECGs** from **7 international databases** across **5 countries**:

```
QTc = QT + 125/RR − 125
```

where QT is in milliseconds and RR in seconds.

> **Manuscript submitted:** Marconi A. *Derivation and Validation of an Improved Method for Correcting the QT Interval.* Submitted to the Journal of the American College of Cardiology (JACC), February 2026.

### Key Findings

- **2- to 10-fold reduction** in false positive QT prolongation diagnoses compared to Bazett, depending on reference methodology
- **Sensitivity ≥99.5%** on formula-independent reference standards
- **NNT = 23**: for every 23 patients screened with the Marconi formula instead of Bazett, one unnecessary cardiac workup is avoided
- **Zero excess risk** among reclassified patients (all adjusted OR ≤1.0) on 588,374 hospital ECGs with linked outcomes
- **100% sensitivity and specificity** on the LUDB external validation dataset with expert-annotated QT intervals (n=180)

### Use the Formula

**Python**
```python
def marconi_qtc(qt_ms, rr_s):
    """Marconi QTc correction. QT in ms, RR in seconds."""
    return qt_ms + 125 / rr_s - 125
```

**R**
```r
marconi_qtc <- function(qt_ms, rr_s) qt_ms + 125 / rr_s - 125
```

**JavaScript**
```javascript
const marconiQTc = (qt_ms, rr_s) => qt_ms + 125 / rr_s - 125;
```

**Excel / Google Sheets**
```
=A1 + 125/B1 - 125
```
where A1 = QT (ms) and B1 = RR (s).

## Overview

Kepler-ECG applies symbolic regression to discover interpretable mathematical equations from ECG data. Just as Kepler discovered simple laws governing planetary motion from Tycho Brahe's observations, this project aims to find simple equations that capture cardiac physiology from large-scale electrocardiographic recordings.

The analysis pipeline spans 5 phases, from data preprocessing through external validation and clinical outcome analysis:

1. **Data preprocessing** — Standardized ingestion and quality filtering across 7 international ECG databases
2. **Feature extraction** — Automated wave delineation (QT, RR, HR) using NeuroKit2
3. **Symbolic regression discovery** — Formula derivation using PySR with anti-overfitting constraints
4. **Internal validation** — Cross-dataset testing, leave-one-dataset-out (LODO), and stratified analyses
5. **External validation** — Independent gold-standard testing on expert-annotated datasets (LUDB, QT Database)

## Datasets

| Dataset | Country | Setting | Raw ECGs | After Filtering | Role |
|---------|---------|---------|----------|-----------------|------|
| PTB-XL | Germany | Clinical | 21,799 | 15,831 | Derivation |
| Chapman-Shaoxing | China | Clinical | 45,152 | 29,797 | Derivation |
| CPSC-2018 | China | Clinical | 6,877 | 5,769 | Derivation |
| Georgia 12-Lead | USA | Clinical | 10,344 | 7,642 | Derivation |
| MIMIC-IV-ECG | USA | Clinical | 800,036 | 593,775 | Derivation + Outcomes |
| CODE-15 | Brazil | Screening | 345,779 | 231,132 | Derivation |
| LUDB | Russia | Mixed | 200 | 180 | External Validation |
| QT Database | PhysioNet | Mixed | 3,066 beats | 3,066 beats | External Validation |

All datasets are publicly available. See the manuscript for access details and citations.

## Installation

```bash
git clone https://github.com/AleMarco1/kepler-ecg.git
cd kepler-ecg

pip install -r requirements.txt
```

### Dependencies

- Python 3.10+
- PySR 1.5.9 (Julia 1.10 backend)
- NeuroKit2 0.2.7
- pandas 2.3
- SciPy 1.15

## Project Structure

```
kepler-ecg/
├── configs/                  # Configuration files
├── data/
│   ├── raw/                  # Raw datasets (not included, see Datasets)
│   ├── processed/            # Preprocessed signals
│   └── features/             # Extracted features
├── docs/                     # Documentation
├── notebooks/                # Jupyter notebooks
├── results/
│   ├── figures/              # Publication figures
│   └── tables/               # Results tables
├── scripts/                  # Analysis pipeline scripts
├── src/kepler_ecg/           # Source code
│   ├── data/                 # Data loading and ingestion
│   ├── preprocessing/        # Signal processing and QC
│   ├── features/             # Feature extraction
│   ├── discovery/            # Symbolic regression
│   ├── validation/           # Validation and diagnostics
│   └── utils/                # Utilities
└── tests/                    # Unit tests
```

## Reproducibility

The full analysis pipeline can be reproduced by running the scripts in sequence. Each script corresponds to a phase of the study design described in the manuscript. Detailed instructions are provided in the `docs/` directory.

## Citation

If you use this work, please cite:

```bibtex
@article{marconi2026qtc,
  title={Derivation and Validation of an Improved Method for Correcting the QT Interval},
  author={Marconi, Alessandro},
  journal={Journal of the American College of Cardiology},
  year={2026},
  note={Submitted}
}
```

## References

- Wagner P, et al. [PTB-XL, a large publicly available electrocardiography dataset.](https://doi.org/10.1038/s41597-020-0495-6) *Sci Data.* 2020;7:154.
- Cranmer M. [Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl.](https://doi.org/10.48550/arXiv.2305.01582) *arXiv.* 2023.
- Makowski D, et al. [NeuroKit2: A Python toolbox for neurophysiological signal processing.](https://doi.org/10.3758/s13428-020-01516-y) *Behav Res Methods.* 2021;53:1689–1696.
- Bazett HC. An analysis of the time-relations of electrocardiograms. 1920;7:353–370.

## License

MIT License — see [LICENSE](LICENSE) file.

## Author

Alessandro Marconi — Independent Researcher, Verona, Italy

📧 ale1marconi@gmail.com
