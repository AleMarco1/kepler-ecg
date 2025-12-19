# Kepler-ECG 🫀

**Discovering Interpretable Cardiac Laws from ECG using Symbolic Regression**

[![Python 3.10+](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

## Overview

Kepler-ECG is a research project applying symbolic regression and Minimum Description Length (MDL) principles to discover interpretable mathematical laws from ECG data. Just as Kepler discovered simple laws governing planetary motion from Tycho Brahe's observations, we aim to find simple equations that capture cardiac physiology.

## Project Goals (v2.0)

The project follows a **multi-stream discovery approach**:

| Stream | Goal | Output |
|--------|------|--------|
| **Stream A** | Map which diagnoses can be compressed into equations | Compressibility Map (71 diagnoses) |
| **Stream B** | Discover equations for continuous targets | Cardiac Age, EF estimation formulas |
| **Stream C** | Improve existing formulas | Corrected Bazett, Sokolow-Lyon |

## Project Structure

```
kepler-ecg/
├── configs/                 # Configuration files
├── data/
│   ├── raw/                 # Raw datasets
│   │   ├── ptb-xl/         # Primary dataset (21,837 ECGs)
│   │   ├── cpsc-2018/      # External validation
│   │   ├── georgia/        # External validation
│   │   ├── chapman/        # External validation
│   │   ├── mit-bih/        # Multi-scale (Holter)
│   │   └── ltaf/           # Circadian analysis
│   ├── processed/          # Preprocessed signals
│   ├── features/           # Extracted features
│   └── external/           # External resources
├── docs/                    # Documentation
├── models/
│   ├── autoencoder/        # Learned feature models
│   └── ecg_generator/      # Causal validation
├── notebooks/              # Jupyter notebooks
├── results/
│   ├── laws/               # Discovered equations
│   ├── figures/            # Plots and visualizations
│   └── tables/             # Results tables
├── scripts/                # Utility scripts
├── src/kepler_ecg/         # Main source code
│   ├── data/               # Data loading
│   ├── preprocessing/      # Signal processing
│   ├── features/           # Feature extraction
│   ├── discovery/          # Law discovery
│   │   ├── stream_a_compressibility/
│   │   ├── stream_b_continuous/
│   │   └── stream_c_correction/
│   ├── validation/         # Law validation
│   └── utils/              # Utilities
└── tests/                  # Unit tests
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/kepler-ecg.git
cd kepler-ecg

# Install with Poetry
poetry install

# Activate environment
poetry shell

# Download PTB-XL dataset
python scripts/download_ptbxl.py
```

## Quick Start

```python
# Load and explore data
from kepler_ecg.data import load_ptbxl

records = load_ptbxl("data/raw/ptb-xl")
print(f"Loaded {len(records)} ECG records")
```

## Datasets

| Dataset | Records | Duration | Purpose |
|---------|---------|----------|---------|
| PTB-XL | 21,837 | 10s | Primary training |
| CPSC 2018 | 6,877 | 6-60s | External validation |
| Georgia | 10,344 | 10s | External validation |
| Chapman | 10,646 | 10s | External validation |

## Development

```bash
# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Check linting
flake8 src/ tests/
```

## References

- [PTB-XL Paper](https://www.nature.com/articles/s41597-020-0495-6)
- [PySR Documentation](https://astroautomata.com/PySR/)
- [Symbolic Regression for Scientific Discovery](https://arxiv.org/abs/2305.01582)

## License

MIT License - see [LICENSE](LICENSE) file.

## Authors

Alessandro - Kepler-ECG Project
