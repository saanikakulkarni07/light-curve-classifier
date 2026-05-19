# Light Curve Classifier: Transient & Variable Star Classification

Classifying variable stars and transients from multi-band photometric light curves using machine learning. Trained on PLAsTiCC (Photometric LSST Astronomical Time-Series Classification Challenge) simulated LSST data — 14 astrophysical classes across galactic and extragalactic sources.

## Motivation

The Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST) will generate millions of transient alerts per night. Automated classification from photometric light curves alone — without spectroscopic follow-up — is essential for real-time science. This project builds and compares two approaches: hand-crafted feature engineering with a Random Forest, and a recurrent neural network (GRU) operating directly on raw light curve sequences.

## Data

- **Source:** [PLAsTiCC Kaggle Competition](https://www.kaggle.com/competitions/PLAsTiCC-2018/data)
- **Training set:** ~7,800 labeled objects across 14 classes
- **Test set:** ~3.5M objects (used for scaling, not for initial development)
- **Bands:** LSST ugrizy (6 passbands)
- **Features per observation:** MJD, passband, flux, flux_err, detected flag

### Classes (14)

| Class ID | Name | Type |
|----------|------|------|
| 6 | Microlensing | Galactic |
| 15 | TDE (Tidal Disruption Event) | Extragalactic |
| 16 | Eclipsing Binary | Galactic |
| 42 | SN II | Extragalactic |
| 52 | SN Iax | Extragalactic |
| 53 | Mira | Galactic |
| 62 | SN Ibc | Extragalactic |
| 64 | Kilonova | Extragalactic |
| 65 | M-dwarf Flare | Galactic |
| 67 | SN Ia-91bg | Extragalactic |
| 88 | AGN | Extragalactic |
| 90 | SN Ia | Extragalactic |
| 92 | RR Lyrae | Galactic |
| 95 | SLSN-I | Extragalactic |

### Data Download

1. Go to the [PLAsTiCC Data page](https://www.kaggle.com/competitions/PLAsTiCC-2018/data)
2. Download `training_set.csv`, `training_set_metadata.csv`, `test_set_metadata.csv`, and optionally `test_set.csv` (15 GB)
3. Place the CSV files in `data/raw/`

## Models

### Random Forest Baseline

Trained on ~130 hand-crafted features per object:
- **Per-band (x6):** statistical moments, amplitude, rise/fall rates, Lomb-Scargle period/power/FAP, Stetson J variability index
- **Cross-band:** color features (u-g, g-r, r-i, i-z, z-y flux ratios)
- **Metadata:** host galaxy photo-z, distance modulus, Milky Way E(B-V), deep drilling field flag
- **Pipeline:** StandardScaler -> RandomForest (500 trees, balanced class weights)

### Bidirectional GRU

Trained directly on raw multi-band light curve sequences:
- **Input:** observations sorted by MJD, each timestep = (delta_t, flux, flux_err, passband one-hot) -> 9 features
- **Architecture:** 2-layer bidirectional GRU (hidden=128) -> BatchNorm -> FC(128) -> 14-class output
- **Training:** AdamW, ReduceLROnPlateau, early stopping, class-weighted CrossEntropyLoss

## Project Structure

```
light-curve-classifier/
├── config.py                         # Centralized configuration
├── src/
│   ├── data.py                       # PLAsTiCC data loading
│   ├── preprocessing.py              # Light curve cleaning, RNN input prep
│   ├── features.py                   # Lomb-Scargle, statistical, temporal, color features
│   ├── models.py                     # RF pipeline + GRU + training loop
│   ├── metrics.py                    # Weighted log-loss, per-class metrics
│   └── utils.py                      # Plotting, label encoding
├── notebooks/
│   ├── 01_data_access/               # Load and verify PLAsTiCC data
│   ├── 02_eda/                       # Class distributions, light curve gallery
│   ├── 03_feature_engineering/       # Statistical + periodicity features
│   ├── 04_modeling/                  # RF baseline, GRU, model comparison
│   └── 05_analysis/                  # Per-class analysis, misclassification study
├── figures/
└── lab-notes/
```

## Setup

```bash
git clone https://github.com/saanikakulkarni07/light-curve-classifier.git
cd light-curve-classifier
pip install -e .
```

Run notebooks in order from `01_data_access/` through `05_analysis/`. Designed to run on [NASA Fornax Science Console](https://fornax.sci.stsci.edu/).

## Evaluation

- **Primary metric:** Weighted multi-class log-loss (PLAsTiCC competition metric)
- **Secondary:** Macro F1, per-class precision/recall, 14x14 confusion matrix
