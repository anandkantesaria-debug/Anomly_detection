# PRONOSTIA / FEMTO-ST Bearing Failure Monitor

An end-to-end industrial Prognostics & Health Management (PHM) project built for the **PRONOSTIA (IEEE PHM 2012)** bearing vibration dataset.

## What It Produces
1. **Continuous Health Indicator (0–100%)**: Unsupervised PCA + Isolation Forest degradation index calibrated on each bearing's healthy run-in period.
2. **Remaining Useful Life (RUL)**: `HistGradientBoostingRegressor` predicting remaining life (% and minutes) with strict bearing-level train/test splits.
3. **Early-Warning Failure Flag**: 3-stage state machine (`HEALTHY` → `WARNING` → `CRITICAL`) with persistence hysteresis to prevent single-spike false alarms.
4. **Interactive Streamlit Dashboard**: Live replay of bearing vibration waveforms, FFT spectra, Health Index curves, RUL countdowns, and alert triggers.

## Quick Start

```powershell
# 1. Install dependencies
uv sync

# 2a. Option A: Fast local test with synthetic PRONOSTIA-format data (seconds)
uv run pronostia-download --synthetic

# 2b. Option B: Download the real PRONOSTIA dataset from GitHub (~2.5 GB full, or use --learning-only)
uv run pronostia-download --learning-only

# 3. Extract vibration features, train models, and evaluate
uv run pronostia-train

# 4. Launch the interactive monitoring dashboard
uv run streamlit run app/dashboard.py
```

## Project Layout
- `src/pronostia/download.py` — Downloads the real IEEE PHM 2012 PRONOSTIA dataset or generates a physics-based synthetic PRONOSTIA dataset.
- `src/pronostia/features.py` — 25.6 kHz time-domain, Welch PSD frequency-band, and Hilbert envelope feature extraction (handles both `,` and `;` CSV delimiters).
- `src/pronostia/model.py` — Health Indicator, RUL regressor, alert state machine, and evaluation metrics.
- `src/pronostia/train.py` — End-to-end feature extraction, bearing-level split training, and evaluation CLI.
- `app/dashboard.py` — Interactive Streamlit + Plotly industrial monitoring dashboard.
