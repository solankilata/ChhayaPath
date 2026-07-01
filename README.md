# 🌑 ChhayaPath — Tracing the Shadow Every Anomaly Leaves

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![BAH2026](https://img.shields.io/badge/BAH_2026-Challenge_07-blueviolet?style=for-the-badge)

**Bharatiya Antariksh Hackathon 2026 | Team ChhayaPath**

*Every planet casts a shadow across its star. ChhayaPath finds it.*

</div>

---

## 🔭 The Problem

When a planet passes in front of its host star, it blocks a tiny fraction of the star's light — a transit. This shows up as a brief, subtle dip in the star's brightness over time, called a **light curve**.

But detecting it is hard.

Real telescope data from missions like **Kepler** and **TESS** is messy — buried in:
- Instrument noise and systematic errors
- Natural stellar variability (starspots, flares, pulsations)
- Data gaps from satellite orientation changes
- Cosmic ray hits and calibration drift

Most existing tools are built for one dataset, one signal type, one threshold. They break on real-world noise.

**ChhayaPath doesn't.**

---

## 💡 Our Approach

We built a **generalized anomaly detection framework for space time-series data** — not a narrow transit classifier, but a pipeline that can detect any meaningful deviation from normal stellar behavior.

### The core idea: Ensemble three detectors, trust none alone

```
Raw Light Curve
      │
      ▼
┌─────────────────────────────────────────────┐
│              Preprocessing                   │
│   Detrend → Normalize → Interpolate gaps     │
└─────────────────────────────────────────────┘
      │
      ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Z-Score    │  │    ARIMA     │  │     LSTM     │
│  Threshold   │  │  Residuals   │  │  Sequence    │
│  Detector    │  │  Detector    │  │   Model      │
└──────────────┘  └──────────────┘  └──────────────┘
      │                 │                  │
      └─────────────────┼──────────────────┘
                        ▼
              ┌──────────────────┐
              │  Ensemble Fusion │
              │  Confidence Score│
              └──────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Ranked Anomaly  │
              │  Output + Plots  │
              └──────────────────┘
```

**Why three detectors?**
- Z-score catches sharp, obvious dips fast
- ARIMA catches gradual trend deviations the Z-score misses
- LSTM learns the temporal pattern of a specific star and flags anything that breaks it

No single detector wins every case. The ensemble does.

---

## 🛰️ Datasets

| Dataset | Source | What we use it for |
|--------|--------|--------------------|
| Kepler Q1–Q17 Light Curves | [NASA MAST](https://archive.stsci.edu) via `lightkurve` | Primary exoplanet transit detection |
| TESS Full Frame Images | [NASA MAST](https://archive.stsci.edu) via `lightkurve` | Cross-validation on shorter baselines |
| Aditya-L1 SoLEXS X-ray Flux | [ISRO PRADAN](https://pradan.issdc.gov.in) | Cross-domain validation — same pipeline, solar flares |

> The Aditya-L1 validation is not just a bonus — it proves the framework generalizes beyond any one mission or signal type. One architecture, two completely different space datasets.

---

## 🧠 Model Architecture

### 1. Preprocessing Module (`src/preprocess.py`)
- Cofactor-based detrending using `lightkurve`'s built-in flatten
- Sigma-clipping to remove cosmic ray outliers
- Linear interpolation for short gaps, NaN masking for long ones
- Min-max normalization per observation quarter

### 2. Z-Score Detector (`src/detectors/zscore_detector.py`)
- Rolling window mean and standard deviation
- Flags any point beyond `n` sigma from the rolling baseline
- Mission-phase-aware: separate sigma thresholds per quarter

### 3. ARIMA Residual Detector (`src/detectors/arima_detector.py`)
- Fits ARIMA(p,d,q) on the detrended flux
- Flags large residuals between predicted and actual values
- Auto-order selection via AIC minimization

### 4. LSTM Sequence Model (`src/detectors/lstm_detector.py`)
- Sliding window input: 128 time steps → predicts next 16
- Trained per-star to learn individual stellar behavior
- Reconstruction error as the anomaly score
- Architecture: `LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(16)`

### 5. Ensemble Fusion (`src/ensemble.py`)
- Weighted average of normalized scores from all three detectors
- Weights tuned on a labeled transit catalog (Kepler KOI list)
- Output: ranked list of candidate anomalies with timestamps + confidence

---

## 📁 Project Structure

```
ChhayaPath/
│
├── data/                     # Raw and processed data (gitignored)
│   ├── raw/
│   └── processed/
│
├── notebooks/                # Exploration and experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_zscore_baseline.ipynb
│   ├── 03_arima_detector.ipynb
│   ├── 04_lstm_model.ipynb
│   └── 05_ensemble_evaluation.ipynb
│
├── src/                      # Core pipeline code
│   ├── preprocess.py
│   ├── ensemble.py
│   └── detectors/
│       ├── zscore_detector.py
│       ├── arima_detector.py
│       └── lstm_detector.py
│
├── outputs/                  # Saved models, plots, results
│   ├── models/
│   └── plots/
│
├── dashboard/                # Streamlit interactive app
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/LATASOLANKI/ChhayaPath.git
cd ChhayaPath

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the pipeline

```bash
# Download a Kepler light curve and run the full pipeline
python src/pipeline.py --target "Kepler-7" --mission kepler

# Launch the interactive dashboard
streamlit run dashboard/app.py
```

---

## 📊 Results

| Metric | Z-Score | ARIMA | LSTM | **Ensemble** |
|--------|---------|-------|------|-------------|
| Precision | — | — | — | **—** |
| Recall | — | — | — | **—** |
| F1 Score | — | — | — | **—** |

> Results will be updated as training completes.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data Acquisition | `lightkurve`, `astroquery` |
| Preprocessing & Stats | `numpy`, `pandas`, `scipy`, `statsmodels` |
| Deep Learning | `tensorflow`, `keras` |
| Classical ML | `scikit-learn` |
| Visualization | `matplotlib`, `plotly` |
| Dashboard | `streamlit` |
| Dev Environment | Python 3.11, VS Code, Git |

---

## 👥 Team

**Team ChhayaPath** — Bharatiya Antariksh Hackathon 2026

| Role | Name | Institution |
|------|------|-------------|
| Team Leader | Lata Solanki | IIT Guwahati (Online BSc DS & AI) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*"Every anomaly leaves a shadow — a dip in starlight, a spike in X-ray flux. ChhayaPath traces both."*

**Built for BAH 2026 | ISRO × Hack2Skill**

</div>
