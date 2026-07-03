"""
ChhayaPath — ARIMA Residual Anomaly Detector
Models the expected flux using ARIMA, then flags
points where actual flux deviates too far from prediction.
Catches gradual/subtle anomalies the Z-score misses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def check_stationarity(flux: np.ndarray) -> bool:
    """
    ADF test to check if series is stationary.
    ARIMA needs stationary input (or d>0 differencing).
    """
    result  = adfuller(flux[:5000], autolag='AIC')
    p_value = result[1]
    print(f"📊 ADF Stationarity Test: p-value = {p_value:.6f} "
          f"({'stationary ✅' if p_value < 0.05 else 'non-stationary ⚠️'})")
    return p_value < 0.05


def fit_arima_chunked(flux: np.ndarray,
                      chunk_size: int = 1000,
                      order: tuple = (2, 0, 2)) -> np.ndarray:
    """
    Fits ARIMA on rolling chunks of the flux series.

    Why chunks?
    ARIMA on 60,000+ points is very slow. We fit it on
    overlapping windows and stitch the residuals together.

    Parameters:
        flux       : 1D array of normalized flux values
        chunk_size : number of points per ARIMA fit
        order      : (p, d, q) ARIMA order

    Returns:
        residuals : actual flux minus ARIMA prediction
    """
    n          = len(flux)
    residuals  = np.zeros(n)
    predicted  = np.zeros(n)

    print(f"\n⚙️  Fitting ARIMA{order} on {n} points in chunks of {chunk_size}...")
    print(f"   Total chunks: {n // chunk_size + 1}")

    for start in range(0, n, chunk_size):
        end    = min(start + chunk_size, n)
        chunk  = flux[start:end]

        # Skip chunks with too many NaNs
        if np.sum(np.isnan(chunk)) > len(chunk) * 0.3:
            residuals[start:end] = 0
            continue

        # Fill remaining NaNs with median
        chunk_clean = np.where(np.isnan(chunk),
                               np.nanmedian(chunk), chunk)

        try:
            model  = ARIMA(chunk_clean, order=order)
            result = model.fit()
            pred   = result.fittedvalues

            residuals[start:end] = chunk_clean - pred
            predicted[start:end] = pred

        except Exception as e:
            # If ARIMA fails on a chunk, residuals stay 0
            residuals[start:end] = 0

        if (start // chunk_size) % 5 == 0:
            pct = min(end / n * 100, 100)
            print(f"   Progress: {pct:.0f}%")

    print("✅ ARIMA fitting complete!")
    return residuals, predicted


def detect_arima_anomalies(df: pd.DataFrame,
                           chunk_size: int = 1000,
                           order: tuple = (2, 0, 2),
                           sigma_thresh: float = 3.0) -> pd.DataFrame:
    """
    Full ARIMA anomaly detection pipeline.

    Flags points where |residual| > sigma_thresh * std(residuals)
    AND residual is negative (we want dips, not spikes).

    Returns:
        df with added columns: 'arima_residual', 'arima_flag'
    """
    df    = df.copy().dropna(subset=['flux']).reset_index(drop=True)
    flux  = df['flux'].values

    # Stationarity check
    check_stationarity(flux)

    # Fit ARIMA and get residuals
    residuals, predicted = fit_arima_chunked(
        flux, chunk_size=chunk_size, order=order
    )

    df['arima_residual']  = residuals
    df['arima_predicted'] = predicted

    # Flag anomalies: negative residuals beyond threshold
    res_std  = np.nanstd(residuals)
    res_mean = np.nanmean(residuals)

    df['arima_flag'] = (
        (residuals < res_mean - sigma_thresh * res_std)
    ).astype(int)

    n_flagged = df['arima_flag'].sum()
    print(f"\n🔍 ARIMA Detector (σ={sigma_thresh})")
    print(f"   Total points   : {len(df)}")
    print(f"   Flagged        : {n_flagged} ({n_flagged/len(df)*100:.3f}%)")

    return df


def plot_arima_detections(df: pd.DataFrame, star_name: str):
    """
    Three-panel plot:
    1. Light curve with ARIMA flags
    2. ARIMA predicted vs actual
    3. Residuals with threshold line
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    flagged = df[df['arima_flag'] == 1]

    # ── Panel 1: Light curve + flags ──────────────────────────────────────────
    axes[0].plot(df['time'], df['flux'],
                 color='steelblue', lw=0.4, alpha=0.7, label='Flux')
    axes[0].scatter(flagged['time'], flagged['flux'],
                    color='crimson', s=6, zorder=5,
                    label=f'ARIMA Flagged ({len(flagged)})')
    axes[0].axhline(1.0, color='gray', lw=0.5, linestyle='--')
    axes[0].set_ylabel('Normalized Flux')
    axes[0].set_title(f'{star_name} — ARIMA Residual Anomaly Detection')
    axes[0].legend(loc='upper right', fontsize=8)

    # ── Panel 2: ARIMA fit vs actual ──────────────────────────────────────────
    axes[1].plot(df['time'], df['flux'],
                 color='steelblue', lw=0.3, alpha=0.5, label='Actual')
    axes[1].plot(df['time'], df['arima_predicted'],
                 color='green', lw=0.5, alpha=0.8, label='ARIMA Predicted')
    axes[1].set_ylabel('Flux')
    axes[1].legend(loc='upper right', fontsize=8)

    # ── Panel 3: Residuals ────────────────────────────────────────────────────
    res_std  = np.nanstd(df['arima_residual'])
    res_mean = np.nanmean(df['arima_residual'])
    thresh   = res_mean - 3.0 * res_std

    axes[2].plot(df['time'], df['arima_residual'],
                 color='darkorange', lw=0.4, alpha=0.8, label='Residual')
    axes[2].axhline(thresh, color='red', lw=1.0,
                    linestyle='--', label=f'−3σ = {thresh:.5f}')
    axes[2].axhline(0, color='gray', lw=0.5, linestyle='--')
    axes[2].set_ylabel('Residual')
    axes[2].set_xlabel('Time (BKJD days)')
    axes[2].legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR,
                        star_name.replace(' ', '_') + '_arima.png')
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"📊 ARIMA plot saved: {path}")


# ── Run directly to test ───────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

    STAR           = "Kepler-7"
    processed_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'processed',
        'Kepler_7_clean.csv'
    )

    df = pd.read_csv(processed_path)
    print(f"✅ Loaded {len(df)} preprocessed points")

    # Run ARIMA detector
    df = detect_arima_anomalies(
        df,
        chunk_size=500,
        order=(2, 0, 2),
        sigma_thresh=3.0
    )

    # Plot
    plot_arima_detections(df, STAR)

    print(f"\n✅ ARIMA detector complete!")
    print(f"   Flagged: {df['arima_flag'].sum()} anomalies")