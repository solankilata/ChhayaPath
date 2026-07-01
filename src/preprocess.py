"""
ChhayaPath — Preprocessing Module
Cleans raw light curves: detrend, normalize, remove outliers, fill gaps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import os

# ── Output folders ─────────────────────────────────────────────────────────────
BASE_DIR      = os.path.join(os.path.dirname(__file__), '..')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUTS_DIR   = os.path.join(BASE_DIR, 'outputs')
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR,   exist_ok=True)


def load_raw_csv(star_name: str) -> pd.DataFrame:
    """Load a previously saved raw CSV from data/raw/"""
    filename = star_name.replace(' ', '_').replace('-', '_') + '.csv'
    filepath = os.path.join(BASE_DIR, 'data', 'raw', filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No raw data found at {filepath}. Run data_loader.py first.")

    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} data points for {star_name}")
    return df


def remove_outliers(df: pd.DataFrame, sigma: float = 4.0) -> pd.DataFrame:
    """
    Remove extreme outlier points (cosmic rays, instrument glitches).
    Anything beyond sigma * std from mean is clipped out.
    """
    flux   = df['flux'].values
    mean   = np.nanmean(flux)
    std    = np.nanstd(flux)

    mask   = np.abs(flux - mean) < sigma * std
    clean  = df[mask].copy()

    removed = len(df) - len(clean)
    print(f"🧹 Outlier removal: {removed} points removed ({removed/len(df)*100:.2f}%)")
    return clean


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill short gaps in time series using linear interpolation.
    Long gaps (>0.5 days) are left as NaN.
    """
    df = df.copy().sort_values('time').reset_index(drop=True)

    # Create a uniform time grid
    time_min  = df['time'].min()
    time_max  = df['time'].max()
    cadence   = np.nanmedian(np.diff(df['time'].values))  # typical gap between points
    time_grid = np.arange(time_min, time_max, cadence)

    # Interpolate flux onto uniform grid
    flux_interp = np.interp(
        time_grid,
        df['time'].values,
        df['flux'].values,
        left=np.nan,
        right=np.nan
    )

    filled = pd.DataFrame({'time': time_grid, 'flux': flux_interp})
    print(f"📐 Gap filling: {len(df)} → {len(filled)} points on uniform grid")
    return filled


def detrend(df: pd.DataFrame, window_fraction: float = 0.05) -> pd.DataFrame:
    """
    Remove long-term stellar trends using a rolling median baseline.
    Window = fraction of total length (default 5%).
    This isolates short-duration transit dips from slow stellar variability.
    """
    df     = df.copy()
    n      = len(df)
    window = max(int(n * window_fraction), 51)
    if window % 2 == 0:
        window += 1  # must be odd for centered rolling

    flux        = pd.Series(df['flux'].values)
    baseline    = flux.rolling(window=window, center=True, min_periods=1).median()
    detrended   = flux / baseline  # divide out the trend

    df['flux']      = detrended.values
    df['baseline']  = baseline.values

    print(f"📉 Detrending: rolling median window = {window} points")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize flux so median = 1.0
    Standard practice for transit detection.
    """
    df       = df.copy()
    median   = np.nanmedian(df['flux'].values)
    df['flux'] = df['flux'].values / median
    print(f"🔢 Normalization: median flux set to 1.0 (was {median:.6f})")
    return df


def plot_comparison(raw_df, clean_df, star_name: str):
    """
    Side-by-side plot of raw vs cleaned light curve.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    # Raw
    axes[0].plot(raw_df['time'], raw_df['flux'],
                 color='steelblue', lw=0.4, alpha=0.7)
    axes[0].set_title(f'{star_name} — Raw Light Curve')
    axes[0].set_ylabel('Raw Flux')

    # Cleaned
    axes[1].plot(clean_df['time'], clean_df['flux'],
                 color='darkorange', lw=0.4, alpha=0.8)
    axes[1].set_title(f'{star_name} — After Preprocessing (transit dips now visible!)')
    axes[1].set_ylabel('Normalized Flux')
    axes[1].set_xlabel('Time (BKJD days)')

    plt.tight_layout()

    plot_path = os.path.join(OUTPUTS_DIR, star_name.replace(' ', '_') + '_preprocessed.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"📊 Comparison plot saved to {plot_path}")


def save_processed(df: pd.DataFrame, star_name: str) -> str:
    """Save cleaned light curve to data/processed/"""
    filename = star_name.replace(' ', '_').replace('-', '_') + '_clean.csv'
    filepath = os.path.join(PROCESSED_DIR, filename)
    df[['time', 'flux']].dropna().to_csv(filepath, index=False)
    print(f"💾 Cleaned data saved: {filepath} ({len(df)} rows)")
    return filepath


def preprocess_pipeline(star_name: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    Load → Remove outliers → Fill gaps → Detrend → Normalize → Save
    """
    print(f"\n{'='*55}")
    print(f"  ChhayaPath Preprocessing: {star_name}")
    print(f"{'='*55}")

    raw_df   = load_raw_csv(star_name)
    df       = remove_outliers(raw_df, sigma=4.0)
    df       = fill_gaps(df)
    df       = detrend(df, window_fraction=0.05)
    df       = normalize(df)
    filepath = save_processed(df, star_name)

    plot_comparison(raw_df, df, star_name)

    print(f"\n✅ Preprocessing complete! Clean data at:\n   {filepath}")
    return df


# ── Run directly to test ───────────────────────────────────────────────────────
if __name__ == '__main__':
    df = preprocess_pipeline("Kepler-7")