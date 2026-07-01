"""
ChhayaPath — Data Loader
Downloads and saves Kepler/TESS light curves for exoplanet detection.
"""

import lightkurve as lk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Output folder ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DIR  = os.path.join(DATA_DIR, 'raw')
os.makedirs(RAW_DIR, exist_ok=True)


def download_kepler_lightcurve(target_name: str, quarter: int = None):
    """
    Downloads a Kepler light curve for a given star.
    
    Parameters:
        target_name : e.g. "Kepler-7", "KIC 11446443"
        quarter     : Kepler quarter (1-17). None = all quarters.
    
    Returns:
        lc : LightCurve object
    """
    print(f"\n🔭 Searching for {target_name} in Kepler archive...")

    search = lk.search_lightcurve(
        target_name,
        mission='Kepler',
        quarter=quarter,
        author='Kepler'
    )

    print(search)

    if len(search) == 0:
        raise ValueError(f"No Kepler data found for {target_name}")

    print(f"\n⬇️  Downloading {len(search)} light curve(s)...")
    lc_collection = search.download_all()

    # Stitch all quarters into one continuous light curve
    lc = lc_collection.stitch()
    print(f"✅ Downloaded! Total data points: {len(lc)}")

    return lc


def save_lightcurve(lc, star_name: str):
    """
    Saves a light curve as CSV to data/raw/
    
    Parameters:
        lc        : LightCurve object
        star_name : name used for the filename
    """
    filename = star_name.replace(' ', '_').replace('-', '_') + '.csv'
    filepath = os.path.join(RAW_DIR, filename)

    df = lc.to_pandas().reset_index()

    # Keep only what we need
    cols = [c for c in ['time', 'flux', 'flux_err'] if c in df.columns]
    df = df[cols].dropna()

    df.to_csv(filepath, index=False)
    print(f"💾 Saved to {filepath} ({len(df)} rows)")

    return filepath


def plot_lightcurve(lc, star_name: str):
    """
    Quick plot to visually verify the downloaded data.
    """
    plt.figure(figsize=(14, 4))
    plt.plot(lc.time.value, lc.flux.value,
             color='steelblue', lw=0.5, alpha=0.8)
    plt.xlabel('Time (BKJD days)')
    plt.ylabel('Normalized Flux')
    plt.title(f'{star_name} — Kepler Light Curve')
    plt.tight_layout()

    plot_path = os.path.join(
        os.path.dirname(__file__), '..', 'outputs',
        star_name.replace(' ', '_') + '_raw.png'
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"📊 Plot saved to {plot_path}")


# ── Run directly to test ───────────────────────────────────────────────────────
if __name__ == '__main__':

    # Kepler-7b is a hot Jupiter — big, obvious transits. Perfect for testing.
    TARGET = "Kepler-7"

    lc       = download_kepler_lightcurve(TARGET)
    filepath = save_lightcurve(lc, TARGET)
    plot_lightcurve(lc, TARGET)

    print("\n🎉 Data loader working! Raw light curve saved.")
    print(f"   File: {filepath}")