"""
ChhayaPath — Z-Score Anomaly Detector
Flags points that deviate significantly from a rolling baseline.
First and fastest detector in our ensemble.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def compute_zscore(df: pd.DataFrame,
                   window: int = 200,
                   sigma_thresh: float = 3.0) -> pd.DataFrame:
    """
    Computes rolling Z-score for each flux point.

    Z = (flux - rolling_mean) / rolling_std

    Points below -sigma_thresh are flagged as transit candidates
    (we look for DIPS, so we care about negative Z scores).

    Parameters:
        df           : DataFrame with 'time' and 'flux' columns
        window       : rolling window size in data points
        sigma_thresh : how many sigma below mean = anomaly

    Returns:
        df with added columns: 'zscore', 'zscore_flag'
    """
    df = df.copy().dropna(subset=['flux']).reset_index(drop=True)

    flux = pd.Series(df['flux'].values)

    rolling_mean = flux.rolling(window=window, center=True, min_periods=1).mean()
    rolling_std  = flux.rolling(window=window, center=True, min_periods=1).std()

    # Avoid division by zero
    rolling_std  = rolling_std.replace(0, np.nan).fillna(1e-10)

    zscore = (flux - rolling_mean) / rolling_std

    df['zscore']      = zscore.values
    df['zscore_flag'] = (zscore < -sigma_thresh).astype(int)

    n_flagged = df['zscore_flag'].sum()
    print(f"🔍 Z-Score Detector (σ={sigma_thresh}, window={window})")
    print(f"   Total points : {len(df)}")
    print(f"   Flagged      : {n_flagged} ({n_flagged/len(df)*100:.3f}%)")

    return df


def get_transit_candidates(df: pd.DataFrame,
                           min_duration: int = 3) -> pd.DataFrame:
    """
    Groups consecutive flagged points into individual transit events.

    Parameters:
        df           : output from compute_zscore()
        min_duration : minimum consecutive flagged points to count as a transit

    Returns:
        DataFrame of transit candidates with start/end/depth/duration
    """
    flags  = df['zscore_flag'].values
    times  = df['time'].values
    fluxes = df['flux'].values

    candidates = []
    i = 0
    while i < len(flags):
        if flags[i] == 1:
            j = i
            while j < len(flags) and flags[j] == 1:
                j += 1
            duration = j - i
            if duration >= min_duration:
                segment_flux = fluxes[i:j]
                candidates.append({
                    'start_time'  : times[i],
                    'end_time'    : times[j-1],
                    'duration_pts': duration,
                    'depth'       : 1.0 - np.min(segment_flux),
                    'min_flux'    : np.min(segment_flux),
                    'zscore_min'  : df['zscore'].values[i:j].min()
                })
            i = j
        else:
            i += 1

    candidates_df = pd.DataFrame(candidates)

    if len(candidates_df) > 0:
        candidates_df = candidates_df.sort_values('depth', ascending=False)
        print(f"\n🪐 Transit Candidates Found: {len(candidates_df)}")
        print(candidates_df.head(10).to_string(index=False))
    else:
        print("⚠️  No transit candidates found. Try lowering sigma_thresh.")

    return candidates_df


def plot_zscore_detections(df: pd.DataFrame,
                           candidates: pd.DataFrame,
                           star_name: str):
    """
    Plots the light curve with Z-score flags overlaid in red.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # ── Top: Light curve with flagged points ───────────────────────────────────
    axes[0].plot(df['time'], df['flux'],
                 color='steelblue', lw=0.4, alpha=0.7, label='Flux')

    flagged = df[df['zscore_flag'] == 1]
    axes[0].scatter(flagged['time'], flagged['flux'],
                    color='red', s=6, zorder=5, label=f'Flagged ({len(flagged)})')

    axes[0].axhline(1.0, color='gray', lw=0.5, linestyle='--')
    axes[0].set_ylabel('Normalized Flux')
    axes[0].set_title(f'{star_name} — Z-Score Anomaly Detection')
    axes[0].legend(loc='upper right', fontsize=8)

    # ── Bottom: Z-score over time ──────────────────────────────────────────────
    axes[1].plot(df['time'], df['zscore'],
                 color='darkorange', lw=0.4, alpha=0.8)
    axes[1].axhline(0,    color='gray',  lw=0.5, linestyle='--')
    axes[1].axhline(-3.0, color='red',   lw=1.0, linestyle='--', label='−3σ threshold')
    axes[1].set_ylabel('Z-Score')
    axes[1].set_xlabel('Time (BKJD days)')
    axes[1].legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR,
                        star_name.replace(' ', '_') + '_zscore.png')
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"📊 Z-Score plot saved: {path}")


# ── Run directly to test ───────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from preprocess import preprocess_pipeline

    STAR = "Kepler-7"

    # Load preprocessed data (or run pipeline if not done yet)
    processed_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'processed',
        'Kepler_7_clean.csv'
    )

    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        print(f"✅ Loaded processed data: {len(df)} points")
    else:
        df = preprocess_pipeline(STAR)

    # Run Z-score detector
    df         = compute_zscore(df, window=200, sigma_thresh=3.0)
    candidates = get_transit_candidates(df, min_duration=3)

    # Plot
    plot_zscore_detections(df, candidates, STAR)

    print(f"\n✅ Z-Score detector complete!")
    print(f"   Candidates: {len(candidates)}")