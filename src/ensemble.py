"""
ChhayaPath — Ensemble Fusion Module
Combines Z-Score + ARIMA + LSTM into one ranked anomaly list.
Only flags points where multiple detectors agree.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add parent to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detectors.zscore_detector  import compute_zscore
from src.detectors.arima_detector   import detect_arima_anomalies
from src.detectors.lstm_detector    import (train_lstm,
                                            detect_lstm_anomalies)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def normalize_score(series: pd.Series) -> pd.Series:
    """Scale any score column to [0, 1] range."""
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-10:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def run_ensemble(df: pd.DataFrame,
                 weights: dict = None,
                 final_thresh: float = 0.5) -> pd.DataFrame:
    """
    Full ensemble pipeline:
    1. Run all three detectors
    2. Normalize each score to [0,1]
    3. Weighted average → ensemble confidence score
    4. Flag points above final_thresh

    Parameters:
        df           : preprocessed light curve DataFrame
        weights      : dict with keys 'zscore','arima','lstm'
        final_thresh : minimum ensemble score to flag as transit

    Returns:
        df with all detector columns + 'ensemble_score' + 'ensemble_flag'
    """

    if weights is None:
        weights = {'zscore': 0.25, 'arima': 0.35, 'lstm': 0.40}

    print("\n" + "="*60)
    print("  ChhayaPath Ensemble Detector")
    print("="*60)
    print(f"  Weights → Z-score:{weights['zscore']}  "
          f"ARIMA:{weights['arima']}  LSTM:{weights['lstm']}")
    print(f"  Final threshold : {final_thresh}")
    print("="*60)

    df = df.copy().dropna(subset=['flux']).reset_index(drop=True)

    # ── 1. Z-Score ─────────────────────────────────────────────────────────────
    print("\n[1/3] Running Z-Score detector...")
    df = compute_zscore(df, window=200, sigma_thresh=3.0)
    # Score = absolute Z-score (inverted so dips score high)
    df['zscore_score'] = df['zscore'].clip(upper=0).abs()

    # ── 2. ARIMA ───────────────────────────────────────────────────────────────
    print("\n[2/3] Running ARIMA detector...")
    df = detect_arima_anomalies(
        df, chunk_size=500, order=(2,0,2), sigma_thresh=3.0
    )
    # Score = absolute negative residual
    df['arima_score'] = df['arima_residual'].clip(upper=0).abs()

    # ── 3. LSTM ────────────────────────────────────────────────────────────────
    print("\n[3/3] Training & running LSTM detector...")
    model, scaler, history, flux_scaled = train_lstm(
        df, window=64, train_frac=0.7, epochs=20, batch_size=256
    )
    df = detect_lstm_anomalies(
        df, model, scaler, flux_scaled,
        window=64, sigma_thresh=3.0
    )
    df['lstm_score'] = df['lstm_error']

    # ── 4. Normalize all scores to [0,1] ───────────────────────────────────────
    print("\n[4/4] Fusing scores...")
    df['zscore_norm'] = normalize_score(df['zscore_score'])
    df['arima_norm']  = normalize_score(df['arima_score'])
    df['lstm_norm']   = normalize_score(df['lstm_score'])

    # ── 5. Weighted ensemble score ─────────────────────────────────────────────
    df['ensemble_score'] = (
        weights['zscore'] * df['zscore_norm'] +
        weights['arima']  * df['arima_norm']  +
        weights['lstm']   * df['lstm_norm']
    )

    # ── 6. Final flag ──────────────────────────────────────────────────────────
    df['ensemble_flag'] = (df['ensemble_score'] >= final_thresh).astype(int)

    n_flagged = df['ensemble_flag'].sum()
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE RESULTS")
    print(f"  Z-Score flags  : {df['zscore_flag'].sum()}")
    print(f"  ARIMA flags    : {df['arima_flag'].sum()}")
    print(f"  LSTM flags     : {df['lstm_flag'].sum()}")
    print(f"  ENSEMBLE flags : {n_flagged}  ← final answer")
    print(f"{'='*60}")

    return df


def get_ranked_candidates(df: pd.DataFrame,
                          min_gap: int = 50) -> pd.DataFrame:
    """
    Groups ensemble-flagged points into individual transit events.
    Ranks them by ensemble confidence score.

    Parameters:
        df      : output from run_ensemble()
        min_gap : minimum point gap between separate transit events

    Returns:
        DataFrame of ranked transit candidates
    """
    flagged = df[df['ensemble_flag'] == 1].copy()

    if len(flagged) == 0:
        print("⚠️  No ensemble candidates found.")
        return pd.DataFrame()

    # Group consecutive flags into events
    events    = []
    indices   = flagged.index.tolist()
    group     = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] - indices[i-1] <= min_gap:
            group.append(indices[i])
        else:
            events.append(group)
            group = [indices[i]]
    events.append(group)

    candidates = []
    for grp in events:
        seg = df.loc[grp]
        candidates.append({
            'center_time'     : seg['time'].median(),
            'start_time'      : seg['time'].min(),
            'end_time'        : seg['time'].max(),
            'duration_days'   : seg['time'].max() - seg['time'].min(),
            'min_flux'        : seg['flux'].min(),
            'depth_pct'       : (1.0 - seg['flux'].min()) * 100,
            'ensemble_score'  : seg['ensemble_score'].max(),
            'zscore_contrib'  : seg['zscore_norm'].max(),
            'arima_contrib'   : seg['arima_norm'].max(),
            'lstm_contrib'    : seg['lstm_norm'].max(),
            'n_points'        : len(grp),
        })

    candidates_df = pd.DataFrame(candidates)
    candidates_df = candidates_df.sort_values(
        'ensemble_score', ascending=False
    ).reset_index(drop=True)

    print(f"\n🪐 TOP TRANSIT CANDIDATES (ranked by confidence):")
    print(candidates_df[['center_time','depth_pct',
                          'ensemble_score','duration_days']
                        ].head(15).to_string(index=True))

    # Save
    out_path = os.path.join(OUTPUTS_DIR, 'transit_candidates.csv')
    candidates_df.to_csv(out_path, index=False)
    print(f"\n💾 Full candidate list saved: {out_path}")

    return candidates_df


def plot_ensemble(df: pd.DataFrame,
                  candidates: pd.DataFrame,
                  star_name: str):
    """
    Four-panel ensemble summary plot:
    1. Individual detector scores (normalized)
    2. Ensemble score with threshold
    3. Light curve with top candidates marked
    4. Zoom into the top candidate transit
    """
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=False)

    t = df['time'].values

    # ── Panel 1: Normalized scores ────────────────────────────────────────────
    axes[0].plot(t, df['zscore_norm'],
                 color='steelblue',   lw=0.4, alpha=0.7, label='Z-Score')
    axes[0].plot(t, df['arima_norm'],
                 color='green',       lw=0.4, alpha=0.7, label='ARIMA')
    axes[0].plot(t, df['lstm_norm'],
                 color='purple',      lw=0.4, alpha=0.5, label='LSTM')
    axes[0].set_ylabel('Normalized Score')
    axes[0].set_title('Individual Detector Scores (Normalized)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_xlim(t[0], t[-1])

    # ── Panel 2: Ensemble score ───────────────────────────────────────────────
    axes[1].plot(t, df['ensemble_score'],
                 color='darkorange', lw=0.5, alpha=0.9, label='Ensemble Score')
    thresh = 0.5
    axes[1].axhline(thresh, color='red', lw=1.2,
                    linestyle='--', label=f'Threshold ({thresh})')
    axes[1].fill_between(t, df['ensemble_score'], thresh,
                         where=df['ensemble_score'] >= thresh,
                         alpha=0.3, color='red', label='Flagged region')
    axes[1].set_ylabel('Ensemble Score')
    axes[1].set_title('Ensemble Confidence Score')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_xlim(t[0], t[-1])

    # ── Panel 3: Light curve + top candidates ─────────────────────────────────
    axes[2].plot(t, df['flux'],
                 color='steelblue', lw=0.4, alpha=0.7, label='Flux')

    if len(candidates) > 0:
        top15 = candidates.head(15)
        for _, row in top15.iterrows():
            axes[2].axvline(row['center_time'],
                            color='red', lw=1.0, alpha=0.6)

        # Mark the top candidate
        best = candidates.iloc[0]
        axes[2].axvspan(best['start_time'], best['end_time'],
                        alpha=0.2, color='gold', label='Top candidate')

    axes[2].axhline(1.0, color='gray', lw=0.5, linestyle='--')
    axes[2].set_ylabel('Normalized Flux')
    axes[2].set_title(f'{star_name} — Ensemble Flagged Transits (red lines)')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].set_xlim(t[0], t[-1])

    # ── Panel 4: Zoom into best candidate ────────────────────────────────────
    if len(candidates) > 0:
        best     = candidates.iloc[0]
        pad      = 5.0  # days either side
        zoom_t0  = best['center_time'] - pad
        zoom_t1  = best['center_time'] + pad
        mask     = (df['time'] >= zoom_t0) & (df['time'] <= zoom_t1)
        zoom_df  = df[mask]

        axes[3].plot(zoom_df['time'], zoom_df['flux'],
                     color='steelblue', lw=1.0, marker='o',
                     markersize=2, label='Flux')
        axes[3].axvline(best['center_time'],
                        color='red', lw=1.5, linestyle='--',
                        label=f"Transit @ {best['center_time']:.2f} BKJD")
        axes[3].axhline(1.0, color='gray', lw=0.5, linestyle='--')
        axes[3].set_ylabel('Normalized Flux')
        axes[3].set_xlabel('Time (BKJD days)')
        axes[3].set_title(f'Zoom: Top Candidate Transit  '
                          f'(depth={best["depth_pct"]:.3f}%,  '
                          f'score={best["ensemble_score"]:.3f})')
        axes[3].legend(loc='upper right', fontsize=8)
    else:
        axes[3].set_visible(False)

    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR,
                        star_name.replace(' ', '_') + '_ensemble.png')
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"\n📊 Ensemble plot saved: {path}")


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == '__main__':

    STAR = "Kepler-7"

    processed_path = os.path.join(
        os.path.dirname(__file__), '..',
        'data', 'processed', 'Kepler_7_clean.csv'
    )

    df = pd.read_csv(processed_path)
    print(f"✅ Loaded {len(df)} preprocessed points for {STAR}")

    # Run full ensemble
    df         = run_ensemble(df, final_thresh=0.5)
    candidates = get_ranked_candidates(df, min_gap=50)

    # Plot
    plot_ensemble(df, candidates, STAR)

    print("\n ENSEMBLE COMPLETE!")
    print(f"   Final transit candidates: {len(candidates)}")