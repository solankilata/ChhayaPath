"""
ChhayaPath — LSTM Anomaly Detector
Learns the 'normal' pattern of a star's light curve,
then flags anything that breaks that pattern.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'models')
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# ── Sliding window helper ──────────────────────────────────────────────────────
def create_sequences(flux: np.ndarray,
                     window: int = 64) -> tuple:
    """
    Converts 1D flux array into sliding windows for LSTM.

    Each sample: window flux values → predict next 1 value
    LSTM learns what 'normal' stellar behavior looks like.

    Returns:
        X : (n_samples, window, 1)
        y : (n_samples,)
    """
    X, y = [], []
    for i in range(len(flux) - window):
        X.append(flux[i : i + window])
        y.append(flux[i + window])
    return np.array(X)[..., np.newaxis], np.array(y)


def build_lstm_model(window: int = 64) -> tf.keras.Model:
    """
    LSTM architecture:
    Input(window) → LSTM(64) → Dropout → LSTM(32) → Dense(1)

    Trained to PREDICT the next flux value.
    High prediction error = anomaly = possible transit.
    """
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


def train_lstm(df: pd.DataFrame,
               window: int = 64,
               train_frac: float = 0.7,
               epochs: int = 20,
               batch_size: int = 256) -> tuple:
    """
    Trains LSTM on the 'normal' (non-transit) portion of the light curve.

    Strategy:
    - Train on first 70% of data (mostly normal baseline)
    - Test/detect on the full series

    Returns:
        model   : trained Keras model
        scaler  : fitted MinMaxScaler
        history : training history
    """
    flux = df['flux'].dropna().values

    # Scale flux to [0, 1] for LSTM stability
    scaler     = MinMaxScaler()
    flux_scaled = scaler.fit_transform(flux.reshape(-1, 1)).flatten()

    # Split
    train_end  = int(len(flux_scaled) * train_frac)
    train_flux = flux_scaled[:train_end]

    print(f"\n🧠 Building LSTM model (window={window})...")
    X_train, y_train = create_sequences(train_flux, window)
    print(f"   Training samples : {len(X_train)}")
    print(f"   Training on first {train_frac*100:.0f}% of data")

    model = build_lstm_model(window)

    early_stop = EarlyStopping(
        monitor='val_loss', patience=3,
        restore_best_weights=True, verbose=1
    )

    print(f"\n🏋️  Training LSTM for up to {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model
    model_path = os.path.join(MODELS_DIR, 'lstm_kepler7.keras')
    model.save(model_path)
    print(f"💾 Model saved: {model_path}")

    return model, scaler, history, flux_scaled


def detect_lstm_anomalies(df: pd.DataFrame,
                          model: tf.keras.Model,
                          scaler: MinMaxScaler,
                          flux_scaled: np.ndarray,
                          window: int = 64,
                          sigma_thresh: float = 3.0) -> pd.DataFrame:
    """
    Runs the trained LSTM over the full light curve.
    Computes reconstruction error (predicted vs actual).
    Flags points where error > sigma_thresh * std(errors).

    Returns:
        df with 'lstm_error' and 'lstm_flag' columns
    """
    df = df.copy().dropna(subset=['flux']).reset_index(drop=True)

    print(f"\n🔍 Running LSTM inference on full light curve...")
    X_all, y_all = create_sequences(flux_scaled, window)

    y_pred = model.predict(X_all, batch_size=512, verbose=1)
    errors = np.abs(y_all - y_pred.flatten())

    # Pad the first `window` points (no prediction possible there)
    pad          = np.zeros(window)
    errors_full  = np.concatenate([pad, errors])

    # Align lengths
    n = min(len(df), len(errors_full))
    df = df.iloc[:n].copy()
    df['lstm_error'] = errors_full[:n]

    # Flag anomalies
    err_mean = np.nanmean(errors_full)
    err_std  = np.nanstd(errors_full)
    thresh   = err_mean + sigma_thresh * err_std

    df['lstm_flag'] = (df['lstm_error'] > thresh).astype(int)

    n_flagged = df['lstm_flag'].sum()
    print(f"\n✅ LSTM Detection Results (σ={sigma_thresh})")
    print(f"   Total points : {len(df)}")
    print(f"   Flagged      : {n_flagged} ({n_flagged/len(df)*100:.3f}%)")
    print(f"   Error thresh : {thresh:.6f}")

    return df


def plot_lstm_detections(df: pd.DataFrame,
                         history,
                         star_name: str):
    """
    Three-panel plot:
    1. Training loss curve
    2. Light curve with LSTM flags
    3. LSTM reconstruction error
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    # ── Panel 1: Training loss ─────────────────────────────────────────────────
    axes[0].plot(history.history['loss'],
                 color='steelblue', label='Train Loss')
    axes[0].plot(history.history['val_loss'],
                 color='darkorange', label='Val Loss')
    axes[0].set_title('LSTM Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()

    # ── Panel 2: Light curve + LSTM flags ─────────────────────────────────────
    flagged = df[df['lstm_flag'] == 1]
    axes[1].plot(df['time'], df['flux'],
                 color='steelblue', lw=0.4, alpha=0.7, label='Flux')
    axes[1].scatter(flagged['time'], flagged['flux'],
                    color='purple', s=6, zorder=5,
                    label=f'LSTM Flagged ({len(flagged)})')
    axes[1].axhline(1.0, color='gray', lw=0.5, linestyle='--')
    axes[1].set_ylabel('Normalized Flux')
    axes[1].set_title(f'{star_name} — LSTM Anomaly Detection')
    axes[1].legend(loc='upper right', fontsize=8)

    # ── Panel 3: Reconstruction error ─────────────────────────────────────────
    err_mean = df['lstm_error'].mean()
    err_std  = df['lstm_error'].std()
    thresh   = err_mean + 3.0 * err_std

    axes[2].plot(df['time'], df['lstm_error'],
                 color='purple', lw=0.4, alpha=0.8, label='LSTM Error')
    axes[2].axhline(thresh, color='red', lw=1.0,
                    linestyle='--', label=f'+3σ threshold')
    axes[2].set_ylabel('Reconstruction Error')
    axes[2].set_xlabel('Time (BKJD days)')
    axes[2].legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR,
                        star_name.replace(' ', '_') + '_lstm.png')
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"📊 LSTM plot saved: {path}")


# ── Run directly to test ───────────────────────────────────────────────────────
if __name__ == '__main__':
    STAR = "Kepler-7"

    processed_path = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'data', 'processed', 'Kepler_7_clean.csv'
    )

    df = pd.read_csv(processed_path)
    print(f"✅ Loaded {len(df)} preprocessed points")

    # Train
    model, scaler, history, flux_scaled = train_lstm(
        df,
        window=64,
        train_frac=0.7,
        epochs=20,
        batch_size=256
    )

    # Detect
    df = detect_lstm_anomalies(
        df, model, scaler, flux_scaled,
        window=64,
        sigma_thresh=3.0
    )

    # Plot
    plot_lstm_detections(df, history, STAR)

    print(f"\n🎉 LSTM detector complete!")
    print(f"   Flagged: {df['lstm_flag'].sum()} anomalies")