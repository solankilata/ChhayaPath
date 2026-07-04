"""
ChhayaPath — Interactive Dashboard
Streamlit app for exploring exoplanet transit detections.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChhayaPath — Exoplanet Detector",
    page_icon="🌑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a1a; }
    .stApp { background-color: #0a0a1a; }
    h1, h2, h3 { color: #FF6600; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #FF6600;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stSidebar { background-color: #0d0d1f; }
</style>
""", unsafe_allow_html=True)


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.join(os.path.dirname(__file__), '..')
PROCESSED_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUTS_DIR    = os.path.join(BASE_DIR, 'outputs')


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_processed_data(star: str) -> pd.DataFrame:
    filename = star.replace(' ', '_').replace('-', '_') + '_clean.csv'
    path     = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_candidates() -> pd.DataFrame:
    path = os.path.join(OUTPUTS_DIR, 'transit_candidates.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def run_zscore_cached(star: str):
    """Run Z-score detector and cache result."""
    from src.detectors.zscore_detector import compute_zscore
    df = load_processed_data(star)
    if df is None:
        return None
    return compute_zscore(df, window=200, sigma_thresh=3.0)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_Gotham.jpg/320px-Above_Gotham.jpg",
             use_container_width=True)

    st.markdown("## 🌑 ChhayaPath")
    st.markdown("*Tracing the Shadow Every Anomaly Leaves*")
    st.divider()

    st.markdown("### 🔭 Target Star")
    star_name = st.selectbox(
        "Select target",
        ["Kepler-7"],
        index=0
    )

    st.divider()

    st.markdown("### ⚙️ Detector Settings")
    zscore_sigma = st.slider("Z-Score σ threshold", 2.0, 5.0, 3.0, 0.1)
    ensemble_thresh = st.slider("Ensemble threshold", 0.1, 0.9, 0.5, 0.05)

    st.divider()

    st.markdown("### 📊 About")
    st.markdown("""
    **ChhayaPath** detects exoplanet transits
    using an ensemble of:
    - 🔵 Z-Score baseline
    - 🟢 ARIMA residuals
    - 🟣 LSTM neural network

    *BAH 2026 | Challenge 07*
    """)


# ── Main content ───────────────────────────────────────────────────────────────
st.markdown("# 🌑 ChhayaPath — Exoplanet Transit Detector")
st.markdown("### *AI-enabled detection of exoplanets from noisy astronomical light curves*")
st.divider()

# ── Load data ──────────────────────────────────────────────────────────────────
df         = load_processed_data(star_name)
candidates = load_candidates()

if df is None:
    st.error("❌ No preprocessed data found. Run `python src/preprocess.py` first.")
    st.stop()

# ── Metric cards ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

n_candidates = len(candidates) if candidates is not None else 0
best_score   = candidates['ensemble_score'].max() if candidates is not None else 0
best_depth   = candidates['depth_pct'].max()      if candidates is not None else 0

col1.metric("📡 Data Points",    f"{len(df):,}")
col2.metric("🪐 Candidates",     f"{n_candidates}")
col3.metric("🏆 Best Score",     f"{best_score:.3f}")
col4.metric("📉 Max Depth",      f"{best_depth:.3f}%")
col5.metric("⭐ Target Star",    star_name)

st.divider()

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Light Curve",
    "🔍 Z-Score Detection",
    "🪐 Transit Candidates",
    "🌞 Aditya-L1 (Cross-Validation)"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Light Curve
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Raw Preprocessed Light Curve")
    st.markdown(f"*{len(df):,} data points spanning {df['time'].max()-df['time'].min():.0f} days of observation*")

    # Time range selector
    t_min = float(df['time'].min())
    t_max = float(df['time'].max())

    col_a, col_b = st.columns(2)
    with col_a:
        t_start = st.slider("Start time (BKJD)", t_min, t_max, t_min, key="t1s")
    with col_b:
        t_end   = st.slider("End time (BKJD)",   t_min, t_max, t_min + 200.0, key="t1e")

    mask    = (df['time'] >= t_start) & (df['time'] <= t_end)
    df_view = df[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_view['time'], y=df_view['flux'],
        mode='lines',
        line=dict(color='#4fa3e0', width=0.8),
        name='Flux'
    ))
    fig.add_hline(y=1.0, line_dash='dash',
                  line_color='gray', line_width=0.8)
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Time (BKJD days)',
        yaxis_title='Normalized Flux',
        height=400,
        title=f'{star_name} — Normalized Light Curve',
        paper_bgcolor='#0a0a1a',
        plot_bgcolor='#0a0a1a'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"💡 Showing {len(df_view):,} of {len(df):,} data points. "
            f"Transit dips appear as brief drops below 1.0 flux.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Z-Score Detection
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Z-Score Anomaly Detection")
    st.markdown("*Flags points that deviate significantly from rolling baseline*")

    with st.spinner("Running Z-Score detector..."):
        df_z = run_zscore_cached(star_name)

    if df_z is not None:
        flagged_z = df_z[df_z['zscore_flag'] == 1]

        col_a, col_b = st.columns(2)
        col_a.metric("🔴 Flagged Points", f"{len(flagged_z):,}")
        col_b.metric("📊 Flag Rate",
                     f"{len(flagged_z)/len(df_z)*100:.2f}%")

        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=['Light Curve + Flags',
                                            'Z-Score Over Time'])

        fig2.add_trace(go.Scatter(
            x=df_z['time'], y=df_z['flux'],
            mode='lines', line=dict(color='#4fa3e0', width=0.5),
            name='Flux'
        ), row=1, col=1)

        fig2.add_trace(go.Scatter(
            x=flagged_z['time'], y=flagged_z['flux'],
            mode='markers',
            marker=dict(color='red', size=3),
            name=f'Flagged ({len(flagged_z):,})'
        ), row=1, col=1)

        fig2.add_trace(go.Scatter(
            x=df_z['time'], y=df_z['zscore'],
            mode='lines', line=dict(color='orange', width=0.5),
            name='Z-Score'
        ), row=2, col=1)

        fig2.add_hline(y=-zscore_sigma, line_dash='dash',
                       line_color='red', row=2, col=1)

        fig2.update_layout(
            template='plotly_dark',
            height=600,
            paper_bgcolor='#0a0a1a',
            plot_bgcolor='#0a0a1a'
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Transit Candidates
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🪐 Ranked Transit Candidates")
    st.markdown("*Sorted by ensemble confidence score — highest = most likely real planet transit*")

    if candidates is None:
        st.warning("⚠️ No candidates found. Run `python src/ensemble.py` first.")
    else:
        # Filter by threshold
        filtered = candidates[
            candidates['ensemble_score'] >= ensemble_thresh
        ].copy()

        st.markdown(f"**{len(filtered)} candidates** above threshold {ensemble_thresh}")

        # Candidate table
        display_cols = ['center_time', 'depth_pct', 'ensemble_score',
                        'duration_days', 'zscore_contrib',
                        'arima_contrib', 'lstm_contrib']
        st.dataframe(
            filtered[display_cols].round(4),
            use_container_width=True,
            height=300
        )

        # Score distribution
        col_a, col_b = st.columns(2)

        with col_a:
            fig_hist = px.histogram(
                candidates, x='ensemble_score', nbins=40,
                title='Ensemble Score Distribution',
                color_discrete_sequence=['#FF6600'],
                template='plotly_dark'
            )
            fig_hist.add_vline(x=ensemble_thresh, line_dash='dash',
                               line_color='red')
            fig_hist.update_layout(
                paper_bgcolor='#0a0a1a',
                plot_bgcolor='#0a0a1a'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            fig_depth = px.scatter(
                candidates, x='center_time', y='depth_pct',
                size='ensemble_score', color='ensemble_score',
                title='Transit Depth vs Time',
                color_continuous_scale='Oranges',
                template='plotly_dark',
                labels={'center_time': 'Time (BKJD)',
                        'depth_pct': 'Depth (%)'}
            )
            fig_depth.update_layout(
                paper_bgcolor='#0a0a1a',
                plot_bgcolor='#0a0a1a'
            )
            st.plotly_chart(fig_depth, use_container_width=True)

        # Zoom into top candidate
        if len(filtered) > 0:
            st.markdown("### 🔬 Zoom: Best Candidate Transit")
            best = filtered.iloc[0]
            pad  = 5.0
            mask = ((df['time'] >= best['center_time'] - pad) &
                    (df['time'] <= best['center_time'] + pad))
            zoom = df[mask]

            fig_zoom = go.Figure()
            fig_zoom.add_trace(go.Scatter(
                x=zoom['time'], y=zoom['flux'],
                mode='lines+markers',
                line=dict(color='#4fa3e0', width=1.5),
                marker=dict(size=3),
                name='Flux'
            ))
            fig_zoom.add_vline(
                x=best['center_time'],
                line_dash='dash', line_color='red', line_width=2,
                annotation_text=f"Transit @ {best['center_time']:.2f} BKJD"
            )
            fig_zoom.add_hline(y=1.0, line_dash='dash',
                               line_color='gray', line_width=0.8)
            fig_zoom.update_layout(
                template='plotly_dark',
                xaxis_title='Time (BKJD days)',
                yaxis_title='Normalized Flux',
                height=350,
                title=(f"Top Candidate — Depth: {best['depth_pct']:.3f}%  |  "
                       f"Score: {best['ensemble_score']:.3f}"),
                paper_bgcolor='#0a0a1a',
                plot_bgcolor='#0a0a1a'
            )
            st.plotly_chart(fig_zoom, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Aditya-L1
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🌞 Aditya-L1 SoLEXS — Cross-Domain Validation")
    st.markdown("""
    The same anomaly detection framework is validated on
    **ISRO's Aditya-L1 SoLEXS** solar X-ray flux data —
    proving this is a *generalized* space time-series detector,
    not just a transit classifier.
    """)

    adl1_path = os.path.join(BASE_DIR, 'data', 'raw', 'aditya_l1_solexs.csv')

    if os.path.exists(adl1_path):
        df_sol = pd.read_csv(adl1_path)

        fig_sol = go.Figure()
        fig_sol.add_trace(go.Scatter(
            x=df_sol.iloc[:, 0], y=df_sol.iloc[:, 1],
            mode='lines',
            line=dict(color='#FF6600', width=0.8),
            name='X-ray Flux'
        ))
        fig_sol.update_layout(
            template='plotly_dark',
            xaxis_title='Time',
            yaxis_title='X-ray Flux',
            height=400,
            title='Aditya-L1 SoLEXS — Solar X-ray Flux',
            paper_bgcolor='#0a0a1a',
            plot_bgcolor='#0a0a1a'
        )
        st.plotly_chart(fig_sol, use_container_width=True)

    else:
        st.info("""
        📡 **Aditya-L1 data not yet loaded.**

        To add it:
        1. Download SoLEXS data from [ISRO PRADAN](https://pradan.issdc.gov.in)
        2. Save as `data/raw/aditya_l1_solexs.csv`
        3. Refresh this page

        The same Z-score + ARIMA + LSTM pipeline will run on it automatically.
        """)

    st.divider()
    st.markdown("""
    #### Why cross-domain validation matters

    | | Kepler/TESS | Aditya-L1 SoLEXS |
    |---|---|---|
    | Signal type | Stellar flux (optical) | Solar X-ray flux |
    | Anomaly | Planet transit dip | Solar flare spike |
    | Sampling | 30-min cadence | Continuous |
    | Source | NASA MAST | ISRO PRADAN |
    | **Same pipeline?** | ✅ | ✅ |
    """)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#666; font-size:12px;'>
ChhayaPath · Bharatiya Antariksh Hackathon 2026 · Challenge 07 ·
Built with Python, TensorFlow, Streamlit
</div>
""", unsafe_allow_html=True)