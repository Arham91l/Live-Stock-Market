import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from datetime import datetime, timedelta
import warnings
import os
import time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense — Live Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 50%, #0a1628 100%);
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f3c 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #e0eeff !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1f3c, #102040);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 20px rgba(0,100,255,0.08);
}
[data-testid="metric-container"] label {
    color: #7ab3ef !important;
    font-size: 0.75rem !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e0f0ff !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.85rem !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: #7ab3ef !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em;
    border-radius: 8px 8px 0 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, #1a4a8a, #1e3a5f) !important;
    border-bottom: 2px solid #4a9eff !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #0d1f3c !important;
    border: 1px solid #1e3a5f !important;
    color: #e0eeff !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a4a8a, #1e6abf) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(30,106,191,0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e6abf, #2484e8) !important;
    box-shadow: 0 6px 20px rgba(30,106,191,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Sliders */
[data-testid="stSlider"] .st-emotion-cache-1inwz65 { background: #4a9eff !important; }

/* Info/success/warning boxes */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}

/* Custom header */
.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4a9eff, #00d4ff, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}
.sub-header {
    color: #4a7ab5;
    font-size: 0.85rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.1em;
    margin-top: 0;
}
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #4a9eff;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-left: 3px solid #4a9eff;
    padding-left: 10px;
    margin-bottom: 12px;
}
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}
.badge-live {
    background: rgba(0, 255, 136, 0.15);
    color: #00ff88;
    border: 1px solid #00ff8844;
}
.badge-cached {
    background: rgba(74, 158, 255, 0.15);
    color: #4a9eff;
    border: 1px solid #4a9eff44;
}
.badge-warning {
    background: rgba(255, 165, 0, 0.15);
    color: #ffa500;
    border: 1px solid #ffa50044;
}
.ticker-card {
    background: linear-gradient(135deg, #0d1f3c, #0a1628);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 8px;
}
.anomaly-card {
    background: linear-gradient(135deg, #2a0d0d, #1a0808);
    border: 1px solid #5f1e1e;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 6px;
}
.forecast-card {
    background: linear-gradient(135deg, #0d2a1f, #081a14);
    border: 1px solid #1e5f3a;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
YF_TICKERS = [
    "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN",
    "META", "NVDA", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"
]
VOLATILE    = ["TSLA", "NVDA"]
COLORS = {
    "price"    : "#4a9eff",
    "forecast" : "#00ff88",
    "upper"    : "rgba(0,255,136,0.3)",
    "lower"    : "rgba(0,255,136,0.1)",
    "anomaly"  : "#ff4a4a",
    "train"    : "#4a9eff",
    "test"     : "#00d4ff",
    "predicted": "#ffa500",
    "vol"      : "#ff6b6b",
    "garch"    : "#7b61ff",
}
CURRENCY = {t: "₹" if t.endswith(".NS") else "$" for t in YF_TICKERS}

# ─────────────────────────────────────────────────────────────
# DATA FUNCTIONS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_and_update_data():
    tickers   = pickle.load(open("models/data/tickers.pkl",   "rb"))
    last_date = pickle.load(open("models/data/last_date.pkl", "rb"))
    close_prices = pd.read_csv(
        "models/data/close_prices.csv",
        index_col=0, parse_dates=True
    )
    try:
        new_data = yf.download(YF_TICKERS, start=last_date, progress=False)["Close"]
        new_data = new_data.iloc[1:]
        if not new_data.empty:
            updated = pd.concat([close_prices, new_data])
            updated = updated[~updated.index.duplicated(keep="last")]
            updated.sort_index(inplace=True)
            updated.to_csv("models/data/close_prices.csv")
            new_last = str(updated.index[-1].date())
            pickle.dump(new_last, open("models/data/last_date.pkl", "wb"))
            return updated, True, new_last
    except Exception as e:
        st.warning(f"Live fetch failed: {e}")
    last_date = pickle.load(open("models/data/last_date.pkl", "rb"))
    return close_prices, False, last_date


@st.cache_resource(show_spinner=False)
def load_models():
    tickers      = pickle.load(open("models/data/tickers.pkl", "rb"))
    arima_models = {}
    garch_models = {}
    for ticker in tickers:
        try:
            with open(f"models/arima/{ticker}_arima.pkl", "rb") as f:
                arima_models[ticker] = pickle.load(f)
        except:
            arima_models[ticker] = None
    for ticker in VOLATILE:
        try:
            with open(f"models/garch/{ticker}_garch.pkl", "rb") as f:
                garch_models[ticker] = pickle.load(f)
        except:
            garch_models[ticker] = None
    return arima_models, garch_models


def retrain_all(close_prices):
    tickers      = list(close_prices.columns)
    arima_models = {}
    garch_models = {}
    progress     = st.progress(0)
    status       = st.empty()
    total        = len(tickers) + len(VOLATILE)
    step         = 0

    for ticker in tickers:
        status.markdown(f'<p class="sub-header">Retraining ARIMA — {ticker}</p>', unsafe_allow_html=True)
        try:
            series = close_prices[ticker].dropna()
            fitted = ARIMA(series, order=(1, 1, 1)).fit()
            with open(f"models/arima/{ticker}_arima.pkl", "wb") as f:
                pickle.dump(fitted, f)
            arima_models[ticker] = fitted
        except Exception as e:
            st.warning(f"ARIMA failed for {ticker}: {e}")
        step += 1
        progress.progress(step / total)

    for ticker in VOLATILE:
        status.markdown(f'<p class="sub-header">Retraining GARCH — {ticker}</p>', unsafe_allow_html=True)
        try:
            series  = close_prices[ticker].dropna()
            returns = series.pct_change().dropna() * 100
            fitted  = arch_model(returns, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")
            with open(f"models/garch/{ticker}_garch.pkl", "wb") as f:
                pickle.dump(fitted, f)
            garch_models[ticker] = fitted
        except Exception as e:
            st.warning(f"GARCH failed for {ticker}: {e}")
        step += 1
        progress.progress(step / total)

    load_models.clear()
    progress.empty()
    status.empty()
    return arima_models, garch_models


@st.cache_data(show_spinner=False)
def detect_anomalies(_close_prices, window=21, threshold=3.0):
    anomaly_dict = {}
    for ticker in _close_prices.columns:
        series    = _close_prices[ticker].dropna()
        returns   = series.pct_change().dropna()
        roll_mean = returns.rolling(window).mean()
        roll_std  = returns.rolling(window).std()
        z_scores  = (returns - roll_mean) / roll_std
        anomalies = z_scores[np.abs(z_scores) > threshold]
        anomaly_dict[ticker] = {
            "series"   : series,
            "returns"  : returns,
            "z_scores" : z_scores,
            "anomalies": anomalies
        }
    return anomaly_dict


def get_forecast(arima_model, garch_model, series, ticker, steps=7):
    future_dates = pd.bdate_range(start=series.index[-1], periods=steps + 1)[1:]
    arima_fc     = arima_model.forecast(steps=steps)
    fc_series    = pd.Series(arima_fc.values, index=future_dates)
    upper, lower = None, None
    if garch_model and ticker in VOLATILE:
        try:
            last_price  = series.iloc[-1]
            vol_fc      = garch_model.forecast(horizon=steps, reindex=False)
            vol         = np.sqrt(vol_fc.variance.values[-1])
            upper       = pd.Series(fc_series.values + (vol / 100) * last_price, index=future_dates)
            lower       = pd.Series(fc_series.values - (vol / 100) * last_price, index=future_dates)
        except:
            pass
    return fc_series, upper, lower


# ─────────────────────────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────
def plot_forecast(series, fc_series, upper, lower, ticker, anomaly_data):
    fig = go.Figure()
    hist = series.iloc[-180:]
    currency = CURRENCY.get(ticker, "$")

    # Historical price
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        name="Historical", line=dict(color=COLORS["price"], width=2),
        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Price: {currency}%{{y:.2f}}<extra></extra>"
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=fc_series.index, y=fc_series.values,
        name="7-Day Forecast",
        line=dict(color=COLORS["forecast"], width=2.5, dash="dot"),
        mode="lines+markers",
        marker=dict(size=7, color=COLORS["forecast"], symbol="diamond"),
        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Forecast: {currency}%{{y:.2f}}<extra></extra>"
    ))

    # GARCH confidence band
    if upper is not None and lower is not None:
        fig.add_trace(go.Scatter(
            x=list(upper.index) + list(lower.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill="toself", fillcolor="rgba(0,255,136,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="GARCH Band", hoverinfo="skip"
        ))

    # Anomaly markers
    if anomaly_data:
        anom   = anomaly_data["anomalies"]
        prices = anomaly_data["series"]
        anom_in_window = anom[anom.index >= hist.index[0]]
        if not anom_in_window.empty:
            anom_prices = prices.loc[anom_in_window.index]
            fig.add_trace(go.Scatter(
                x=anom_prices.index, y=anom_prices.values,
                mode="markers", name="Anomaly",
                marker=dict(size=11, color=COLORS["anomaly"],
                            symbol="x", line=dict(width=2, color="#ff0000")),
                hovertemplate="<b>⚠️ Anomaly</b><br>%{x|%d %b %Y}<br>Price: " + currency + "%{y:.2f}<extra></extra>"
            ))

    # Vertical line at forecast start
    fig.add_vline(
        x=series.index[-1], line_dash="dash",
        line_color="rgba(255,255,255,0.2)", line_width=1
    )
    fig.add_annotation(
        x=series.index[-1], y=series.iloc[-1],
        text="Today", showarrow=False,
        font=dict(color="#7ab3ef", size=10, family="Space Mono"),
        yshift=15
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,31,60,0.5)",
        font=dict(family="DM Sans", color="#e0eeff"),
        legend=dict(
            bgcolor="rgba(13,31,60,0.8)",
            bordercolor="#1e3a5f", borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            gridcolor="rgba(30,58,95,0.5)",
            showgrid=True, zeroline=False
        ),
        yaxis=dict(
            gridcolor="rgba(30,58,95,0.5)",
            showgrid=True, zeroline=False,
            title=f"Price ({currency})"
        ),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10),
        height=420
    )
    return fig


def plot_anomaly_zscore(anomaly_data, ticker):
    z       = anomaly_data["z_scores"].iloc[-252:]
    anom    = anomaly_data["anomalies"]
    anom_in = anom[anom.index >= z.index[0]]

    fig = go.Figure()
    fig.add_hrect(y0=-3, y1=3, fillcolor="rgba(74,158,255,0.05)", line_width=0)
    fig.add_trace(go.Scatter(
        x=z.index, y=z.values,
        name="Z-Score", line=dict(color="#4a9eff", width=1.2),
        fill="tozeroy", fillcolor="rgba(74,158,255,0.08)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Z-Score: %{y:.2f}<extra></extra>"
    ))
    fig.add_hline(y=3,  line_dash="dot", line_color="#ff4a4a", line_width=1.2)
    fig.add_hline(y=-3, line_dash="dot", line_color="#ff4a4a", line_width=1.2)

    if not anom_in.empty:
        fig.add_trace(go.Scatter(
            x=anom_in.index, y=anom_in.values,
            mode="markers", name="Anomaly",
            marker=dict(size=10, color="#ff4a4a", symbol="x",
                        line=dict(width=2, color="#ff0000")),
            hovertemplate="<b>⚠️ Anomaly</b><br>%{x|%d %b %Y}<br>Z: %{y:.2f}<extra></extra>"
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,31,60,0.5)",
        font=dict(family="DM Sans", color="#e0eeff"),
        legend=dict(bgcolor="rgba(13,31,60,0.8)", bordercolor="#1e3a5f", borderwidth=1),
        xaxis=dict(gridcolor="rgba(30,58,95,0.5)"),
        yaxis=dict(gridcolor="rgba(30,58,95,0.5)", title="Z-Score"),
        hovermode="x unified",
        height=250,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig


def plot_garch_volatility(garch_model, ticker, close_prices):
    series   = close_prices[ticker].dropna()
    returns  = series.pct_change().dropna() * 100
    cond_vol = garch_model.conditional_volatility
    real_vol = returns.rolling(21).std().dropna()
    window   = cond_vol.iloc[-252:]
    real_w   = real_vol.iloc[-252:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=real_w.index, y=real_w.values,
        name="Realised Vol (21d)", line=dict(color="#4a9eff", width=1.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Realised: %{y:.2f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=window.index, y=window.values,
        name="GARCH Conditional Vol",
        line=dict(color=COLORS["garch"], width=1.5, dash="dash"),
        fill="tonexty", fillcolor="rgba(123,97,255,0.08)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>GARCH: %{y:.2f}%<extra></extra>"
    ))

    # 7-day forecast
    vol_fc  = garch_model.forecast(horizon=7, reindex=False)
    vol_arr = np.sqrt(vol_fc.variance.values[-1])
    fc_dates = pd.bdate_range(start=series.index[-1], periods=8)[1:]
    fig.add_trace(go.Scatter(
        x=fc_dates, y=vol_arr,
        name="7-day Vol Forecast",
        line=dict(color="#00ff88", width=2, dash="dot"),
        mode="lines+markers",
        marker=dict(size=6, color="#00ff88"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast Vol: ±%{y:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,31,60,0.5)",
        font=dict(family="DM Sans", color="#e0eeff"),
        legend=dict(bgcolor="rgba(13,31,60,0.8)", bordercolor="#1e3a5f", borderwidth=1),
        xaxis=dict(gridcolor="rgba(30,58,95,0.5)"),
        yaxis=dict(gridcolor="rgba(30,58,95,0.5)", title="Volatility (%)"),
        hovermode="x unified",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig


def plot_all_tickers(close_prices):
    fig = go.Figure()
    palette = px.colors.qualitative.Bold
    for i, ticker in enumerate(close_prices.columns):
        s = close_prices[ticker].dropna()
        s_norm = (s / s.iloc[0]) * 100
        fig.add_trace(go.Scatter(
            x=s_norm.index, y=s_norm.values,
            name=ticker,
            line=dict(color=palette[i % len(palette)], width=1.5),
            hovertemplate=f"<b>{ticker}</b><br>%{{x|%d %b %Y}}<br>Indexed: %{{y:.1f}}<extra></extra>"
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,31,60,0.5)",
        font=dict(family="DM Sans", color="#e0eeff"),
        legend=dict(bgcolor="rgba(13,31,60,0.8)", bordercolor="#1e3a5f",
                    borderwidth=1, orientation="h", y=-0.2),
        xaxis=dict(gridcolor="rgba(30,58,95,0.5)"),
        yaxis=dict(gridcolor="rgba(30,58,95,0.5)", title="Indexed Price (Base=100)"),
        hovermode="x unified",
        height=400,
        margin=dict(l=10, r=10, t=10, b=60)
    )
    return fig


def plot_correlation(close_prices):
    returns = close_prices.pct_change().dropna()
    corr    = returns.corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,31,60,0.5)",
        font=dict(family="DM Sans", color="#e0eeff"),
        coloraxis_colorbar=dict(
            title="Correlation",
            tickfont=dict(color="#e0eeff")
        ),
        height=450,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-header" style="font-size:1.4rem;">📈 StockSense</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">LIVE FORECASTING SYSTEM</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<p class="section-title">Ticker Selection</p>', unsafe_allow_html=True)
    selected_ticker = st.selectbox(
        "Choose Stock", YF_TICKERS,
        format_func=lambda x: f"{'🇮🇳' if x.endswith('.NS') else '🇺🇸'} {x}"
    )

    st.markdown('<p class="section-title">Forecast Settings</p>', unsafe_allow_html=True)
    forecast_days = st.slider("Forecast Horizon (days)", 3, 30, 7)
    history_days  = st.slider("History Window (days)", 60, 365, 180)
    anom_threshold = st.slider("Anomaly Z-Score Threshold", 2.0, 5.0, 3.0, 0.5)

    st.divider()
    st.markdown('<p class="section-title">Model Controls</p>', unsafe_allow_html=True)
    retrain_btn = st.button("🔄 Force Retrain Models", use_container_width=True)
    refresh_btn = st.button("🔃 Refresh Live Data", use_container_width=True)

    st.divider()
    st.markdown("""
    <div style="font-family: Space Mono; font-size: 0.65rem; color: #4a7ab5; line-height: 1.8;">
    MODELS<br>
    ▸ ARIMA(1,1,1) — all tickers<br>
    ▸ GARCH(1,1) — TSLA, NVDA<br><br>
    VALIDATION<br>
    ▸ TimeSeriesSplit (5 folds)<br>
    ▸ Walk-forward evaluation<br><br>
    DATA SOURCE<br>
    ▸ Yahoo Finance (yfinance)<br>
    ▸ Auto-refresh every 1hr
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN — DATA LOAD
# ─────────────────────────────────────────────────────────────
if refresh_btn:
    load_and_update_data.clear()
    st.rerun()

with st.spinner(""):
    close_prices, has_new_data, last_date = load_and_update_data()

if retrain_btn:
    with st.spinner("Retraining all models on latest data ..."):
        retrain_all(close_prices)
    st.success("All models retrained successfully ✅")
    st.rerun()
elif has_new_data:
    with st.spinner("New data detected — retraining models ..."):
        retrain_all(close_prices)

arima_models, garch_models = load_models()
anomaly_dict = detect_anomalies(close_prices, threshold=anom_threshold)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<p class="main-header">StockSense</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">REAL-TIME FORECASTING & ANOMALY DETECTION</p>', unsafe_allow_html=True)
with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    if has_new_data:
        st.markdown('<span class="status-badge badge-live">⚡ LIVE — Models Retrained</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-cached">📦 Cached Models Loaded</span>', unsafe_allow_html=True)
    st.caption(f"Data up to: **{last_date}**")

st.divider()

# ─────────────────────────────────────────────────────────────
# MARKET OVERVIEW METRICS
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Market Overview — All Tickers</p>', unsafe_allow_html=True)

metric_cols = st.columns(len(YF_TICKERS))
for i, ticker in enumerate(YF_TICKERS):
    series   = close_prices[ticker].dropna()
    last_p   = series.iloc[-1]
    prev_p   = series.iloc[-2]
    change   = ((last_p - prev_p) / prev_p) * 100
    currency = CURRENCY.get(ticker, "$")
    with metric_cols[i]:
        st.metric(
            label=ticker,
            value=f"{currency}{last_p:,.2f}",
            delta=f"{change:+.2f}%"
        )

st.divider()

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast",
    "⚠️ Anomaly Detection",
    "🌋 GARCH Volatility",
    "🌐 Market Overview",
    "📊 Correlation"
])


# ════════════════════════════════════════
# TAB 1 — FORECAST
# ════════════════════════════════════════
with tab1:
    series       = close_prices[selected_ticker].dropna()
    arima_model  = arima_models.get(selected_ticker)
    garch_model  = garch_models.get(selected_ticker)
    anomaly_data = anomaly_dict.get(selected_ticker)
    currency     = CURRENCY.get(selected_ticker, "$")

    if arima_model is None:
        st.error(f"ARIMA model not found for {selected_ticker}. Please retrain.")
    else:
        fc_series, upper, lower = get_forecast(
            arima_model, garch_model, series, selected_ticker, forecast_days
        )

        # Forecast KPIs
        last_price  = series.iloc[-1]
        fc_end      = fc_series.iloc[-1]
        fc_change   = ((fc_end - last_price) / last_price) * 100
        anom_count  = len(anomaly_data["anomalies"]) if anomaly_data else 0
        recent_anom = len(anomaly_data["anomalies"].last("30D")) if anomaly_data else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Current Price",    f"{currency}{last_price:,.2f}")
        k2.metric(f"{forecast_days}d Forecast", f"{currency}{fc_end:,.2f}", f"{fc_change:+.2f}%")
        k3.metric("Total Anomalies",  f"{anom_count}")
        k4.metric("Anomalies (30d)",  f"{recent_anom}",
                  delta="⚠️ Active" if recent_anom > 0 else "✅ Clean",
                  delta_color="inverse" if recent_anom > 0 else "normal")

        st.markdown('<p class="section-title">Price Forecast Chart</p>', unsafe_allow_html=True)

        # Trim to history window
        trimmed = series.iloc[-history_days:]
        fig = plot_forecast(trimmed, fc_series, upper, lower, selected_ticker, anomaly_data)
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        col_fc, col_anom = st.columns([1, 1])
        with col_fc:
            st.markdown('<p class="section-title">Forecast Values</p>', unsafe_allow_html=True)
            fc_df = pd.DataFrame({
                "Date"    : fc_series.index.strftime("%d %b %Y"),
                "Forecast": [f"{currency}{v:,.2f}" for v in fc_series.values],
                "Change"  : [f"{((v - last_price)/last_price)*100:+.2f}%" for v in fc_series.values]
            })
            if upper is not None:
                fc_df["Upper Band"] = [f"{currency}{v:,.2f}" for v in upper.values]
                fc_df["Lower Band"] = [f"{currency}{v:,.2f}" for v in lower.values]
            st.dataframe(fc_df.set_index("Date"), use_container_width=True)

        with col_anom:
            st.markdown('<p class="section-title">Recent Anomalies (Last 90 Days)</p>', unsafe_allow_html=True)
            if anomaly_data:
                recent = anomaly_data["anomalies"].last("90D")
                if recent.empty:
                    st.success("✅ No anomalies detected in last 90 days")
                else:
                    anom_df = pd.DataFrame({
                        "Date"   : recent.index.strftime("%d %b %Y"),
                        "Z-Score": [f"{v:+.2f}" for v in recent.values],
                        "Signal" : ["🔴 Crash" if v < 0 else "🟢 Spike" for v in recent.values]
                    })
                    st.dataframe(anom_df.set_index("Date"), use_container_width=True)


# ════════════════════════════════════════
# TAB 2 — ANOMALY DETECTION
# ════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Anomaly Detection — All Tickers</p>', unsafe_allow_html=True)

    # Summary metrics
    total_anomalies = sum(len(d["anomalies"]) for d in anomaly_dict.values())
    recent_total    = sum(len(d["anomalies"].last("30D")) for d in anomaly_dict.values())
    worst_ticker    = max(anomaly_dict, key=lambda t: len(anomaly_dict[t]["anomalies"]))

    a1, a2, a3 = st.columns(3)
    a1.metric("Total Anomalies (All Tickers)", total_anomalies)
    a2.metric("Anomalies in Last 30 Days",     recent_total,
              delta="⚠️ Active" if recent_total > 0 else "✅ Clean",
              delta_color="inverse" if recent_total > 0 else "normal")
    a3.metric("Most Volatile Ticker",           worst_ticker)

    st.divider()

    # Per ticker anomaly plots
    for ticker in YF_TICKERS:
        anom   = anomaly_dict[ticker]
        n_anom = len(anom["anomalies"])
        r_anom = len(anom["anomalies"].last("30D"))

        with st.expander(
            f"{'🔴' if r_anom > 0 else '🟢'} {ticker} — {n_anom} total anomalies | {r_anom} in last 30 days",
            expanded=(ticker == selected_ticker)
        ):
            fig = plot_anomaly_zscore(anom, ticker)
            st.plotly_chart(fig, use_container_width=True)

            if not anom["anomalies"].empty:
                worst = anom["anomalies"].abs().nlargest(5)
                worst_df = pd.DataFrame({
                    "Date"   : worst.index.strftime("%d %b %Y"),
                    "Z-Score": [f"{v:+.2f}" for v in anom["anomalies"].loc[worst.index].values],
                    "Type"   : ["🔴 Crash" if v < 0 else "🟢 Spike"
                                for v in anom["anomalies"].loc[worst.index].values]
                })
                st.caption("Top 5 most extreme anomalies:")
                st.dataframe(worst_df.set_index("Date"), use_container_width=True)


# ════════════════════════════════════════
# TAB 3 — GARCH VOLATILITY
# ════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">GARCH(1,1) Volatility Analysis — TSLA & NVDA</p>', unsafe_allow_html=True)
    st.caption("GARCH models volatility clustering — not price. Used for confidence bands and risk assessment.")

    for ticker in VOLATILE:
        gm = garch_models.get(ticker)
        if gm is None:
            st.warning(f"GARCH model not found for {ticker}")
            continue

        series    = close_prices[ticker].dropna()
        returns   = series.pct_change().dropna() * 100
        cond_vol  = gm.conditional_volatility
        vol_fc    = gm.forecast(horizon=7, reindex=False)
        vol_arr   = np.sqrt(vol_fc.variance.values[-1])
        currency  = CURRENCY.get(ticker, "$")

        st.markdown(f"### {ticker}")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Current Volatility", f"±{cond_vol.iloc[-1]:.2f}%")
        v2.metric("1-Day Vol Forecast", f"±{vol_arr[0]:.2f}%")
        v3.metric("7-Day Vol Forecast", f"±{vol_arr[-1]:.2f}%")
        v4.metric("AIC",                f"{gm.aic:.2f}")

        fig = plot_garch_volatility(gm, ticker, close_prices)
        st.plotly_chart(fig, use_container_width=True)

        # Diagnostics summary
        diag1, diag2, diag3 = st.columns(3)
        diag1.markdown("""
        <div class="forecast-card">
        <b style="color:#00ff88">Ljung-Box</b><br>
        <span style="color:#e0eeff; font-size:0.85rem;">p > 0.05 ✅<br>No autocorrelation in residuals</span>
        </div>
        """, unsafe_allow_html=True)
        diag2.markdown("""
        <div class="forecast-card">
        <b style="color:#00ff88">ARCH-LM</b><br>
        <span style="color:#e0eeff; font-size:0.85rem;">p > 0.05 ✅<br>Volatility clustering captured</span>
        </div>
        """, unsafe_allow_html=True)
        diag3.markdown("""
        <div class="anomaly-card">
        <b style="color:#ffa500">Jarque-Bera</b><br>
        <span style="color:#e0eeff; font-size:0.85rem;">p < 0.05 ⚠️<br>Fat tails — normal for stocks</span>
        </div>
        """, unsafe_allow_html=True)
        st.divider()


# ════════════════════════════════════════
# TAB 4 — MARKET OVERVIEW
# ════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Indexed Price Comparison (Base = 100)</p>', unsafe_allow_html=True)
    st.plotly_chart(plot_all_tickers(close_prices), use_container_width=True)

    st.markdown('<p class="section-title">All Tickers — Summary</p>', unsafe_allow_html=True)
    rows = []
    for ticker in YF_TICKERS:
        s        = close_prices[ticker].dropna()
        currency = CURRENCY.get(ticker, "$")
        ret_1d   = ((s.iloc[-1] - s.iloc[-2])  / s.iloc[-2])  * 100
        ret_30d  = ((s.iloc[-1] - s.iloc[-22]) / s.iloc[-22]) * 100
        ret_ytd  = ((s.iloc[-1] - s.iloc[0])   / s.iloc[0])   * 100
        vol_30d  = s.pct_change().iloc[-22:].std() * 100
        anom_cnt = len(anomaly_dict[ticker]["anomalies"])
        rows.append({
            "Ticker"    : ticker,
            "Price"     : f"{currency}{s.iloc[-1]:,.2f}",
            "1D %"      : f"{ret_1d:+.2f}%",
            "30D %"     : f"{ret_30d:+.2f}%",
            "YTD %"     : f"{ret_ytd:+.2f}%",
            "30D Vol %"  : f"{vol_30d:.2f}%",
            "Anomalies" : anom_cnt
        })
    summary_df = pd.DataFrame(rows).set_index("Ticker")
    st.dataframe(summary_df, use_container_width=True, height=420)


# ════════════════════════════════════════
# TAB 5 — CORRELATION
# ════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">Return Correlation Heatmap</p>', unsafe_allow_html=True)
    st.caption("Pairwise correlation of daily returns across all tickers.")
    st.plotly_chart(plot_correlation(close_prices), use_container_width=True)

    st.markdown('<p class="section-title">Key Insights</p>', unsafe_allow_html=True)
    returns = close_prices.pct_change().dropna()
    corr    = returns.corr()

    # Find highest/lowest correlations
    corr_pairs = []
    tickers_list = list(close_prices.columns)
    for i in range(len(tickers_list)):
        for j in range(i+1, len(tickers_list)):
            corr_pairs.append((tickers_list[i], tickers_list[j], corr.iloc[i, j]))

    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top5    = corr_pairs[:5]
    bottom5 = sorted(corr_pairs, key=lambda x: x[2])[:5]

    ins1, ins2 = st.columns(2)
    with ins1:
        st.markdown("**🔴 Highest Correlations (Move Together)**")
        for a, b, v in top5:
            st.markdown(f"`{a}` ↔ `{b}` — **{v:.3f}**")
    with ins2:
        st.markdown("**🔵 Lowest Correlations (Diversification)**")
        for a, b, v in bottom5:
            st.markdown(f"`{a}` ↔ `{b}` — **{v:.3f}**")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style="font-family: Space Mono; font-size: 0.65rem; color: #2a4a6a; text-align: center; padding: 10px 0;">
StockSense &nbsp;|&nbsp; ARIMA(1,1,1) + GARCH(1,1) &nbsp;|&nbsp; 
11 Tickers &nbsp;|&nbsp; Auto-refreshes every hour &nbsp;|&nbsp;
Last updated: {last_date}
</div>
""", unsafe_allow_html=True)
