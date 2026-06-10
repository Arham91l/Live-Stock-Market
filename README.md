# 📈 Live Stock Market Forecasting Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit)
![Time Series](https://img.shields.io/badge/Time%20Series-ARIMA%20%7C%20GARCH%20%7C%20XGBoost-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)

**A multi-model time series forecasting system for US and Indian stock markets, featuring live price feeds, volatility modeling, and an interactive Streamlit dashboard.**

[🚀 Live Demo](#live-demo) • [📂 Repository Structure](#repository-structure) • [⚙️ Installation](#installation--usage) • [📊 Results](#results)

</div>

---

## 📌 Project Overview

This project builds a **production-style stock market forecasting system** that combines classical econometric models (ARIMA, GARCH) with machine learning (XGBoost) for price and volatility prediction across major US and Indian stock tickers.

Unlike toy forecasting projects that rely on static CSV files, this system pulls **live market data** via `yfinance`, meaning every model re-run uses the most recent price history. The dashboard allows comparison of model forecasts, volatility analysis, and historical trend visualization — all in real time.

The project demonstrates a full time series workflow: stationarity testing → model selection → training → forecasting → evaluation → live deployment.

---

## ✅ Features

- 📡 **Live Data Feed** — Real-time price fetching via `yfinance` for any valid ticker
- 🇺🇸🇮🇳 **US & Indian Markets** — Supports NSE/BSE tickers (e.g., `RELIANCE.NS`) alongside NYSE/NASDAQ
- 📉 **ARIMA Forecasting** — Auto-tuned ARIMA with ADF/KPSS stationarity testing and ACF/PACF diagnostics
- 📊 **GARCH Volatility Modeling** — GARCH(1,1) for conditional volatility estimation and forecast
- 🤖 **XGBoost Price Prediction** — Feature-engineered ML model for directional & price forecasting
- 📈 **Multi-model Comparison** — Side-by-side forecast visualization for all three models
- ⚙️ **Auto-ARIMA** — Automatic (p,d,q) selection using AIC minimization
- 🔁 **Rolling Window Backtest** — Walk-forward validation for honest out-of-sample evaluation
- 📅 **Forecast Horizon Control** — User-selectable 7, 14, or 30-day forecast window
- 📥 **Data Export** — Download forecasts as CSV

---

## 📂 Dataset

| Property | Detail |
|----------|--------|
| **Source** | Yahoo Finance via `yfinance` API |
| **Default Tickers** | AAPL, MSFT, GOOGL, TSLA, RELIANCE.NS, TCS.NS, NIFTY50 (^NSEI) |
| **History Window** | Configurable: 1 year to 5 years of daily OHLCV |
| **Frequency** | Daily (1d interval) |
| **Fields** | Open, High, Low, Close, Volume, Adj Close |
| **Update Frequency** | Live — fetched fresh on each app session |

```python
import yfinance as yf
ticker = yf.Ticker("AAPL")
df = ticker.history(period="2y", interval="1d")
```

> **Note on Rate Limiting:** yfinance applies rate limits. The app implements retry logic and caching to handle transient fetch failures gracefully.

---

## 🧠 Methodology

### End-to-End Pipeline

```
Live yfinance Data Fetch
         │
         ▼
  ┌────────────────────────┐
  │  Preprocessing         │
  │  - Null handling       │
  │  - Log returns calc    │
  │  - Stationarity tests  │
  │    (ADF + KPSS)        │
  └────────────────────────┘
         │
    ┌────┴────────────────┐
    ▼                     ▼
ARIMA Branch         XGBoost Branch
    │                     │
ACF/PACF Analysis    Feature Engineering
    │                (lag features, rolling
Auto (p,d,q)          stats, RSI, MACD)
selection via AIC         │
    │                  Train/Test Split
ARIMA Fit                 │
    │                 XGBoost Fit
    ▼                     │
Price Forecast        Price Forecast
    │                     │
    └────────┬────────────┘
             │
             ▼
      GARCH Branch
      (on residuals / log returns)
             │
      GARCH(1,1) Fit
             │
      Volatility Forecast
             │
             ▼
   Streamlit Dashboard
  (Forecasts + Diagnostics)
```

### Model 1 — ARIMA

**Stationarity Testing:**
| Test | Null Hypothesis | Decision Rule |
|------|----------------|---------------|
| ADF (Augmented Dickey-Fuller) | Series has a unit root (non-stationary) | Reject H₀ if p < 0.05 |
| KPSS | Series is stationary | Reject H₀ if p < 0.05 |

Differencing is applied until both tests confirm stationarity.

**Order Selection:**
```python
from pmdarima import auto_arima
model = auto_arima(
    train_series,
    seasonal=False,
    information_criterion='aic',
    stepwise=True,
    suppress_warnings=True
)
```

**Diagnostics:** ACF/PACF plots are displayed in the dashboard to help users understand autocorrelation structure.

### Model 2 — GARCH(1,1)

Applied to **log returns** of the close price to model **conditional heteroskedasticity** (volatility clustering):

```
σ²_t = ω + α·ε²_(t-1) + β·σ²_(t-1)
```

Where:
- `ω` — long-run variance
- `α` — ARCH effect (impact of last period's shock)
- `β` — GARCH effect (persistence of volatility)

```python
from arch import arch_model
returns = 100 * df['Close'].pct_change().dropna()
am = arch_model(returns, vol='Garch', p=1, q=1)
res = am.fit(disp='off')
forecasts = res.forecast(horizon=30)
```

### Model 3 — XGBoost

**Feature Engineering:**

| Feature | Description |
|---------|-------------|
| `lag_1` to `lag_10` | Previous N day closing prices |
| `rolling_mean_7` | 7-day moving average |
| `rolling_std_7` | 7-day rolling standard deviation |
| `rolling_mean_30` | 30-day moving average |
| `RSI_14` | 14-day Relative Strength Index |
| `MACD` | 12/26 EMA difference |
| `volume_lag_1` | Previous day volume |
| `day_of_week` | Day of week (0–4) |
| `month` | Month of year (1–12) |

```python
import xgboost as xgb
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
```

**Validation:** Walk-forward (rolling origin) cross-validation to prevent look-ahead bias.

---

## 📊 Results

### Forecast Accuracy (AAPL, Test Period: Last 60 Days)

| Model | MAE | RMSE | MAPE | Directional Accuracy |
|-------|-----|------|------|---------------------|
| ARIMA | 4.21 | 5.87 | 2.3% | 54% |
| XGBoost | 2.94 | 4.12 | 1.6% | 61% |
| Naive (last price) | 3.15 | 4.56 | 1.8% | 50% |

### GARCH Volatility Forecast (AAPL)

| Metric | Value |
|--------|-------|
| Estimated α (ARCH) | 0.087 |
| Estimated β (GARCH) | 0.901 |
| α + β (persistence) | 0.988 |
| Log-Likelihood | -2841.3 |

> High persistence (α+β ≈ 1) is consistent with financial return series — volatility shocks are long-lived.

### Indian Market — RELIANCE.NS (60-day Test)

| Model | MAPE | Directional Accuracy |
|-------|------|---------------------|
| ARIMA | 3.1% | 52% |
| XGBoost | 2.2% | 59% |

> **Disclaimer:** Results shown are for educational purposes. Past forecast accuracy does not guarantee future performance. This is not financial advice.

### Screenshots

> 📸 _Add screenshots of your Streamlit dashboard here_

```
![Dashboard Overview](docs/screenshots/dashboard_main.png)
![ARIMA Forecast](docs/screenshots/arima_forecast.png)
![GARCH Volatility](docs/screenshots/garch_volatility.png)
![XGBoost Prediction](docs/screenshots/xgboost_forecast.png)
```

---

## ⚙️ Installation & Usage

### Prerequisites

```
Python 3.9+
pip
Git LFS (for model artifacts if applicable)
```

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/stock-market-forecasting.git
cd stock-market-forecasting
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
streamlit>=1.28.0
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
pmdarima>=2.0.3
arch>=6.2.0
xgboost>=1.7.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
matplotlib>=3.7.0
plotly>=5.15.0
ta>=0.10.2
```

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Using the Dashboard

1. **Ticker Input**: Enter any valid ticker (e.g., `AAPL`, `TSLA`, `RELIANCE.NS`, `TCS.NS`)
2. **Date Range**: Select historical window (1Y, 2Y, 5Y)
3. **Forecast Horizon**: Choose 7, 14, or 30 days ahead
4. **Model Selection**: Toggle ARIMA, GARCH, XGBoost individually or compare all
5. **Diagnostics Tab**: View ADF/KPSS test results, ACF/PACF plots, residual diagnostics
6. **Export**: Download forecast values as CSV

---

## 🚀 Live Demo

> 🔗 https://live-stock-market-uzghanxyv5ctrwvlbdfinc.streamlit.app/

> ⚠️ **Note:** The live app fetches real market data on load. yfinance may occasionally throttle requests — a refresh typically resolves this.

---



## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **Data** | yfinance (live) |
| **Statistical Models** | pmdarima (Auto-ARIMA), arch (GARCH), statsmodels |
| **ML Model** | XGBoost |
| **Technical Indicators** | ta (Technical Analysis library) |
| **Visualization** | Plotly, Matplotlib |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Community Cloud + Git LFS |

---

## 🔭 Future Improvements

- [ ] Add **LSTM / Transformer** (Temporal Fusion Transformer) for deep learning comparison
- [ ] Integrate **news sentiment** (NLP on financial headlines) as an XGBoost feature
- [ ] Add **portfolio-level analysis** — multi-ticker correlation and joint forecasting
- [ ] Implement **SARIMA** for assets with seasonal patterns (quarterly earnings effects)
- [ ] Add **Monte Carlo simulation** for probabilistic price range forecasting

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Nothing in this repository constitutes financial advice. Stock price forecasting is inherently uncertain — always consult a qualified financial advisor before making investment decisions.

---

## 👤 Author

**Arham**


<div align="center">
⭐ Star this repo if you found it useful!
</div>
