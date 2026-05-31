# 📈 StockSense — Real-Time Stock Forecasting & Anomaly Detection

A production-ready time series forecasting system that fetches live stock market data, trains statistical models, detects anomalies, and visualizes forecasts on an interactive dashboard — deployed on Streamlit Cloud.

---

## 🌐 Live Demo

🔗 **[stocksense.streamlit.app](https://arham91l-live-stock-market.streamlit.app)**

---

## 📌 Project Overview

StockSense is an end-to-end time series machine learning project that:

- Fetches **live stock price data** from Yahoo Finance
- Trains **ARIMA(1,1,1)** models for price forecasting across 11 tickers
- Applies **GARCH(1,1)** for volatility modelling on high-volatility stocks
- Detects **price anomalies** using Z-score analysis
- **Auto-retrains** models when new market data is detected
- Presents everything on a **colorful interactive Streamlit dashboard**

---

## 🗂️ Dataset

| Source | Details |
|---|---|
| **Yahoo Finance** (`yfinance`) | Historical OHLCV data from 2022 → present |
| **Update frequency** | Auto-fetches on every app startup |
| **Tickers covered** | 7 US stocks + 4 Indian stocks |

### Tickers

| Market | Tickers |
|---|---|
| 🇺🇸 US | `AAPL` `TSLA` `MSFT` `GOOGL` `AMZN` `META` `NVDA` |
| 🇮🇳 India (NSE) | `RELIANCE.NS` `TCS.NS` `INFY.NS` `HDFCBANK.NS` |

---

## 🧠 Models & Methodology

### 1. Stationarity Testing
Before model selection, each ticker's price series is tested for stationarity using:

- **ADF Test** (Augmented Dickey-Fuller) — tests for unit root
- **KPSS Test** — tests for trend stationarity

Both tests confirmed all 11 tickers are **non-stationary in levels, stationary after 1st differencing** → `d = 1` for all tickers.

### 2. ARIMA(1,1,1) — All 11 Tickers
- `p = 1` identified from PACF cutoff at lag 1
- `d = 1` from stationarity tests
- `q = 1` identified from ACF cutoff at lag 1
- Validated using **TimeSeriesSplit (5 folds)** with walk-forward evaluation

### 3. GARCH(1,1) — TSLA & NVDA Only
- Applied to high-volatility tickers where price uncertainty is significant
- Predicts **conditional volatility** (not price)
- Used to generate **confidence bands** around ARIMA forecasts
- Validated using Ljung-Box, ARCH-LM, and Jarque-Bera tests

### 4. Anomaly Detection — Z-Score Method
- Computes rolling 21-day Z-score on daily returns
- Flags observations beyond ±3 standard deviations
- Configurable threshold via sidebar slider

---

## 📊 Model Performance

Validated using **TimeSeriesSplit (5-fold)** cross-validation with walk-forward forecasting:

| Ticker | Avg MAE | Avg RMSE | Avg MAPE% |
|---|---|---|---|
| TCS.NS | 29.12 | 40.34 | **1.05%** ✅ |
| HDFCBANK.NS | 7.36 | 10.34 | **1.08%** ✅ |
| RELIANCE.NS | 12.24 | 16.99 | **1.20%** ✅ |
| INFY.NS | 14.73 | 20.42 | **1.20%** ✅ |
| MSFT | 3.49 | 4.75 | **1.26%** ✅ |
| AAPL | 1.97 | 2.75 | **1.36%** ✅ |
| GOOGL | 1.76 | 2.44 | **1.42%** ✅ |
| AMZN | 2.30 | 3.17 | **1.55%** ✅ |
| META | 5.56 | 8.30 | **1.76%** ✅ |
| NVDA | 1.17 | 1.65 | **2.37%** 🟡 |
| TSLA | 6.14 | 8.49 | **2.93%** 🟡 |

> **Note:** High MAE/RMSE for Indian stocks (RELIANCE, TCS, INFY) is due to INR scale — MAPE confirms strong performance. TSLA and NVDA have higher MAPE due to AI-driven volatility, addressed with GARCH confidence bands.

### GARCH Diagnostics (TSLA & NVDA)

| Test | Result | Interpretation |
|---|---|---|
| Ljung-Box | p > 0.05 ✅ | No autocorrelation in residuals |
| ARCH-LM | p > 0.05 ✅ | Volatility clustering fully captured |
| Jarque-Bera | p < 0.05 ⚠️ | Fat tails — expected for volatile stocks |

---

## 🖥️ Dashboard Features

| Tab | Features |
|---|---|
| 📈 **Forecast** | Live price chart, 7-day ARIMA forecast, GARCH confidence bands, anomaly markers, forecast table |
| ⚠️ **Anomaly Detection** | Z-score chart per ticker, worst anomaly dates, 30-day active count |
| 🌋 **GARCH Volatility** | Conditional vs realised volatility, 7-day vol forecast, diagnostics summary |
| 🌐 **Market Overview** | Indexed price comparison (base=100), 1D/30D/YTD returns, volatility table |
| 📊 **Correlation** | Return correlation heatmap, highest/lowest correlated pairs |

### Sidebar Controls
- **Ticker selector** — choose any of the 11 stocks
- **Forecast horizon** — 3 to 30 days
- **History window** — 60 to 365 days (chart zoom)
- **Anomaly threshold** — Z-score from 2.0 to 5.0
- **Force Retrain** — manually retrain all models
- **Refresh Live Data** — fetch latest prices immediately

---

## ⚙️ Live Retraining Architecture

```
App startup
    ↓
Check last_date in models/data/last_date.pkl
    ↓
Fetch new data from yfinance (last_date → today)
    ↓
New data found?
    YES → append to close_prices.csv
          retrain all 11 ARIMA + 2 GARCH models
          save updated .pkl files
          update last_date
    NO  → load saved models directly
    ↓
Render dashboard with fresh forecasts
```

Cache refreshes every 1 hour automatically via `@st.cache_data(ttl=3600)`.

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Arham91l/Live-Stock-Market.git
cd Live-Stock-Market
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy yfinance plotly statsmodels arch scikit-learn
```

### 3. Run the app
```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 📁 Project Structure

```
Live-Stock-Market/
│
├── app.py                     # Main Streamlit dashboard
├── requirements.txt           # Python dependencies
├── README.md
│
├── .streamlit/
│   └── config.toml            # Streamlit theme config
│
├── models/
│   ├── arima/                 # Trained ARIMA models (.pkl)
│   │   ├── AAPL_arima.pkl
│   │   ├── TSLA_arima.pkl
│   │   └── ... (11 total)
│   ├── garch/                 # Trained GARCH models (.pkl)
│   │   ├── TSLA_garch.pkl
│   │   └── NVDA_garch.pkl
│   └── data/                  # Cached data files
│       ├── close_prices.csv
│       ├── tickers.pkl
│       ├── last_date.pkl
│       └── last_prices.pkl
│
└── stock.ipynb                # Training notebook (Google Colab)
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data | `yfinance`, `pandas` |
| Statistical Models | `statsmodels` (ARIMA), `arch` (GARCH) |
| Validation | `scikit-learn` (TimeSeriesSplit) |
| Visualization | `plotly`, `streamlit` |
| Deployment | Streamlit Cloud, Git LFS |

---

## 📓 Training Notebook

All model training was done in **Google Colab** (`stock.ipynb`) and includes:

- Data fetching and preprocessing
- Stationarity testing (ADF + KPSS) with plots
- ACF/PACF analysis for order selection
- ARIMA training with TimeSeriesSplit cross-validation
- GARCH fitting and diagnostic tests
- Model serialization (`.pkl` files)

---

## 🔑 Key Design Decisions

**Why ARIMA over XGBoost?**
XGBoost was evaluated on TSLA and NVDA but achieved MAPE of 14–19% vs ARIMA's 2–3%. Tree-based models require richer feature sets beyond price history to outperform statistical baselines on financial time series. ARIMA was retained for all tickers.

**Why GARCH only for TSLA & NVDA?**
GARCH models volatility clustering — meaningful only for tickers with significant heteroskedasticity. TSLA and NVDA show clear volatility bursts (earnings, AI news) while other tickers are sufficiently stable for ARIMA alone.

**Why TimeSeriesSplit over 80/20?**
Standard train-test split gives a single performance snapshot. TimeSeriesSplit across 5 folds with walk-forward evaluation gives a more robust, realistic estimate of model performance across different market conditions.

---

## 👤 Author

**Arham** — B.Tech Student  
📧 [GitHub](https://github.com/Arham91l)

---

## Link
https://live-stock-market-uzghanxyv5ctrwvlbdfinc.streamlit.app/

---

## 📄 License

This project is for educational purposes. Stock forecasts are not financial advice.
