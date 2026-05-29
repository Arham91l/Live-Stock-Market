# 📈 Intelligent Stock Market Forecasting System

## Overview

An end-to-end time series forecasting system that analyzes and predicts stock market behavior using classical statistical models and machine learning techniques.

The project combines:

* ARIMA forecasting
* GARCH volatility modeling
* XGBoost forecasting
* Stationarity analysis
* Time-series cross-validation
* Anomaly detection
* Automated data updates and retraining

The system works on both US and Indian equities and is designed as a production-oriented forecasting pipeline.

---

## Stocks Analyzed

### US Stocks

* AAPL
* TSLA
* MSFT
* GOOGL
* AMZN
* META
* NVDA

### Indian Stocks

* RELIANCE.NS
* TCS.NS
* INFY.NS
* HDFCBANK.NS

---

## Project Workflow

```text
Historical Stock Data
        ↓
Stationarity Analysis
        ↓
ADF & KPSS Tests
        ↓
Differencing
        ↓
ACF / PACF Analysis
        ↓
ARIMA Forecasting
        ↓
Volatility Modeling (GARCH)
        ↓
Feature Engineering
        ↓
XGBoost Forecasting
        ↓
Model Evaluation
        ↓
Anomaly Detection
        ↓
Live Data Updates
        ↓
Automatic Retraining
```

---

## Features

### 1. Data Collection

Historical stock data is downloaded using Yahoo Finance.

* Multiple tickers supported
* Daily closing prices
* Automated updates
* Incremental data fetching

---

### 2. Stationarity Analysis

Implemented:

* Augmented Dickey-Fuller (ADF) Test
* KPSS Test
* First-order differencing
* Rolling statistics analysis

Used to determine whether time series satisfy ARIMA assumptions.

---

### 3. ACF & PACF Diagnostics

Used to:

* Identify autoregressive patterns
* Identify moving-average patterns
* Select ARIMA parameters

---

### 4. ARIMA Forecasting

Implemented:

* ARIMA(1,1,1)
* Walk-forward validation
* Multi-stock forecasting
* TimeSeriesSplit cross-validation

Evaluation metrics:

* MAE
* RMSE
* MAPE

---

### 5. XGBoost Forecasting

For highly volatile stocks:

* TSLA
* NVDA

Features engineered:

* Lag features
* Rolling mean
* Rolling volatility
* Log returns

Benefits:

* Captures nonlinear patterns
* Handles volatile market behavior

---

### 6. GARCH Volatility Modeling

Implemented:

* GARCH(1,1)

Used to model:

* Conditional variance
* Volatility clustering
* Market risk

Provides volatility forecasts and confidence intervals.

---

### 7. Hybrid ARIMA-GARCH Model

Combines:

* ARIMA mean forecasts
* GARCH volatility estimates

Produces:

* Forecasted price
* Confidence bands
* Risk-aware predictions

---

### 8. Anomaly Detection

Rolling Z-Score based anomaly detection.

Detects:

* Sudden crashes
* Market shocks
* Abnormal returns
* Earnings-event spikes

---

### 9. Automated Data Pipeline

Features:

* Incremental Yahoo Finance updates
* Dataset synchronization
* Duplicate handling
* Metadata tracking
* Automatic retraining support

---

## Model Evaluation Strategy

This project uses:

### TimeSeriesSplit Cross Validation

Unlike random train-test splitting, data is evaluated chronologically.

```text
Fold 1:
Train → Test

Fold 2:
Train --------> Test

Fold 3:
Train -----------------> Test
```

This better simulates real-world forecasting.

---

## Technologies Used

### Python Libraries

* pandas
* numpy
* matplotlib
* yfinance
* statsmodels
* arch
* xgboost
* scikit-learn
* pickle
* joblib

---

## Key Concepts Demonstrated

* Time Series Forecasting
* Stationarity Testing
* ARIMA Modeling
* SARIMA Concepts
* GARCH Volatility Forecasting
* Feature Engineering
* Machine Learning Forecasting
* Cross Validation for Time Series
* Walk-Forward Validation
* Anomaly Detection
* Automated Retraining Pipelines

---

## Future Improvements

* Streamlit Dashboard Deployment
* Real-Time Forecasting Interface
* Technical Indicators (RSI, MACD)
* LSTM Forecasting
* Transformer-based Time Series Models
* Portfolio Optimization Module

---

## Author

**Shaikh Arham Ahmed**

Bachelor's in Artificial Intelligence & Machine Learning

Focused on:

* Machine Learning
* Time Series Forecasting
* Financial Analytics
* AI Engineering
