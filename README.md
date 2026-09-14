# 📈 Multi-Asset Stock Price Prediction using Deep Learning and Machine Learning

## 🚀 Project Overview

This project explores the application of **Deep Learning and Machine Learning techniques for financial time-series forecasting**, with the objective of predicting future stock prices from historical market data.

The study compares multiple predictive approaches, including:

- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- CNN-LSTM
- Attention-LSTM
- Random Forest
- XGBoost

The models are trained on historical daily closing prices of five major technology companies and evaluated using **Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE)**.

The primary objective is to understand how different model architectures capture **temporal dependencies, market trends, volatility, and sequential patterns** in financial data.

---

## 🎯 Objectives

The main objectives of this project are to:

- Analyze historical stock price trends and market fluctuations.
- Build time-series forecasting models using both Machine Learning and Deep Learning.
- Capture temporal dependencies using recurrent neural networks.
- Compare traditional Machine Learning models with sequential Deep Learning architectures.
- Evaluate model performance using RMSE, MAE, and MAPE.
- Analyze stock market volatility over time.
- Generate future stock price forecasts using the best-performing model.
- Understand the strengths and limitations of different forecasting architectures.

---

## 📊 Dataset

The historical stock price data was obtained from the **Yahoo Finance API** using the `yfinance` Python library.

### Stocks Analyzed

| Company | Ticker |
|---|---|
| Apple | AAPL |
| Microsoft | MSFT |
| Google | GOOGL |
| Amazon | AMZN |
| Tesla | TSLA |

The dataset contains daily closing prices covering approximately **January 2015 to December 2023**.

The resulting dataset contains:

- **2,264 trading-day observations**
- **5 stock price features**
- Daily closing prices for all five companies
- No missing values

The data is structured as a time series, with historical observations used to learn patterns for future predictions.

---

## 🔄 Data Preprocessing

Before model training, several preprocessing steps were performed.

### 1. Missing Value Analysis

The dataset was checked for missing values across all five stocks.

No missing values were identified.

### 2. Feature Scaling

Stock prices were normalized using **MinMaxScaler**, transforming the values into a common range between 0 and 1.

This scaling helps neural networks train more efficiently and improves numerical stability.

### 3. Sliding Window Sequence Generation

A **60-day sliding window** was implemented.

The previous 60 trading days are used as the input sequence for predicting the following day's stock price.

```text
Previous 60 Days
       ↓
 ┌─────────────────────────────┐
 │ Day 1 ... Day 60            │
 │ AAPL MSFT GOOGL AMZN TSLA   │
 └─────────────────────────────┘
       ↓
 Next-Day Stock Price
