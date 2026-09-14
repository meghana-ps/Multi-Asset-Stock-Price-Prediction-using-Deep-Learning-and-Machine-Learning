# 📈 Multi-Asset Stock Price Prediction using Deep Learning and Machine Learning

## 🚀 Project Overview

This project explores the application of Deep Learning and Machine Learning techniques for financial time-series forecasting, with the objective of predicting future stock prices from historical market data.

The study compares multiple predictive approaches, including Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), CNN-LSTM, Attention-LSTM, Random Forest, and XGBoost.

The models are trained on historical daily closing prices of five major technology companies and evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).

The primary objective is to understand how different model architectures capture temporal dependencies, market trends, volatility, and sequential patterns in financial data.

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

The historical stock price data was obtained from the Yahoo Finance API using the `yfinance` Python library.

### Stocks Analyzed

| Company | Ticker |
|---|---|
| Apple | AAPL |
| Microsoft | MSFT |
| Google | GOOGL |
| Amazon | AMZN |
| Tesla | TSLA |

The dataset covers approximately January 2015 to December 2023 and contains daily closing prices for the five selected technology companies.

Each observation represents a trading day, while each stock is represented as a separate feature. The date is used as the time-series index.

The selected companies provide a multi-asset financial time-series dataset suitable for comparing forecasting approaches across different stocks.

---

## 🔄 Data Preprocessing

Several preprocessing steps were performed before model training.

### 1. Missing Value Analysis

The dataset was examined for missing values across all stock price variables.

No missing values were identified in the dataset used for modeling.

### 2. Feature Scaling

Stock prices were normalized using `MinMaxScaler`, transforming the values into a common range between 0 and 1.

This scaling improves numerical stability and helps neural networks train more effectively.

### 3. Sliding Window Sequence Generation

A 60-day sliding window was used to transform the historical stock prices into sequential training samples.

The previous 60 trading days were used to predict the next day's closing price.

```text
Previous 60 Trading Days
            ↓
    Sequential Input
            ↓
       ML / DL Model
            ↓
   Next-Day Stock Price
---

---

## 4. Train-Test Split

The sequential dataset was divided into training and testing sets using an 80–20 split, where 80% of the data was used for model training and the remaining 20% was used for evaluation on unseen data.

The input features and target values were divided into `X_train`, `X_test`, `y_train`, and `y_test`.

The target variable represents the **next-day closing price** of the stock being predicted.

---

# 🧠 Machine Learning & Deep Learning Models

Six different predictive models were implemented and compared:

- LSTM
- GRU
- Random Forest
- XGBoost
- CNN-LSTM
- Attention-LSTM

The models were evaluated to determine how effectively different architectures could learn patterns and temporal dependencies from historical stock price data.

---

## 1. LSTM — Long Short-Term Memory

Long Short-Term Memory (LSTM) is a recurrent neural network architecture designed to learn long-term dependencies in sequential data.

LSTM was selected because stock prices are sequential and depend on historical patterns. The architecture can retain relevant information over multiple time steps while reducing the impact of irrelevant information.

### Model Architecture

The implemented LSTM model consists of:

- LSTM layer — 64 units
- Dropout — 20%
- LSTM layer — 32 units
- Dropout — 20%
- Dense output layer
- Adam optimizer
- Mean Squared Error (MSE) loss

The model was trained for 20 epochs with a batch size of 32.

### Performance

```text
RMSE: 17.7782
MAE:  14.6153
MAPE: 8.5265%
