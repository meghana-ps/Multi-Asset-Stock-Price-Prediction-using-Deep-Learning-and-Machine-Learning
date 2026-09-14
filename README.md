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

The dataset covers the period from January 2015 to December 2023 and contains daily closing prices for the five selected technology companies.

Each row represents a trading day, while each column corresponds to the closing price of a specific company. The date is used as the time-series index.

The selected companies provide a multi-asset financial time-series dataset suitable for comparing different forecasting approaches.

---

## 🔄 Data Preprocessing

Several preprocessing steps were performed before model training.

### 1. Missing Value Analysis

The dataset was examined for missing values across the stock price variables.

No missing values were identified in the dataset used for modeling.

### 2. Feature Scaling

Stock prices were normalized using `MinMaxScaler`, transforming the values into a common range between 0 and 1.

This scaling helps provide numerical stability and supports effective neural network training.

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
```

### 4. Train-Test Split

The sequential dataset was divided into training and testing sets using an 80–20 split.

80% of the data was used for model training.
20% of the data was used for evaluation on unseen data.

The input features and target values were divided into X_train, X_test, y_train, and y_test.

The target variable represents the next-day closing price of the stock being predicted.


## Machine Learning & Deep Learning Models

Six different predictive models were implemented and compared:

LSTM
GRU
Random Forest
XGBoost
CNN-LSTM
Attention-LSTM

The models were evaluated to determine how effectively different architectures could learn patterns and temporal dependencies from historical stock price data.

### 1. LSTM — Long Short-Term Memory

Long Short-Term Memory (LSTM) is a recurrent neural network architecture designed to learn long-term dependencies in sequential data.

LSTM was selected because stock prices are sequential and depend on historical patterns. The architecture can retain relevant information over multiple time steps while reducing the impact of irrelevant information.

Model Architecture

The implemented LSTM model consists of:

LSTM layer — 64 units
Dropout — 20%
LSTM layer — 32 units
Dropout — 20%
Dense output layer
Adam optimizer
Mean Squared Error (MSE) loss

The model was trained for 20 epochs with a batch size of 32.

Performance
RMSE: 17.7782
MAE:  14.6153
MAPE: 8.5265%

The LSTM model captured the general trends in stock prices but produced higher prediction errors compared with the GRU model.

### 2. GRU — Gated Recurrent Unit

Gated Recurrent Unit (GRU) is a recurrent neural network architecture designed to capture temporal dependencies in sequential data.

GRU provides a comparatively simpler recurrent architecture while retaining the ability to learn important patterns from historical observations.

Model Architecture

The implemented GRU model consists of:

GRU layer — 64 units
Dropout — 20%
GRU layer — 32 units
Dense output layer
Adam optimizer
Mean Squared Error (MSE) loss

The model was trained for 20 epochs with a batch size of 32.

Performance
RMSE: 9.9561
MAE:  7.2083
MAPE: 4.1377%

GRU achieved the best overall performance among all the models evaluated in this project.

The lower error values indicate that the GRU predictions were closer to the actual stock prices compared with the other tested models.

### 3. Random Forest

Random Forest was implemented as a traditional Machine Learning baseline for comparison with the Deep Learning models.

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions.

It can capture nonlinear relationships in data, but it does not inherently account for the sequential nature of time-series observations.

Model Configuration

The Random Forest model was implemented using:

Random Forest Regressor
100 decision trees (n_estimators=100)

The sequential input data was transformed into a suitable tabular representation before model training.

Performance
RMSE: 39.2629
MAE:  31.2417
MAPE: 17.5553%

Random Forest produced the highest prediction error among the evaluated models.

This indicates that the model was less effective at capturing the temporal dependencies present in the financial time-series data.

### 4. XGBoost

XGBoost was implemented as another traditional Machine Learning benchmark.

XGBoost is a gradient-boosting algorithm that combines multiple decision trees to model complex nonlinear relationships.

Model Configuration

The model was implemented using:

XGBoost Regressor
100 estimators

The sequential input data was transformed into a tabular representation before training.

Performance
RMSE: 18.7103
MAE:  13.8070
MAPE: 7.9972%

XGBoost performed substantially better than Random Forest.

However, its performance remained below the best-performing recurrent Deep Learning model, GRU.

### 5. CNN-LSTM

CNN-LSTM combines Convolutional Neural Networks (CNN) with Long Short-Term Memory (LSTM) networks.

The CNN component is used to identify local patterns within the time-series data, while the LSTM component captures temporal dependencies.

Model Architecture

The implemented CNN-LSTM architecture consists of:

Conv1D layer — 64 filters
Kernel size — 3
ReLU activation
MaxPooling1D
LSTM layer — 50 units
Dense output layer
Adam optimizer
Mean Squared Error (MSE) loss
Performance
RMSE: 15.6540
MAE:  12.2325
MAPE: 6.6736%

CNN-LSTM achieved the second-best RMSE among the evaluated models.

The combination of convolutional feature extraction and recurrent sequence modeling allowed the model to capture both local patterns and temporal dependencies.

### 6. Attention-LSTM

Attention-LSTM extends the traditional LSTM architecture by incorporating an attention mechanism.

The attention mechanism allows the model to assign different importance to different time steps in the input sequence.

This enables the model to focus on potentially more relevant historical observations when generating predictions.

Performance
RMSE: 29.8928
MAE:  24.1991
MAPE: 14.7664%

Despite the additional attention mechanism, the Attention-LSTM model performed worse than the simpler recurrent architectures.

This indicates that increased architectural complexity did not necessarily result in better forecasting performance for this dataset.
