# Finance-Analysis

📈 Machine Learning–Based Trading Strategy with Walk-Forward Validation
📌 Project Overview

This project implements an end-to-end machine learning pipeline for financial time-series forecasting and trading strategy evaluation.
The goal is to predict daily stock returns and generate risk-aware trading signals, evaluated using walk-forward validation to avoid look-ahead bias.

The project focuses on robust evaluation, feature engineering, and realistic performance metrics, rather than purely predictive accuracy.

🧠 Key Concepts

Time-series feature engineering (returns, momentum, volatility)

Walk-forward (rolling window) training and testing

Threshold-based trading signals

Risk-adjusted performance evaluation

Overfitting diagnostics for financial ML

🛠️ Tech Stack

Python

pandas, numpy

scikit-learn

matplotlib

yfinance

joblib

📊 Data

Historical daily price data for AAPL

Source: Yahoo Finance (via yfinance)

Period: 2018–2024

🧩 Feature Engineering

The model uses engineered features commonly applied in quantitative finance:

Lagged returns

Rolling mean returns (momentum)

Rolling volatility

Multiple lookback windows (126, 252, 504 trading days)

🤖 Models

Linear Regression (baseline)

Random Forest Regressor

Gradient Boosting Regressor (final model)

🔁 Validation Strategy

Walk-forward validation with rolling windows

No random shuffling (time order preserved)

Models retrained at each step using only past data

📈 Evaluation Metrics

RMSE & MAE

Directional Accuracy

Maximum Drawdown

Sharpe Ratio

Cumulative strategy vs buy-and-hold returns

Permutation test for overfitting detection


🔍 Overfitting Analysis

Rolling directional accuracy (65–90%) reflects market regime changes

In-sample vs out-of-sample error comparison

Permutation test confirms predictive signal beyond randomness


⚠️ Limitations

Single-asset focus (AAPL)

Transaction costs and slippage not modeled

Not intended as financial advice

🚀 Future Work

Multi-asset extension

Volatility-adjusted position sizing

Application to other domains (e.g. healthcare time series)
