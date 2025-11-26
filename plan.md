# Stock Predictor POC Plan

## Goal
Test multiple machine learning models for stock price prediction using JEDI.DE ETF data.

## Current Implementation
- Fetches historical data using yfinance
- Creates technical indicators (SMA, EMA)
- Trains models to predict next day's closing price
- Evaluates with multiple metrics (MSE, RMSE, MAE, R², MAPE)

## Models to Test
- Random Forest (current)
- Linear Regression
- Support Vector Regression (SVR)
- Gradient Boosting (XGBoost/LightGBM)

## Simple Implementation Steps
- [x] Update code to support multiple models
- [x] Add model comparison functionality
- [x] Test each model on JEDI.DE data
- [x] Compare accuracy metrics across models
- [x] Keep implementation simple, no MLflow

## New Requirements
- [ ] Add hyperparameter tuning for best performing models
- [ ] Implement cross-validation for robust evaluation
- [ ] Add more technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Create backtesting functionality for historical performance
- [ ] Add model persistence (save/load trained models)
- [ ] Implement prediction for future dates
- [ ] Add visualization of predictions vs actual prices
- [ ] Create command-line interface for different tickers

## Success Criteria
- All models run without errors
- Clear comparison of model performances
- Reproducible results
- Simple codebase for easy experimentation