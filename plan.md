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
1. Update code to support multiple models
2. Add model comparison functionality
3. Test each model on JEDI.DE data
4. Compare accuracy metrics across models
5. Keep implementation simple, no MLflow

## Success Criteria
- All models run without errors
- Clear comparison of model performances
- Reproducible results
- Simple codebase for easy experimentation