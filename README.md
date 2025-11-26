# ML-Based Stock Price Predictor POC

A machine learning proof of concept that compares multiple models for predicting JEDI.DE ETF prices using technical indicators.

## Features

- Fetches historical stock data using yfinance
- Creates technical indicators (Simple Moving Averages, Exponential Moving Averages)
- Compares multiple models: Random Forest, Linear Regression, SVR, Gradient Boosting
- Comprehensive model evaluation with multiple metrics (MSE, RMSE, MAE, R², MAPE)
- Standalone testing functionality (src/test_predictor.py)
- Simple accuracy reporting tool (src/get_accuracy.py)

## Requirements

- Python 3.x
- yfinance, pandas, numpy, scikit-learn

Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

Compare multiple models:
```bash
uv run src/test_predictor.py
```

Get simple model accuracy percentage (Random Forest):
```bash
uv run src/get_accuracy.py
```

## Expected Output

The comparison will:
1. Fetch JEDI.DE ETF data from 2020-2024
2. Create technical indicators (SMA_10, SMA_50, EMA_10, EMA_50)
3. Train and evaluate multiple models
4. Display comparison of accuracy metrics for each model

**src/test_predictor.py**: Model comparison with detailed metrics
**src/get_accuracy.py**: Simple accuracy percentage for Random Forest

Typical accuracies: Random Forest/Linear Regression/Gradient Boosting ~93%, SVR ~14%
