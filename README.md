# ML-Based Stock Price Predictor POC

A machine learning proof of concept that predicts stock prices for JEDI.DE using Random Forest regression with technical indicators.

## Features

- Fetches historical stock data using yfinance
- Creates technical indicators (Simple Moving Averages, Exponential Moving Averages)
- Trains Random Forest model to predict next day's closing price
- Evaluates model performance using Mean Squared Error
- Uses MLflow for experiment tracking (when available)

## Requirements

- Python 3.x
- yfinance, pandas, numpy, scikit-learn
- mlflow (optional, for experiment tracking)

Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

Run the main predictor:
```bash
uv run python src/main.py
```

Test the core functionality without MLflow:
```bash
uv run python test_predictor.py
```

## Expected Output

The model will:
1. Fetch JEDI.DE stock data from 2020-2024
2. Create technical indicators (SMA_10, SMA_50, EMA_10, EMA_50)
3. Train a Random Forest model
4. Display MSE and sample predictions vs actual values

Sample output shows predictions within ~1-2% accuracy of actual prices.
