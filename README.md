# ML-Based Stock Price Predictor POC

A machine learning proof of concept that predicts stock prices for JEDI.DE using Random Forest regression with technical indicators.

## Features

- Fetches historical stock data using yfinance
- Creates technical indicators (Simple Moving Averages, Exponential Moving Averages)
- Trains Random Forest model to predict next day's closing price
- Comprehensive model evaluation with multiple metrics (MSE, RMSE, MAE, R², MAPE)
- MLflow integration for experiment tracking and model logging (src/main.py)
- Standalone testing functionality without MLflow dependencies (src/test_predictor.py)
- Simple accuracy reporting tool (src/get_accuracy.py)

## Requirements

- Python 3.x
- yfinance, pandas, numpy, scikit-learn
- mlflow (optional, for experiment tracking)

Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

Run the main predictor with MLflow tracking:
```bash
uv run src/main.py
```

Test the core functionality without MLflow dependencies:
```bash
uv run src/test_predictor.py
```

Get simple model accuracy percentage:
```bash
uv run src/get_accuracy.py
```

## Expected Output

The model will:
1. Fetch JEDI.DE stock data from 2020-2024
2. Create technical indicators (SMA_10, SMA_50, EMA_10, EMA_50)
3. Train a Random Forest model
4. Display comprehensive evaluation metrics:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R-squared Score
   - Model Accuracy (R² × 100)
   - Mean Absolute Percentage Error (MAPE)
   - Sample predictions vs actual values

**src/main.py**: Includes MLflow experiment tracking and model logging
**src/test_predictor.py**: Standalone testing with detailed metrics display
**src/get_accuracy.py**: Simple accuracy percentage output

Typical model accuracy ranges from 85-95% depending on market conditions.
