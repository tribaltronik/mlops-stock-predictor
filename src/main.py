import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    """
    Preprocess data: create features and target.
    """
    # Create features: moving averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_10'] = data['Close'].ewm(span=10).mean()
    data['EMA_50'] = data['Close'].ewm(span=50).mean()

    # Target: next day's close price
    data['Target'] = data['Close'].shift(-1)

    # Drop NaN values
    data.dropna(inplace=True)

    # Features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50']
    X = data[features]
    y = data['Target']

    return X, y

def train_model(X_train, y_train):
    """
    Train a Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    ticker = 'JEDI.DE'
    start_date = '2020-01-01'
    end_date = '2024-01-01'

    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    if data.empty:
        print("No data fetched. Check ticker or dates.")
        return

    # Preprocess
    X, y = preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy_percentage = r2 * 100
    
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"R-squared Score: {r2:.6f}")
    print(f"Model Accuracy: {accuracy_percentage:.2f}%")

    # MLflow logging
    mlflow.set_experiment("Stock Prediction")
    with mlflow.start_run():
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("accuracy_percentage", accuracy_percentage)
        mlflow.sklearn.log_model(model, "model")

    print("Model saved with MLflow.")

if __name__ == "__main__":
    main()