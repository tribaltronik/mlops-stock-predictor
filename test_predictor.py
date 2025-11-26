import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

    print(f"Fetching data for {ticker} from {start_date} to {end_date}")
    
    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    if data.empty:
        print("No data fetched. Check ticker or dates.")
        return

    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Preprocess
    X, y = preprocess_data(data)
    print(f"Features shape after preprocessing: {X.shape}")
    print(f"Target shape after preprocessing: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train model
    print("Training Random Forest model...")
    model = train_model(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    # Show some predictions vs actual
    print("\nSample predictions vs actual:")
    for i in range(min(5, len(y_test))):
        print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_test.iloc[i]:.2f}")

    print("Stock prediction model completed successfully!")

if __name__ == "__main__":
    main()