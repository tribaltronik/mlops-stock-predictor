import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


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
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_10"] = data["Close"].ewm(span=10).mean()
    data["EMA_50"] = data["Close"].ewm(span=50).mean()

    # Target: next day's close price
    data["Target"] = data["Close"].shift(-1)

    # Drop NaN values
    data.dropna(inplace=True)

    # Features
    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA_10",
        "SMA_50",
        "EMA_10",
        "EMA_50",
    ]
    X = data[features]
    y = data["Target"]

    return X, y


def train_model(X_train, y_train, model_type="random_forest"):
    """
    Train a model based on type.
    """
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "svr":
        model = SVR(kernel="rbf")
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model_accuracy(model, X_test, y_test):
    """
    Calculate comprehensive accuracy metrics for the regression model.

    Returns:
        dict: Dictionary containing various accuracy metrics
    """
    y_pred = model.predict(X_test)

    # Calculate various metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Calculate R-squared as percentage (accuracy-like metric)
    accuracy_percentage = r2 * 100

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "accuracy_percentage": accuracy_percentage,
        "mape": mape,
        "predictions": y_pred,
        "actual": y_test,
    }


def return_model_accuracy():
    """
    Main function to return model accuracy metrics.
    """
    ticker = "JEDI.DE"
    start_date = "2020-01-01"
    end_date = "2024-01-01"

    print(f"Fetching data for {ticker} from {start_date} to {end_date}")

    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    if data.empty:
        print("No data fetched. Check ticker or dates.")
        return None

    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Preprocess
    X, y = preprocess_data(data)
    print(f"Features shape after preprocessing: {X.shape}")
    print(f"Target shape after preprocessing: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train model
    print("Training Random Forest model...")
    model = train_model(X_train, y_train)

    # Evaluate model accuracy
    accuracy_metrics = evaluate_model_accuracy(model, X_test, y_test)

    print("\n" + "=" * 50)
    print("MODEL ACCURACY METRICS")
    print("=" * 50)
    print(f"Mean Squared Error (MSE): {accuracy_metrics['mse']:.6f}")
    print(f"Root Mean Squared Error (RMSE): {accuracy_metrics['rmse']:.6f}")
    print(f"Mean Absolute Error (MAE): {accuracy_metrics['mae']:.6f}")
    print(f"R-squared Score: {accuracy_metrics['r2_score']:.6f}")
    print(f"Model Accuracy (R² × 100): {accuracy_metrics['accuracy_percentage']:.2f}%")
    print(f"Mean Absolute Percentage Error (MAPE): {accuracy_metrics['mape']:.2f}%")

    # Show some predictions vs actual
    print("\nSample predictions vs actual:")
    for i in range(min(5, len(y_test))):
        print(
            f"Predicted: {accuracy_metrics['predictions'][i]:.2f}, Actual: {accuracy_metrics['actual'].iloc[i]:.2f}"
        )

    print("Stock prediction model accuracy evaluation completed!")
    return accuracy_metrics


def get_model_accuracy():
    """
    Simple function to return just the model accuracy percentage.
    This is the main function to call for getting model accuracy.
    """
    ticker = "JEDI.DE"
    start_date = "2020-01-01"
    end_date = "2024-01-01"

    try:
        # Fetch data
        data = fetch_stock_data(ticker, start_date, end_date)
        if data.empty:
            return None

        # Preprocess
        X, y = preprocess_data(data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = train_model(X_train, y_train)

        # Get accuracy
        accuracy_metrics = evaluate_model_accuracy(model, X_test, y_test)

        return accuracy_metrics["accuracy_percentage"]

    except Exception as e:
        print(f"Error calculating model accuracy: {e}")
        return None


def compare_models():
    """
    Compare multiple models on the same data.
    """
    ticker = "JEDI.DE"
    start_date = "2020-01-01"
    end_date = "2024-01-01"

    models = ["random_forest", "linear_regression", "svr", "gradient_boosting"]
    results = {}

    print(f"Fetching data for {ticker} from {start_date} to {end_date}")

    # Fetch data once
    data = fetch_stock_data(ticker, start_date, end_date)
    if data.empty:
        print("No data fetched.")
        return None

    # Preprocess once
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Data shape: {data.shape}")
    print(f"Training set: {X_train.shape[0]} samples")
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for model_type in models:
        print(f"\nTesting {model_type.replace('_', ' ').title()}...")
        try:
            model = train_model(X_train, y_train, model_type)
            accuracy_metrics = evaluate_model_accuracy(model, X_test, y_test)
            results[model_type] = accuracy_metrics
            print(f"  Accuracy: {accuracy_metrics['accuracy_percentage']:.2f}%")
            print(f"  R² Score: {accuracy_metrics['r2_score']:.4f}")
            print(f"  RMSE: {accuracy_metrics['rmse']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model, metrics in results.items():
        print(
            f"{model.replace('_', ' ').title()}: {metrics['accuracy_percentage']:.2f}%"
        )

    return results


def main():
    # Run comparison
    results = compare_models()
    if results:
        print("\nModel comparison completed!")
    else:
        print("Unable to perform model comparison")


if __name__ == "__main__":
    main()
