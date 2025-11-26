# Model Accuracy Report

## Overview
The stock prediction model has been enhanced to provide comprehensive accuracy metrics for evaluating its performance.

## Model Details
- **Algorithm**: Random Forest Regressor
- **Target**: Next day's closing price prediction
- **Features**: Open, High, Low, Close, Volume, SMA_10, SMA_50, EMA_10, EMA_50
- **Data**: JEDI.DE stock data from 2020-01-01 to 2024-01-01

## Accuracy Metrics Implemented

### 1. R-squared Score (Primary Accuracy Metric)
- **Definition**: Coefficient of determination representing the proportion of variance explained by the model
- **Range**: -∞ to 1 (typically 0 to 1 for good models)
- **Interpretation**: Higher values indicate better model fit
- **Usage**: Converted to percentage for "accuracy" reporting

### 2. Mean Squared Error (MSE)
- **Definition**: Average of squared differences between predicted and actual values
- **Units**: Price² (euro² in this case)
- **Interpretation**: Lower values indicate better predictions

### 3. Root Mean Squared Error (RMSE)
- **Definition**: Square root of MSE, same units as target variable
- **Units**: Price (euro in this case)
- **Interpretation**: More interpretable error metric than MSE

### 4. Mean Absolute Error (MAE)
- **Definition**: Average of absolute differences between predicted and actual values
- **Units**: Price (euro in this case)
- **Interpretation**: Robust to outliers, easy to interpret

### 5. Mean Absolute Percentage Error (MAPE)
- **Definition**: Average of absolute percentage errors
- **Units**: Percentage (%)
- **Interpretation**: Scale-independent metric, easy to understand

## How to Get Model Accuracy

### Method 1: Simple Accuracy Call
```bash
python get_accuracy.py
```
Returns: Model accuracy as percentage (e.g., "85.23%")

### Method 2: Comprehensive Metrics
```bash
python test_predictor.py
```
Returns: All accuracy metrics with detailed output

### Method 3: Using the Functions
```python
from test_predictor import get_model_accuracy, return_model_accuracy

# Get just the accuracy percentage
accuracy = get_model_accuracy()
print(f"Model Accuracy: {accuracy}%")

# Get comprehensive metrics
metrics = return_model_accuracy()
print(metrics)
```

## Files Modified

1. **test_predictor.py**: Enhanced with comprehensive accuracy metrics
   - Added `r2_score` and `mean_absolute_error` imports
   - Added `evaluate_model_accuracy()` function
   - Added `return_model_accuracy()` function  
   - Added `get_model_accuracy()` simple function
   - Enhanced main output with all metrics

2. **src/main.py**: Updated with improved accuracy logging
   - Added comprehensive metrics calculation
   - Enhanced MLflow logging with all metrics
   - Improved console output

3. **get_accuracy.py**: New standalone script for simple accuracy retrieval

## Model Accuracy Interpretation

### For Regression Models:
- **R² > 0.8**: Excellent model performance
- **R² 0.6-0.8**: Good model performance  
- **R² 0.4-0.6**: Moderate model performance
- **R² < 0.4**: Poor model performance

### Current Implementation:
The model returns accuracy as R² × 100%, which provides an intuitive percentage representation of how well the model explains the variance in stock price movements.

## Next Steps for Model Improvement

1. **Hyperparameter Tuning**: Optimize Random Forest parameters
2. **Feature Engineering**: Add more technical indicators
3. **Cross Validation**: Implement time series cross-validation
4. **Ensemble Methods**: Combine multiple models
5. **Feature Selection**: Identify most important features
6. **Regular Retraining**: Update model with new data regularly