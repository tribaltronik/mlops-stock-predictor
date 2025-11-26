#!/usr/bin/env python3
"""
Simple script to return the model accuracy for the stock predictor.
This script can be run directly to get the model accuracy percentage.
"""

import sys
import os

def get_model_accuracy_simple():
    """
    Return just the model accuracy percentage as a simple number.
    """
    try:
        # Import the test function from the same directory
        from test_predictor import get_model_accuracy
        
        accuracy = get_model_accuracy()
        if accuracy is not None:
            return f"{accuracy:.2f}%"
        else:
            return "Error: Unable to calculate accuracy"
            
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    result = get_model_accuracy_simple()
    print(result)