import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def evaluate_model(predictions, actual):
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    return rmse

def plot_predictions(actual, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, color='blue', label='Actual Price')
    plt.plot(predicted, color='red', label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()