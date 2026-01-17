import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def max_drawdown(returns):
    returns = pd.Series(returns)  # convert to Pandas
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def directional_accuracy(y_true, y_pred):
    """
    Percentage of times the model correctly predicts
    the direction of returns (up/down).
    """
    correct = np.sign(y_true) == np.sign(y_pred)
    return correct.mean()