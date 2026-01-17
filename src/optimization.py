import numpy as np


def simple_portfolio_optimization(returns):
    """
    Simple mean-variance inspired optimization.
    """
    mean_return = np.mean(returns)
    risk = np.std(returns)
    score = mean_return / risk if risk != 0 else 0
    return score
