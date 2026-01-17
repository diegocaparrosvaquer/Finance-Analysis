import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

FEATURES = [
    "RollingMean_5",
    "RollingMean_20",
    "Volatility_20"
]
TARGET = "Return"

def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    return train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
