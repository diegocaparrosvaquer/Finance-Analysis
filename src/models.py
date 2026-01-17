import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def train_models(df: pd.DataFrame):
    features = [
        "RollingMean_5",
        "RollingMean_20",
        "Volatility_20"
    ]
    target = "Return"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return lr, rf, X_test, y_test


if __name__ == "__main__":
    df = pd.read_csv("data/raw/aapl_features.csv")
    lr, rf, X_test, y_test = train_models(df)
    print("Models trained successfully.")
