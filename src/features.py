import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create financial features for ML.
    Automatically uses 'Adj Close' if present, otherwise 'Close'.
    """
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

    # Returns
    df['Return'] = df[price_col].pct_change()

    # Rolling features
    df['RollingMean_5'] = df[price_col].rolling(5).mean()
    df['RollingMean_20'] = df[price_col].rolling(20).mean()
    df['Volatility_20'] = df['Return'].rolling(20).std()

    df = df.dropna()
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/aapl.csv")
    df = create_features(df)
    df.to_csv("data/raw/aapl_features.csv", index=False)
    print("Feature dataset saved.")
