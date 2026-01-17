import pandas as pd
import yfinance as yf

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical market data for a given ticker.
    """
    df = yf.download(ticker, start=start, end=end)
    df = df.reset_index()  # Move 'Date' from index to column
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning of financial time-series data.
    """
    # Drop any completely empty rows (just in case)
    df = df.dropna()
    
    # Ensure correct column names (remove MultiIndex if present)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df
