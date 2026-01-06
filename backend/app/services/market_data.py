import yfinance as yf
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
from app.core.config import settings

def fetch_historical_prices(tickers: List[str], years: int = 1) -> pd.DataFrame:
    """
    Fetches historical adjusted close prices for the given tickers.
    """
    if not tickers:
        return pd.DataFrame()

    start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
    
    # yfinance handles multiple tickers space-separated
    data = yf.download(tickers, start=start_date, progress=False)['Adj Close']
    
    # If only one ticker, it returns Series, we want DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
        data.columns = tickers
        
    return data

def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Get the latest price for a list of tickers.
    """
    if not tickers:
        return {}
    
    # Using Ticker.info or fast download for just the last day
    data = yf.download(tickers, period="1d", progress=False)['Adj Close']
    
    if isinstance(data, pd.Series):
        return {tickers[0]: data.iloc[-1]}
    
    prices = {}
    for ticker in tickers:
        try:
            # Check if ticker is in columns (yfinance structure varies by version)
            if ticker in data.columns:
                 prices[ticker] = data[ticker].iloc[-1]
        except Exception:
            prices[ticker] = 0.0
            
    return prices
