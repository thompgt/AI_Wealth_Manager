import pandas as pd
import numpy as np
from pypfopt import efficient_frontier, risk_models, expected_returns
from app.services.market_data import fetch_historical_prices
from app.core.config import settings
from typing import Dict, Any, List

def calculate_portfolio_metrics(holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates detailed risk metrics for a portfolio.
    
    Args:
        holdings: List of dicts with 'symbol' and 'value' (or quantity)
    """
    if not holdings:
        return {"error": "No holdings provided"}

    tickers = [h['symbol'] for h in holdings]
    weights_map = {}
    total_value = sum(h['value'] for h in holdings)
    
    if total_value == 0:
        return {"error": "Portfolio value is zero"}

    for h in holdings:
        weights_map[h['symbol']] = h['value'] / total_value

    # 1. Fetch Data
    try:
        prices_df = fetch_historical_prices(tickers, years=settings.LOOKBACK_PERIOD_YEARS)
    except Exception as e:
        return {"error": f"Failed to fetch market data: {str(e)}"}

    if prices_df.empty:
        return {"error": "No market data found for tickers"}

    # Clean data: drop columns with too many NaNs, fill remaining
    prices_df = prices_df.dropna(axis=1, how='all').fillna(method='ffill').dropna()
    
    available_tickers = prices_df.columns.tolist()
    # Filter weights to only include available tickers and re-normalize
    valid_weights = np.array([weights_map[t] for t in available_tickers])
    if valid_weights.sum() == 0:
         return {"error": "No valid data for these holdings"}
         
    valid_weights = valid_weights / valid_weights.sum()

    # 2. Risk Calculations using PyPortfolioOpt
    
    # Expected Returns (mu) and Sample Covariance (S)
    mu = expected_returns.mean_historical_return(prices_df)
    S = risk_models.sample_cov(prices_df)

    # Portfolio Performance
    # return, volatility, sharpe
    # Manually calc portfolio vol and return since we have specific weights
    
    # Expected Annual Return
    port_return = np.sum(mu * valid_weights)
    
    # Annual Volatility
    port_volatility = np.sqrt(np.dot(valid_weights.T, np.dot(S, valid_weights)))
    
    # Sharpe Ratio
    sharpe = (port_return - settings.RISK_FREE_RATE) / port_volatility

    # Diversification Score (HHI - Herfindahl-Hirschman Index)
    # Lower is better diversified. Range 1/N to 1.
    # We can inverse or map it to a 0-100 score.
    hhi = np.sum(valid_weights ** 2)
    diversification_score = 100 * (1 - hhi) # Rough approximation

    return {
        "annual_return": round(port_return, 4),
        "annual_volatility": round(port_volatility, 4),
        "sharpe_ratio": round(sharpe, 2),
        "diversification_score": round(diversification_score, 2),
        "market_data_tickers": available_tickers
    }
