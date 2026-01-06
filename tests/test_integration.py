import pytest
from app.analytics.risk_engine import calculate_portfolio_metrics
from app.services.news_service import search_financial_news

# We can mock yfinance if we don't want real network calls, 
# but the user asked to "Include local testing" implying they want to see it work.
# We'll do a simple integration test.

def test_risk_calculation_integration():
    """
    Test risk engine with real data (requires internet).
    """
    holdings = [
        {"symbol": "AAPL", "value": 1000},
        {"symbol": "MSFT", "value": 1000}
    ]
    
    metrics = calculate_portfolio_metrics(holdings)
    
    assert "annual_return" in metrics
    assert "sharpe_ratio" in metrics
    assert metrics["market_data_tickers"] == ["AAPL", "MSFT"]

def test_news_search_integration():
    """
    Test news fetcher (requires internet).
    """
    results = search_financial_news(["AAPL"], max_results=1)
    assert len(results) > 0
    assert "title" in results[0]
