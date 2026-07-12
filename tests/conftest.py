import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import pytest


def make_price_frame(tickers, days=130, start_price=100.0, daily_drift=0.001):
    """A small deterministic synthetic price history, shaped like what
    services.market_data.fetch_historical_prices returns (a DataFrame
    indexed by date, one column per ticker)."""
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    data = {}
    for i, ticker in enumerate(tickers):
        # Give each ticker a slightly different deterministic trend so
        # ratio-based signals (rising/falling) have something to compute.
        drift = daily_drift * (1 + 0.1 * i)
        prices = [start_price * (1 + drift) ** day for day in range(days)]
        data[ticker] = prices
    return pd.DataFrame(data, index=index)


class FakeTicker:
    """Stand-in for yfinance.Ticker -- returns a fixed, reasonable .info dict
    so suitability/diagnostics/stock_research pure-logic paths can run
    without any network access."""

    DEFAULT_INFO = {
        "sector": "Technology",
        "marketCap": 500_000_000_000,
        "exchange": "NMS",
        "fullExchangeName": "Nasdaq Global Select",
        "beta": 1.0,
        "trailingPE": 20.0,
        "priceToBook": 5.0,
        "dividendYield": 1.8,
    }

    def __init__(self, symbol):
        self.symbol = symbol

    @property
    def info(self):
        return dict(self.DEFAULT_INFO)


@pytest.fixture
def fake_yfinance(monkeypatch):
    """Patches yf.Ticker in every agent module that calls it directly."""
    import agents.diagnostics as diagnostics
    import agents.stock_research as stock_research
    import agents.suitability as suitability

    monkeypatch.setattr(diagnostics.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(stock_research.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(suitability.yf, "Ticker", FakeTicker)


@pytest.fixture
def patched_external_calls(monkeypatch, fake_yfinance):
    """
    Full offline double for every network/LLM dependency the orchestrator's
    graph touches, so the E2E graph test is deterministic and doesn't require
    a live GEMINI_API_KEY or internet access. Each agent module imports these
    helpers by name at module load time (`from services.market_data import
    fetch_historical_prices`), so every import site must be patched
    individually -- patching services.market_data itself would not affect
    already-bound references in agents/*.py.
    """
    import agents.diagnostics as diagnostics
    import agents.finance_report as finance_report
    import agents.market_regime as market_regime
    import agents.stock_research as stock_research
    import agents.tax_awareness as tax_awareness
    import orchestrator

    def fake_fetch_historical_prices(tickers, years=1):
        return make_price_frame(tickers, days=130)

    def fake_get_current_prices(tickers):
        return {t: 250.0 for t in tickers}

    def fake_search_financial_news(keywords, max_results=5):
        return []

    def fail_llm(*args, **kwargs):
        raise RuntimeError("LLM disabled in tests -- exercising the deterministic fallback path")

    monkeypatch.setattr(diagnostics, "fetch_historical_prices", fake_fetch_historical_prices)
    monkeypatch.setattr(market_regime, "fetch_historical_prices", fake_fetch_historical_prices)
    monkeypatch.setattr(market_regime, "search_financial_news", fake_search_financial_news)
    monkeypatch.setattr(market_regime, "_invoke_llm", fail_llm)
    monkeypatch.setattr(stock_research, "_llm_rank", fail_llm)
    monkeypatch.setattr(finance_report, "_call_llm", fail_llm)
    monkeypatch.setattr(tax_awareness, "get_current_prices", fake_get_current_prices)
    monkeypatch.setattr(orchestrator, "get_current_prices", fake_get_current_prices)
