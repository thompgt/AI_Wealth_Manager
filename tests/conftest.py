import atexit
import os
import shutil
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Test isolation.
#
# These MUST be set before anything imports `config` or `db`, because db.py
# builds its SQLAlchemy engine at import time from settings.NEON_DATABASE_URL.
# conftest.py is imported before any test module, so this is the only place
# it can happen.
#
# Previously the suite ran against ./wealth_manager.db -- the developer's real
# database. Tests seeded it, wrote agent_runs and reports into it, and their
# behaviour depended on whatever state a previous manual run had left behind.
# A test suite that mutates the dev database is both unsafe and
# non-reproducible.
# ---------------------------------------------------------------------------
_TMP_DIR = tempfile.mkdtemp(prefix="wealth-manager-tests-")
atexit.register(shutil.rmtree, _TMP_DIR, True)

os.environ["NEON_DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
os.environ["CHECKPOINT_DB_PATH"] = os.path.join(_TMP_DIR, "checkpoints.sqlite")
os.environ["ENVIRONMENT"] = "development"
os.environ["GEMINI_API_KEY"] = "DUMMY_API_KEY"
# Keep the backoff loop from adding real wall-clock time to the suite.
os.environ["LLM_MAX_ATTEMPTS"] = "1"

import pandas as pd  # noqa: E402
import pytest  # noqa: E402


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
    """Patch the single place that now talks to yfinance for ticker metadata.

    All three agents used to call `yf.Ticker(...).info` themselves, so this
    fixture had to patch three module-level `yf` references. They now share
    services.market_data.get_ticker_info, so there is exactly one seam -- and
    its in-process cache has to be cleared around each test so a real lookup
    from another test can't leak in (or a fake one leak out).
    """
    import services.market_data as market_data

    market_data.clear_ticker_info_cache()
    monkeypatch.setattr(market_data.yf, "Ticker", FakeTicker)
    yield
    market_data.clear_ticker_info_cache()


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

    def fail_llm(*args, **kwargs):
        raise RuntimeError("LLM disabled in tests -- exercising the deterministic fallback path")

    def fake_search_financial_news(keywords, max_results=5):
        return []

    monkeypatch.setattr(diagnostics, "fetch_historical_prices", fake_fetch_historical_prices)
    monkeypatch.setattr(market_regime, "fetch_historical_prices", fake_fetch_historical_prices)
    monkeypatch.setattr(market_regime, "search_financial_news", fake_search_financial_news)
    monkeypatch.setattr(market_regime, "_invoke_llm", fail_llm)
    monkeypatch.setattr(stock_research, "_llm_rank", fail_llm)
    monkeypatch.setattr(finance_report, "_call_llm", fail_llm)
    monkeypatch.setattr(tax_awareness, "get_current_prices", fake_get_current_prices)
    monkeypatch.setattr(orchestrator, "get_current_prices", fake_get_current_prices)
