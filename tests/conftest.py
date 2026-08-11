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


def make_security_info(symbol, **overrides):
    """A reasonable, eligible SecurityInfo, so suitability's screens pass
    unless a test deliberately makes them fail."""
    from services.providers.base import SecurityInfo

    fields = {
        "symbol": symbol,
        "name": f"{symbol} Inc.",
        "sector": "Technology",
        "industry": "Software",
        "exchange": "NMS",
        "market_cap": 500_000_000_000.0,
        "beta": 1.0,
        "pe_ratio": 20.0,
        "pb_ratio": 5.0,
        "dividend_yield": 1.8,
        "avg_dollar_volume": 500_000_000.0,
        "quote_type": "EQUITY",
        "provider": "test",
    }
    fields.update(overrides)
    return SecurityInfo(**{k: v for k, v in fields.items() if k in SecurityInfo.__dataclass_fields__})


@pytest.fixture
def fake_security_info(monkeypatch):
    """Offline reference data.

    Market data moved behind a provider chain, so the old seam -- patching
    `yfinance.Ticker` on services.market_data -- no longer exists; there is no
    `yf` attribute on that module at all. The seam now is `get_security_info`,
    patched both where it is defined and where `agents.suitability` bound it by
    name at import time. Its in-process cache is cleared around each test so a
    real lookup cannot leak in (or a fake one leak out).
    """
    import agents.suitability as suitability
    import services.market_data as market_data

    market_data.clear_ticker_info_cache()

    def fake_get_security_info(ticker):
        return make_security_info(str(ticker).upper())

    monkeypatch.setattr(market_data, "get_security_info", fake_get_security_info)
    monkeypatch.setattr(suitability, "get_security_info", fake_get_security_info)
    yield
    market_data.clear_ticker_info_cache()


def make_quote(symbol, price=250.0):
    from services.providers.base import Quote

    return Quote(symbol=symbol, price=price, provider="test", as_of=_pd_now())


def _pd_now():
    import datetime as _dt

    return _dt.datetime.now().replace(microsecond=0)


@pytest.fixture
def patched_external_calls(monkeypatch, fake_security_info):
    """
    Full offline double for every network/LLM dependency the orchestrator's
    graph touches, so the E2E graph test is deterministic and doesn't require
    a live GEMINI_API_KEY or internet access.

    Each agent module imports these helpers by name at module load time
    (`from services.market_data import fetch_historical_prices`), so every
    import site must be patched individually -- patching services.market_data
    itself would not affect references already bound in agents/*.py.

    The LLM is disabled at `get_chat_model` rather than at each agent's own
    call wrapper. Those wrappers have been renamed more than once, and a
    monkeypatch naming a function that no longer exists fails loudly at setup
    -- but only after the rename, which is how this fixture came to reference
    three functions that had all been gone for a while.
    """
    import agents.diagnostics as diagnostics
    import agents.finance_report as finance_report
    import agents.market_regime as market_regime
    import agents.rebalance as rebalance_agent
    import agents.stock_research as stock_research
    from services.llm import LLMUnavailable
    from services.news_service import NewsResult

    def fake_fetch_historical_prices(tickers, years=1):
        return make_price_frame(tickers, days=130)

    def fake_get_quotes(tickers, **kwargs):
        return {t: make_quote(t) for t in tickers if str(t).upper() != "CASH"}

    def unavailable_llm(*args, **kwargs):
        raise LLMUnavailable(
            "LLM disabled in tests -- exercising the deterministic fallback path"
        )

    def fake_get_market_news(keywords=None, max_results=6):
        return NewsResult(degraded=True, reason="news disabled in tests")

    monkeypatch.setattr(diagnostics, "fetch_historical_prices", fake_fetch_historical_prices)
    monkeypatch.setattr(market_regime, "fetch_historical_prices", fake_fetch_historical_prices)
    monkeypatch.setattr(stock_research, "fetch_historical_prices", fake_fetch_historical_prices)
    monkeypatch.setattr(market_regime, "get_market_news", fake_get_market_news)
    monkeypatch.setattr(stock_research, "get_quotes", fake_get_quotes)
    monkeypatch.setattr(rebalance_agent, "get_quotes", fake_get_quotes)

    for module in (market_regime, stock_research, finance_report):
        monkeypatch.setattr(module, "get_chat_model", unavailable_llm)
