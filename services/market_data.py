"""Market data access: historical prices, current prices, and ticker metadata.

Everything that touches yfinance goes through this module so caching,
failure handling, and logging are consistent.

Three production bugs previously lived here and are fixed below:

1.  `get_current_prices` cached a price of 0.0 whenever a fetch failed. That
    0.0 was then served from cache for the next 24 hours, so a single
    transient yfinance hiccup silently valued a client's entire holding at
    $0 -- which cascades into wrong concentration percentages, wrong flaws,
    and wrong position sizing, with no error anywhere. Failed lookups are now
    never cached and are reported to the caller.
2.  `fetch_historical_prices` wrote the cache with one SELECT + one INSERT
    per (ticker, day). The macro basket alone is 12 tickers x ~126 trading
    days, i.e. ~1,500 sequential round trips per run. It now reads existing
    keys in one query and bulk-inserts the remainder.
3.  yfinance `Ticker.info` was fetched independently by three agents, and
    re-fetched from scratch on every research retry -- up to ~120 HTTP
    round trips per run for the same ~30 symbols. `get_ticker_info` adds a
    process-level TTL cache shared by all callers.
"""

from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from config import settings
from db import MarketDataCache, SessionLocal
from logging_setup import get_logger

logger = get_logger(__name__)


def _utcnow() -> datetime:
    """Naive UTC 'now'.

    The DB columns are naive (SQLAlchemy DateTime without timezone) and the
    seed data is naive UTC, so comparisons must stay naive. `datetime.utcnow()`
    is deprecated in 3.12+, hence going through an aware object and dropping
    the tzinfo explicitly rather than reintroducing the deprecated call.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _cache_is_fresh(fetched_at: datetime) -> bool:
    if fetched_at is None:
        return False
    return _utcnow() - fetched_at < timedelta(hours=settings.MARKET_DATA_CACHE_TTL_HOURS)


def _get_cached_current_price(db: Session, ticker: str) -> Optional[float]:
    row = (
        db.query(MarketDataCache)
        .filter(MarketDataCache.ticker == ticker)
        .order_by(MarketDataCache.as_of_date.desc())
        .first()
    )
    # A cached 0.0 is meaningless as a price and only ever arises from a bad
    # write; treat it as a miss so a stale bad row self-heals.
    if row and row.close_price and row.close_price > 0 and _cache_is_fresh(row.fetched_at):
        return row.close_price
    return None


def _bulk_upsert_cache(db: Session, rows: List[Tuple[str, datetime, float]]) -> None:
    """Insert/update many (ticker, as_of_date, price) rows with two queries
    instead of two per row."""
    if not rows:
        return

    now = _utcnow()
    tickers = {t for t, _, _ in rows}
    dates = [d for _, d, _ in rows]

    existing = (
        db.query(MarketDataCache)
        .filter(
            MarketDataCache.ticker.in_(tickers),
            MarketDataCache.as_of_date >= min(dates),
            MarketDataCache.as_of_date <= max(dates),
        )
        .all()
    )
    by_key = {(row.ticker, row.as_of_date): row for row in existing}

    new_rows = []
    for ticker, as_of_date, price in rows:
        found = by_key.get((ticker, as_of_date))
        if found is not None:
            found.close_price = price
            found.fetched_at = now
        else:
            new_rows.append(
                MarketDataCache(
                    ticker=ticker, as_of_date=as_of_date, close_price=price, fetched_at=now
                )
            )
    if new_rows:
        db.bulk_save_objects(new_rows)


def fetch_historical_prices(tickers: List[str], years: float = 1) -> pd.DataFrame:
    """
    Fetches historical adjusted close prices for the given tickers.
    Every fetched close is written through to market_data_cache.
    Returns an empty DataFrame (never raises) if the download fails.
    """
    if not tickers:
        return pd.DataFrame()

    start_date = (datetime.now() - timedelta(days=int(years * 365))).strftime("%Y-%m-%d")

    # yfinance handles multiple tickers space-separated.
    # auto_adjust=False is required to get an "Adj Close" column at all --
    # newer yfinance defaults to auto_adjust=True, which returns a
    # pre-adjusted "Close" column and drops "Adj Close" entirely.
    try:
        raw = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)
        data = raw["Adj Close"]
    except Exception as exc:  # noqa: BLE001 -- data source failures are expected
        logger.warning("Historical price download failed for %s: %s", tickers, exc)
        return pd.DataFrame()

    # If only one ticker, it returns a Series; we want a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
        data.columns = tickers

    if data.empty:
        logger.warning("Historical price download returned no rows for %s", tickers)
        return data

    rows: List[Tuple[str, datetime, float]] = []
    for ticker in data.columns:
        for as_of_date, price in data[ticker].dropna().items():
            if pd.isna(price) or float(price) <= 0:
                continue
            rows.append((str(ticker), pd.Timestamp(as_of_date).to_pydatetime(), float(price)))

    db = SessionLocal()
    try:
        _bulk_upsert_cache(db, rows)
        db.commit()
    except Exception as exc:  # noqa: BLE001 -- caching is best-effort
        db.rollback()
        logger.warning("Failed to write %d price rows to cache: %s", len(rows), exc)
    finally:
        db.close()

    return data


def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Get the latest price for a list of tickers, serving from
    market_data_cache when a fresh-enough (within
    settings.MARKET_DATA_CACHE_TTL_HOURS) entry already exists.

    Tickers whose price could not be determined are OMITTED from the result
    rather than returned as 0.0, so callers can distinguish "this is worth
    nothing" from "we don't know what this is worth". Failures are never
    written to the cache.
    """
    if not tickers:
        return {}

    db = SessionLocal()
    try:
        prices: Dict[str, float] = {}
        missing: List[str] = []
        for ticker in tickers:
            cached = _get_cached_current_price(db, ticker)
            if cached is not None:
                prices[ticker] = cached
            else:
                missing.append(ticker)

        if not missing:
            return prices

        try:
            raw = yf.download(missing, period="5d", progress=False, auto_adjust=False)
            data = raw["Adj Close"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Current price download failed for %s: %s", missing, exc)
            return prices

        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = missing

        today = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        to_cache: List[Tuple[str, datetime, float]] = []
        unresolved: List[str] = []
        for ticker in missing:
            price = None
            if ticker in data.columns:
                series = data[ticker].dropna()
                if not series.empty:
                    candidate = float(series.iloc[-1])
                    if candidate > 0:
                        price = candidate
            if price is None:
                unresolved.append(ticker)
                continue
            prices[ticker] = price
            to_cache.append((ticker, today, price))

        if unresolved:
            logger.warning(
                "No usable current price for %s -- omitted from the result and NOT cached, "
                "so downstream valuation treats them as unknown rather than $0.",
                unresolved,
            )

        try:
            _bulk_upsert_cache(db, to_cache)
            db.commit()
        except Exception as exc:  # noqa: BLE001 -- caching is best-effort
            db.rollback()
            logger.warning("Failed to cache current prices: %s", exc)

        return prices
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Ticker metadata (`Ticker.info`) with a process-level TTL cache.
# ---------------------------------------------------------------------------

_info_cache: Dict[str, Tuple[datetime, Optional[Dict[str, Any]]]] = {}
_info_lock = Lock()


def get_ticker_info(ticker: str) -> Optional[Dict[str, Any]]:
    """Best-effort fetch of yfinance's `.info` dict, cached in-process.

    Returns None on any failure (network error, delisted ticker, empty or
    unexpected response) so callers can treat "couldn't verify" as its own
    explicit case instead of crashing. Negative results are cached too, for a
    shorter window, so a delisted symbol doesn't get retried on every
    candidate loop.
    """
    ttl = timedelta(minutes=settings.TICKER_INFO_CACHE_TTL_MINUTES)
    negative_ttl = min(ttl, timedelta(minutes=5))
    now = _utcnow()

    with _info_lock:
        entry = _info_cache.get(ticker)
        if entry is not None:
            cached_at, value = entry
            age_limit = ttl if value is not None else negative_ttl
            if now - cached_at < age_limit:
                return value

    try:
        info = yf.Ticker(ticker).info
        if not info or not isinstance(info, dict):
            info = None
    except Exception as exc:  # noqa: BLE001 -- data source failures are expected
        logger.debug("yfinance .info fetch failed for %s: %s", ticker, exc)
        info = None

    with _info_lock:
        _info_cache[ticker] = (now, info)
    return info


def clear_ticker_info_cache() -> None:
    """Used by tests and by the demo notebook to guarantee a cold start."""
    with _info_lock:
        _info_cache.clear()
