"""Market data facade: quotes, history and security reference data.

Everything in the system that needs a price goes through here. Agents never
touch a provider directly, which is what makes the whole suite runnable with
no network and no API keys — and what made a yfinance field rename a
one-adapter change rather than three silently-broken agents.

Three rules this module enforces on behalf of every caller:

1. **A failed lookup is never cached and never returned as zero.** The
   original bug was cheap to write and expensive to have: a transient fetch
   failure cached 0.0, that 0.0 was served for the next 24 hours, and a
   client's entire holding was valued at nothing. Concentration percentages,
   flaws and position sizes all silently followed. Unresolvable symbols are
   omitted from the result and reported.

2. **Staleness is data, not a footnote.** Every quote carries its age and its
   source. A run that sized positions off a two-day-old close records that,
   the report says so, and `/health` can see it.

3. **Batch, don't loop.** A screen touches hundreds of symbols. Writing the
   price cache row-by-row cost ~1,500 sequential round trips per run; reading
   `.info` per agent cost ~120 HTTP calls for the same 30 symbols. Both are
   now one query and one shared cache respectively.
"""

from collections import OrderedDict
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from config import settings
from db import MarketDataCache, SessionLocal, engine, utcnow
from logging_setup import get_logger
from metrics import market_data_requests, stale_quotes
from services.providers import AllProvidersFailed, Quote, SecurityInfo, call_with_failover

logger = get_logger(__name__)


def _cache_is_fresh(fetched_at: Optional[datetime]) -> bool:
    if fetched_at is None:
        return False
    return utcnow() - fetched_at < timedelta(hours=settings.MARKET_DATA_CACHE_TTL_HOURS)


# --- Price cache -------------------------------------------------------------


def _bulk_upsert_cache(db: Session, rows: Sequence[Tuple[str, datetime, float, str]]) -> None:
    """Write many (ticker, date, price, provider) rows in one statement.

    On Postgres this is a real ON CONFLICT upsert. On SQLite it is a read of
    the affected key range followed by one bulk insert — which is safe now
    that (ticker, as_of_date) is unique, and was not before: the previous
    read-then-write with no constraint let two concurrent runs duplicate bars,
    silently double-weighting a day in every return series computed from them.
    """
    if not rows:
        return

    now = utcnow()
    # Collapse duplicates within the batch first; a single statement cannot
    # update the same key twice.
    deduped: Dict[Tuple[str, datetime], Tuple[float, str]] = {}
    for ticker, as_of, price, provider in rows:
        deduped[(ticker, as_of)] = (price, provider)

    payload = [
        {
            "ticker": ticker,
            "as_of_date": as_of,
            "close_price": price,
            "provider": provider,
            "fetched_at": now,
        }
        for (ticker, as_of), (price, provider) in deduped.items()
    ]

    if engine.dialect.name == "postgresql":
        statement = pg_insert(MarketDataCache).values(payload)
        db.execute(
            statement.on_conflict_do_update(
                index_elements=["ticker", "as_of_date"],
                set_={
                    "close_price": statement.excluded.close_price,
                    "provider": statement.excluded.provider,
                    "fetched_at": statement.excluded.fetched_at,
                },
            )
        )
        return

    tickers = {t for t, _ in deduped}
    dates = [d for _, d in deduped]
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

    fresh = []
    for (ticker, as_of), (price, provider) in deduped.items():
        found = by_key.get((ticker, as_of))
        if found is not None:
            found.close_price = price
            found.provider = provider
            found.fetched_at = now
        else:
            fresh.append(
                MarketDataCache(
                    ticker=ticker,
                    as_of_date=as_of,
                    close_price=price,
                    provider=provider,
                    fetched_at=now,
                )
            )
    if fresh:
        db.bulk_save_objects(fresh)


def _cached_quotes(db: Session, tickers: Sequence[str]) -> Dict[str, Quote]:
    """Most recent cached bar per ticker, in one query rather than one each."""
    if not tickers:
        return {}
    rows = (
        db.query(MarketDataCache)
        .filter(MarketDataCache.ticker.in_(list(tickers)))
        .order_by(MarketDataCache.ticker, MarketDataCache.as_of_date.desc())
        .all()
    )
    best: Dict[str, Quote] = {}
    for row in rows:
        if row.ticker in best:
            continue  # ordered descending, so the first is the newest
        price = float(row.close_price or 0)
        # A cached 0.0 is meaningless as a price and only ever arises from a
        # bad write; treat it as a miss so a stale bad row self-heals.
        if price <= 0 or not _cache_is_fresh(row.fetched_at):
            continue
        best[row.ticker] = Quote(
            symbol=row.ticker,
            price=price,
            as_of=row.as_of_date,
            provider=row.provider or "cache",
            stale=True,
        )
    return best


# --- Public API --------------------------------------------------------------


def get_quotes(tickers: Sequence[str], *, allow_cache: bool = True) -> Dict[str, Quote]:
    """Latest price per ticker, with provenance and staleness.

    Unresolvable tickers are omitted. Callers must treat a missing key as
    "unknown value" and decide explicitly what to do — `load_client_state`
    drops the holding from the run and says so, rather than valuing it at $0.
    """
    tickers = [t.upper().strip() for t in tickers if t and t.upper() != "CASH"]
    if not tickers:
        return {}

    db = SessionLocal()
    try:
        resolved: Dict[str, Quote] = {}
        if allow_cache:
            resolved = _cached_quotes(db, tickers)
            for _ in resolved:
                market_data_requests.labels("cache", "quotes", "cache_hit").inc()

        missing = [t for t in tickers if t not in resolved]
        if missing:
            try:
                live = call_with_failover(
                    "quotes",
                    lambda provider: provider.get_quotes(missing),
                    accept=lambda result: bool(result),
                )
            except AllProvidersFailed as exc:
                logger.warning(
                    "No provider could quote %s (%s). These symbols are omitted from the "
                    "result, so downstream valuation treats them as unknown rather than $0.",
                    missing, exc,
                )
                live = {}

            resolved.update(live)

            today = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            to_cache = [
                (quote.symbol, today, quote.price, quote.provider)
                for quote in live.values()
            ]
            try:
                _bulk_upsert_cache(db, to_cache)
                db.commit()
            except Exception as exc:  # noqa: BLE001 -- caching is best-effort
                db.rollback()
                logger.warning("Failed to cache %d quotes: %s", len(to_cache), exc)

        unresolved = [t for t in tickers if t not in resolved]
        if unresolved:
            logger.warning(
                "No usable price for %s -- omitted from the result and NOT cached.", unresolved
            )

        for quote in resolved.values():
            if quote.is_stale():
                stale_quotes.inc()

        return resolved
    finally:
        db.close()


def get_current_prices(tickers: Sequence[str]) -> Dict[str, float]:
    """Plain symbol -> price mapping for callers that need nothing else."""
    return {symbol: quote.price for symbol, quote in get_quotes(tickers).items()}


def quote_provenance(quotes: Dict[str, Quote]) -> Dict[str, object]:
    """Summarize where a run's prices came from, for the audit record.

    Persisted onto `agent_runs.data_provenance`, which is what makes
    "recompute this recommendation from the data it actually saw" a possible
    request rather than an aspiration.
    """
    if not quotes:
        return {"symbols": 0, "providers": [], "oldest_trading_days": None, "stale_symbols": []}
    ages = {symbol: quote.age_trading_days() for symbol, quote in quotes.items()}
    return {
        "symbols": len(quotes),
        "providers": sorted({quote.provider for quote in quotes.values()}),
        "oldest_trading_days": max(ages.values()),
        "stale_symbols": sorted(s for s, quote in quotes.items() if quote.is_stale()),
        # The oldest print in the set, not the newest: a report claiming data
        # "as of" its freshest input overstates how current the analysis is.
        "as_of": min(quote.as_of for quote in quotes.values()).isoformat(),
    }


def fetch_historical_prices(tickers: Sequence[str], years: float = 1) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker.

    Returns an empty DataFrame rather than raising: a missing history degrades
    the analytics that need it (volatility, correlation, drawdown) while
    leaving the rest of the run usable, and every consumer already handles the
    empty case. The failure is logged and recorded as degradation.
    """
    tickers = [t.upper().strip() for t in tickers if t and t.upper() != "CASH"]
    if not tickers:
        return pd.DataFrame()

    try:
        data = call_with_failover(
            "history",
            lambda provider: provider.get_history(list(tickers), years),
            accept=lambda frame: frame is not None and not frame.empty,
        )
    except AllProvidersFailed as exc:
        logger.warning("No provider returned history for %s: %s", tickers, exc)
        return pd.DataFrame()

    rows: List[Tuple[str, datetime, float, str]] = []
    for ticker in data.columns:
        for as_of, price in data[ticker].dropna().items():
            if pd.isna(price) or float(price) <= 0:
                continue
            rows.append(
                (str(ticker), pd.Timestamp(as_of).to_pydatetime(), float(price), "history")
            )

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


# --- Security reference data -------------------------------------------------


class _BoundedTTLCache:
    """LRU + TTL cache for security metadata.

    Bounded because a worker screening a large universe over days would
    otherwise grow an unbounded dict — the previous cache had no eviction at
    all. Negative results are cached for a shorter window so a delisted symbol
    is not retried on every pass of every screen, but also does not stay
    poisoned for an hour after a transient failure.
    """

    def __init__(self, max_entries: int):
        self._data: "OrderedDict[str, Tuple[datetime, Optional[SecurityInfo]]]" = OrderedDict()
        self._max = max_entries
        self._lock = Lock()

    def get(self, key: str, ttl: timedelta, negative_ttl: timedelta):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False, None
            cached_at, value = entry
            limit = ttl if value is not None else negative_ttl
            if utcnow() - cached_at >= limit:
                del self._data[key]
                return False, None
            self._data.move_to_end(key)
            return True, value

    def put(self, key: str, value: Optional[SecurityInfo]) -> None:
        with self._lock:
            self._data[key] = (utcnow(), value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_info_cache = _BoundedTTLCache(settings.TICKER_INFO_CACHE_MAX_ENTRIES)


def get_security_info(ticker: str) -> Optional[SecurityInfo]:
    """Normalized reference data and fundamentals, cached in-process.

    Returns None when no provider could describe the symbol, so callers can
    treat "couldn't verify" as its own case. Suitability fails closed on it,
    which is the right direction: an unverifiable security is not one to
    recommend.
    """
    ticker = ticker.upper().strip()
    ttl = timedelta(minutes=settings.TICKER_INFO_CACHE_TTL_MINUTES)
    negative_ttl = min(ttl, timedelta(minutes=5))

    hit, cached = _info_cache.get(ticker, ttl, negative_ttl)
    if hit:
        market_data_requests.labels("cache", "info", "cache_hit").inc()
        return cached

    try:
        info = call_with_failover(
            "info",
            lambda provider: provider.get_security_info(ticker),
            # A provider that supplies identity but no market cap has not
            # given us enough to make a suitability call, so keep looking.
            accept=lambda result: result is not None and result.is_usable(),
        )
    except AllProvidersFailed as exc:
        logger.debug("No provider could describe %s: %s", ticker, exc)
        info = None

    _info_cache.put(ticker, info)
    return info


def get_ticker_info(ticker: str) -> Optional[dict]:
    """Backwards-compatible dict view of `get_security_info`.

    Kept so the demo notebook and any external caller written against the old
    signature keep working. New code should use `get_security_info`, whose
    field names do not change when a vendor renames theirs.
    """
    info = get_security_info(ticker)
    if info is None:
        return None
    return {
        "sector": info.sector,
        "industry": info.industry,
        "exchange": info.exchange,
        "marketCap": info.market_cap,
        "beta": info.beta,
        "trailingPE": info.pe_ratio,
        "priceToBook": info.pb_ratio,
        "dividendYield": info.dividend_yield,
        "longName": info.name,
    }


def clear_ticker_info_cache() -> None:
    """Used by tests and by the demo notebook to guarantee a cold start."""
    _info_cache.clear()
