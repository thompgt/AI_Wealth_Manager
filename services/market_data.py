from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from config import settings
from db import MarketDataCache, SessionLocal


def _cache_is_fresh(fetched_at: datetime) -> bool:
    return datetime.utcnow() - fetched_at < timedelta(hours=settings.MARKET_DATA_CACHE_TTL_HOURS)


def _get_cached_current_price(db: Session, ticker: str) -> Optional[float]:
    row = (
        db.query(MarketDataCache)
        .filter(MarketDataCache.ticker == ticker)
        .order_by(MarketDataCache.as_of_date.desc())
        .first()
    )
    if row and _cache_is_fresh(row.fetched_at):
        return row.close_price
    return None


def _upsert_cache(db: Session, ticker: str, as_of_date: datetime, price: float) -> None:
    existing = (
        db.query(MarketDataCache)
        .filter(MarketDataCache.ticker == ticker, MarketDataCache.as_of_date == as_of_date)
        .first()
    )
    if existing:
        existing.close_price = price
        existing.fetched_at = datetime.utcnow()
    else:
        db.add(
            MarketDataCache(
                ticker=ticker,
                as_of_date=as_of_date,
                close_price=price,
                fetched_at=datetime.utcnow(),
            )
        )


def fetch_historical_prices(tickers: List[str], years: int = 1) -> pd.DataFrame:
    """
    Fetches historical adjusted close prices for the given tickers.
    Every fetched close is written through to market_data_cache.
    """
    if not tickers:
        return pd.DataFrame()

    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    # yfinance handles multiple tickers space-separated.
    # auto_adjust=False is required to get an "Adj Close" column at all --
    # newer yfinance defaults to auto_adjust=True, which returns a
    # pre-adjusted "Close" column and drops "Adj Close" entirely.
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)["Adj Close"]

    # If only one ticker, it returns a Series; we want a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
        data.columns = tickers

    db = SessionLocal()
    try:
        for ticker in data.columns:
            for as_of_date, price in data[ticker].dropna().items():
                if pd.isna(price):
                    continue
                _upsert_cache(db, str(ticker), pd.Timestamp(as_of_date).to_pydatetime(), float(price))
        db.commit()
    finally:
        db.close()

    return data


def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Get the latest price for a list of tickers, serving from
    market_data_cache when a fresh-enough (within
    settings.MARKET_DATA_CACHE_TTL_HOURS) entry already exists.
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

        if missing:
            data = yf.download(missing, period="1d", progress=False, auto_adjust=False)["Adj Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame()
                data.columns = missing

            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            for ticker in missing:
                try:
                    price = float(data[ticker].dropna().iloc[-1]) if ticker in data.columns else 0.0
                except Exception:
                    price = 0.0
                prices[ticker] = price
                _upsert_cache(db, ticker, today, price)
            db.commit()

        return prices
    finally:
        db.close()
