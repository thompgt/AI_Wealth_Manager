"""Tiingo adapter.

A second licensed source, and the reason failover is worth having at all: two
independent vendors disagreeing about a close is a detectable event, whereas a
single vendor's bad print is indistinguishable from the truth. Configure
`TIINGO_API_KEY` to put it in the chain.
"""

from datetime import timedelta
from typing import Dict, List, Optional

import httpx
import pandas as pd

from config import settings
from db import utcnow
from logging_setup import get_logger
from services.providers.base import MarketDataProvider, Quote, SecurityInfo
from services.resilience import ProviderError

logger = get_logger(__name__)

BASE_URL = "https://api.tiingo.com"


class TiingoProvider(MarketDataProvider):
    name = "tiingo"
    requires_credentials = True
    supports_fundamentals = False  # fundamentals are a paid add-on

    def __init__(self, api_key: Optional[str] = None, timeout: Optional[float] = None):
        self.api_key = api_key or settings.TIINGO_API_KEY
        self.timeout = timeout or settings.MARKET_DATA_TIMEOUT_SECONDS

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, **params):
        headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"}
        try:
            response = httpx.get(
                f"{BASE_URL}{path}", params=params, headers=headers, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"tiingo request failed: {exc}", provider=self.name) from exc

        if response.status_code == 429:
            raise ProviderError("tiingo rate limit exceeded", transient=True, provider=self.name)
        if response.status_code in (401, 403):
            raise ProviderError(
                f"tiingo rejected the API key ({response.status_code})",
                transient=False,
                provider=self.name,
            )
        if response.status_code == 404:
            raise ProviderError(
                f"tiingo has no data for {path}", transient=False, provider=self.name
            )
        if response.status_code >= 500:
            raise ProviderError(
                f"tiingo server error {response.status_code}", transient=True, provider=self.name
            )
        response.raise_for_status()
        return response.json()

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        if not symbols:
            return {}
        # The IEX endpoint takes a comma-separated batch, which keeps a wide
        # screen to one request instead of one per name.
        payload = self._get("/iex", tickers=",".join(symbols))
        quotes: Dict[str, Quote] = {}
        for entry in payload or []:
            symbol = (entry.get("ticker") or "").upper()
            # `last` is absent outside market hours; `prevClose` is the honest
            # fallback and is still a real print.
            price = entry.get("last") or entry.get("tngoLast") or entry.get("prevClose")
            if not symbol or not price or float(price) <= 0:
                continue
            timestamp = entry.get("timestamp")
            as_of = (
                pd.Timestamp(timestamp).to_pydatetime().replace(tzinfo=None)
                if timestamp
                else utcnow()
            )
            quotes[symbol] = Quote(
                symbol=symbol, price=float(price), as_of=as_of, provider=self.name
            )
        return quotes

    def get_history(self, symbols: List[str], years: float = 1.0) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        end = utcnow().date()
        start = end - timedelta(days=int(years * 365))

        series: Dict[str, pd.Series] = {}
        for symbol in symbols:
            try:
                rows = self._get(
                    f"/tiingo/daily/{symbol}/prices",
                    startDate=str(start),
                    endDate=str(end),
                    # Split- and dividend-adjusted. An unadjusted series turns
                    # a 2:1 split into a -50% day and poisons every volatility
                    # and correlation figure computed from it.
                    format="json",
                )
            except ProviderError as exc:
                if exc.transient:
                    raise
                logger.debug("tiingo has no history for %s: %s", symbol, exc)
                continue
            if not rows:
                continue
            index = pd.to_datetime([r["date"] for r in rows]).tz_localize(None)
            closes = [float(r.get("adjClose") or r.get("close")) for r in rows]
            series[symbol] = pd.Series(closes, index=index)

        if not series:
            raise ProviderError(f"tiingo returned no history for {symbols}", provider=self.name)
        return pd.DataFrame(series).sort_index()

    def get_security_info(self, symbol: str) -> Optional[SecurityInfo]:
        try:
            meta = self._get(f"/tiingo/daily/{symbol}")
        except ProviderError as exc:
            if exc.transient:
                raise
            return None
        if not meta:
            return None
        return SecurityInfo(
            symbol=symbol,
            provider=self.name,
            name=meta.get("name"),
            exchange=meta.get("exchangeCode"),
            country="US",
            fetched_at=utcnow(),
            # Identity only on the free tier. Saying so explicitly stops the
            # screener from reading an absent market cap as a small one and
            # filtering the name out for the wrong reason.
            warnings=["tiingo free tier supplies no fundamentals"],
        )
