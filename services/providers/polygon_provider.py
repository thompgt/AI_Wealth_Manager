"""Polygon.io adapter.

A licensed vendor with an SLA, point-in-time correctness and terms that permit
commercial use — none of which the free default offers. Configure
`POLYGON_API_KEY` and it leads the failover chain automatically.

Implemented against the REST API directly rather than the vendor SDK: the
surface used here is three endpoints, and an extra dependency that pins its
own HTTP stack is a poor trade for that.
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

BASE_URL = "https://api.polygon.io"


class PolygonProvider(MarketDataProvider):
    name = "polygon"
    requires_credentials = True
    supports_fundamentals = True

    def __init__(self, api_key: Optional[str] = None, timeout: Optional[float] = None):
        self.api_key = api_key or settings.POLYGON_API_KEY
        self.timeout = timeout or settings.MARKET_DATA_TIMEOUT_SECONDS

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, **params) -> dict:
        params["apiKey"] = self.api_key
        try:
            response = httpx.get(f"{BASE_URL}{path}", params=params, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise ProviderError(f"polygon request failed: {exc}", provider=self.name) from exc

        if response.status_code == 429:
            raise ProviderError("polygon rate limit exceeded", transient=True, provider=self.name)
        if response.status_code in (401, 403):
            # Not transient: retrying a bad key just burns the retry budget on
            # every ticker of every run.
            raise ProviderError(
                f"polygon rejected the API key ({response.status_code})",
                transient=False,
                provider=self.name,
            )
        if response.status_code == 404:
            raise ProviderError(
                f"polygon has no data for {path}", transient=False, provider=self.name
            )
        if response.status_code >= 500:
            raise ProviderError(
                f"polygon server error {response.status_code}", transient=True, provider=self.name
            )
        response.raise_for_status()
        return response.json()

    # -- prices ---------------------------------------------------------------

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        if not symbols:
            return {}
        # One grouped snapshot call rather than one call per symbol: the
        # per-symbol form is the fastest way to exhaust a rate limit on a
        # screen of any width.
        payload = self._get(
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            tickers=",".join(symbols),
        )
        quotes: Dict[str, Quote] = {}
        for entry in payload.get("tickers", []) or []:
            symbol = entry.get("ticker")
            day = entry.get("day") or {}
            prev = entry.get("prevDay") or {}
            # Before the open, `day.c` is 0 rather than absent. Falling back to
            # the previous close is what stops a pre-market run from valuing
            # the whole book at zero.
            price = day.get("c") or prev.get("c")
            if not symbol or not price or float(price) <= 0:
                continue
            nanos = entry.get("updated")
            as_of = (
                pd.Timestamp(nanos, unit="ns").to_pydatetime().replace(tzinfo=None)
                if nanos
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
                payload = self._get(
                    f"/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}",
                    adjusted="true",  # splits and dividends, or the returns are wrong
                    sort="asc",
                    limit=50000,
                )
            except ProviderError as exc:
                if exc.transient:
                    raise
                # A single delisted symbol must not fail the whole batch.
                logger.debug("polygon has no history for %s: %s", symbol, exc)
                continue
            rows = payload.get("results") or []
            if not rows:
                continue
            index = pd.to_datetime([r["t"] for r in rows], unit="ms").tz_localize(None)
            series[symbol] = pd.Series([float(r["c"]) for r in rows], index=index)

        if not series:
            raise ProviderError(f"polygon returned no history for {symbols}", provider=self.name)
        return pd.DataFrame(series).sort_index()

    # -- fundamentals ---------------------------------------------------------

    def get_security_info(self, symbol: str) -> Optional[SecurityInfo]:
        try:
            payload = self._get(f"/v3/reference/tickers/{symbol}")
        except ProviderError as exc:
            if exc.transient:
                raise
            return None

        result = payload.get("results") or {}
        if not result:
            return None

        market_cap = result.get("market_cap")
        shares = result.get("weighted_shares_outstanding") or result.get("share_class_shares_outstanding")

        return SecurityInfo(
            symbol=symbol,
            provider=self.name,
            name=result.get("name"),
            sector=(result.get("sic_description") or None),
            exchange=result.get("primary_exchange"),
            currency=result.get("currency_name", "usd").upper(),
            country=result.get("locale", "us").upper(),
            quote_type="ETF" if result.get("type") == "ETF" else "EQUITY",
            market_cap=float(market_cap) if market_cap else None,
            # Polygon's reference endpoint carries identity, not ratios. The
            # screener fills valuation and quality metrics from whichever
            # provider supplies them; leaving these None is honest rather than
            # inventing values from an incomplete source.
            fetched_at=utcnow(),
            warnings=[] if market_cap else ["polygon reference data has no market cap"],
            raw={"shares_outstanding": shares} if shares else {},
        )
