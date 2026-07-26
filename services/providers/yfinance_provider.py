"""Yahoo Finance adapter (via yfinance).

The free default, and the reason this system runs with no API keys at all. It
is also unauthenticated, unofficial, has no SLA, rate-limits aggressively, and
renames `.info` fields between minor releases — so it belongs behind the
provider interface with everything else, not wired directly into the agents
where a field rename silently turns every P/E into `None`.

Where a licensed provider is configured, this sits at the end of the failover
chain rather than leading it.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from db import utcnow
from logging_setup import get_logger
from services.providers.base import MarketDataProvider, Quote, SecurityInfo
from services.resilience import ProviderError

logger = get_logger(__name__)


def _num(raw: Dict[str, Any], *keys: str) -> Optional[float]:
    """First usable numeric value among several candidate keys.

    yfinance moves fields between releases (`trailingPE` vs `trailing_pe`) and
    returns `'Infinity'` strings for undefined ratios. Accepting a list of
    aliases here means a rename degrades one field rather than the whole
    fundamentals record.
    """
    for key in keys:
        value = raw.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        # NaN and infinities reach analytics as poison: a NaN P/E propagates
        # through a median and takes the whole screen with it.
        if number != number or number in (float("inf"), float("-inf")):
            continue
        return number
    return None


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"
    requires_credentials = False
    supports_fundamentals = True

    def is_available(self) -> bool:
        return True

    # -- prices ---------------------------------------------------------------

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        if not symbols:
            return {}
        try:
            raw = yf.download(
                symbols, period="5d", progress=False, auto_adjust=False, threads=False
            )
        except Exception as exc:  # noqa: BLE001 -- normalized for the failover layer
            raise ProviderError(f"yfinance quote download failed: {exc}", provider=self.name) from exc

        frame = self._adj_close(raw, symbols)
        if frame is None or frame.empty:
            # Not transient: we asked for a 5-day window on symbols that
            # should be trading. Retrying a delisted or misspelled ticker
            # three times costs ~6s per name, which on a wide screen is the
            # difference between a slow run and a timed-out one.
            raise ProviderError(
                f"yfinance returned no rows for {symbols}", transient=False, provider=self.name
            )

        quotes: Dict[str, Quote] = {}
        for symbol in symbols:
            if symbol not in frame.columns:
                continue
            series = frame[symbol].dropna()
            if series.empty:
                continue
            price = float(series.iloc[-1])
            if price <= 0:
                continue
            quotes[symbol] = Quote(
                symbol=symbol,
                price=price,
                as_of=pd.Timestamp(series.index[-1]).to_pydatetime().replace(tzinfo=None),
                provider=self.name,
            )
        return quotes

    def get_history(self, symbols: List[str], years: float = 1.0) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        period_days = max(5, int(years * 365))
        try:
            raw = yf.download(
                symbols,
                period=f"{period_days}d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"yfinance history download failed: {exc}", provider=self.name
            ) from exc

        frame = self._adj_close(raw, symbols)
        if frame is None or frame.empty:
            raise ProviderError(f"yfinance returned no history for {symbols}", provider=self.name)
        return frame.dropna(how="all")

    @staticmethod
    def _adj_close(raw: pd.DataFrame, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Pull the adjusted-close frame out of yfinance's shape-shifting result.

        `yf.download` returns a plain frame for one symbol and a MultiIndex for
        several, and `auto_adjust` (whose default flipped to True in a minor
        release) decides whether `Adj Close` exists at all. Passing
        auto_adjust=False keeps the column, but the fallback to `Close` here
        means a future default change degrades rather than breaks.
        """
        if raw is None or raw.empty:
            return None
        frame = None
        if isinstance(raw.columns, pd.MultiIndex):
            for field in ("Adj Close", "Close"):
                if field in raw.columns.get_level_values(0):
                    frame = raw[field]
                    break
        else:
            for field in ("Adj Close", "Close"):
                if field in raw.columns:
                    frame = raw[[field]]
                    frame.columns = symbols[:1]
                    break
        if frame is None:
            return None
        if isinstance(frame, pd.Series):
            frame = frame.to_frame()
            frame.columns = symbols[:1]
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        return frame

    # -- fundamentals ---------------------------------------------------------

    def get_security_info(self, symbol: str) -> Optional[SecurityInfo]:
        try:
            info = yf.Ticker(symbol).info
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"yfinance info fetch failed for {symbol}: {exc}", provider=self.name
            ) from exc

        if not info or not isinstance(info, dict):
            return None

        warnings: List[str] = []
        market_cap = _num(info, "marketCap")
        if market_cap is None:
            warnings.append("no market capitalisation reported")

        # Dollar volume, not share volume: 10M shares of a $2 stock and 10M of
        # a $400 stock are entirely different liquidity situations, and only
        # the dollar figure tells you whether a position can be exited.
        avg_volume = _num(info, "averageDailyVolume3Month", "averageVolume", "averageVolume10days")
        last_price = _num(info, "currentPrice", "regularMarketPrice", "previousClose")
        avg_dollar_volume = (
            avg_volume * last_price if avg_volume is not None and last_price is not None else None
        )

        yield_raw = _num(info, "dividendYield")
        # yfinance has reported this both as a fraction and as a percentage
        # across releases. Anything above 1.0 is a percentage -- a 100%+
        # dividend yield is not a thing this screen will legitimately see.
        dividend_yield = (
            yield_raw / 100.0 if yield_raw is not None and yield_raw > 1.0 else yield_raw
        )

        debt_to_equity = _num(info, "debtToEquity")
        if debt_to_equity is not None and debt_to_equity > 5:
            # Reported as a percentage (150 meaning 1.5x).
            debt_to_equity = debt_to_equity / 100.0

        return SecurityInfo(
            symbol=symbol,
            provider=self.name,
            name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            exchange=info.get("exchange"),
            currency=info.get("currency", "USD"),
            country=info.get("country"),
            quote_type=info.get("quoteType"),
            market_cap=market_cap,
            avg_dollar_volume=avg_dollar_volume,
            beta=_num(info, "beta", "beta3Year"),
            pe_ratio=_num(info, "trailingPE"),
            forward_pe=_num(info, "forwardPE"),
            pb_ratio=_num(info, "priceToBook"),
            ps_ratio=_num(info, "priceToSalesTrailing12Months"),
            peg_ratio=_num(info, "pegRatio", "trailingPegRatio"),
            dividend_yield=dividend_yield,
            profit_margin=_num(info, "profitMargins"),
            return_on_equity=_num(info, "returnOnEquity"),
            debt_to_equity=debt_to_equity,
            revenue_growth=_num(info, "revenueGrowth"),
            earnings_growth=_num(info, "earningsGrowth", "earningsQuarterlyGrowth"),
            free_cash_flow=_num(info, "freeCashflow"),
            expense_ratio=_num(info, "annualReportExpenseRatio", "netExpenseRatio"),
            fetched_at=utcnow(),
            warnings=warnings,
        )
