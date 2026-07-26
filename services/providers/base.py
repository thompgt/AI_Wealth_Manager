"""Market data provider interface.

Every provider returns the same shapes, and every shape carries provenance:
which provider served it and how old it is. That is not bookkeeping — it is
the difference between a system that sizes a position against a price and one
that can say afterwards *which* price, from *where*, and *when*. A quote with
no source and no timestamp is indistinguishable from a guess.

Two conventions every implementation must honour:

* **Unknown is not zero.** A symbol whose price could not be determined is
  omitted from the result, never returned as 0.0. Callers must be able to
  distinguish "worth nothing" from "we don't know", because those lead to
  opposite decisions.
* **Missing is not absent.** A fundamental that genuinely has no value (a P/E
  for a company with no earnings) is `None`, and so is one we failed to fetch.
  Where the difference matters, it belongs in `SecurityInfo.warnings`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import settings
from db import utcnow


@dataclass(frozen=True)
class Quote:
    """A price with the context needed to judge whether to trust it."""

    symbol: str
    price: float
    as_of: datetime
    provider: str
    currency: str = "USD"
    # True when the value came from a cache or a stale bar rather than a live
    # fetch. Surfaced all the way to the API response, because a client whose
    # allocation was computed off a two-day-old close deserves to know.
    stale: bool = False

    @property
    def age(self) -> timedelta:
        return utcnow() - self.as_of

    def age_minutes(self) -> float:
        return self.age.total_seconds() / 60.0

    def age_trading_days(self) -> int:
        """Sessions elapsed since this price printed.

        The right unit for a daily close. Clock time would mark every Monday
        morning quote as ~65 hours stale and every long-weekend run as worse,
        which makes the staleness signal useless precisely because it fires
        constantly. Weekends are excluded; exchange holidays are not, so this
        can overstate staleness by a day around a holiday -- erring toward
        flagging is the safe direction for a number used to decide whether to
        trust a price.
        """
        return int(
            np.busday_count(
                self.as_of.date(),
                max(utcnow().date(), self.as_of.date()),
            )
        )

    def is_stale(self, max_trading_days: Optional[int] = None) -> bool:
        limit = max_trading_days if max_trading_days is not None else settings.QUOTE_MAX_AGE_TRADING_DAYS
        return self.age_trading_days() > limit


@dataclass
class SecurityInfo:
    """Reference data and fundamentals for one instrument.

    Flat and explicit rather than a passthrough of a provider's raw dict:
    yfinance renames `.info` keys between minor releases, and three agents
    reading raw keys meant a rename silently turned every P/E into None with
    no error. Normalizing here means a provider change is a change in one
    adapter.
    """

    symbol: str
    provider: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    currency: str = "USD"
    country: Optional[str] = None
    quote_type: Optional[str] = None  # EQUITY | ETF | MUTUALFUND | INDEX

    market_cap: Optional[float] = None
    avg_dollar_volume: Optional[float] = None
    beta: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    profit_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    free_cash_flow: Optional[float] = None
    expense_ratio: Optional[float] = None

    fetched_at: datetime = field(default_factory=utcnow)
    warnings: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def is_usable(self) -> bool:
        """Enough identity to make a suitability decision about.

        A record with no exchange and no market cap is not a security we
        verified; it is a symbol we failed to look up. Suitability fails
        closed on this, which is the correct direction.
        """
        return bool(self.exchange) and self.market_cap is not None


@dataclass
class NewsItem:
    title: str
    url: str
    snippet: str = ""
    source: Optional[str] = None
    published_at: Optional[datetime] = None


class MarketDataProvider(ABC):
    """One data vendor.

    Implementations raise `ProviderError` on failure rather than returning
    empty results, so the failover layer can tell "this provider is broken"
    from "this provider says there is nothing". Swallowing errors inside an
    adapter makes the circuit breaker blind.
    """

    name: str = "abstract"
    # Providers that need no credentials are always available as a last
    # resort, which is what keeps the system runnable with no API keys.
    requires_credentials: bool = False
    supports_fundamentals: bool = True

    @abstractmethod
    def is_available(self) -> bool:
        """False when required credentials are absent."""

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Latest price per symbol. Unresolvable symbols are omitted."""

    @abstractmethod
    def get_history(self, symbols: List[str], years: float = 1.0) -> pd.DataFrame:
        """Adjusted daily closes: DatetimeIndex rows, one column per symbol.

        Adjusted, not raw: a split or dividend in an unadjusted series shows
        up as a one-day return of -50%, which poisons every volatility,
        correlation and drawdown figure computed from it.
        """

    def get_security_info(self, symbol: str) -> Optional[SecurityInfo]:
        """Reference data and fundamentals. None when unavailable."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{type(self).__name__} name={self.name!r}>"
