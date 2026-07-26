"""The investable universe: seed catalogue and refresh.

The previous system could only ever recommend from a 34-name constant in
`stock_research.py` — all US large-cap equities. That is not a tuning
parameter, it is a ceiling on what the system is capable of advising. A
client who needs duration, international exposure, or simply a cheap core
holding could not be served, because the recommendation space did not contain
the instruments. Worse, the diagnostics agent would correctly flag "100% of
your equity is in one sector" and the research agent had no sector fund to
fix it with.

The universe now lives in the `securities` table, spans asset classes, and is
refreshed from the provider chain. The seed below is a starting catalogue, not
a fixed list: rows can be added, deactivated or restricted per firm without a
deploy, and `refresh_fundamentals` keeps the screening inputs current.

Seed selection principles:

* **Build the asset allocation first.** Broad, liquid, low-cost index funds
  covering each asset class an IPS can target. Without these, a target
  allocation is unimplementable and the rebalancer has nothing to buy.
* **Then the satellites.** Sector funds so a concentration flaw has a
  remedy, factor funds so a tilt is expressible, and individual equities
  spread across all eleven GICS sectors so single-name research has breadth.
* **Liquidity over cleverness.** Every seed name trades enough that a
  client-sized order is not itself the market. Nothing here is an
  edge; it is a competent, implementable opportunity set.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db import Security, session_scope, utcnow
from logging_setup import get_logger
from services.market_data import fetch_historical_prices, get_security_info

logger = get_logger(__name__)


@dataclass(frozen=True)
class SeedSecurity:
    ticker: str
    name: str
    asset_class: str
    security_type: str = "equity"
    region: str = "US"
    sector: Optional[str] = None


# --- Core building blocks ----------------------------------------------------
# One or two credible options per asset class, so any target allocation an IPS
# can express has an instrument that implements it.

_CORE = [
    SeedSecurity("VTI", "Vanguard Total Stock Market ETF", "us_equity", "etf"),
    SeedSecurity("VOO", "Vanguard S&P 500 ETF", "us_equity", "etf"),
    SeedSecurity("SPY", "SPDR S&P 500 ETF Trust", "us_equity", "etf"),
    SeedSecurity("IWM", "iShares Russell 2000 ETF", "us_equity", "etf"),
    SeedSecurity("VB", "Vanguard Small-Cap ETF", "us_equity", "etf"),
    SeedSecurity("VO", "Vanguard Mid-Cap ETF", "us_equity", "etf"),
    SeedSecurity("QQQ", "Invesco QQQ Trust", "us_equity", "etf"),

    SeedSecurity("VEA", "Vanguard FTSE Developed Markets ETF", "intl_equity", "etf", "Global ex-US"),
    SeedSecurity("VXUS", "Vanguard Total International Stock ETF", "intl_equity", "etf", "Global ex-US"),
    SeedSecurity("IEFA", "iShares Core MSCI EAFE ETF", "intl_equity", "etf", "Developed ex-US"),
    SeedSecurity("VGK", "Vanguard FTSE Europe ETF", "intl_equity", "etf", "Europe"),

    SeedSecurity("VWO", "Vanguard FTSE Emerging Markets ETF", "em_equity", "etf", "Emerging"),
    SeedSecurity("IEMG", "iShares Core MSCI Emerging Markets ETF", "em_equity", "etf", "Emerging"),

    SeedSecurity("BND", "Vanguard Total Bond Market ETF", "fixed_income", "etf"),
    SeedSecurity("AGG", "iShares Core US Aggregate Bond ETF", "fixed_income", "etf"),
    SeedSecurity("BNDX", "Vanguard Total International Bond ETF", "fixed_income", "etf", "Global ex-US"),
    SeedSecurity("TLT", "iShares 20+ Year Treasury Bond ETF", "fixed_income", "etf"),
    SeedSecurity("IEF", "iShares 7-10 Year Treasury Bond ETF", "fixed_income", "etf"),
    SeedSecurity("SHY", "iShares 1-3 Year Treasury Bond ETF", "fixed_income", "etf"),
    SeedSecurity("TIP", "iShares TIPS Bond ETF", "fixed_income", "etf"),
    SeedSecurity("LQD", "iShares Investment Grade Corporate Bond ETF", "fixed_income", "etf"),
    SeedSecurity("HYG", "iShares High Yield Corporate Bond ETF", "fixed_income", "etf"),
    SeedSecurity("MUB", "iShares National Muni Bond ETF", "fixed_income", "etf"),
    SeedSecurity("SGOV", "iShares 0-3 Month Treasury Bond ETF", "cash", "etf"),

    SeedSecurity("VNQ", "Vanguard Real Estate ETF", "real_assets", "etf", sector="Real Estate"),
    SeedSecurity("SCHH", "Schwab US REIT ETF", "real_assets", "etf", sector="Real Estate"),
    SeedSecurity("GLD", "SPDR Gold Shares", "commodity", "etf"),
    SeedSecurity("IAU", "iShares Gold Trust", "commodity", "etf"),
    SeedSecurity("DBC", "Invesco DB Commodity Index Tracking Fund", "commodity", "etf"),
]

# Sector funds. These exist so that "your equity sleeve is 100% Technology" has
# an actionable answer -- the diagnosis was previously unfixable because the
# universe contained no diversifying instrument.
_SECTOR_FUNDS = [
    SeedSecurity("XLK", "Technology Select Sector SPDR", "us_equity", "etf", sector="Technology"),
    SeedSecurity("XLF", "Financial Select Sector SPDR", "us_equity", "etf", sector="Financial Services"),
    SeedSecurity("XLV", "Health Care Select Sector SPDR", "us_equity", "etf", sector="Healthcare"),
    SeedSecurity("XLE", "Energy Select Sector SPDR", "us_equity", "etf", sector="Energy"),
    SeedSecurity("XLI", "Industrial Select Sector SPDR", "us_equity", "etf", sector="Industrials"),
    SeedSecurity("XLY", "Consumer Discretionary Select Sector SPDR", "us_equity", "etf", sector="Consumer Cyclical"),
    SeedSecurity("XLP", "Consumer Staples Select Sector SPDR", "us_equity", "etf", sector="Consumer Defensive"),
    SeedSecurity("XLU", "Utilities Select Sector SPDR", "us_equity", "etf", sector="Utilities"),
    SeedSecurity("XLB", "Materials Select Sector SPDR", "us_equity", "etf", sector="Basic Materials"),
    SeedSecurity("XLRE", "Real Estate Select Sector SPDR", "real_assets", "etf", sector="Real Estate"),
    SeedSecurity("XLC", "Communication Services Select Sector SPDR", "us_equity", "etf", sector="Communication Services"),
]

# Factor and income funds, so an IPS can express a tilt rather than only a
# weight.
_FACTOR_FUNDS = [
    SeedSecurity("VTV", "Vanguard Value ETF", "us_equity", "etf"),
    SeedSecurity("VUG", "Vanguard Growth ETF", "us_equity", "etf"),
    SeedSecurity("VBR", "Vanguard Small-Cap Value ETF", "us_equity", "etf"),
    SeedSecurity("MTUM", "iShares MSCI USA Momentum Factor ETF", "us_equity", "etf"),
    SeedSecurity("QUAL", "iShares MSCI USA Quality Factor ETF", "us_equity", "etf"),
    SeedSecurity("USMV", "iShares MSCI USA Min Vol Factor ETF", "us_equity", "etf"),
    SeedSecurity("VIG", "Vanguard Dividend Appreciation ETF", "us_equity", "etf"),
    SeedSecurity("VYM", "Vanguard High Dividend Yield ETF", "us_equity", "etf"),
    SeedSecurity("SCHD", "Schwab US Dividend Equity ETF", "us_equity", "etf"),
    SeedSecurity("DGRO", "iShares Core Dividend Growth ETF", "us_equity", "etf"),
]

# Individual equities across all eleven sectors. Sectors are refreshed from the
# provider rather than trusted from here; the annotation is a fallback for when
# reference data is unavailable, so a screen never silently loses a whole
# sector to an "Unknown" bucket.
_EQUITIES = [
    # Technology
    ("AAPL", "Apple Inc.", "Technology"), ("MSFT", "Microsoft Corp.", "Technology"),
    ("NVDA", "NVIDIA Corp.", "Technology"), ("AVGO", "Broadcom Inc.", "Technology"),
    ("ORCL", "Oracle Corp.", "Technology"), ("CRM", "Salesforce Inc.", "Technology"),
    ("ADBE", "Adobe Inc.", "Technology"), ("AMD", "Advanced Micro Devices", "Technology"),
    ("CSCO", "Cisco Systems", "Technology"), ("ACN", "Accenture plc", "Technology"),
    ("INTU", "Intuit Inc.", "Technology"), ("TXN", "Texas Instruments", "Technology"),
    ("QCOM", "Qualcomm Inc.", "Technology"), ("IBM", "IBM Corp.", "Technology"),
    ("NOW", "ServiceNow Inc.", "Technology"), ("AMAT", "Applied Materials", "Technology"),
    ("MU", "Micron Technology", "Technology"), ("LRCX", "Lam Research", "Technology"),
    ("PANW", "Palo Alto Networks", "Technology"), ("SNPS", "Synopsys Inc.", "Technology"),
    # Communication Services
    ("GOOGL", "Alphabet Inc.", "Communication Services"), ("META", "Meta Platforms", "Communication Services"),
    ("NFLX", "Netflix Inc.", "Communication Services"), ("DIS", "Walt Disney Co.", "Communication Services"),
    ("CMCSA", "Comcast Corp.", "Communication Services"), ("VZ", "Verizon Communications", "Communication Services"),
    ("T", "AT&T Inc.", "Communication Services"), ("TMUS", "T-Mobile US", "Communication Services"),
    ("EA", "Electronic Arts", "Communication Services"),
    # Consumer Cyclical
    ("AMZN", "Amazon.com Inc.", "Consumer Cyclical"), ("TSLA", "Tesla Inc.", "Consumer Cyclical"),
    ("HD", "Home Depot", "Consumer Cyclical"), ("MCD", "McDonald's Corp.", "Consumer Cyclical"),
    ("NKE", "Nike Inc.", "Consumer Cyclical"), ("LOW", "Lowe's Companies", "Consumer Cyclical"),
    ("SBUX", "Starbucks Corp.", "Consumer Cyclical"), ("TJX", "TJX Companies", "Consumer Cyclical"),
    ("BKNG", "Booking Holdings", "Consumer Cyclical"), ("GM", "General Motors", "Consumer Cyclical"),
    ("F", "Ford Motor Co.", "Consumer Cyclical"),
    # Consumer Defensive
    ("PG", "Procter & Gamble", "Consumer Defensive"), ("KO", "Coca-Cola Co.", "Consumer Defensive"),
    ("PEP", "PepsiCo Inc.", "Consumer Defensive"), ("COST", "Costco Wholesale", "Consumer Defensive"),
    ("WMT", "Walmart Inc.", "Consumer Defensive"), ("MDLZ", "Mondelez International", "Consumer Defensive"),
    ("CL", "Colgate-Palmolive", "Consumer Defensive"), ("MO", "Altria Group", "Consumer Defensive"),
    ("KMB", "Kimberly-Clark", "Consumer Defensive"), ("GIS", "General Mills", "Consumer Defensive"),
    # Healthcare
    ("UNH", "UnitedHealth Group", "Healthcare"), ("JNJ", "Johnson & Johnson", "Healthcare"),
    ("LLY", "Eli Lilly and Co.", "Healthcare"), ("ABBV", "AbbVie Inc.", "Healthcare"),
    ("MRK", "Merck & Co.", "Healthcare"), ("PFE", "Pfizer Inc.", "Healthcare"),
    ("TMO", "Thermo Fisher Scientific", "Healthcare"), ("ABT", "Abbott Laboratories", "Healthcare"),
    ("DHR", "Danaher Corp.", "Healthcare"), ("BMY", "Bristol-Myers Squibb", "Healthcare"),
    ("AMGN", "Amgen Inc.", "Healthcare"), ("GILD", "Gilead Sciences", "Healthcare"),
    ("CVS", "CVS Health Corp.", "Healthcare"), ("MDT", "Medtronic plc", "Healthcare"),
    ("ISRG", "Intuitive Surgical", "Healthcare"),
    # Financial Services
    ("BRK-B", "Berkshire Hathaway", "Financial Services"), ("JPM", "JPMorgan Chase", "Financial Services"),
    ("V", "Visa Inc.", "Financial Services"), ("MA", "Mastercard Inc.", "Financial Services"),
    ("BAC", "Bank of America", "Financial Services"), ("WFC", "Wells Fargo", "Financial Services"),
    ("GS", "Goldman Sachs", "Financial Services"), ("MS", "Morgan Stanley", "Financial Services"),
    ("SCHW", "Charles Schwab", "Financial Services"), ("BLK", "BlackRock Inc.", "Financial Services"),
    ("AXP", "American Express", "Financial Services"), ("C", "Citigroup Inc.", "Financial Services"),
    ("SPGI", "S&P Global Inc.", "Financial Services"), ("CB", "Chubb Ltd.", "Financial Services"),
    ("PGR", "Progressive Corp.", "Financial Services"),
    # Industrials
    ("CAT", "Caterpillar Inc.", "Industrials"), ("HON", "Honeywell International", "Industrials"),
    ("UNP", "Union Pacific", "Industrials"), ("BA", "Boeing Co.", "Industrials"),
    ("GE", "General Electric", "Industrials"), ("RTX", "RTX Corp.", "Industrials"),
    ("DE", "Deere & Co.", "Industrials"), ("LMT", "Lockheed Martin", "Industrials"),
    ("UPS", "United Parcel Service", "Industrials"), ("MMM", "3M Co.", "Industrials"),
    ("ETN", "Eaton Corp.", "Industrials"), ("EMR", "Emerson Electric", "Industrials"),
    # Energy
    ("XOM", "Exxon Mobil", "Energy"), ("CVX", "Chevron Corp.", "Energy"),
    ("COP", "ConocoPhillips", "Energy"), ("SLB", "Schlumberger NV", "Energy"),
    ("EOG", "EOG Resources", "Energy"), ("PSX", "Phillips 66", "Energy"),
    ("MPC", "Marathon Petroleum", "Energy"), ("WMB", "Williams Companies", "Energy"),
    # Utilities
    ("NEE", "NextEra Energy", "Utilities"), ("DUK", "Duke Energy", "Utilities"),
    ("SO", "Southern Co.", "Utilities"), ("D", "Dominion Energy", "Utilities"),
    ("AEP", "American Electric Power", "Utilities"), ("EXC", "Exelon Corp.", "Utilities"),
    ("SRE", "Sempra", "Utilities"),
    # Basic Materials
    ("LIN", "Linde plc", "Basic Materials"), ("SHW", "Sherwin-Williams", "Basic Materials"),
    ("APD", "Air Products & Chemicals", "Basic Materials"), ("ECL", "Ecolab Inc.", "Basic Materials"),
    ("NEM", "Newmont Corp.", "Basic Materials"), ("FCX", "Freeport-McMoRan", "Basic Materials"),
    ("DOW", "Dow Inc.", "Basic Materials"),
    # Real Estate
    ("PLD", "Prologis Inc.", "Real Estate"), ("AMT", "American Tower", "Real Estate"),
    ("EQIX", "Equinix Inc.", "Real Estate"), ("SPG", "Simon Property Group", "Real Estate"),
    ("O", "Realty Income Corp.", "Real Estate"), ("PSA", "Public Storage", "Real Estate"),
]


def seed_catalogue() -> List[SeedSecurity]:
    seeds = list(_CORE) + list(_SECTOR_FUNDS) + list(_FACTOR_FUNDS)
    seeds += [
        SeedSecurity(
            ticker,
            name,
            "real_assets" if sector == "Real Estate" else "us_equity",
            "equity",
            "US",
            sector,
        )
        for ticker, name, sector in _EQUITIES
    ]
    # Deduplicate on ticker while preserving order -- a name appearing in two
    # lists (a REIT in both the sector funds and the equities) must not create
    # two rows and violate the unique constraint.
    seen = set()
    unique = []
    for seed in seeds:
        if seed.ticker in seen:
            continue
        seen.add(seed.ticker)
        unique.append(seed)
    return unique


def seed_universe(db: Optional[Session] = None) -> int:
    """Insert any catalogue entries missing from `securities`.

    Idempotent, and never overwrites an existing row: a firm may have edited
    asset class or set a restriction, and re-running the seed must not undo
    that.
    """

    def _apply(session: Session) -> int:
        existing = {row.ticker for row in session.query(Security.ticker).all()}
        added = 0
        for seed in seed_catalogue():
            if seed.ticker in existing:
                continue
            session.add(
                Security(
                    ticker=seed.ticker,
                    name=seed.name,
                    asset_class=seed.asset_class,
                    security_type=seed.security_type,
                    region=seed.region,
                    sector=seed.sector,
                    is_active=True,
                )
            )
            added += 1
        return added

    if db is not None:
        count = _apply(db)
        db.flush()
        return count
    with session_scope() as session:
        return _apply(session)


# --- Refresh -----------------------------------------------------------------


def refresh_fundamentals(
    tickers: Optional[Sequence[str]] = None,
    *,
    max_age_hours: int = 24,
    limit: Optional[int] = None,
) -> int:
    """Pull reference data and fundamentals into `securities`.

    Only refreshes rows older than `max_age_hours`, so a nightly job over a
    large universe does not re-fetch everything it already has. Returns the
    number of rows updated.
    """
    cutoff = utcnow() - timedelta(hours=max_age_hours)
    updated = 0

    with session_scope() as session:
        query = session.query(Security).filter(Security.is_active.is_(True))
        if tickers:
            query = query.filter(Security.ticker.in_([t.upper() for t in tickers]))
        else:
            query = query.filter(
                (Security.fundamentals_updated_at.is_(None))
                | (Security.fundamentals_updated_at < cutoff)
            )
        if limit:
            query = query.limit(limit)

        for security in query.all():
            info = get_security_info(security.ticker)
            if info is None:
                logger.debug("No provider could describe %s; leaving prior data in place.",
                             security.ticker)
                continue

            # Only overwrite with values we actually got. Blanking a field
            # because this particular provider does not serve it would make
            # every screen worse after a failover.
            if info.name:
                security.name = info.name
            if info.sector:
                security.sector = info.sector
            if info.industry:
                security.industry = info.industry
            if info.exchange:
                security.exchange = info.exchange
            if info.currency:
                security.currency = info.currency
            # Correct the seed annotation from the provider's own
            # classification. Whether a row is a fund decides which peer group
            # it is scored against and whether the equity market-cap floor
            # applies to it, so a wrong guess here quietly distorts the screen.
            if info.quote_type:
                resolved = {"ETF": "etf", "MUTUALFUND": "fund", "EQUITY": "equity"}.get(
                    info.quote_type.upper()
                )
                if resolved:
                    security.security_type = resolved

            for field in (
                "market_cap", "avg_dollar_volume", "beta", "pe_ratio", "forward_pe",
                "pb_ratio", "ps_ratio", "peg_ratio", "dividend_yield", "profit_margin",
                "return_on_equity", "debt_to_equity", "revenue_growth", "earnings_growth",
                "free_cash_flow", "expense_ratio",
            ):
                value = getattr(info, field)
                if value is not None:
                    setattr(security, field, value)

            security.fundamentals_updated_at = utcnow()
            updated += 1

    logger.info("Refreshed fundamentals for %d securities.", updated)
    return updated


def refresh_price_metrics(
    tickers: Optional[Sequence[str]] = None,
    *,
    batch_size: int = 40,
    lookback_years: float = 1.0,
) -> int:
    """Compute momentum, volatility and drawdown from price history.

    Cached on the security row so a screen across hundreds of names does not
    recompute hundreds of return series per run -- which was the single
    largest cost in making the universe wide enough to be useful.
    """
    with session_scope() as session:
        query = session.query(Security).filter(Security.is_active.is_(True))
        if tickers:
            query = query.filter(Security.ticker.in_([t.upper() for t in tickers]))
        symbols = [row.ticker for row in query.all()]

    if not symbols:
        return 0

    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start:start + batch_size]
        frame = fetch_historical_prices(batch, years=lookback_years)
        if frame.empty:
            logger.warning("No price history for batch starting at %s", batch[0])
            continue
        for symbol in batch:
            if symbol not in frame.columns:
                continue
            series = frame[symbol].dropna()
            computed = compute_price_metrics(series)
            if computed:
                metrics[symbol] = computed

    if not metrics:
        return 0

    updated = 0
    with session_scope() as session:
        for security in session.query(Security).filter(Security.ticker.in_(list(metrics))).all():
            values = metrics[security.ticker]
            for key, value in values.items():
                setattr(security, key, value)
            security.metrics_updated_at = utcnow()
            updated += 1

    logger.info("Refreshed price metrics for %d securities.", updated)
    return updated


def compute_price_metrics(series: pd.Series) -> Dict[str, Optional[float]]:
    """Momentum, volatility and max drawdown from an adjusted close series.

    Requires ~6 months of data before reporting anything: a volatility
    computed from three weeks of prices is not a worse estimate, it is a
    different and misleading quantity, and a screen would rank on it as if it
    were comparable to a full-year figure.
    """
    series = series.dropna()
    if len(series) < 120:
        return {}

    returns = series.pct_change().dropna()
    if returns.empty:
        return {}

    trading_days = 252
    volatility = float(returns.std() * np.sqrt(trading_days))

    def _momentum(days: int) -> Optional[float]:
        if len(series) <= days:
            return None
        past = float(series.iloc[-days - 1])
        if past <= 0:
            return None
        return float(series.iloc[-1] / past - 1.0)

    running_max = series.cummax()
    drawdown = float((series / running_max - 1.0).min())

    return {
        # 12-month momentum conventionally skips the most recent month, whose
        # short-term reversal works against the medium-term trend the factor
        # is trying to capture.
        "momentum_12m": _momentum(252 - 21) if len(series) > 252 - 21 else _momentum(len(series) - 1),
        "momentum_6m": _momentum(126),
        "volatility_1y": volatility if volatility == volatility else None,
        "max_drawdown_1y": drawdown if drawdown == drawdown else None,
    }


def active_universe(db: Session, asset_classes: Optional[Iterable[str]] = None) -> List[Security]:
    """Investable rows, excluding anything the firm has restricted."""
    query = db.query(Security).filter(
        Security.is_active.is_(True), Security.is_restricted.is_(False)
    )
    if asset_classes:
        query = query.filter(Security.asset_class.in_(list(asset_classes)))
    return query.all()
