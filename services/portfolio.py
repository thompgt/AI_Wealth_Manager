"""Live portfolio valuation and exposure analysis.

One place that answers "what does this client actually own, and what is it
worth right now". Everything downstream — diagnostics, drift, rebalancing,
performance — reads from here, so there is exactly one definition of market
value, one definition of a weight, and one policy for what happens when a
price is unavailable.

That last point is the one that used to go wrong. A symbol with no price was
valued at $0, which is not a conservative assumption — it is a wrong one that
*shrinks the denominator*, inflating every other position's weight and
manufacturing a concentration flaw that does not exist. Unpriced positions are
now excluded from the base and reported explicitly, so the analysis is
computed on what it could actually see and says what it could not.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from db import Account, ClientProfile, Position, Security, to_float
from logging_setup import get_logger
from services.market_data import get_quotes
from services.providers.base import Quote

logger = get_logger(__name__)

CASH_ASSET_CLASS = "cash"


@dataclass
class HoldingView:
    symbol: str
    account_id: int
    account_name: str
    tax_treatment: str
    quantity: float
    price: float
    market_value: float
    cost_basis: float
    asset_class: str
    sector: Optional[str]
    security_type: str
    beta: Optional[float]
    stale_price: bool = False

    @property
    def unrealized_gain(self) -> float:
        return self.market_value - self.cost_basis


@dataclass
class PortfolioView:
    """A client's complete position, priced, with everything derived from it."""

    client_id: int
    holdings: List[HoldingView] = field(default_factory=list)
    # Total cash, including proceeds that have not settled yet.
    cash_by_account: Dict[int, float] = field(default_factory=dict)
    # Cash committed to unsettled trades. Still an asset, not yet spendable.
    # Conflating these two was a real bug: subtracting holds from the balance
    # made a portfolio appear to lose the entire proceeds of every sale for a
    # day, so a rebalance looked like it destroyed half the client's money.
    held_by_account: Dict[int, float] = field(default_factory=dict)
    # Symbols held but not priced. Excluded from every weight below; naming
    # them is what stops the analysis from being quietly wrong.
    unpriced_symbols: List[str] = field(default_factory=list)
    quotes: Dict[str, Quote] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def cash(self) -> float:
        """All cash the client owns. The valuation figure."""
        return sum(self.cash_by_account.values())

    @property
    def spendable_cash(self) -> float:
        """Cash that can fund a trade today. The sizing figure."""
        return sum(
            max(0.0, balance - self.held_by_account.get(account_id, 0.0))
            for account_id, balance in self.cash_by_account.items()
        )

    @property
    def invested_value(self) -> float:
        return sum(h.market_value for h in self.holdings)

    @property
    def total_value(self) -> float:
        """The denominator for every weight in the system.

        Includes cash. A portfolio that is 40% cash is 40% cash, and computing
        position weights against the invested sleeve alone would report a 25%
        position as 42% and trip a limit that was never breached.
        """
        return self.invested_value + self.cash

    def weights(self) -> Dict[str, float]:
        total = self.total_value
        if total <= 0:
            return {}
        by_symbol: Dict[str, float] = {}
        for holding in self.holdings:
            by_symbol[holding.symbol] = by_symbol.get(holding.symbol, 0.0) + holding.market_value
        weights = {symbol: value / total for symbol, value in by_symbol.items()}
        if self.cash > 0:
            weights["CASH"] = self.cash / total
        return weights

    def market_values(self) -> Dict[str, float]:
        by_symbol: Dict[str, float] = {}
        for holding in self.holdings:
            by_symbol[holding.symbol] = by_symbol.get(holding.symbol, 0.0) + holding.market_value
        if self.cash > 0:
            by_symbol["CASH"] = self.cash
        return by_symbol

    def asset_class_weights(self) -> Dict[str, float]:
        total = self.total_value
        if total <= 0:
            return {}
        buckets: Dict[str, float] = {}
        for holding in self.holdings:
            buckets[holding.asset_class] = buckets.get(holding.asset_class, 0.0) + holding.market_value
        if self.cash > 0:
            buckets[CASH_ASSET_CLASS] = buckets.get(CASH_ASSET_CLASS, 0.0) + self.cash
        return {key: value / total for key, value in buckets.items()}

    def is_diversified_fund(self, holding: HoldingView) -> bool:
        """A broad fund carries no single-sector exposure.

        Sector-specific funds (XLK, XLE) are seeded with a sector and are
        treated like any other holding. A total-market or bond fund is not:
        bucketing it as "Unclassified" and then applying the sector
        concentration limit reported a properly diversified portfolio as 91%
        concentrated in a sector that does not exist.
        """
        return holding.security_type in ("etf", "fund") and not holding.sector

    def sector_weights(self) -> Dict[str, float]:
        """Sector exposure as a share of *invested* value, not total.

        Cash has no sector, so including it in the denominator would report a
        portfolio that is 100% technology plus 40% cash as 60% technology --
        understating a concentration that is exactly as severe as it looks.

        Broad funds appear under `_diversified` rather than in a sector
        bucket. This does understate true sector exposure, since a total
        market fund really is ~30% technology; correcting that needs fund
        holdings look-through, which no provider in the chain supplies. The
        alternative -- pretending the exposure is zero *and* flagging it as a
        concentration -- was strictly worse.
        """
        invested = self.invested_value
        if invested <= 0:
            return {}
        buckets: Dict[str, float] = {}
        for holding in self.holdings:
            if self.is_diversified_fund(holding):
                key = "_diversified"
            else:
                key = holding.sector or "Unclassified"
            buckets[key] = buckets.get(key, 0.0) + holding.market_value
        return {key: value / invested for key, value in buckets.items()}

    def portfolio_beta(self) -> Optional[float]:
        """Value-weighted beta over the holdings that report one.

        Cash is included at beta 0 -- it genuinely dampens portfolio beta, and
        omitting it would overstate the market sensitivity of a
        cash-heavy book. Holdings with no beta are excluded from both sides so
        they are not silently treated as beta 1.
        """
        total = self.total_value
        if total <= 0:
            return None
        covered = self.cash
        weighted = 0.0
        for holding in self.holdings:
            if holding.beta is None:
                continue
            covered += holding.market_value
            weighted += holding.beta * holding.market_value
        if covered <= 0:
            return None
        # Less than 60% coverage makes the number more misleading than useful.
        if covered / total < 0.6:
            return None
        return round(weighted / covered, 3)

    def by_account(self, account_id: int) -> List[HoldingView]:
        return [h for h in self.holdings if h.account_id == account_id]

    def available_cash(self, account_id: Optional[int] = None) -> float:
        """Spendable cash, net of settlement holds."""
        if account_id is None:
            return self.spendable_cash
        balance = self.cash_by_account.get(account_id, 0.0)
        return max(0.0, balance - self.held_by_account.get(account_id, 0.0))

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_value": round(self.total_value, 2),
            "invested_value": round(self.invested_value, 2),
            "cash": round(self.cash, 2),
            "position_count": len({h.symbol for h in self.holdings}),
            "weights": {k: round(v, 4) for k, v in self.weights().items()},
            "asset_class_weights": {k: round(v, 4) for k, v in self.asset_class_weights().items()},
            "sector_weights": {k: round(v, 4) for k, v in self.sector_weights().items()},
            "portfolio_beta": self.portfolio_beta(),
            "unpriced_symbols": self.unpriced_symbols,
            "warnings": self.warnings,
        }


def load_portfolio(
    db: Session,
    client: ClientProfile,
    *,
    account_ids: Optional[Sequence[int]] = None,
) -> PortfolioView:
    """Price every position the client holds and derive the exposures."""
    accounts_query = db.query(Account).filter(
        Account.client_id == client.id, Account.is_active.is_(True)
    )
    if account_ids:
        accounts_query = accounts_query.filter(Account.id.in_(list(account_ids)))
    accounts = accounts_query.all()

    view = PortfolioView(client_id=client.id)
    if not accounts:
        view.warnings.append(
            "This client has no active accounts, so there is nothing to analyse."
        )
        return view

    account_by_id = {a.id: a for a in accounts}
    positions = (
        db.query(Position).filter(Position.account_id.in_(list(account_by_id))).all()
    )

    total_held = 0.0
    for account in accounts:
        held = to_float(account.pending_cash_hold)
        view.cash_by_account[account.id] = max(0.0, to_float(account.cash_balance))
        view.held_by_account[account.id] = max(0.0, held)
        total_held += held

    if total_held > 0:
        # A hold reduces what can be *spent*, never what is *owned*. Sizing
        # against the raw balance is how the same dollar gets deployed twice
        # by two runs a minute apart.
        view.warnings.append(
            f"${total_held:,.0f} of cash is unsettled and cannot fund new purchases until "
            "settlement. It is still counted in the portfolio's value."
        )

    symbols = sorted({p.symbol for p in positions if to_float(p.quantity) > 0})
    quotes = get_quotes(symbols) if symbols else {}
    view.quotes = quotes

    reference = {
        s.ticker: s
        for s in db.query(Security).filter(Security.ticker.in_(symbols)).all()
    } if symbols else {}

    unpriced: List[str] = []
    for position in positions:
        quantity = to_float(position.quantity)
        if quantity <= 0:
            continue
        quote = quotes.get(position.symbol)
        if quote is None:
            if position.symbol not in unpriced:
                unpriced.append(position.symbol)
            continue

        security = reference.get(position.symbol)
        account = account_by_id[position.account_id]
        average_cost = to_float(position.average_cost)
        view.holdings.append(
            HoldingView(
                symbol=position.symbol,
                account_id=account.id,
                account_name=account.name,
                tax_treatment=account.tax_treatment,
                quantity=quantity,
                price=quote.price,
                market_value=quantity * quote.price,
                cost_basis=quantity * average_cost,
                asset_class=security.asset_class if security else "us_equity",
                sector=security.sector if security else None,
                security_type=security.security_type if security else "equity",
                beta=security.beta if security else None,
                stale_price=quote.is_stale(),
            )
        )

    if unpriced:
        view.unpriced_symbols = sorted(unpriced)
        view.warnings.append(
            f"No current price was available for {', '.join(view.unpriced_symbols)}. "
            "These positions are excluded from all weights and totals below rather than "
            "valued at zero, so the percentages describe only the portion of the "
            "portfolio that could be priced."
        )

    stale = sorted({h.symbol for h in view.holdings if h.stale_price})
    if stale:
        view.warnings.append(
            f"Prices for {', '.join(stale)} are older than the freshness threshold; "
            "valuations for those positions may not reflect current market levels."
        )

    unclassified = sorted({
        h.symbol for h in view.holdings if h.sector is None and h.security_type == "equity"
    })
    if unclassified:
        view.warnings.append(
            f"No sector classification for {', '.join(unclassified)}; these are grouped as "
            "'Unclassified' and may understate a real sector concentration."
        )

    return view


@dataclass
class DriftEntry:
    asset_class: str
    current_weight: float
    target_weight: float
    drift: float
    band: float
    breached: bool
    dollar_gap: float  # positive means underweight, i.e. needs buying

    def to_dict(self) -> Dict[str, object]:
        return {
            "asset_class": self.asset_class,
            "current_weight": round(self.current_weight, 4),
            "target_weight": round(self.target_weight, 4),
            "drift": round(self.drift, 4),
            "band": round(self.band, 4),
            "breached": self.breached,
            "dollar_gap": round(self.dollar_gap, 2),
        }


def compute_drift(view: PortfolioView, policy) -> List[DriftEntry]:
    """Distance from the target allocation, by asset class.

    Bands exist because rebalancing is not free. Trading back to target on
    every basis point of drift hands the difference to the spread and, in a
    taxable account, to the tax on gains realized for no reason. A band says
    "this is close enough that acting costs more than it fixes".
    """
    total = view.total_value
    if total <= 0:
        return []

    current = view.asset_class_weights()
    classes = set(current) | set(policy.target_allocation)

    entries: List[DriftEntry] = []
    for asset_class in sorted(classes):
        current_weight = current.get(asset_class, 0.0)
        target_weight = policy.target_for(asset_class)
        drift = current_weight - target_weight
        band = policy.band_for(asset_class)
        entries.append(
            DriftEntry(
                asset_class=asset_class,
                current_weight=current_weight,
                target_weight=target_weight,
                drift=drift,
                band=band,
                breached=abs(drift) > band,
                dollar_gap=(target_weight - current_weight) * total,
            )
        )
    return sorted(entries, key=lambda e: -abs(e.drift))


def concentration_breaches(view: PortfolioView, policy) -> List[Dict[str, object]]:
    """Positions and sectors over their policy limits, with the dollars to fix.

    Reporting the excess in dollars rather than percentage points is
    deliberate: "trim $84,000 of AAPL" is executable and "you are 12 points
    over" is not.
    """
    breaches: List[Dict[str, object]] = []
    total = view.total_value
    if total <= 0:
        return breaches

    for symbol, weight in view.weights().items():
        if symbol == "CASH":
            continue
        if weight > policy.max_position_pct:
            breaches.append(
                {
                    "kind": "position",
                    "key": symbol,
                    "weight": round(weight, 4),
                    "limit": policy.max_position_pct,
                    "excess_value": round((weight - policy.max_position_pct) * total, 2),
                }
            )

    invested = view.invested_value
    for sector, weight in view.sector_weights().items():
        # `_diversified` is the broad-fund bucket, not a sector, and
        # `Unclassified` means reference data was missing -- neither is a
        # concentration a trim can fix, and proposing one would sell a
        # diversified fund to cure a problem it does not have.
        if sector in ("_diversified", "Unclassified"):
            continue
        if weight > policy.max_sector_pct:
            breaches.append(
                {
                    "kind": "sector",
                    "key": sector,
                    "weight": round(weight, 4),
                    "limit": policy.max_sector_pct,
                    "excess_value": round((weight - policy.max_sector_pct) * invested, 2),
                }
            )

    cash_weight = view.cash / total if total else 0.0
    if cash_weight > policy.max_cash_pct:
        breaches.append(
            {
                "kind": "cash_high",
                "key": "CASH",
                "weight": round(cash_weight, 4),
                "limit": policy.max_cash_pct,
                "excess_value": round((cash_weight - policy.max_cash_pct) * total, 2),
            }
        )
    elif cash_weight < policy.min_cash_pct:
        breaches.append(
            {
                "kind": "cash_low",
                "key": "CASH",
                "weight": round(cash_weight, 4),
                "limit": policy.min_cash_pct,
                "excess_value": round((policy.min_cash_pct - cash_weight) * total, 2),
            }
        )

    return sorted(breaches, key=lambda b: -abs(float(b["excess_value"])))
