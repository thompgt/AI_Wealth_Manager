"""Rebalancing: turning a diagnosis into executable trades.

The previous system could only propose buys from cash. It would correctly
identify "45% of this portfolio is in one position, against a 25% limit" and
then have no mechanism to do anything about it — the only lever available was
to buy *more* things, diluting the concentration slowly and only while cash
lasted. The headline finding was structurally unfixable, which is a strange
property for a portfolio manager.

This module proposes sells and trims as well as buys, and it is where the
several constraints that pull against each other get resolved:

* **Sells before buys, always.** A buy funded by a sale that has not happened
  yet overdraws the account. Proposals carry a sequence number and the
  executor honours it.
* **Trim to the limit, not to zero.** A position 45% of the book against a 25%
  cap needs $X sold, not liquidation. Selling the whole thing is a larger
  taxable event than the breach requires and discards a holding the client may
  have good reason to own.
* **Tax cost is weighed, not ignored.** In a taxable account every trim is a
  realized gain. Where the same exposure can be reduced from a tax-deferred
  account, it is — and where it cannot, the estimated tax is attached to the
  proposal so a human can decide whether the fix is worth it.
* **The cash floor is respected.** The old sizing logic deployed the entire
  cash balance, which created the very liquidity flaw the diagnostics agent
  warns about. Investable cash is what remains above the policy floor.
* **Small trades are dropped.** A $40 rebalancing trade costs more in spread
  than the drift it corrects.

Nothing here executes. It produces proposals, which the guardrails then filter
and a human approves.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from config import settings
from db import Account, ClientProfile, Security
from logging_setup import get_logger
from services import tax_lots
from services.policy import ResolvedPolicy
from services.portfolio import (
    DriftEntry,
    HoldingView,
    PortfolioView,
    concentration_breaches,
)

logger = get_logger(__name__)


@dataclass
class TradeProposalDraft:
    """One proposed trade, with the reason it exists attached.

    `rationale` is not decoration. A proposal a human is asked to approve must
    say which constraint it serves, or the approval is a formality.
    """

    symbol: str
    side: str  # BUY | SELL
    account_id: int
    notional: float
    quantity: Optional[float] = None
    rationale: str = ""
    addresses_flaw: Optional[str] = None
    confidence: Optional[float] = None
    estimated_tax_cost: Optional[float] = None
    current_weight: Optional[float] = None
    target_weight: Optional[float] = None
    # Sells sort ahead of buys so the cash exists before it is spent.
    sequence: int = 0
    tax_detail: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "account_id": self.account_id,
            "notional": round(self.notional, 2),
            "quantity": round(self.quantity, 6) if self.quantity is not None else None,
            "rationale": self.rationale,
            "addresses_flaw": self.addresses_flaw,
            "confidence": self.confidence,
            "estimated_tax_cost": (
                round(self.estimated_tax_cost, 2) if self.estimated_tax_cost is not None else None
            ),
            "current_weight": round(self.current_weight, 4) if self.current_weight else None,
            "target_weight": round(self.target_weight, 4) if self.target_weight else None,
            "sequence": self.sequence,
            "tax_detail": self.tax_detail,
        }


@dataclass
class RebalancePlan:
    proposals: List[TradeProposalDraft] = field(default_factory=list)
    # Things the plan deliberately did not do, and why. As important as the
    # proposals: "we left the concentration alone because trimming it would
    # realize $180k of short-term gain" is a decision, and it should be a
    # visible one.
    deferred: List[Dict[str, object]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def sells(self) -> List[TradeProposalDraft]:
        return [p for p in self.proposals if p.side == "SELL"]

    @property
    def buys(self) -> List[TradeProposalDraft]:
        return [p for p in self.proposals if p.side == "BUY"]

    def gross_notional(self) -> float:
        return sum(p.notional for p in self.proposals)

    def estimated_tax(self) -> float:
        return sum(p.estimated_tax_cost or 0.0 for p in self.proposals)

    def to_dict(self) -> Dict[str, object]:
        return {
            "proposals": [p.to_dict() for p in sorted(self.proposals, key=lambda p: p.sequence)],
            "sell_count": len(self.sells),
            "buy_count": len(self.buys),
            "gross_notional": round(self.gross_notional(), 2),
            "estimated_tax_cost": round(self.estimated_tax(), 2),
            "deferred": self.deferred,
            "notes": self.notes,
        }


def _investable_cash(view: PortfolioView, policy: ResolvedPolicy) -> float:
    """Cash available to deploy, after reserving the policy floor.

    Two subtractions, for different reasons. The floor is a policy reserve:
    the old sizing distributed the entire cash balance, which took cash to
    zero and manufactured the very liquidity flaw the diagnostics agent warns
    about on the next run. Settlement holds are a mechanical constraint: that
    money exists but cannot be spent today.
    """
    total = view.total_value
    if total <= 0:
        return 0.0
    floor = policy.min_cash_pct * total
    return max(0.0, min(view.cash - floor, view.spendable_cash))


def _round_quantity(notional: float, price: float) -> Optional[float]:
    """Convert dollars to shares under the fractional-share policy.

    Whole shares round *down*: rounding up a buy overdraws the account, and
    rounding up a sell sells shares that are not there.
    """
    if price <= 0:
        return None
    raw = notional / price
    if settings.ALLOW_FRACTIONAL_SHARES:
        return round(raw, 6) if raw > 0 else None
    whole = float(int(raw))
    return whole if whole >= 1 else None


def plan_rebalance(
    db: Session,
    client: ClientProfile,
    view: PortfolioView,
    policy: ResolvedPolicy,
    *,
    candidates: Optional[Sequence[Dict[str, object]]] = None,
    candidate_provider: Optional[Callable[[str], Sequence[Dict[str, object]]]] = None,
    # High enough that a first rebalance of a badly misallocated portfolio can
    # finish in one pass. A cap of a dozen looks prudent but leaves the client
    # stranded between two allocations -- half-traded is a position nobody
    # chose. The cap exists to bound a runaway plan, not to pace a correct one.
    max_proposals: int = 24,
) -> RebalancePlan:
    """Build a plan that moves the portfolio toward its policy.

    `candidates` are screened new ideas, used to fill an underweight when no
    existing holding can. Without them the plan can still trim breaches and
    top up existing positions — a rebalance does not require new research.
    """
    plan = RebalancePlan()
    total = view.total_value
    if total <= 0:
        plan.notes.append("Portfolio has no priced value; no trades can be proposed.")
        return plan

    accounts = {
        a.id: a
        for a in db.query(Account).filter(
            Account.client_id == client.id, Account.is_active.is_(True)
        ).all()
    }
    if not accounts:
        plan.notes.append("This client has no active accounts to trade in.")
        return plan

    reference = {
        s.ticker: s for s in db.query(Security).filter(Security.is_active.is_(True)).all()
    }

    # --- 1. Trim positions and sectors over their limits ---------------------
    #
    # Breaches overlap: a portfolio that is 51% AAPL and 100% Technology
    # breaches both limits, and the *same* AAPL shares cure both. Treating
    # them as independent produced two full sets of sells -- $460k of trades
    # on a $723k portfolio, with the second set rejected at execution for
    # trying to sell shares the first set had already committed.
    #
    # `planned` tracks dollars already scheduled per holding, and each
    # subsequent breach is credited with the sales that reduce it. Position
    # breaches are handled first because they are the more specific claim on
    # the same shares.
    proceeds = 0.0
    planned: Dict[tuple, float] = {}

    def planned_for_holding(holding) -> float:
        return planned.get((holding.account_id, holding.symbol), 0.0)

    breaches = sorted(
        [b for b in concentration_breaches(view, policy) if b["kind"] in ("position", "sector")],
        key=lambda b: 0 if b["kind"] == "position" else 1,
    )

    for breach in breaches:
        excess = float(breach["excess_value"])

        # Credit sales already planned against the holdings this breach covers.
        covered = _holdings_for_breach(view, breach)
        already = sum(planned_for_holding(h) for h in covered)
        excess -= already
        if excess < policy.min_position_notional:
            if already > 0:
                plan.notes.append(
                    f"The {breach['kind']} limit on {breach['key']} is already satisfied by "
                    f"${already:,.0f} of trims proposed for an overlapping breach."
                )
            continue

        targets = _trim_targets(view, breach, excess, planned_for_holding)
        for holding, amount in targets:
            if amount < policy.min_position_notional:
                continue
            account = accounts.get(holding.account_id)
            if account is None:
                continue

            estimate = tax_lots.estimate_sale(
                db, account, holding.symbol,
                _round_quantity(amount, holding.price) or 0.0,
                holding.price,
                method=policy.lot_selection_method,
            )
            tax_cost = estimate.estimated_tax() if account.tax_treatment == "taxable" else 0.0

            # A trim that costs more in tax than the risk it removes is not
            # obviously correct. Surface it as a deferred decision rather than
            # either forcing it through or silently skipping it.
            if tax_cost > amount * 0.25 and account.tax_treatment == "taxable":
                plan.deferred.append(
                    {
                        "symbol": holding.symbol,
                        "action": "TRIM",
                        "amount": round(amount, 2),
                        "reason": (
                            f"Trimming ${amount:,.0f} of {holding.symbol} would realize "
                            f"${estimate.realized_gain:,.0f} of gain and an estimated "
                            f"${tax_cost:,.0f} of tax -- more than 25% of the trade. The "
                            f"{breach['kind']} concentration is left in place for a human "
                            f"to weigh; consider trimming across tax years instead."
                        ),
                        "estimated_tax": round(tax_cost, 2),
                    }
                )
                continue

            quantity = _round_quantity(amount, holding.price)
            if quantity is None or quantity <= 0:
                continue

            plan.proposals.append(
                TradeProposalDraft(
                    symbol=holding.symbol,
                    side="SELL",
                    account_id=account.id,
                    notional=quantity * holding.price,
                    quantity=quantity,
                    sequence=0,
                    current_weight=holding.market_value / total,
                    target_weight=float(breach["limit"]),
                    addresses_flaw=f"{breach['kind']} concentration in {breach['key']}",
                    rationale=(
                        f"{breach['key']} is {float(breach['weight']):.0%} of the portfolio "
                        f"against a {float(breach['limit']):.0%} policy limit. Selling "
                        f"{quantity:g} shares (${quantity * holding.price:,.0f}) brings it "
                        f"back to the limit rather than exiting the position."
                    ),
                    estimated_tax_cost=tax_cost,
                    tax_detail=estimate.to_dict(),
                )
            )
            filled = quantity * holding.price
            planned[(holding.account_id, holding.symbol)] = (
                planned.get((holding.account_id, holding.symbol), 0.0) + filled
            )
            proceeds += filled

    # --- 2. Raise cash if it is below the floor ------------------------------
    cash_now = view.cash + proceeds
    floor = policy.min_cash_pct * total
    if cash_now < floor:
        shortfall = floor - cash_now
        plan.notes.append(
            f"Cash is ${cash_now:,.0f} against a ${floor:,.0f} policy floor; "
            f"${shortfall:,.0f} of the proposed buys are withheld to restore it."
        )

    # --- 3. Deploy investable cash into underweight asset classes ------------
    #
    # Tracked per account, because cash cannot move between them: an IRA's
    # sale proceeds can only be reinvested inside that IRA. A single
    # portfolio-wide pool over-promises what any one account can fund.
    cash_by_account: Dict[int, float] = {
        account_id: view.available_cash(account_id) for account_id in accounts
    }
    for proposal in plan.proposals:
        if proposal.side == "SELL":
            cash_by_account[proposal.account_id] = (
                cash_by_account.get(proposal.account_id, 0.0) + proposal.notional
            )

    # Hold the policy cash floor back from the largest balance rather than
    # spreading it, so a small account is not made untradeable by its share.
    floor = policy.min_cash_pct * total
    if cash_by_account and floor > 0:
        largest = max(cash_by_account, key=lambda a: cash_by_account[a])
        cash_by_account[largest] = max(0.0, cash_by_account[largest] - floor)

    investable = sum(cash_by_account.values())
    if investable < policy.min_position_notional:
        plan.notes.append(
            f"Only ${investable:,.0f} is investable after reserving the "
            f"{policy.min_cash_pct:.0%} cash floor -- below the ${policy.min_position_notional:,.0f} "
            "minimum trade size, so no purchases are proposed."
        )
        _sequence(plan)
        return plan

    # Gaps are measured against the portfolio *as the sells will leave it*.
    # Sizing buys off pre-trade weights is wrong whenever the plan also
    # trims: having just sold $275k of equity, the equity sleeve is no longer
    # overweight, and the cash raised has to be re-deployed against the
    # allocation that will actually exist.
    drift = _projected_drift(view, policy, planned)
    breached = [d for d in drift if d.breached and d.dollar_gap > 0]
    if breached:
        drift = breached
    else:
        plan.notes.append(
            "Every asset class is within its drift band after the proposed trims; cash is "
            "being deployed to the largest underweights rather than to correct a breach."
        )
        drift = [d for d in drift if d.dollar_gap > 0][:4]

    if not drift:
        plan.notes.append("The portfolio is at its target allocation; no purchases proposed.")
        _sequence(plan)
        return plan

    by_class = _candidates_by_asset_class(candidates or [], reference)
    remaining = investable
    # Purchases planned so far, so each subsequent buy is sized against the
    # portfolio this plan is building rather than the one it started from.
    bought: Dict[str, float] = {}

    for entry in drift:
        if remaining < policy.min_position_notional:
            break
        # Fill each gap in full, largest drift first, until cash runs out.
        # Never more than the gap itself: overshooting a target just creates
        # the opposite breach for the next run to undo.
        #
        # An earlier version scaled each allocation by the *remaining* cash
        # rather than the remaining gaps, which compounds -- every asset class
        # after the first got a fraction of a fraction, and a portfolio with
        # ample cash still finished 30% in cash against a 20% ceiling. When
        # cash is genuinely insufficient, filling the worst breaches first is
        # the right priority anyway.
        allocation = min(entry.dollar_gap, remaining)
        if allocation < policy.min_position_notional:
            continue

        picks = by_class.get(entry.asset_class, [])
        if not picks and candidate_provider is not None:
            # A global screen ranks the whole universe, so a small sleeve like
            # emerging markets can fall below the cut-off and its gap goes
            # unfilled for want of a candidate rather than for want of a
            # suitable one. Ask specifically.
            picks = list(candidate_provider(entry.asset_class) or [])
        if not picks:
            plan.deferred.append(
                {
                    "asset_class": entry.asset_class,
                    "action": "BUY",
                    "amount": round(allocation, 2),
                    "reason": (
                        f"{entry.asset_class} is {abs(entry.drift):.1%} below target but no "
                        f"eligible instrument was found for it in the screened universe. "
                        f"${allocation:,.0f} was left in cash."
                    ),
                }
            )
            continue

        # Cap how many names one asset class gets, so a single underweight
        # does not produce a dozen tiny positions.
        chosen = picks[: max(1, min(3, int(allocation // max(policy.min_position_notional, 1))))]
        per_pick = allocation / len(chosen)

        for pick in chosen:
            symbol = str(pick["ticker"])
            price = float(pick.get("price") or 0.0)
            if price <= 0:
                continue

            account_id = _account_for_buy(accounts, cash_by_account, per_pick, policy)
            if account_id is None:
                continue

            headroom = _position_headroom(view, symbol, policy, total, bought)
            sector = _sector_of(symbol, reference)
            sector_room = _sector_headroom(view, sector, policy, planned, bought, reference)

            amount = min(per_pick, headroom, sector_room, remaining, cash_by_account[account_id])
            if amount < policy.min_position_notional:
                if sector_room < policy.min_position_notional and sector:
                    # The case this catches is a rebalance undoing its own
                    # work: having just trimmed a technology concentration,
                    # buying the highest-scoring name -- which is also a
                    # technology name -- puts the sector straight back over
                    # its limit. Checking only the position cap misses it,
                    # because the individual position is small.
                    plan.deferred.append(
                        {
                            "symbol": symbol,
                            "action": "BUY",
                            "amount": round(per_pick, 2),
                            "reason": (
                                f"Buying {symbol} would push {sector} exposure past the "
                                f"{policy.max_sector_pct:.0%} sector limit, undoing the trim "
                                f"proposed above."
                            ),
                        }
                    )
                elif headroom < policy.min_position_notional:
                    plan.deferred.append(
                        {
                            "symbol": symbol,
                            "action": "BUY",
                            "amount": round(per_pick, 2),
                            "reason": (
                                f"{symbol} is already at or near the "
                                f"{policy.max_position_pct:.0%} position limit; no room to add."
                            ),
                        }
                    )
                continue

            quantity = _round_quantity(amount, price)
            if quantity is None or quantity <= 0:
                continue

            notional = quantity * price
            bought[symbol] = bought.get(symbol, 0.0) + notional
            cash_by_account[account_id] -= notional
            plan.proposals.append(
                TradeProposalDraft(
                    symbol=symbol,
                    side="BUY",
                    account_id=account_id,
                    notional=notional,
                    quantity=quantity,
                    sequence=1,
                    current_weight=view.weights().get(symbol, 0.0),
                    target_weight=entry.target_weight,
                    addresses_flaw=(
                        f"{entry.asset_class} is {abs(entry.drift):.1%} below its "
                        f"{entry.target_weight:.0%} target"
                    ),
                    confidence=pick.get("confidence"),
                    rationale=str(
                        pick.get("rationale")
                        or f"Adds {entry.asset_class} exposure, which is "
                           f"{abs(entry.drift):.1%} below its policy target."
                    ),
                    estimated_tax_cost=0.0,  # buying realizes nothing
                )
            )
            remaining -= notional

        if len(plan.proposals) >= max_proposals:
            plan.notes.append(
                f"Proposal list capped at {max_proposals}; further adjustments will be "
                "proposed on the next run."
            )
            break

    _sequence(plan)
    return plan


def _projected_drift(
    view: PortfolioView, policy: ResolvedPolicy, planned: Dict[tuple, float]
) -> List[DriftEntry]:
    """Drift as it will stand once the planned sells have executed.

    Total value is unchanged by a sale -- the dollars move from a holding into
    cash -- so only the composition shifts. Recomputing here is what stops the
    planner from buying against an overweight it has already cured.
    """
    total = view.total_value
    if total <= 0:
        return []

    values: Dict[str, float] = {}
    freed = 0.0
    for holding in view.holdings:
        sold = planned.get((holding.account_id, holding.symbol), 0.0)
        remaining_value = max(0.0, holding.market_value - sold)
        freed += holding.market_value - remaining_value
        values[holding.asset_class] = values.get(holding.asset_class, 0.0) + remaining_value
    values["cash"] = values.get("cash", 0.0) + view.cash + freed

    entries: List[DriftEntry] = []
    for asset_class in sorted(set(values) | set(policy.target_allocation)):
        current_weight = values.get(asset_class, 0.0) / total
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


def _merge_duplicates(plan: RebalancePlan) -> None:
    """Combine proposals for the same symbol, account and side into one trade.

    A portfolio breaching both its position limit and its sector limit
    generates a trim for each, and although the amounts are correctly netted,
    presenting them as two separate sales of the same stock is wrong in two
    ways: the client reads it as two decisions, and the executor places two
    orders where one would do, paying the spread twice.
    """
    merged: Dict[tuple, TradeProposalDraft] = {}
    for proposal in plan.proposals:
        key = (proposal.account_id, proposal.symbol, proposal.side)
        existing = merged.get(key)
        if existing is None:
            merged[key] = proposal
            continue
        existing.notional += proposal.notional
        if existing.quantity is not None and proposal.quantity is not None:
            existing.quantity = round(existing.quantity + proposal.quantity, 6)
        if existing.estimated_tax_cost is not None and proposal.estimated_tax_cost is not None:
            existing.estimated_tax_cost += proposal.estimated_tax_cost
        # Keep both reasons: the trade serves both constraints, and dropping
        # one would understate why it is being made.
        if proposal.addresses_flaw and proposal.addresses_flaw not in (existing.addresses_flaw or ""):
            existing.addresses_flaw = f"{existing.addresses_flaw}; {proposal.addresses_flaw}"
            existing.rationale = (
                f"{existing.rationale} This trade also addresses: {proposal.addresses_flaw}."
            )
    plan.proposals = list(merged.values())


def _sequence(plan: RebalancePlan) -> None:
    """Merge duplicates, then number so sells always execute before buys."""
    _merge_duplicates(plan)
    for index, proposal in enumerate(sorted(plan.proposals, key=lambda p: (p.side != "SELL",))):
        proposal.sequence = index


def _holdings_for_breach(view: PortfolioView, breach: Dict[str, object]) -> List[HoldingView]:
    """The holdings a given breach is measured over."""
    key = str(breach["key"])
    if breach["kind"] == "position":
        return [h for h in view.holdings if h.symbol == key]
    return [
        h
        for h in view.holdings
        if not view.is_diversified_fund(h) and (h.sector or "Unclassified") == key
    ]


def _trim_targets(
    view: PortfolioView,
    breach: Dict[str, object],
    excess: float,
    already_planned,
) -> List[tuple]:
    """Which holdings to sell from, and how much. Sheltered accounts first.

    Reducing an exposure from an IRA costs nothing in tax; reducing the
    identical exposure from a taxable account costs real money. Where the same
    security is held in both, preferring the shelter is free -- and it is a
    saving the old single-bucket model could not even represent, because it
    had no notion of which account a share sat in.
    """
    matching = _holdings_for_breach(view, breach)
    if not matching:
        return []

    ordered = sorted(
        matching,
        key=lambda h: (
            0 if h.tax_treatment != "taxable" else 1,   # shelter first
            -h.unrealized_gain,                          # then the least gain
        ),
    )

    targets = []
    remaining = excess
    for holding in ordered:
        if remaining <= 0:
            break
        # Never plan to sell more of a holding than remains unsold after
        # earlier breaches took their share of it.
        sellable = holding.market_value - already_planned(holding)
        if sellable <= 0:
            continue
        amount = min(remaining, sellable)
        targets.append((holding, amount))
        remaining -= amount
    return targets


def _position_headroom(
    view: PortfolioView,
    symbol: str,
    policy: ResolvedPolicy,
    total: float,
    bought: Optional[Dict[str, float]] = None,
) -> float:
    """Dollars that can be added to `symbol` before it breaches its cap.

    Sizing without this is how a rebalance meant to fix a concentration
    creates a new one.
    """
    current = view.weights().get(symbol, 0.0) * total
    current += (bought or {}).get(symbol, 0.0)
    return max(0.0, policy.max_position_pct * total - current)


def _sector_of(symbol: str, reference: Dict[str, Security]) -> Optional[str]:
    """The sector a buy would add to, or None for a diversified fund."""
    security = reference.get(symbol.upper())
    if security is None:
        return None
    if security.security_type in ("etf", "fund") and not security.sector:
        return None
    return security.sector


def _sector_headroom(
    view: PortfolioView,
    sector: Optional[str],
    policy: ResolvedPolicy,
    planned_sells: Dict[tuple, float],
    bought: Dict[str, float],
    reference: Dict[str, Security],
) -> float:
    """Dollars addable to `sector` before it breaches the sector limit.

    Accounts for the trims this plan already proposes, so a trim genuinely
    creates room rather than the buy loop treating the pre-trade exposure as
    binding. Broad funds carry no sector and are unconstrained here.
    """
    if sector is None:
        return float("inf")

    invested = view.invested_value
    exposure = 0.0
    for holding in view.holdings:
        if view.is_diversified_fund(holding) or (holding.sector or "Unclassified") != sector:
            continue
        sold = planned_sells.get((holding.account_id, holding.symbol), 0.0)
        exposure += max(0.0, holding.market_value - sold)
        invested -= sold

    for symbol, amount in bought.items():
        if _sector_of(symbol, reference) == sector:
            exposure += amount
        invested += amount

    if invested <= 0:
        return 0.0
    # Adding `x` raises both the numerator and the denominator, so the limit
    # solves to: (exposure + x) / (invested + x) <= cap.
    cap = policy.max_sector_pct
    if cap >= 1.0:
        return float("inf")
    allowed = (cap * invested - exposure) / (1 - cap)
    return max(0.0, allowed)


def _account_for_buy(
    accounts: Dict[int, Account],
    cash_by_account: Dict[int, float],
    amount: float,
    policy: ResolvedPolicy,
) -> Optional[int]:
    """Which account funds this buy, given what each has left.

    Cash does not move between accounts -- an IRA cannot fund a purchase in a
    taxable account -- so choosing on portfolio-wide cash silently strands
    every dollar sitting in the other account. In the run this was found on,
    $123k of IRA sale proceeds went undeployed while a buy was rejected for
    insufficient funds in the brokerage account.

    Prefer an account that can fund the whole amount; failing that, the one
    with the most cash, and let the caller size down to what is there.
    """
    fundable = [
        account_id
        for account_id, available in cash_by_account.items()
        if account_id in accounts and available >= amount
    ]
    if fundable:
        return max(fundable, key=lambda a: cash_by_account[a])

    partial = [
        account_id
        for account_id, available in cash_by_account.items()
        if account_id in accounts and available >= policy.min_position_notional
    ]
    if not partial:
        return None
    return max(partial, key=lambda a: cash_by_account[a])


def _candidates_by_asset_class(
    candidates: Sequence[Dict[str, object]], reference: Dict[str, Security]
) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for candidate in candidates:
        ticker = str(candidate.get("ticker", "")).upper()
        security = reference.get(ticker)
        asset_class = (
            str(candidate.get("asset_class"))
            if candidate.get("asset_class")
            else (security.asset_class if security else "us_equity")
        )
        grouped.setdefault(asset_class, []).append(candidate)
    return grouped
