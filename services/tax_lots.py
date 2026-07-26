"""Tax lot accounting: basis, holding period, realized gains and wash sales.

The previous implementation had a single average cost per holding and treated
any sale in the last 30 days as a wash sale. Both are wrong in ways that
change recommendations:

* **Average cost gives the wrong realized gain for any partial sale.** Sell
  half a position bought across three purchases and the answer depends on
  which shares you sold. That is not a rounding difference — with HIFO versus
  FIFO on a long-held position the realized gain can differ by multiples, and
  the tax on it is real money.

* **The wash-sale rule applies only to losses.** The old check filtered on
  `transaction_type == "SELL"` and never looked at the price, so a *profitable*
  sale blocked re-buying that name for a month. It was a guardrail firing on a
  condition the IRS rule does not describe, and it burned research retries
  doing it.

Three further things the old model could not express, all fixed here:

* **The window is ±30 days, not 30 days after.** A purchase in the 30 days
  *before* a loss sale triggers the rule just as surely. Checking only forward
  misses half the cases.
* **The rule spans accounts, including IRAs.** Repurchasing in an IRA after a
  taxable loss sale disallows the loss *permanently* — there is no basis
  adjustment to recover it later. That is the most expensive version of this
  mistake and the old single-bucket model could not even represent it.
* **A disallowed loss is deferred, not destroyed.** It is added to the basis of
  the replacement lot, and the holding period tacks on. Recording that is the
  difference between warning about wash sales and accounting for them.

Nothing here is tax advice, and the system says so. It is bookkeeping accurate
enough that the advice built on top of it is not wrong for arithmetic reasons.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from db import Account, ClientProfile, Execution, Position, TaxLot, to_float, utcnow
from logging_setup import get_logger

logger = get_logger(__name__)

# IRC §1091: the disallowance window runs 30 days before through 30 days after
# the loss sale -- 61 days inclusive of the sale date.
WASH_SALE_WINDOW_DAYS = 30

# IRC §1222: "more than one year" for long-term treatment. A position bought
# on 1 January and sold on the following 1 January is short-term; it needs one
# more day. Off-by-one here changes the tax rate applied to the whole gain.
LONG_TERM_DAYS = 365

LOT_METHODS = ("HIFO", "FIFO", "LIFO", "MINTAX")


@dataclass
class LotSelection:
    """Which lots a sale would consume, and what it would realize."""

    lot_id: int
    quantity: float
    cost_per_share: float
    acquired_at: datetime
    proceeds: float
    gain: float
    term: str

    @property
    def is_loss(self) -> bool:
        return self.gain < 0


@dataclass
class SaleEstimate:
    """The tax consequence of a proposed sale, before it happens.

    Computed ahead of execution so the rebalancer can weigh a trade's tax cost
    against its portfolio benefit, rather than discovering it afterwards.
    """

    symbol: str
    quantity: float
    proceeds: float
    cost_basis: float
    realized_gain: float
    short_term_gain: float
    long_term_gain: float
    selections: List[LotSelection] = field(default_factory=list)
    wash_sale_risk: bool = False
    wash_sale_note: Optional[str] = None
    shortfall: float = 0.0  # requested quantity beyond what is held

    def estimated_tax(self, short_rate: float = 0.37, long_rate: float = 0.20) -> float:
        """Rough tax owed. Positive means a cost, negative means a benefit.

        Deliberately a blunt estimate at top marginal rates: this is used to
        *rank* trades by tax cost, not to file anything. Pretending to
        precision here -- state tax, NIIT, bracket position, carryforwards --
        would imply an accuracy the system does not have and cannot verify.
        """
        return self.short_term_gain * short_rate + self.long_term_gain * long_rate

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "quantity": round(self.quantity, 6),
            "proceeds": round(self.proceeds, 2),
            "cost_basis": round(self.cost_basis, 2),
            "realized_gain": round(self.realized_gain, 2),
            "short_term_gain": round(self.short_term_gain, 2),
            "long_term_gain": round(self.long_term_gain, 2),
            "estimated_tax": round(self.estimated_tax(), 2),
            "lots_consumed": len(self.selections),
            "wash_sale_risk": self.wash_sale_risk,
            "wash_sale_note": self.wash_sale_note,
        }


# --- Reading lots ------------------------------------------------------------


def open_lots(db: Session, account_id: int, symbol: Optional[str] = None) -> List[TaxLot]:
    """Lots with shares remaining, oldest first."""
    query = db.query(TaxLot).filter(
        TaxLot.account_id == account_id,
        TaxLot.remaining_quantity > 0,
    )
    if symbol:
        query = query.filter(TaxLot.symbol == symbol.upper())
    return query.order_by(TaxLot.acquired_at.asc()).all()


def term_for(acquired_at: datetime, disposed_at: Optional[datetime] = None) -> str:
    disposed_at = disposed_at or utcnow()
    return "long" if (disposed_at - acquired_at).days > LONG_TERM_DAYS else "short"


def select_lots(
    lots: Sequence[TaxLot],
    quantity: float,
    price: float,
    *,
    method: str = "HIFO",
    as_of: Optional[datetime] = None,
) -> Tuple[List[LotSelection], float]:
    """Choose which lots a sale consumes. Returns (selections, shortfall).

    The method is a real decision, not a preference:

    * **HIFO** sells the highest-cost shares first, minimizing the realized
      gain. The sensible default in a taxable account.
    * **FIFO** is the IRS default absent a specific identification, and what
      some custodians enforce.
    * **MINTAX** goes further than HIFO by preferring long-term gains over
      short-term ones at the same dollar gain, since the rate difference
      usually dominates the basis difference.
    """
    as_of = as_of or utcnow()
    method = method.upper()
    if method not in LOT_METHODS:
        logger.warning("Unknown lot method %r; falling back to HIFO.", method)
        method = "HIFO"

    available = [lot for lot in lots if to_float(lot.remaining_quantity) > 0]

    if method == "FIFO":
        ordered = sorted(available, key=lambda lot: lot.acquired_at)
    elif method == "LIFO":
        ordered = sorted(available, key=lambda lot: lot.acquired_at, reverse=True)
    elif method == "MINTAX":
        # Losses first (they offset gains), then long-term gains, then
        # short-term. Within each group, highest cost basis first.
        def rank(lot: TaxLot) -> Tuple[int, float]:
            cost = to_float(lot.cost_per_share)
            is_long = term_for(lot.acquired_at, as_of) == "long"
            if cost > price:
                group = 0  # a loss
            elif is_long:
                group = 1
            else:
                group = 2
            return (group, -cost)

        ordered = sorted(available, key=rank)
    else:  # HIFO
        ordered = sorted(available, key=lambda lot: to_float(lot.cost_per_share), reverse=True)

    selections: List[LotSelection] = []
    remaining = quantity
    for lot in ordered:
        if remaining <= 1e-9:
            break
        take = min(remaining, to_float(lot.remaining_quantity))
        if take <= 0:
            continue
        cost = to_float(lot.cost_per_share)
        proceeds = take * price
        selections.append(
            LotSelection(
                lot_id=lot.id,
                quantity=take,
                cost_per_share=cost,
                acquired_at=lot.acquired_at,
                proceeds=proceeds,
                gain=proceeds - take * cost,
                term=term_for(lot.acquired_at, as_of),
            )
        )
        remaining -= take

    return selections, max(0.0, remaining)


def estimate_sale(
    db: Session,
    account: Account,
    symbol: str,
    quantity: float,
    price: float,
    *,
    method: str = "HIFO",
    as_of: Optional[datetime] = None,
) -> SaleEstimate:
    """What selling `quantity` shares would realize, before doing it."""
    as_of = as_of or utcnow()
    symbol = symbol.upper()
    lots = open_lots(db, account.id, symbol)
    selections, shortfall = select_lots(lots, quantity, price, method=method, as_of=as_of)

    proceeds = sum(s.proceeds for s in selections)
    basis = sum(s.quantity * s.cost_per_share for s in selections)
    short_gain = sum(s.gain for s in selections if s.term == "short")
    long_gain = sum(s.gain for s in selections if s.term == "long")

    estimate = SaleEstimate(
        symbol=symbol,
        quantity=sum(s.quantity for s in selections),
        proceeds=proceeds,
        cost_basis=basis,
        realized_gain=proceeds - basis,
        short_term_gain=short_gain,
        long_term_gain=long_gain,
        selections=selections,
        shortfall=shortfall,
    )

    # A sale at a loss creates a *forward* wash-sale exposure: repurchasing
    # within 30 days would disallow it. Only meaningful in a taxable account.
    if estimate.realized_gain < 0 and account.tax_treatment == "taxable":
        recent = _recent_purchases(db, account.client_id, symbol, as_of, WASH_SALE_WINDOW_DAYS)
        if recent:
            estimate.wash_sale_risk = True
            estimate.wash_sale_note = (
                f"{symbol} was purchased within the last {WASH_SALE_WINDOW_DAYS} days, so "
                f"realizing this loss would trigger the wash-sale rule on those shares."
            )

    return estimate


# --- Wash sales --------------------------------------------------------------


@dataclass
class WashSaleFinding:
    symbol: str
    blocked: bool
    reason: str
    loss_amount: float
    sold_at: datetime
    days_remaining: int
    permanent: bool = False  # true when the replacement would land in an IRA

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "blocked": self.blocked,
            "reason": self.reason,
            "loss_amount": round(self.loss_amount, 2),
            "sold_at": self.sold_at.isoformat(),
            "days_remaining": self.days_remaining,
            "permanent": self.permanent,
        }


def _recent_purchases(
    db: Session, client_id: int, symbol: str, as_of: datetime, days: int
) -> List[Execution]:
    """Buys of `symbol` across *all* the client's accounts within the window.

    Across all accounts deliberately: the rule follows the taxpayer, not the
    account. A repurchase in an IRA after a taxable loss sale still triggers
    it, and does so in the worst way.
    """
    account_ids = [a.id for a in db.query(Account.id).filter(Account.client_id == client_id).all()]
    if not account_ids:
        return []
    return (
        db.query(Execution)
        .filter(
            Execution.account_id.in_(account_ids),
            Execution.symbol == symbol.upper(),
            Execution.side == "BUY",
            Execution.executed_at >= as_of - timedelta(days=days),
            Execution.executed_at <= as_of + timedelta(days=days),
        )
        .all()
    )


def check_wash_sale(
    db: Session,
    client_id: int,
    symbol: str,
    *,
    target_account: Optional[Account] = None,
    as_of: Optional[datetime] = None,
) -> Optional[WashSaleFinding]:
    """Would buying `symbol` now trigger a wash sale? None if not.

    Looks for a *loss* disposal of the same symbol within the trailing 30 days
    in a taxable account. This is the check the previous version got wrong in
    three separate ways -- it ignored whether the sale was at a loss, it looked
    at a transaction log rather than at realized results, and it had no notion
    of which account either side happened in.
    """
    as_of = as_of or utcnow()
    symbol = symbol.upper()
    window_start = as_of - timedelta(days=WASH_SALE_WINDOW_DAYS)

    taxable_account_ids = [
        a.id
        for a in db.query(Account.id)
        .filter(Account.client_id == client_id, Account.tax_treatment == "taxable")
        .all()
    ]
    if not taxable_account_ids:
        # No taxable account means no deductible loss to disallow.
        return None

    loss_lots = (
        db.query(TaxLot)
        .filter(
            TaxLot.account_id.in_(taxable_account_ids),
            TaxLot.symbol == symbol,
            TaxLot.disposed_at.isnot(None),
            TaxLot.disposed_at >= window_start,
            TaxLot.disposed_at <= as_of,
            TaxLot.realized_gain < 0,
        )
        .all()
    )
    if not loss_lots:
        return None

    total_loss = sum(to_float(lot.realized_gain) for lot in loss_lots)
    most_recent = max(lot.disposed_at for lot in loss_lots)
    days_remaining = WASH_SALE_WINDOW_DAYS - (as_of - most_recent).days

    permanent = bool(target_account and target_account.tax_treatment != "taxable")
    if permanent:
        reason = (
            f"{symbol} was sold at a loss of ${abs(total_loss):,.0f} "
            f"{(as_of - most_recent).days} days ago in a taxable account. Repurchasing it "
            f"in a {target_account.account_type} account would disallow that loss "
            f"permanently -- there is no basis adjustment to recover it later. "
            f"Blocked for {days_remaining} more day(s)."
        )
    else:
        reason = (
            f"{symbol} was sold at a loss of ${abs(total_loss):,.0f} "
            f"{(as_of - most_recent).days} days ago. Repurchasing within "
            f"{WASH_SALE_WINDOW_DAYS} days disallows the loss for now; it would be added "
            f"to the basis of the new shares instead. Blocked for {days_remaining} more "
            f"day(s)."
        )

    return WashSaleFinding(
        symbol=symbol,
        blocked=True,
        reason=reason,
        loss_amount=total_loss,
        sold_at=most_recent,
        days_remaining=max(0, days_remaining),
        permanent=permanent,
    )


# --- Writing lots ------------------------------------------------------------


def record_purchase(
    db: Session,
    account: Account,
    symbol: str,
    quantity: float,
    price: float,
    *,
    executed_at: Optional[datetime] = None,
    execution_id: Optional[int] = None,
) -> TaxLot:
    """Open a lot and apply any wash-sale basis adjustment it inherits.

    If this purchase replaces shares sold at a loss inside the window, the
    disallowed loss is added to this lot's basis. That is the mechanism by
    which the deduction is deferred rather than lost, and skipping it would
    overstate the gain on this lot when it is eventually sold.
    """
    executed_at = executed_at or utcnow()
    symbol = symbol.upper()
    cost_per_share = price

    finding = check_wash_sale(db, account.client_id, symbol, target_account=account, as_of=executed_at)
    disallowed = None
    replaced_lot = None
    if finding and not finding.permanent and quantity > 0:
        # Only the replaced shares carry the adjustment. Buying back fewer
        # shares than were sold defers only a proportional share of the loss.
        replaced_lot = (
            db.query(TaxLot)
            .filter(
                TaxLot.symbol == symbol,
                TaxLot.disposed_at.isnot(None),
                TaxLot.realized_gain < 0,
                TaxLot.disposed_at >= executed_at - timedelta(days=WASH_SALE_WINDOW_DAYS),
            )
            .order_by(TaxLot.disposed_at.desc())
            .first()
        )
        if replaced_lot is not None:
            sold_quantity = to_float(replaced_lot.quantity)
            matched = min(quantity, sold_quantity) if sold_quantity > 0 else 0.0
            if matched > 0 and sold_quantity > 0:
                disallowed = abs(to_float(replaced_lot.realized_gain)) * (matched / sold_quantity)
                cost_per_share = price + (disallowed / quantity)
                replaced_lot.wash_sale_disallowed = Decimal(str(round(disallowed, 6)))
                logger.info(
                    "Wash sale on %s: $%.2f of loss deferred into the new lot's basis.",
                    symbol, disallowed,
                )

    lot = TaxLot(
        org_id=account.org_id,
        account_id=account.id,
        symbol=symbol,
        quantity=Decimal(str(quantity)),
        remaining_quantity=Decimal(str(quantity)),
        cost_per_share=Decimal(str(round(cost_per_share, 6))),
        acquired_at=executed_at,
        execution_id=execution_id,
    )
    db.add(lot)
    db.flush()

    if replaced_lot is not None and disallowed:
        replaced_lot.replacement_lot_id = lot.id

    _rebuild_position(db, account, symbol)
    return lot


def record_sale(
    db: Session,
    account: Account,
    symbol: str,
    quantity: float,
    price: float,
    *,
    method: str = "HIFO",
    executed_at: Optional[datetime] = None,
    execution_id: Optional[int] = None,
) -> SaleEstimate:
    """Consume lots and write the realized result. Returns what was realized.

    Raises if the account does not hold enough shares. Permitting a sale
    beyond the position would create a negative holding, which this system has
    no concept of and no ability to unwind.
    """
    executed_at = executed_at or utcnow()
    symbol = symbol.upper()
    lots = open_lots(db, account.id, symbol)
    by_id = {lot.id: lot for lot in lots}
    selections, shortfall = select_lots(lots, quantity, price, method=method, as_of=executed_at)

    if shortfall > 1e-6:
        held = sum(to_float(lot.remaining_quantity) for lot in lots)
        raise ValueError(
            f"Cannot sell {quantity:g} {symbol}: the account holds {held:g}. "
            "Short positions are not supported."
        )

    for selection in selections:
        lot = by_id[selection.lot_id]
        remaining = to_float(lot.remaining_quantity) - selection.quantity
        lot.remaining_quantity = Decimal(str(round(max(0.0, remaining), 8)))

        # Accumulate rather than overwrite: a lot sold across two trades has
        # realized gain from both, and overwriting would silently discard the
        # earlier one from the tax record.
        prior_gain = to_float(lot.realized_gain)
        prior_proceeds = to_float(lot.proceeds)
        lot.realized_gain = Decimal(str(round(prior_gain + selection.gain, 6)))
        lot.proceeds = Decimal(str(round(prior_proceeds + selection.proceeds, 6)))
        lot.term = selection.term
        if to_float(lot.remaining_quantity) <= 1e-9:
            lot.disposed_at = executed_at
        else:
            # Partially sold lots must still be findable by the wash-sale
            # lookback, which keys on disposed_at.
            lot.disposed_at = executed_at

    proceeds = sum(s.proceeds for s in selections)
    basis = sum(s.quantity * s.cost_per_share for s in selections)
    result = SaleEstimate(
        symbol=symbol,
        quantity=sum(s.quantity for s in selections),
        proceeds=proceeds,
        cost_basis=basis,
        realized_gain=proceeds - basis,
        short_term_gain=sum(s.gain for s in selections if s.term == "short"),
        long_term_gain=sum(s.gain for s in selections if s.term == "long"),
        selections=selections,
    )

    db.flush()
    _rebuild_position(db, account, symbol)
    return result


def _rebuild_position(db: Session, account: Account, symbol: str) -> Optional[Position]:
    """Recompute the position rollup from its open lots.

    Derived, never incremented. An incremental update that drifts from the
    lots is indistinguishable from a correct one until somebody reconciles,
    and by then the wrong number has been used to size trades.
    """
    symbol = symbol.upper()
    lots = open_lots(db, account.id, symbol)
    total_quantity = sum(to_float(lot.remaining_quantity) for lot in lots)
    total_cost = sum(to_float(lot.remaining_quantity) * to_float(lot.cost_per_share) for lot in lots)

    position = (
        db.query(Position)
        .filter(Position.account_id == account.id, Position.symbol == symbol)
        .first()
    )

    if total_quantity <= 1e-9:
        if position is not None:
            db.delete(position)
        return None

    average_cost = total_cost / total_quantity if total_quantity else 0.0
    if position is None:
        position = Position(
            org_id=account.org_id,
            account_id=account.id,
            symbol=symbol,
            quantity=Decimal(str(round(total_quantity, 8))),
            average_cost=Decimal(str(round(average_cost, 6))),
        )
        db.add(position)
    else:
        position.quantity = Decimal(str(round(total_quantity, 8)))
        position.average_cost = Decimal(str(round(average_cost, 6)))
    db.flush()
    return position


def rebuild_all_positions(db: Session, account: Account) -> int:
    """Reconcile every position in an account against its lots."""
    symbols = {
        lot.symbol
        for lot in db.query(TaxLot.symbol).filter(TaxLot.account_id == account.id).all()
    }
    for symbol in symbols:
        _rebuild_position(db, account, symbol)
    return len(symbols)


# --- Analysis ----------------------------------------------------------------


@dataclass
class UnrealizedPosition:
    symbol: str
    account_id: int
    quantity: float
    cost_basis: float
    market_value: float
    unrealized_gain: float
    short_term_gain: float
    long_term_gain: float
    return_pct: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "account_id": self.account_id,
            "quantity": round(self.quantity, 6),
            "cost_basis": round(self.cost_basis, 2),
            "market_value": round(self.market_value, 2),
            "unrealized_gain": round(self.unrealized_gain, 2),
            "short_term_gain": round(self.short_term_gain, 2),
            "long_term_gain": round(self.long_term_gain, 2),
            "return_pct": round(self.return_pct, 4) if self.return_pct is not None else None,
        }


def unrealized_positions(
    db: Session,
    account: Account,
    prices: Dict[str, float],
    *,
    as_of: Optional[datetime] = None,
) -> List[UnrealizedPosition]:
    """Per-symbol unrealized gain, split by holding period.

    Symbols with no price are omitted rather than valued at zero -- the same
    rule the market data layer applies, for the same reason: an unknown value
    and a zero value lead to opposite decisions.
    """
    as_of = as_of or utcnow()
    by_symbol: Dict[str, List[TaxLot]] = {}
    for lot in open_lots(db, account.id):
        by_symbol.setdefault(lot.symbol, []).append(lot)

    output: List[UnrealizedPosition] = []
    for symbol, lots in by_symbol.items():
        price = prices.get(symbol)
        if price is None or price <= 0:
            logger.debug("No price for %s; omitted from unrealized gain analysis.", symbol)
            continue
        quantity = cost = short = long_ = 0.0
        for lot in lots:
            qty = to_float(lot.remaining_quantity)
            lot_cost = qty * to_float(lot.cost_per_share)
            gain = qty * price - lot_cost
            quantity += qty
            cost += lot_cost
            if term_for(lot.acquired_at, as_of) == "long":
                long_ += gain
            else:
                short += gain
        market_value = quantity * price
        output.append(
            UnrealizedPosition(
                symbol=symbol,
                account_id=account.id,
                quantity=quantity,
                cost_basis=cost,
                market_value=market_value,
                unrealized_gain=market_value - cost,
                short_term_gain=short,
                long_term_gain=long_,
                return_pct=(market_value / cost - 1.0) if cost > 0 else None,
            )
        )
    return sorted(output, key=lambda p: -p.market_value)


def harvestable_losses(
    db: Session,
    client: ClientProfile,
    prices: Dict[str, float],
    *,
    min_loss: float = 500.0,
    as_of: Optional[datetime] = None,
) -> List[Dict[str, object]]:
    """Positions sitting at a loss that could be realized to offset gains.

    Restricted to taxable accounts, because harvesting a loss inside an IRA
    accomplishes nothing — there is no deduction to take. The previous system
    reported harvest candidates without knowing the account type, so a
    meaningful fraction of its tax "opportunities" were not opportunities.
    """
    as_of = as_of or utcnow()
    candidates: List[Dict[str, object]] = []

    accounts = (
        db.query(Account)
        .filter(
            Account.client_id == client.id,
            Account.tax_treatment == "taxable",
            Account.is_active.is_(True),
        )
        .all()
    )

    for account in accounts:
        for position in unrealized_positions(db, account, prices, as_of=as_of):
            if position.unrealized_gain >= -abs(min_loss):
                continue
            finding = check_wash_sale(db, client.id, position.symbol, as_of=as_of)
            candidates.append(
                {
                    "symbol": position.symbol,
                    "account_id": account.id,
                    "account_name": account.name,
                    "unrealized_loss": round(position.unrealized_gain, 2),
                    "short_term_loss": round(min(0.0, position.short_term_gain), 2),
                    "long_term_loss": round(min(0.0, position.long_term_gain), 2),
                    "market_value": round(position.market_value, 2),
                    # Harvesting requires *selling*; an existing wash-sale
                    # flag on this symbol means a recent repurchase would
                    # already have disallowed part of it.
                    "wash_sale_conflict": finding.reason if finding else None,
                }
            )

    return sorted(candidates, key=lambda c: c["unrealized_loss"])


def realized_gains(
    db: Session,
    client: ClientProfile,
    *,
    since: Optional[datetime] = None,
    taxable_only: bool = True,
) -> Dict[str, float]:
    """Realized gains year to date, split short versus long.

    Feeds the policy's short-term gain budget: a rebalance that would push
    realized short-term gains past the client's tolerance should know that
    before proposing the trade, not after.
    """
    since = since or datetime(utcnow().year, 1, 1)
    query = db.query(Account).filter(Account.client_id == client.id)
    if taxable_only:
        query = query.filter(Account.tax_treatment == "taxable")
    account_ids = [a.id for a in query.all()]
    if not account_ids:
        return {"short_term": 0.0, "long_term": 0.0, "total": 0.0, "wash_sale_disallowed": 0.0}

    lots = (
        db.query(TaxLot)
        .filter(
            TaxLot.account_id.in_(account_ids),
            TaxLot.disposed_at.isnot(None),
            TaxLot.disposed_at >= since,
        )
        .all()
    )

    short = sum(to_float(lot.realized_gain) for lot in lots if lot.term == "short")
    long_ = sum(to_float(lot.realized_gain) for lot in lots if lot.term == "long")
    disallowed = sum(to_float(lot.wash_sale_disallowed) for lot in lots)

    return {
        "short_term": round(short, 2),
        "long_term": round(long_, 2),
        "total": round(short + long_, 2),
        # Reported separately because it is *not* deductible this year, and a
        # net figure that quietly includes it overstates the deduction.
        "wash_sale_disallowed": round(disallowed, 2),
        # The number that actually matters for a gain budget. `total` includes
        # losses the wash-sale rule deferred, so a caller comparing `total`
        # against a tolerance would believe it had more room than it does.
        "deductible_total": round(short + long_ + disallowed, 2),
    }


def seed_lots_from_positions(
    db: Session,
    account: Account,
    holdings: Iterable[Tuple[str, float, float, Optional[datetime]]],
) -> int:
    """Create opening lots during onboarding or a data import.

    Each tuple is (symbol, quantity, cost_per_share, acquired_at). An unknown
    acquisition date defaults to today, which makes the position short-term --
    the conservative assumption, since wrongly claiming long-term treatment
    understates the tax on a sale.
    """
    created = 0
    for symbol, quantity, cost, acquired in holdings:
        if quantity <= 0:
            continue
        lot = TaxLot(
            org_id=account.org_id,
            account_id=account.id,
            symbol=symbol.upper(),
            quantity=Decimal(str(quantity)),
            remaining_quantity=Decimal(str(quantity)),
            cost_per_share=Decimal(str(cost)),
            acquired_at=acquired or utcnow(),
        )
        db.add(lot)
        created += 1
    db.flush()
    for symbol in {h[0].upper() for h in holdings}:
        _rebuild_position(db, account, symbol)
    return created
