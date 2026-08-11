"""Order lifecycle and simulated execution.

There is no live brokerage adapter here, and that is a deliberate scope
decision rather than an unfinished one: connecting real money to an LLM-driven
process requires a custodian relationship, a compliance regime and an
operational failure plan that no amount of code substitutes for. What this
does provide is the full state machine around an order, executed against a
simulator honest enough that the numbers mean something —

* fills cross the spread rather than printing at the mid,
* commissions are charged,
* proceeds settle on T+1 and cash is held until they do,
* every fill writes tax lots, so the next run's basis and holding periods
  reflect what happened.

The consequence is that a rebalance actually *changes* the portfolio, and the
next analysis sees a portfolio that moved — which is what makes performance
measurement and recommendation scoring possible at all. A system whose
recommendations never touch state can never be evaluated.

`TRADING_ENABLED=false` blocks execution entirely while leaving proposal
generation intact, which is the correct posture for a deployment that wants
advice without letting an automated process mutate positions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

import numpy as np
from sqlalchemy.orm import Session

from config import settings
from db import Account, Execution, Order, Position, to_float, utcnow
from logging_setup import get_logger
from metrics import order_notional, orders_placed
from services import tax_lots
from services.audit import Action, record as record_audit
from services.market_data import get_quotes

logger = get_logger(__name__)


class OrderRejected(Exception):
    """The order cannot be placed. Carries a client-safe reason."""


@dataclass
class Fill:
    symbol: str
    side: str
    quantity: float
    price: float
    gross_amount: float
    commission: float
    slippage_cost: float
    net_amount: float
    venue: str = "SIMULATED"
    broker_execution_id: Optional[str] = None


class BrokerAdapter(ABC):
    """Anything that can fill an order.

    An interface with one implementation, kept because the boundary is where
    the assumptions live: everything above it is written against "an order was
    filled at some price with some cost", not against the simulator's specific
    arithmetic. Adding a real adapter later is then a new class rather than a
    rewrite of the order lifecycle.
    """

    name: str = "abstract"

    @abstractmethod
    def execute(self, order: Order, reference_price: float) -> Fill:
        ...


class SimulatedBroker(BrokerAdapter):
    """Fills immediately at a modelled price.

    The realism that matters is not the tick-by-tick microstructure — it is
    that trading is not free. A simulator that fills at the mid with no
    commission makes every rebalance look profitable and makes a strategy that
    trades constantly look better than one that does not, which is exactly
    backwards.
    """

    name = "simulated"

    def __init__(
        self,
        slippage_bps: Optional[float] = None,
        commission_bps: Optional[float] = None,
        commission_min: Optional[float] = None,
    ):
        self.slippage_bps = settings.SIM_SLIPPAGE_BPS if slippage_bps is None else slippage_bps
        self.commission_bps = (
            settings.SIM_COMMISSION_BPS if commission_bps is None else commission_bps
        )
        self.commission_min = (
            settings.SIM_COMMISSION_MIN_USD if commission_min is None else commission_min
        )

    def execute(self, order: Order, reference_price: float) -> Fill:
        quantity = to_float(order.quantity)
        if quantity <= 0:
            raise OrderRejected("Order quantity must be positive.")
        if reference_price <= 0:
            raise OrderRejected(f"No usable reference price for {order.symbol}.")

        # Slippage always works against the order: buys fill above the
        # reference, sells below. Modelling it as symmetric noise would let it
        # average out to zero over many trades, which is not how crossing a
        # spread works.
        drift = self.slippage_bps / 10_000.0
        fill_price = reference_price * (1 + drift) if order.side == "BUY" else reference_price * (1 - drift)

        if order.order_type == "limit" and order.limit_price is not None:
            limit = to_float(order.limit_price)
            # A limit that the modelled fill price violates does not fill.
            # Filling it anyway would make limit orders meaningless and let a
            # backtest claim prices that were never available.
            if order.side == "BUY" and fill_price > limit:
                raise OrderRejected(
                    f"Limit ${limit:,.2f} is below the modelled fill of ${fill_price:,.2f}; "
                    "the order would not have executed."
                )
            if order.side == "SELL" and fill_price < limit:
                raise OrderRejected(
                    f"Limit ${limit:,.2f} is above the modelled fill of ${fill_price:,.2f}; "
                    "the order would not have executed."
                )

        gross = quantity * fill_price
        commission = max(self.commission_min, gross * self.commission_bps / 10_000.0)
        slippage_cost = abs(fill_price - reference_price) * quantity

        # Net is what leaves or enters the account: a buy costs the gross plus
        # commission, a sell returns the gross minus it.
        net = gross + commission if order.side == "BUY" else gross - commission

        return Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=fill_price,
            gross_amount=gross,
            commission=commission,
            slippage_cost=slippage_cost,
            net_amount=net,
            venue="SIMULATED",
            broker_execution_id=f"sim-{order.id}",
        )


def get_broker() -> BrokerAdapter:
    return SimulatedBroker()


def estimate_trading_cost(notional: float, side: str) -> float:
    """What a trade of this size is expected to cost, before it is placed.

    The same slippage and commission the simulator charges at fill time, so a
    plan and its execution cannot disagree about the cost of trading. They did:
    the planner credited a sell's full gross notional and spent it on buys,
    while execution filled the sell 5 bps below the reference and the buy 5 bps
    above, plus commission on both. Sized off gross, the buys could overdraw at
    fill time -- and the only cost a human saw before approving was the tax.

    Returns a positive number of dollars for either side.
    """
    if notional <= 0:
        return 0.0
    slippage = notional * settings.SIM_SLIPPAGE_BPS / 10_000.0
    commission = max(settings.SIM_COMMISSION_MIN_USD, notional * settings.SIM_COMMISSION_BPS / 10_000.0)
    return slippage + commission


def net_sale_proceeds(notional: float) -> float:
    """Cash a sale of `notional` is expected to actually deliver."""
    return max(0.0, notional - estimate_trading_cost(notional, "SELL"))


def gross_purchase_cost(notional: float) -> float:
    """Cash a purchase of `notional` is expected to actually consume."""
    return notional + estimate_trading_cost(notional, "BUY")


def settlement_date(executed_at) -> "object":
    """T+1 in business days.

    Calendar days would settle a Friday trade on Saturday. Exchange holidays
    are not modelled, so this can be a day optimistic around one -- which is
    the safe direction for a cash hold, since it releases later rather than
    sooner.
    """
    return (
        np.busday_offset(executed_at.date(), settings.SETTLEMENT_DAYS, roll="forward")
        .astype("datetime64[s]")
        .astype(object)
    )


# --- Order lifecycle ---------------------------------------------------------


def create_order(
    db: Session,
    account: Account,
    *,
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    limit_price: Optional[float] = None,
    run_id: Optional[str] = None,
    proposal_id: Optional[int] = None,
    created_by_user_id: Optional[int] = None,
    idempotency_key: Optional[str] = None,
    status: str = "draft",
) -> Order:
    """Create an order after pre-trade checks. Does not execute it.

    The checks here are the ones that must happen before anything is
    recorded — sufficient shares, sufficient cash, a sane size. Discovering
    mid-execution that the account cannot fund a trade leaves a half-applied
    state that has to be unwound by hand.
    """
    symbol = symbol.upper().strip()
    side = side.upper().strip()
    if side not in ("BUY", "SELL"):
        raise OrderRejected(f"Unknown order side {side!r}.")
    if quantity <= 0:
        raise OrderRejected("Order quantity must be positive.")

    if idempotency_key:
        existing = (
            db.query(Order)
            .filter(Order.org_id == account.org_id, Order.idempotency_key == idempotency_key)
            .first()
        )
        if existing is not None:
            # A retried submission returns the original rather than doubling
            # the position. Network retries are routine; duplicated trades are
            # not recoverable by an apology.
            logger.info("Idempotent replay of order %s for key %s", existing.id, idempotency_key)
            return existing

    if not settings.ALLOW_FRACTIONAL_SHARES and quantity != int(quantity):
        raise OrderRejected(
            f"Fractional shares are disabled; {quantity:g} {symbol} is not a whole number."
        )

    quotes = get_quotes([symbol])
    quote = quotes.get(symbol)
    if quote is None:
        raise OrderRejected(
            f"No current price is available for {symbol}, so this order cannot be sized "
            "or funded safely."
        )
    notional = quantity * quote.price

    if notional < settings.MIN_ORDER_NOTIONAL_USD:
        raise OrderRejected(
            f"${notional:,.2f} is below the ${settings.MIN_ORDER_NOTIONAL_USD:,.0f} minimum "
            "order size; the trading cost would exceed the benefit."
        )

    if side == "SELL":
        position = (
            db.query(Position)
            .filter(Position.account_id == account.id, Position.symbol == symbol)
            .first()
        )
        held = to_float(position.quantity) if position else 0.0
        # Shares already committed to other open sells cannot be sold twice.
        committed = _open_sell_quantity(db, account.id, symbol)
        if quantity > held - committed + 1e-9:
            raise OrderRejected(
                f"Cannot sell {quantity:g} {symbol}: the account holds {held:g} with "
                f"{committed:g} already committed to open orders."
            )
    else:
        available = to_float(account.cash_balance) - to_float(account.pending_cash_hold)
        if notional > available + 1e-6:
            raise OrderRejected(
                f"Insufficient settled cash in {account.name}: ${notional:,.2f} required, "
                f"${available:,.2f} available."
            )

    order = Order(
        org_id=account.org_id,
        client_id=account.client_id,
        account_id=account.id,
        proposal_id=proposal_id,
        run_id=run_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=Decimal(str(quantity)),
        limit_price=Decimal(str(limit_price)) if limit_price is not None else None,
        status=status,
        created_by_user_id=created_by_user_id,
        idempotency_key=idempotency_key,
    )

    if side == "BUY":
        # Reserve the cash now, not at execution. A rebalance creates a dozen
        # orders before executing any of them; without a reservation each one
        # validates against the same untouched balance, the batch overdraws
        # the account, and the portfolio appears to gain the overdraft out of
        # thin air. A margin of headroom covers the modelled slippage and
        # commission that the fill will add on top of the quote.
        reserve = notional * 1.01
        order.reserved_cash = Decimal(str(round(reserve, 6)))
        account.pending_cash_hold = Decimal(
            str(round(to_float(account.pending_cash_hold) + reserve, 6))
        )

    db.add(order)
    db.flush()

    record_audit(
        db,
        org_id=account.org_id,
        action=Action.ORDER_CREATED,
        user_id=created_by_user_id,
        entity_type="order",
        entity_id=order.id,
        detail={
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "estimated_notional": round(notional, 2),
            "account_id": account.id,
        },
    )
    return order


def _release_reservation(account: Account, order: Order) -> None:
    """Give back the cash a buy order was holding. Idempotent."""
    reserved = to_float(order.reserved_cash)
    if reserved <= 0:
        return
    account.pending_cash_hold = Decimal(
        str(round(max(0.0, to_float(account.pending_cash_hold) - reserved), 6))
    )
    order.reserved_cash = Decimal("0")


def _open_sell_quantity(db: Session, account_id: int, symbol: str) -> float:
    open_orders = (
        db.query(Order)
        .filter(
            Order.account_id == account_id,
            Order.symbol == symbol,
            Order.side == "SELL",
            Order.status.in_(("draft", "pending_approval", "approved", "submitted")),
        )
        .all()
    )
    return sum(to_float(o.quantity) - to_float(o.filled_quantity) for o in open_orders)


def submit_order(
    db: Session,
    order: Order,
    *,
    approved_by_user_id: Optional[int] = None,
    broker: Optional[BrokerAdapter] = None,
) -> Order:
    """Execute an approved order and apply every consequence of the fill.

    All of it happens in the caller's transaction: the execution row, the tax
    lots, the position rollup and the cash movement either all commit or none
    do. A fill recorded without its cash movement is a portfolio that does not
    balance, and there is no safe way to reconcile that after the fact.
    """
    if not settings.TRADING_ENABLED:
        raise OrderRejected(
            "Trading is disabled on this deployment (TRADING_ENABLED=false). The order was "
            "recorded but not executed."
        )
    if order.status in ("filled", "cancelled", "rejected"):
        raise OrderRejected(f"Order {order.id} is already {order.status}.")

    account = db.query(Account).filter(Account.id == order.account_id).first()
    if account is None:
        raise OrderRejected("The order's account no longer exists.")

    quotes = get_quotes([order.symbol])
    quote = quotes.get(order.symbol)
    if quote is None:
        order.status = "failed"
        order.failure_reason = "No reference price available at execution time."
        _release_reservation(account, order)
        raise OrderRejected(order.failure_reason)

    broker = broker or get_broker()
    try:
        fill = broker.execute(order, quote.price)
    except OrderRejected as exc:
        order.status = "failed"
        order.failure_reason = str(exc)[:500]
        # An order that will never fill must not keep holding its cash.
        _release_reservation(account, order)
        orders_placed.labels(order.side, "failed").inc()
        db.flush()
        raise

    executed_at = utcnow()
    execution = Execution(
        org_id=order.org_id,
        order_id=order.id,
        account_id=account.id,
        symbol=fill.symbol,
        side=fill.side,
        quantity=Decimal(str(fill.quantity)),
        price=Decimal(str(round(fill.price, 6))),
        gross_amount=Decimal(str(round(fill.gross_amount, 6))),
        commission=Decimal(str(round(fill.commission, 6))),
        slippage_cost=Decimal(str(round(fill.slippage_cost, 6))),
        net_amount=Decimal(str(round(fill.net_amount, 6))),
        executed_at=executed_at,
        settles_on=settlement_date(executed_at),
        broker_execution_id=fill.broker_execution_id,
        venue=fill.venue,
    )
    db.add(execution)
    db.flush()

    if fill.side == "BUY":
        tax_lots.record_purchase(
            db, account, fill.symbol, fill.quantity, fill.price,
            executed_at=executed_at, execution_id=execution.id,
        )
        # Cash leaves immediately; a buy does not get to spend money it has
        # not paid yet. The reservation taken at order creation is released in
        # the same step, so the hold does not outlive the order it protected.
        account.cash_balance = Decimal(str(round(to_float(account.cash_balance) - fill.net_amount, 6)))
        _release_reservation(account, order)
    else:
        from services.policy import resolve as resolve_policy  # local: avoids a cycle

        client = account.client
        method = resolve_policy(db, client).lot_selection_method if client else "HIFO"
        result = tax_lots.record_sale(
            db, account, fill.symbol, fill.quantity, fill.price,
            method=method, executed_at=executed_at, execution_id=execution.id,
        )
        execution.realized_gain = Decimal(str(round(result.realized_gain, 6)))
        execution.realized_term = (
            "long" if result.long_term_gain and not result.short_term_gain else "short"
        )
        # Proceeds are credited and immediately available to *reinvest*. This
        # matches how custodians actually treat a cash account: unsettled
        # proceeds may be used to buy the same day, they may not be withdrawn
        # until settlement. Holding them against reinvestment instead broke
        # the ordinary sell-to-fund-a-buy rebalance -- every buy in a plan
        # funded by its own sells was rejected for insufficient cash.
        # `unsettled_proceeds` is what gates a withdrawal.
        account.cash_balance = Decimal(str(round(to_float(account.cash_balance) + fill.net_amount, 6)))

    order.status = "filled"
    order.filled_quantity = Decimal(str(fill.quantity))
    order.average_fill_price = Decimal(str(round(fill.price, 6)))
    order.submitted_at = order.submitted_at or executed_at
    order.filled_at = executed_at
    if approved_by_user_id is not None:
        order.approved_by_user_id = approved_by_user_id

    orders_placed.labels(order.side, "filled").inc()
    order_notional.labels(order.side).inc(fill.gross_amount)

    record_audit(
        db,
        org_id=order.org_id,
        action=Action.ORDER_FILLED,
        user_id=approved_by_user_id,
        entity_type="order",
        entity_id=order.id,
        detail={
            "symbol": fill.symbol,
            "side": fill.side,
            "quantity": fill.quantity,
            "price": round(fill.price, 4),
            "net_amount": round(fill.net_amount, 2),
            "commission": round(fill.commission, 2),
            "slippage_cost": round(fill.slippage_cost, 2),
            "realized_gain": (
                round(to_float(execution.realized_gain), 2) if execution.realized_gain else None
            ),
            "venue": fill.venue,
        },
    )
    db.flush()
    logger.info(
        "Filled %s %g %s at %.2f in account %s",
        fill.side, fill.quantity, fill.symbol, fill.price, account.name,
        extra={"client_id": order.client_id, "run_id": order.run_id},
    )
    return order


def cancel_order(
    db: Session, order: Order, *, user_id: Optional[int] = None, reason: str = ""
) -> Order:
    if order.status in ("filled", "cancelled"):
        raise OrderRejected(f"Order {order.id} is already {order.status} and cannot be cancelled.")
    account = db.query(Account).filter(Account.id == order.account_id).first()
    if account is not None:
        # A cancelled order that keeps its reservation slowly strands the
        # account's cash: the balance is there, nothing can spend it, and
        # nothing explains why.
        _release_reservation(account, order)
    order.status = "cancelled"
    order.cancelled_at = utcnow()
    order.failure_reason = reason[:500] or None
    orders_placed.labels(order.side, "cancelled").inc()
    record_audit(
        db,
        org_id=order.org_id,
        action=Action.ORDER_CANCELLED,
        user_id=user_id,
        entity_type="order",
        entity_id=order.id,
        detail={"reason": reason, "symbol": order.symbol},
    )
    db.flush()
    return order


def unsettled_proceeds(db: Session, account: Account, *, as_of=None) -> float:
    """Sale proceeds not yet settled.

    Available to reinvest, not to withdraw. Withdrawing unsettled proceeds in
    a cash account is a good-faith violation, so any cash-out path must check
    this rather than the raw balance.
    """
    as_of = as_of or utcnow()
    pending = (
        db.query(Execution)
        .filter(
            Execution.account_id == account.id,
            Execution.side == "SELL",
            Execution.settles_on.isnot(None),
            Execution.settles_on > as_of,
        )
        .all()
    )
    return round(sum(to_float(e.net_amount) for e in pending), 2)


def withdrawable_cash(db: Session, account: Account, *, as_of=None) -> float:
    """Cash that could actually leave the account today."""
    balance = to_float(account.cash_balance) - to_float(account.pending_cash_hold)
    return max(0.0, balance - unsettled_proceeds(db, account, as_of=as_of))


def execute_plan(
    db: Session,
    orders: List[Order],
    *,
    approved_by_user_id: Optional[int] = None,
) -> List[Order]:
    """Execute already-created orders, sells first.

    One rejected order does not abort the batch: the others are independently
    valid, and refusing them all because of one failure leaves the portfolio
    in a state nobody chose. Failures are recorded on their own orders.
    """
    ordered = sorted(orders, key=lambda o: (o.side != "SELL", o.id))
    executed: List[Order] = []
    for order in ordered:
        try:
            executed.append(submit_order(db, order, approved_by_user_id=approved_by_user_id))
        except OrderRejected as exc:
            logger.warning("Order %s rejected: %s", order.id, exc)
            order.status = "rejected"
            order.failure_reason = str(exc)[:500]
            account = db.query(Account).filter(Account.id == order.account_id).first()
            if account is not None:
                _release_reservation(account, order)
    db.flush()
    return executed


def execute_proposals(
    db: Session,
    proposals: List[dict],
    *,
    run_id: Optional[str] = None,
    approved_by_user_id: Optional[int] = None,
) -> List[Order]:
    """Create and execute a rebalance plan in sequence order.

    Creating every order up front and executing afterwards does not work, and
    the failure is instructive: a plan's buys are funded by its own sells, so
    at creation time that cash does not exist yet and each buy is rejected for
    insufficient funds. Interleaving creation with execution -- sell, receive
    the proceeds, then create the next buy against the balance that now
    exists -- is the only ordering where a self-funding rebalance works.

    Each proposal is committed as it completes. A failure part-way leaves the
    trades that already executed intact rather than rolling back real fills,
    which is both correct and unavoidable: you cannot un-trade.
    """
    executed: List[Order] = []
    for proposal in sorted(proposals, key=lambda p: (p.get("sequence", 0), p.get("side") != "SELL")):
        account = db.query(Account).filter(Account.id == proposal["account_id"]).first()
        if account is None:
            logger.warning("Proposal for missing account %s skipped.", proposal.get("account_id"))
            continue
        try:
            order = create_order(
                db,
                account,
                symbol=proposal["symbol"],
                side=proposal["side"],
                quantity=proposal["quantity"],
                run_id=run_id,
                proposal_id=proposal.get("proposal_id"),
                created_by_user_id=approved_by_user_id,
                status="approved",
            )
            executed.append(
                submit_order(db, order, approved_by_user_id=approved_by_user_id)
            )
            db.commit()
        except OrderRejected as exc:
            db.rollback()
            logger.warning(
                "Proposal %s %s rejected: %s",
                proposal.get("side"), proposal.get("symbol"), exc,
            )
        except Exception:
            db.rollback()
            logger.exception(
                "Unexpected failure executing %s %s",
                proposal.get("side"), proposal.get("symbol"),
            )
    return executed
