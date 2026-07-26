"""Performance measurement, attribution and recommendation scoring.

The previous system had no notion of whether any of its advice worked. It
produced confident recommendations indefinitely with no feedback signal — no
benchmark, no return history, no record of whether a past pick beat holding
the index. That is the difference between a tool that gives advice and one
that can be held to it.

Three things live here:

* **Return measurement, done twice.** Time-weighted return measures the
  *strategy*, neutralizing the client's deposits and withdrawals. Money-weighted
  return (IRR) measures the *client's experience*, which depends on when they
  put money in. Both are correct answers to different questions, and reporting
  only one is how a portfolio that returned 12% gets described to a client who
  actually made 3%.

* **Attribution.** A return number says what happened; attribution says why.
  Splitting excess return into an allocation effect (being in the right asset
  classes) and a selection effect (picking the right names within them) is what
  distinguishes a good call from a lucky one.

* **Recommendation outcomes.** Each recommendation is scored against the
  benchmark at a fixed horizon. This is the only component that can make the
  system look bad, which is precisely why it matters.

A note on honesty: every function here refuses to compute rather than
extrapolate. Fewer than two snapshots is not a 0% return, an unpriced
benchmark is not a flat benchmark, and thirty days of history does not produce
an annualized figure. Returning `None` with a stated reason is the whole point.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db import (
    CashTransaction,
    ClientProfile,
    PortfolioSnapshot,
    RecommendationOutcome,
    to_float,
    utcnow,
)
from logging_setup import get_logger
from services.market_data import fetch_historical_prices, get_quotes
from services.portfolio import PortfolioView, load_portfolio

logger = get_logger(__name__)

TRADING_DAYS = 252
# Below this many observations an annualized figure is arithmetic without
# meaning: a 3% gain over eight days annualizes to 150%, which is not a
# forecast, an estimate, or in any sense a description of the portfolio.
MIN_OBSERVATIONS_FOR_ANNUALIZED = 60


@dataclass
class PerformanceResult:
    """Returns over one window, with an explicit statement of what is missing."""

    start: Optional[datetime] = None
    end: Optional[datetime] = None
    days: int = 0
    observations: int = 0

    beginning_value: Optional[float] = None
    ending_value: Optional[float] = None
    net_flows: float = 0.0

    time_weighted_return: Optional[float] = None
    money_weighted_return: Optional[float] = None
    annualized_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

    benchmark_ticker: Optional[str] = None
    benchmark_return: Optional[float] = None
    excess_return: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    beta_to_benchmark: Optional[float] = None

    insufficient_data: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        def r(value, digits=4):
            return round(value, digits) if value is not None else None

        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "days": self.days,
            "observations": self.observations,
            "beginning_value": r(self.beginning_value, 2),
            "ending_value": r(self.ending_value, 2),
            "net_flows": round(self.net_flows, 2),
            "time_weighted_return": r(self.time_weighted_return),
            "money_weighted_return": r(self.money_weighted_return),
            "annualized_return": r(self.annualized_return),
            "volatility": r(self.volatility),
            "sharpe_ratio": r(self.sharpe_ratio, 3),
            "max_drawdown": r(self.max_drawdown),
            "benchmark_ticker": self.benchmark_ticker,
            "benchmark_return": r(self.benchmark_return),
            "excess_return": r(self.excess_return),
            "tracking_error": r(self.tracking_error),
            "information_ratio": r(self.information_ratio, 3),
            "beta_to_benchmark": r(self.beta_to_benchmark, 3),
            "insufficient_data": self.insufficient_data,
            "notes": self.notes,
        }


# --- Snapshots ---------------------------------------------------------------


def capture_snapshot(
    db: Session,
    client: ClientProfile,
    *,
    as_of: Optional[datetime] = None,
    view: Optional[PortfolioView] = None,
) -> Optional[PortfolioSnapshot]:
    """Record today's valuation and the day's external flows.

    Idempotent per (client, date): re-running updates the existing row rather
    than creating a second one, so a retried job cannot double-count a day in
    the return series.
    """
    as_of = (as_of or utcnow()).replace(hour=0, minute=0, second=0, microsecond=0)
    view = view or load_portfolio(db, client)

    if view.total_value <= 0 and not view.holdings:
        logger.debug("Client %s has nothing to snapshot.", client.id)
        return None

    # Flows must be measured, not inferred. A $50k jump in market value is
    # either a great day or a wire transfer, and only this table knows which.
    account_ids = [a.id for a in client.accounts]
    flows = 0.0
    if account_ids:
        rows = (
            db.query(CashTransaction)
            .filter(
                CashTransaction.account_id.in_(account_ids),
                CashTransaction.occurred_at >= as_of,
                CashTransaction.occurred_at < as_of + timedelta(days=1),
                CashTransaction.transaction_type.in_(("DEPOSIT", "WITHDRAWAL")),
            )
            .all()
        )
        flows = sum(to_float(r.amount) for r in rows)

    benchmark = _benchmark_for(db, client)
    benchmark_close = None
    if benchmark:
        quotes = get_quotes([benchmark])
        if benchmark in quotes:
            benchmark_close = Decimal(str(round(quotes[benchmark].price, 6)))

    existing = (
        db.query(PortfolioSnapshot)
        .filter(
            PortfolioSnapshot.client_id == client.id,
            PortfolioSnapshot.account_id.is_(None),
            PortfolioSnapshot.as_of == as_of,
        )
        .first()
    )

    holdings_snapshot = {
        holding.symbol: {
            "quantity": round(holding.quantity, 6),
            "price": round(holding.price, 4),
            "value": round(holding.market_value, 2),
            "asset_class": holding.asset_class,
        }
        for holding in view.holdings
    }

    if existing is not None:
        existing.market_value = Decimal(str(round(view.total_value, 6)))
        existing.cash_value = Decimal(str(round(view.cash, 6)))
        existing.net_flow = Decimal(str(round(flows, 6)))
        existing.holdings_snapshot = holdings_snapshot
        existing.benchmark_ticker = benchmark
        existing.benchmark_close = benchmark_close
        db.flush()
        return existing

    snapshot = PortfolioSnapshot(
        org_id=client.org_id,
        client_id=client.id,
        account_id=None,
        as_of=as_of,
        market_value=Decimal(str(round(view.total_value, 6))),
        cash_value=Decimal(str(round(view.cash, 6))),
        net_flow=Decimal(str(round(flows, 6))),
        holdings_snapshot=holdings_snapshot,
        benchmark_ticker=benchmark,
        benchmark_close=benchmark_close,
    )
    db.add(snapshot)
    db.flush()
    return snapshot


def _benchmark_for(db: Session, client: ClientProfile) -> str:
    from services.policy import resolve as resolve_policy

    return resolve_policy(db, client).benchmark_ticker


# --- Returns -----------------------------------------------------------------


def time_weighted_return(values: Sequence[float], flows: Sequence[float]) -> Optional[float]:
    """Geometrically linked periodic returns, neutralizing external flows.

    Each period's return is measured against the beginning value *plus* that
    period's flow, on the assumption the flow arrived at the start. That
    assumption is why daily snapshots matter: over a month it can be materially
    wrong, over a day it barely moves the answer.

    Without the flow adjustment a client wiring in $100k on a flat day shows a
    100% gain, and the whole series is nonsense from that point forward.
    """
    if len(values) < 2:
        return None

    linked = 1.0
    used = 0
    for index in range(1, len(values)):
        begin = values[index - 1]
        flow = flows[index] if index < len(flows) else 0.0
        adjusted_begin = begin + flow
        if adjusted_begin <= 0:
            # A period starting from nothing has no meaningful return; skip it
            # rather than dividing by zero or inventing one.
            continue
        period_return = (values[index] - adjusted_begin) / adjusted_begin
        linked *= 1.0 + period_return
        used += 1

    if used == 0:
        return None
    return linked - 1.0


def money_weighted_return(
    dates: Sequence[datetime], amounts: Sequence[float], max_iterations: int = 100
) -> Optional[float]:
    """Annualized IRR of the client's actual cash flows.

    Solved by bisection rather than Newton's method: IRR is not well behaved
    for sign-alternating flows, and Newton can diverge to a nonsense root that
    then gets reported as a return. Bisection over a bounded bracket either
    converges or honestly fails.
    """
    if len(dates) < 2 or len(amounts) != len(dates):
        return None
    if not (any(a > 0 for a in amounts) and any(a < 0 for a in amounts)):
        # Without both an inflow and an outflow there is no rate to solve for.
        return None

    start = dates[0]
    years = [(d - start).days / 365.25 for d in dates]

    def npv(rate: float) -> float:
        return sum(a / ((1.0 + rate) ** t) for a, t in zip(amounts, years))

    low, high = -0.9999, 10.0
    npv_low, npv_high = npv(low), npv(high)
    if npv_low * npv_high > 0:
        # No sign change in the bracket means no root inside it. Widening the
        # bracket to find one would produce a mathematically valid rate with
        # no financial meaning.
        return None

    for _ in range(max_iterations):
        mid = (low + high) / 2
        value = npv(mid)
        if abs(value) < 1e-7:
            return mid
        if value * npv_low > 0:
            low, npv_low = mid, value
        else:
            high = mid
    return (low + high) / 2


def compute_performance(
    db: Session,
    client: ClientProfile,
    *,
    since: Optional[datetime] = None,
    risk_free_rate: float = 0.04,
) -> PerformanceResult:
    """Full return and risk profile over the available snapshot history."""
    result = PerformanceResult()

    query = db.query(PortfolioSnapshot).filter(
        PortfolioSnapshot.client_id == client.id,
        PortfolioSnapshot.account_id.is_(None),
    )
    if since:
        query = query.filter(PortfolioSnapshot.as_of >= since)
    snapshots = query.order_by(PortfolioSnapshot.as_of.asc()).all()

    if len(snapshots) < 2:
        result.insufficient_data = True
        result.observations = len(snapshots)
        result.notes.append(
            f"Only {len(snapshots)} valuation snapshot(s) exist for this client, so no "
            "return can be computed. Snapshots accumulate daily; performance reporting "
            "becomes available once there are at least two."
        )
        return result

    values = [to_float(s.market_value) for s in snapshots]
    flows = [to_float(s.net_flow) for s in snapshots]
    dates = [s.as_of for s in snapshots]

    result.start, result.end = dates[0], dates[-1]
    result.days = (dates[-1] - dates[0]).days
    result.observations = len(snapshots)
    result.beginning_value = values[0]
    result.ending_value = values[-1]
    result.net_flows = sum(flows)

    result.time_weighted_return = time_weighted_return(values, flows)

    # IRR flows: the opening value is an outflow (money put to work), each
    # deposit an outflow, each withdrawal an inflow, the closing value an
    # inflow.
    irr_dates = [dates[0]]
    irr_amounts = [-values[0]]
    for index in range(1, len(snapshots)):
        if abs(flows[index]) > 1e-6:
            irr_dates.append(dates[index])
            irr_amounts.append(-flows[index])
    irr_dates.append(dates[-1])
    irr_amounts.append(values[-1])
    result.money_weighted_return = money_weighted_return(irr_dates, irr_amounts)

    series = pd.Series(values, index=pd.to_datetime(dates))
    # Flow-adjusted daily returns, so risk statistics are not driven by
    # deposits. A wire transfer is not volatility.
    daily = []
    for index in range(1, len(values)):
        begin = values[index - 1] + flows[index]
        daily.append((values[index] - begin) / begin if begin > 0 else 0.0)
    returns = pd.Series(daily, index=series.index[1:])

    if len(returns) >= MIN_OBSERVATIONS_FOR_ANNUALIZED:
        result.volatility = float(returns.std() * np.sqrt(TRADING_DAYS))
        if result.time_weighted_return is not None and result.days > 0:
            growth = 1.0 + result.time_weighted_return
            if growth > 0:
                result.annualized_return = growth ** (365.25 / result.days) - 1.0
        if result.annualized_return is not None and result.volatility and result.volatility > 0:
            result.sharpe_ratio = (result.annualized_return - risk_free_rate) / result.volatility
    else:
        result.notes.append(
            f"{len(returns)} daily observations is too few to annualize; volatility, "
            f"Sharpe and annualized return need at least "
            f"{MIN_OBSERVATIONS_FOR_ANNUALIZED}. Cumulative return is reported instead."
        )

    running_max = series.cummax()
    drawdown = (series / running_max - 1.0).min()
    result.max_drawdown = float(drawdown) if drawdown == drawdown else None

    _add_benchmark(db, client, result, dates, returns)
    return result


def _add_benchmark(
    db: Session,
    client: ClientProfile,
    result: PerformanceResult,
    dates: Sequence[datetime],
    returns: pd.Series,
) -> None:
    """Compare against the policy benchmark over the same window.

    Uses the benchmark's own price history rather than the closes stored on
    the snapshots, because a snapshot is only written on days a job ran, and a
    benchmark sampled on those days is not the benchmark's return.
    """
    benchmark = _benchmark_for(db, client)
    result.benchmark_ticker = benchmark
    if not benchmark or len(dates) < 2:
        return

    span_days = max(1, (dates[-1] - dates[0]).days)
    history = fetch_historical_prices([benchmark], years=max(0.1, span_days / 365.0 + 0.05))
    if history.empty or benchmark not in history.columns:
        result.notes.append(
            f"No price history was available for the benchmark {benchmark}, so no "
            "comparison is reported. The portfolio's own return above is unaffected."
        )
        return

    prices = history[benchmark].dropna()
    window = prices[(prices.index >= pd.Timestamp(dates[0])) & (prices.index <= pd.Timestamp(dates[-1]))]
    if len(window) < 2:
        result.notes.append(
            f"The benchmark {benchmark} has fewer than two closes inside the measurement "
            "window, so no comparison is reported."
        )
        return

    result.benchmark_return = float(window.iloc[-1] / window.iloc[0] - 1.0)
    if result.time_weighted_return is not None:
        result.excess_return = result.time_weighted_return - result.benchmark_return

    benchmark_returns = window.pct_change().dropna()
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = ["portfolio", "benchmark"]
    # Below ~30 overlapping observations, tracking error and beta have
    # confidence intervals wide enough that quoting a point estimate is
    # misleading precision.
    if len(aligned) < 30:
        result.notes.append(
            f"Only {len(aligned)} trading days overlap between the portfolio's snapshots "
            "and the benchmark, which is too few for a reliable tracking error or beta."
        )
        return

    active = aligned["portfolio"] - aligned["benchmark"]
    tracking_error = float(active.std() * np.sqrt(TRADING_DAYS))
    result.tracking_error = tracking_error
    if tracking_error > 0 and result.excess_return is not None:
        result.information_ratio = result.excess_return / tracking_error

    variance = float(aligned["benchmark"].var())
    if variance > 0:
        covariance = float(aligned["portfolio"].cov(aligned["benchmark"]))
        result.beta_to_benchmark = covariance / variance


# --- Attribution -------------------------------------------------------------


@dataclass
class AttributionEntry:
    asset_class: str
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    allocation_effect: float
    selection_effect: float
    total_effect: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "asset_class": self.asset_class,
            "portfolio_weight": round(self.portfolio_weight, 4),
            "benchmark_weight": round(self.benchmark_weight, 4),
            "portfolio_return": round(self.portfolio_return, 4),
            "benchmark_return": round(self.benchmark_return, 4),
            "allocation_effect": round(self.allocation_effect, 5),
            "selection_effect": round(self.selection_effect, 5),
            "total_effect": round(self.total_effect, 5),
        }


def brinson_attribution(
    portfolio_weights: Dict[str, float],
    portfolio_returns: Dict[str, float],
    benchmark_weights: Dict[str, float],
    benchmark_returns: Dict[str, float],
) -> Tuple[List[AttributionEntry], Dict[str, float]]:
    """Brinson-Hood-Beebower attribution.

    Splits excess return into:

      * **Allocation** -- (wp - wb) x (rb - r_total). Value added by weighting
        an asset class differently from the benchmark.
      * **Selection** -- wb x (rp - rb). Value added by holding better things
        *within* an asset class.
      * **Interaction** -- folded into selection here, which is the common
        two-factor presentation. Reporting a separate interaction term is more
        precise and, in practice, tells an advisor nothing they can act on.

    This is the difference between "we beat the benchmark" and "we beat the
    benchmark because we were overweight bonds, not because we picked good
    stocks" -- and only the second is a repeatable claim.
    """
    classes = set(portfolio_weights) | set(benchmark_weights)
    total_benchmark_return = sum(
        benchmark_weights.get(c, 0.0) * benchmark_returns.get(c, 0.0) for c in classes
    )

    entries: List[AttributionEntry] = []
    for asset_class in sorted(classes):
        wp = portfolio_weights.get(asset_class, 0.0)
        wb = benchmark_weights.get(asset_class, 0.0)
        rp = portfolio_returns.get(asset_class, 0.0)
        rb = benchmark_returns.get(asset_class, 0.0)

        allocation = (wp - wb) * (rb - total_benchmark_return)
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)

        entries.append(
            AttributionEntry(
                asset_class=asset_class,
                portfolio_weight=wp,
                benchmark_weight=wb,
                portfolio_return=rp,
                benchmark_return=rb,
                allocation_effect=allocation,
                selection_effect=selection + interaction,
                total_effect=allocation + selection + interaction,
            )
        )

    totals = {
        "allocation_effect": round(sum(e.allocation_effect for e in entries), 5),
        "selection_effect": round(sum(e.selection_effect for e in entries), 5),
        "total_effect": round(sum(e.total_effect for e in entries), 5),
        "benchmark_return": round(total_benchmark_return, 5),
    }
    return entries, totals


# --- Recommendation outcomes -------------------------------------------------


def record_recommendations(
    db: Session,
    client: ClientProfile,
    run_id: str,
    recommendations: Sequence[Dict[str, object]],
    *,
    horizons: Sequence[int] = (30, 90, 365),
) -> int:
    """Open outcome rows so each recommendation can be scored later.

    Written at recommendation time with the entry price captured then. Trying
    to reconstruct "what was it worth when we said this?" afterwards is
    exactly the kind of retrospective the results should not depend on.
    """
    created = 0
    symbols = [str(r.get("ticker") or r.get("symbol") or "").upper() for r in recommendations]
    symbols = [s for s in symbols if s]
    quotes = get_quotes(symbols) if symbols else {}

    for recommendation in recommendations:
        symbol = str(recommendation.get("ticker") or recommendation.get("symbol") or "").upper()
        if not symbol:
            continue
        quote = quotes.get(symbol)
        if quote is None:
            logger.debug("No entry price for %s; outcome tracking skipped.", symbol)
            continue

        for horizon in horizons:
            exists = (
                db.query(RecommendationOutcome)
                .filter(
                    RecommendationOutcome.run_id == run_id,
                    RecommendationOutcome.symbol == symbol,
                    RecommendationOutcome.horizon_days == horizon,
                )
                .first()
            )
            if exists is not None:
                continue
            db.add(
                RecommendationOutcome(
                    org_id=client.org_id,
                    client_id=client.id,
                    run_id=run_id,
                    symbol=symbol,
                    side=str(recommendation.get("side", "BUY")).upper(),
                    recommended_at=utcnow(),
                    entry_price=Decimal(str(round(quote.price, 6))),
                    confidence=recommendation.get("confidence"),
                    was_executed=bool(recommendation.get("executed")),
                    horizon_days=horizon,
                )
            )
            created += 1

    db.flush()
    return created


def evaluate_outcomes(db: Session, *, as_of: Optional[datetime] = None, limit: int = 500) -> int:
    """Score recommendations whose horizon has elapsed.

    Excess return against the benchmark is the figure that matters. A pick
    that returned 8% while the index returned 12% was a bad call, and a raw
    return number would present it as a good one.
    """
    as_of = as_of or utcnow()
    due = (
        db.query(RecommendationOutcome)
        .filter(RecommendationOutcome.status == "open")
        .order_by(RecommendationOutcome.recommended_at.asc())
        .limit(limit)
        .all()
    )
    due = [
        outcome
        for outcome in due
        if outcome.recommended_at + timedelta(days=outcome.horizon_days) <= as_of
    ]
    if not due:
        return 0

    symbols = sorted({o.symbol for o in due})
    benchmarks = sorted({_benchmark_for(db, o.client) for o in due if o.client})
    quotes = get_quotes(symbols + [b for b in benchmarks if b])

    evaluated = 0
    for outcome in due:
        quote = quotes.get(outcome.symbol)
        entry = to_float(outcome.entry_price)
        if quote is None or entry <= 0:
            # Cannot be scored: mark it void rather than leaving it open
            # forever, and never guess a return.
            outcome.status = "void"
            outcome.evaluated_at = as_of
            continue

        outcome.exit_price = Decimal(str(round(quote.price, 6)))
        outcome.return_pct = round(quote.price / entry - 1.0, 6)

        benchmark = _benchmark_for(db, outcome.client) if outcome.client else None
        if benchmark:
            history = fetch_historical_prices(
                [benchmark], years=max(0.1, outcome.horizon_days / 365.0 + 0.1)
            )
            if not history.empty and benchmark in history.columns:
                prices = history[benchmark].dropna()
                window = prices[prices.index >= pd.Timestamp(outcome.recommended_at)]
                if len(window) >= 2:
                    outcome.benchmark_return_pct = round(
                        float(window.iloc[-1] / window.iloc[0] - 1.0), 6
                    )
                    outcome.excess_return_pct = round(
                        outcome.return_pct - outcome.benchmark_return_pct, 6
                    )

        outcome.status = "evaluated"
        outcome.evaluated_at = as_of
        evaluated += 1

    db.flush()
    logger.info("Evaluated %d recommendation outcome(s).", evaluated)
    return evaluated


def recommendation_scorecard(
    db: Session, client: Optional[ClientProfile] = None, *, org_id: Optional[int] = None
) -> Dict[str, object]:
    """Aggregate hit rate and average excess return.

    Deliberately reports the sample size alongside the hit rate. "62% of our
    picks beat the benchmark" means something over 200 recommendations and
    nothing over eight, and a scorecard that omits the denominator invites the
    second reading.
    """
    query = db.query(RecommendationOutcome).filter(RecommendationOutcome.status == "evaluated")
    if client is not None:
        query = query.filter(RecommendationOutcome.client_id == client.id)
    elif org_id is not None:
        query = query.filter(RecommendationOutcome.org_id == org_id)

    outcomes = query.all()
    if not outcomes:
        return {
            "evaluated": 0,
            "note": "No recommendations have reached their evaluation horizon yet.",
        }

    by_horizon: Dict[int, List[RecommendationOutcome]] = {}
    for outcome in outcomes:
        by_horizon.setdefault(outcome.horizon_days, []).append(outcome)

    summary: Dict[str, object] = {"evaluated": len(outcomes), "horizons": {}}
    for horizon, group in sorted(by_horizon.items()):
        with_benchmark = [o for o in group if o.excess_return_pct is not None]
        beat = [o for o in with_benchmark if o.excess_return_pct > 0]
        summary["horizons"][str(horizon)] = {
            "count": len(group),
            "scored_against_benchmark": len(with_benchmark),
            "beat_benchmark": len(beat),
            "hit_rate": round(len(beat) / len(with_benchmark), 3) if with_benchmark else None,
            "avg_return": round(
                sum(o.return_pct for o in group if o.return_pct is not None) / len(group), 4
            ),
            "avg_excess_return": (
                round(sum(o.excess_return_pct for o in with_benchmark) / len(with_benchmark), 4)
                if with_benchmark
                else None
            ),
            "reliable": len(with_benchmark) >= 30,
        }
    return summary


def backfill_snapshots(
    db: Session, client: ClientProfile, *, days: int = 90
) -> int:
    """Reconstruct a valuation history from current positions and price history.

    A genuine approximation, and labelled as one: it applies *today's* share
    counts to historical prices, so it shows what the current portfolio would
    have done, not what the client actually held. That makes it useful for
    context on a new client and wrong for a performance claim. Real snapshots
    take over from the day the job first runs.
    """
    view = load_portfolio(db, client)
    if not view.holdings:
        return 0

    symbols = sorted({h.symbol for h in view.holdings})
    history = fetch_historical_prices(symbols, years=max(0.1, days / 365.0))
    if history.empty:
        return 0

    quantities: Dict[str, float] = {}
    for holding in view.holdings:
        quantities[holding.symbol] = quantities.get(holding.symbol, 0.0) + holding.quantity

    created = 0
    for timestamp, row in history.iterrows():
        as_of = pd.Timestamp(timestamp).to_pydatetime().replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None
        )
        value = view.cash
        for symbol, quantity in quantities.items():
            price = row.get(symbol)
            if price is None or pd.isna(price):
                continue
            value += quantity * float(price)

        exists = (
            db.query(PortfolioSnapshot)
            .filter(
                PortfolioSnapshot.client_id == client.id,
                PortfolioSnapshot.account_id.is_(None),
                PortfolioSnapshot.as_of == as_of,
            )
            .first()
        )
        if exists is not None:
            continue
        db.add(
            PortfolioSnapshot(
                org_id=client.org_id,
                client_id=client.id,
                account_id=None,
                as_of=as_of,
                market_value=Decimal(str(round(value, 6))),
                cash_value=Decimal(str(round(view.cash, 6))),
                net_flow=Decimal("0"),
                holdings_snapshot={"_reconstructed": True},
            )
        )
        created += 1

    db.flush()
    logger.info(
        "Backfilled %d reconstructed snapshot(s) for client %s. These apply current "
        "holdings to past prices and are not a record of what was actually held.",
        created, client.id,
    )
    return created
