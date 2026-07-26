"""Portfolio Diagnostics: what is actually wrong with this portfolio.

Deterministic throughout. No LLM touches this, by design: the findings here
decide what the research agent looks for and what the guardrails measure
against, so they must be reproducible and explainable to a reviewer who will
not accept "the model thought so".

Beyond the concentration, sector, cash and volatility checks the previous
version performed, this adds the three measurements that were missing and that
change conclusions:

* **Correlation.** A portfolio of twelve names that all move together is one
  position wearing twelve hats. A diversification score computed from position
  weights cannot see that, and will report eight megacap technology names as
  well diversified.
* **Drawdown.** Volatility describes the average day. Peak-to-trough decline
  describes the day the client calls wanting out, which is the number that
  determines whether a mandate survives contact with a bear market.
* **Drift from the policy target.** A flaw is a deviation from *this client's*
  agreed allocation, not from a general notion of prudence. Without a target,
  "you hold no bonds" is an opinion; with one it is a measured 28-point gap
  against a signed mandate.

Every flaw is phrased as a sentence a client can read, because these strings
are what the research agent matches on and what the report prints.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings
from db import ClientProfile as ClientProfileModel
from db import SessionLocal
from logging_setup import get_logger
from services.market_data import fetch_historical_prices, quote_provenance
from services.policy import resolve as resolve_policy
from services.portfolio import compute_drift, concentration_breaches, load_portfolio
from state import AgentState, PortfolioDiagnostics

from agents.runtime import finish, node_run

logger = get_logger(__name__)

TRADING_DAYS = 252

# Two holdings above this correlation are, for risk purposes, the same
# position. Not a flaw alone -- two share classes of one company are
# legitimately correlated -- but a cluster of them is.
HIGH_CORRELATION = 0.80
# Below roughly six months of overlapping history a correlation estimate's
# confidence interval is wide enough that acting on it is noise.
MIN_CORRELATION_OBSERVATIONS = 120
# Fewer observations than this and annualized risk figures are arithmetic
# without meaning.
MIN_RETURN_OBSERVATIONS = 30

NODE_NAME = "diagnostics"

# Client-facing names for the internal asset-class keys. These strings appear
# verbatim in the report, and "em equity" is not a phrase to put in front of a
# client.
ASSET_CLASS_LABELS = {
    "us_equity": "US equity",
    "intl_equity": "International equity",
    "em_equity": "Emerging-market equity",
    "fixed_income": "Fixed income",
    "real_assets": "Real assets",
    "commodity": "Commodities",
    "cash": "Cash",
    "alternative": "Alternatives",
}


def _annualized_stats(
    prices: pd.DataFrame, weights: Dict[str, float], risk_free_rate: float
) -> Dict[str, Optional[float]]:
    """Return, volatility, Sharpe and drawdown for the portfolio as held.

    Computed from the *actual weights*, not an equal-weighted basket. An
    equal-weighted proxy for a portfolio that is half in one name describes a
    portfolio the client does not own and understates its volatility badly.
    """
    empty: Dict[str, Optional[float]] = {
        "annual_return": None,
        "annual_volatility": None,
        "sharpe_ratio": None,
        "max_drawdown": None,
    }
    if prices.empty or not weights:
        return empty

    columns = [c for c in prices.columns if c in weights and weights[c] > 0]
    if not columns:
        return empty

    returns = prices[columns].pct_change().dropna(how="all")
    if len(returns) < MIN_RETURN_OBSERVATIONS:
        return empty

    total = sum(weights[c] for c in columns)
    if total <= 0:
        return empty
    normalized = {c: weights[c] / total for c in columns}

    portfolio_returns = sum(returns[c].fillna(0.0) * normalized[c] for c in columns)
    if not isinstance(portfolio_returns, pd.Series) or portfolio_returns.empty:
        return empty

    volatility = float(portfolio_returns.std() * np.sqrt(TRADING_DAYS))
    mean_return = float(portfolio_returns.mean() * TRADING_DAYS)

    growth = (1.0 + portfolio_returns).cumprod()
    drawdown = float((growth / growth.cummax() - 1.0).min())

    sharpe = (mean_return - risk_free_rate) / volatility if volatility > 0 else None

    return {
        "annual_return": round(mean_return, 4),
        "annual_volatility": round(volatility, 4),
        "sharpe_ratio": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown": round(drawdown, 4) if drawdown == drawdown else None,
    }


def _correlation_analysis(
    prices: pd.DataFrame, weights: Dict[str, float]
) -> Tuple[Optional[float], List[List[str]]]:
    """Weighted average pairwise correlation, and clusters of near-duplicates.

    Weighted because a 45% position's correlation with everything else matters
    far more than a 1% position's; a plain average lets a long tail of tiny
    holdings disguise a concentrated core.
    """
    columns = [c for c in prices.columns if c in weights and weights[c] > 0]
    if len(columns) < 2:
        return None, []

    returns = prices[columns].pct_change().dropna()
    if len(returns) < MIN_CORRELATION_OBSERVATIONS:
        return None, []

    matrix = returns.corr()
    if matrix.isna().all().all():
        return None, []

    weighted_sum = 0.0
    weight_total = 0.0
    clusters: Dict[str, List[str]] = {}

    for index, left in enumerate(columns):
        for right in columns[index + 1:]:
            value = matrix.loc[left, right]
            if value != value:  # NaN
                continue
            pair_weight = weights[left] * weights[right]
            weighted_sum += float(value) * pair_weight
            weight_total += pair_weight
            if value >= HIGH_CORRELATION:
                clusters.setdefault(left, [left]).append(right)

    average = weighted_sum / weight_total if weight_total > 0 else None

    merged: List[List[str]] = []
    for group in clusters.values():
        unique = sorted(set(group))
        if not any(set(unique) <= set(existing) for existing in merged):
            merged.append(unique)

    return (round(average, 3) if average is not None else None), merged


def _diversification_score(weights: Dict[str, float]) -> Tuple[float, Optional[float]]:
    """Herfindahl-based score, plus the effective number of positions.

    The effective count is the more useful of the two: "you hold 12 positions
    but they behave like 2.4" is a sentence a client immediately understands,
    where "your diversification score is 58" is not.
    """
    values = [w for w in weights.values() if w > 0]
    if not values:
        return 0.0, None
    total = sum(values)
    if total <= 0:
        return 0.0, None
    shares = [v / total for v in values]
    hhi = sum(s * s for s in shares)
    effective = 1.0 / hhi if hhi > 0 else None
    return round(100.0 * (1.0 - hhi), 1), (round(effective, 2) if effective else None)


def _build_flaws(
    view,
    policy,
    stats: Dict[str, Optional[float]],
    drift,
    breaches,
    average_correlation: Optional[float],
    clusters: List[List[str]],
    effective_positions: Optional[float],
) -> List[str]:
    """Turn measurements into client-readable findings.

    Ordered by severity: the research agent works down this list and the
    report leads with it. A flaw that names its limit and its magnitude is
    actionable; "you are too concentrated" is not.
    """
    flaws: List[str] = []

    for breach in breaches:
        if breach["kind"] == "position":
            flaws.append(
                f"{breach['key']} is {breach['weight']:.0%} of the portfolio, above the "
                f"{breach['limit']:.0%} single-position limit in this client's policy "
                f"(${breach['excess_value']:,.0f} over)."
            )
        elif breach["kind"] == "sector":
            flaws.append(
                f"{breach['key']} accounts for {breach['weight']:.0%} of invested assets, "
                f"above the {breach['limit']:.0%} sector limit "
                f"(${breach['excess_value']:,.0f} over)."
            )
        elif breach["kind"] == "cash_high":
            flaws.append(
                f"Cash is {breach['weight']:.0%} of the portfolio against a "
                f"{breach['limit']:.0%} ceiling -- ${breach['excess_value']:,.0f} sits "
                f"uninvested and is not working toward the client's goals."
            )
        elif breach["kind"] == "cash_low":
            flaws.append(
                f"Cash is only {breach['weight']:.0%} of the portfolio, below the "
                f"{breach['limit']:.0%} liquidity floor; an unexpected need would force "
                f"selling investments at whatever price is available that day."
            )

    for entry in drift:
        if not entry.breached:
            continue
        direction = "below" if entry.drift < 0 else "above"
        label = ASSET_CLASS_LABELS.get(entry.asset_class, entry.asset_class.replace("_", " "))
        flaws.append(
            f"{label} is {abs(entry.drift):.1%} {direction} its "
            f"{entry.target_weight:.0%} policy target (tolerance band {entry.band:.0%}), a gap "
            f"of ${abs(entry.dollar_gap):,.0f}."
        )

    volatility = stats.get("annual_volatility")
    if (
        volatility is not None
        and policy.max_portfolio_volatility is not None
        and volatility > policy.max_portfolio_volatility
    ):
        flaws.append(
            f"Realized volatility of {volatility:.0%} exceeds the "
            f"{policy.max_portfolio_volatility:.0%} ceiling for a {policy.risk_tier} mandate."
        )

    drawdown = stats.get("max_drawdown")
    if drawdown is not None and drawdown < -0.20:
        flaws.append(
            f"The portfolio fell {abs(drawdown):.0%} peak-to-trough over the measurement "
            f"window. A {policy.risk_tier} mandate should be sized so a decline of that "
            f"depth does not force a change of plan."
        )

    beta = view.portfolio_beta()
    if (
        beta is not None
        and policy.max_portfolio_beta is not None
        and beta > policy.max_portfolio_beta
    ):
        flaws.append(
            f"Portfolio beta of {beta:.2f} is above the {policy.max_portfolio_beta:.2f} "
            f"ceiling for this mandate, so it amplifies market moves in both directions."
        )

    if average_correlation is not None and average_correlation > 0.75:
        flaws.append(
            f"The holdings have an average pairwise correlation of {average_correlation:.2f}, "
            f"so they largely rise and fall together. The portfolio is less diversified than "
            f"its number of positions suggests."
        )

    for cluster in clusters[:3]:
        if len(cluster) >= 3:
            flaws.append(
                f"{', '.join(cluster)} move almost identically (correlation above "
                f"{HIGH_CORRELATION:.0%}); for risk purposes they are close to a single "
                f"position."
            )

    if effective_positions is not None and effective_positions < 4:
        holdings_count = len({h.symbol for h in view.holdings})
        if holdings_count > effective_positions + 1:
            flaws.append(
                f"There are {holdings_count} holdings, but their weights make the portfolio "
                f"behave like roughly {effective_positions:.1f} equally-sized positions."
            )

    if not view.holdings and view.cash > 0:
        flaws.append(
            f"The entire ${view.cash:,.0f} portfolio is held in cash and is not invested "
            f"toward any of the client's stated goals."
        )

    return flaws


def diagnostics_node(state: AgentState) -> dict:
    """Measure the portfolio against this client's policy."""
    diagnostics: Optional[PortfolioDiagnostics] = None

    with node_run(NODE_NAME, state) as ctx:
        profile = state.get("client_profile") or {}
        client_id = profile.get("client_id")

        db = SessionLocal()
        try:
            client = (
                db.query(ClientProfileModel).filter(ClientProfileModel.id == client_id).first()
            )
            if client is None:
                raise ValueError(f"No client profile with id {client_id}")

            policy = resolve_policy(db, client)
            view = load_portfolio(db, client)

            ctx.input_snapshot = {
                "positions": len({h.symbol for h in view.holdings}),
                "total_value": round(view.total_value, 2),
                "policy_version": policy.version,
                "policy_source": policy.source,
            }

            if view.total_value <= 0:
                ctx.degrade(
                    reason="empty_portfolio",
                    detail="No priced positions and no cash balance.",
                    impact=(
                        "There were no priced holdings to analyse, so no diagnostics or "
                        "risk measures could be produced for this run."
                    ),
                )
                ctx.summary = "No priced holdings; diagnostics skipped."
                diagnostics = PortfolioDiagnostics(
                    total_value=0.0, invested_value=0.0, cash=0.0, concentration={},
                    sector_exposure={}, asset_class_exposure={}, diversification_score=0.0,
                    flaws=[], drift=[], breaches=[], correlation_clusters=[],
                    market_data_tickers=[],
                )
            else:
                weights = view.weights()
                symbols = sorted({h.symbol for h in view.holdings})
                prices = (
                    fetch_historical_prices(symbols, years=settings.LOOKBACK_PERIOD_YEARS)
                    if symbols
                    else pd.DataFrame()
                )

                if prices.empty and symbols:
                    ctx.degrade(
                        reason="no_price_history",
                        detail=f"No provider returned history for {symbols}.",
                        impact=(
                            "Return, volatility, drawdown and correlation could not be "
                            "computed for this portfolio. The concentration and allocation "
                            "findings are unaffected."
                        ),
                    )

                stats = _annualized_stats(prices, weights, settings.RISK_FREE_RATE)
                average_correlation, clusters = _correlation_analysis(prices, weights)
                score, effective = _diversification_score(
                    {k: v for k, v in weights.items() if k != "CASH"}
                )
                drift = compute_drift(view, policy)
                breaches = concentration_breaches(view, policy)
                flaws = _build_flaws(
                    view, policy, stats, drift, breaches,
                    average_correlation, clusters, effective,
                )

                diagnostics = PortfolioDiagnostics(
                    total_value=round(view.total_value, 2),
                    invested_value=round(view.invested_value, 2),
                    cash=round(view.cash, 2),
                    concentration={k: round(v, 4) for k, v in weights.items()},
                    sector_exposure={k: round(v, 4) for k, v in view.sector_weights().items()},
                    asset_class_exposure={
                        k: round(v, 4) for k, v in view.asset_class_weights().items()
                    },
                    sharpe_ratio=stats["sharpe_ratio"],
                    annual_return=stats["annual_return"],
                    annual_volatility=stats["annual_volatility"],
                    max_drawdown=stats["max_drawdown"],
                    portfolio_beta=view.portfolio_beta(),
                    diversification_score=score,
                    effective_positions=effective,
                    average_correlation=average_correlation,
                    correlation_clusters=clusters,
                    drift=[d.to_dict() for d in drift],
                    breaches=breaches,
                    flaws=flaws,
                    market_data_tickers=list(prices.columns) if not prices.empty else [],
                )

                ctx.data_provenance = quote_provenance(view.quotes)
                ctx.output_snapshot = {
                    "flaw_count": len(flaws),
                    "breach_count": len(breaches),
                    "diversification_score": score,
                    "effective_positions": effective,
                    "average_correlation": average_correlation,
                }
                ctx.summary = (
                    f"${view.total_value:,.0f} across "
                    f"{len({h.symbol for h in view.holdings})} position(s); {len(flaws)} "
                    f"finding(s), {len(breaches)} policy breach(es)."
                )

                for warning in view.warnings:
                    if "no current price" in warning.lower():
                        ctx.degrade(
                            reason="unpriced_positions", detail=warning, impact=warning
                        )

                logger.info("[Diagnostics] %s", ctx.summary)
                for flaw in flaws[:5]:
                    logger.info("    - %s", flaw)
        finally:
            db.close()

    return finish(ctx, {"portfolio_diagnostics": diagnostics})
