"""Portfolio Diagnostics Agent.

A deterministic/quant LangGraph node -- NO LLM calls. It scores the client's
CURRENT portfolio against their OWN risk tolerance (rather than a fixed
threshold) and produces a PortfolioDiagnostics dict (see state.py) describing
concentration, sector exposure, risk/return stats, and a list of concrete,
plain-language "flaws" that downstream agents (Stock Research, Suitability)
can act on.

Ported from the legacy backend/app/analytics/risk_engine.py
(calculate_portfolio_metrics), adapted to:
  - pull prices through services/market_data.fetch_historical_prices (cached)
  - read RISK_FREE_RATE / LOOKBACK_PERIOD_YEARS from config.settings
  - exclude the "CASH" pseudo-symbol from any price-based calculation
  - scale concentration/cash/volatility flaws to the client's own risk
    tolerance instead of one-size-fits-all thresholds
"""

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

# Allow running this file directly (`python agents/diagnostics.py`) as well
# as importing it as part of the `agents` package -- both need the repo
# root on sys.path so `import config`, `import state`, etc. resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import expected_returns, risk_models

from config import settings
from services.market_data import fetch_historical_prices
from state import AgentRunRecord, AgentState, ClientProfile, PortfolioDiagnostics

# --------------------------------------------------------------------------
# Thresholds (documented, not magic numbers). Concentration limits are
# intentionally kept in lockstep with the Suitability guardrail agent, which
# reuses these exact percentages for its own hard-stop checks.
# --------------------------------------------------------------------------

# Single non-cash position, as a fraction of net worth, above which we flag
# excessive concentration risk. Scaled to how much volatility the client has
# said they can tolerate.
CONCENTRATION_LIMITS: Dict[str, float] = {
    "Conservative": 0.15,
    "Moderate": 0.25,
    "Aggressive": 0.35,
}
DEFAULT_CONCENTRATION_LIMIT = CONCENTRATION_LIMITS["Moderate"]

# Cash allocations above this fraction of net worth are "cash drag" for
# clients who said they want growth (Moderate/Aggressive) -- sitting in cash
# works against their stated objective.
CASH_DRAG_THRESHOLD = 0.30
CASH_DRAG_TOLERANCES = {"Aggressive", "Moderate"}

# Cash allocations below this fraction are a liquidity-buffer risk if the
# client has a goal that could require near-term withdrawals.
CASH_LIQUIDITY_MIN = 0.05
# A "near-term" goal is approximated as: the client has at least one stated
# goal AND a short overall time horizon. This is a best-effort heuristic --
# state.py does not currently tag individual goals with their own horizon.
NEAR_TERM_HORIZON_YEARS = 5

# A single GICS-ish sector (best-effort via yfinance) making up more than
# this fraction of the NON-CASH sleeve is flagged as sector concentration.
SECTOR_CONCENTRATION_THRESHOLD = 0.50

# Volatility mismatch: annualized volatility above this is "too hot" for a
# Conservative client; below this is "too cold" (opportunity cost) for an
# Aggressive client who presumably wants growth.
HIGH_VOLATILITY_THRESHOLD_CONSERVATIVE = 0.25
LOW_VOLATILITY_THRESHOLD_AGGRESSIVE = 0.08

NODE_NAME = "portfolio_diagnostics"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _compute_concentration(holdings: Dict[str, float], total_value: float) -> Dict[str, float]:
    if total_value <= 0:
        return {symbol: 0.0 for symbol in holdings}
    return {symbol: value / total_value for symbol, value in holdings.items()}


def _concentration_flaws(concentration: Dict[str, float], risk_tolerance: str) -> List[str]:
    limit = CONCENTRATION_LIMITS.get(risk_tolerance, DEFAULT_CONCENTRATION_LIMIT)
    flaws = []
    for symbol, frac in concentration.items():
        if symbol == "CASH":
            continue
        if frac > limit:
            flaws.append(
                f"{symbol} is {_pct(frac)} of the portfolio -- exceeds the "
                f"{_pct(limit)} concentration limit for a {risk_tolerance} risk tolerance"
            )
    return flaws


def _cash_flaws(concentration: Dict[str, float], profile: ClientProfile) -> List[str]:
    flaws: List[str] = []
    cash_frac = concentration.get("CASH", 0.0)
    risk_tolerance = profile.get("risk_tolerance", "Moderate")

    if risk_tolerance in CASH_DRAG_TOLERANCES and cash_frac > CASH_DRAG_THRESHOLD:
        flaws.append(
            f"CASH is {_pct(cash_frac)} of the portfolio -- too much cash drag for a "
            f"{risk_tolerance} risk tolerance (threshold {_pct(CASH_DRAG_THRESHOLD)})"
        )

    goals = profile.get("goals") or []
    horizon = profile.get("time_horizon_years", 0) or 0
    has_near_term_goal = bool(goals) and horizon <= NEAR_TERM_HORIZON_YEARS
    if cash_frac < CASH_LIQUIDITY_MIN and has_near_term_goal:
        flaws.append(
            f"CASH is only {_pct(cash_frac)} of the portfolio -- insufficient liquidity "
            f"buffer given a near-term goal ({', '.join(goals)}) and a "
            f"{horizon}-year time horizon"
        )
    return flaws


def _fetch_sector_exposure(non_cash_holdings: Dict[str, float]) -> Dict[str, float]:
    """Best-effort sector lookup via yfinance. Never raises -- unresolvable
    tickers are bucketed into 'Unknown'."""
    non_cash_total = sum(non_cash_holdings.values())
    if non_cash_total <= 0:
        return {}

    sector_values: Dict[str, float] = {}
    for symbol, value in non_cash_holdings.items():
        sector = "Unknown"
        try:
            info = yf.Ticker(symbol).info
            sector = info.get("sector") or "Unknown"
        except Exception:
            sector = "Unknown"
        sector_values[sector] = sector_values.get(sector, 0.0) + value

    return {sector: value / non_cash_total for sector, value in sector_values.items()}


def _sector_flaws(sector_exposure: Dict[str, float]) -> List[str]:
    flaws = []
    for sector, frac in sector_exposure.items():
        if frac > SECTOR_CONCENTRATION_THRESHOLD:
            flaws.append(
                f"{sector} sector is {_pct(frac)} of non-cash holdings -- exceeds the "
                f"{_pct(SECTOR_CONCENTRATION_THRESHOLD)} sector concentration threshold"
            )
    return flaws


def _compute_price_metrics(non_cash_holdings: Dict[str, float]) -> Dict[str, Any]:
    """Returns annual_return, annual_volatility, sharpe_ratio, market_data_tickers.
    Degrades gracefully (returns zeros / empty list) if price data can't be
    fetched for any of the tickers -- never raises."""
    tickers = list(non_cash_holdings.keys())
    result = {
        "annual_return": 0.0,
        "annual_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "market_data_tickers": [],
    }
    if not tickers:
        return result

    try:
        prices_df = fetch_historical_prices(tickers, years=settings.LOOKBACK_PERIOD_YEARS)
    except Exception:
        return result

    if prices_df is None or prices_df.empty:
        return result

    prices_df = prices_df.dropna(axis=1, how="all").ffill().dropna()
    available_tickers = [t for t in prices_df.columns.tolist() if t in non_cash_holdings]
    if not available_tickers or len(prices_df) < 2:
        return result

    weights = np.array([non_cash_holdings[t] for t in available_tickers], dtype=float)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return result
    weights = weights / weight_sum

    prices_df = prices_df[available_tickers]

    try:
        mu = expected_returns.mean_historical_return(prices_df)
        cov = risk_models.sample_cov(prices_df)

        port_return = float(np.sum(mu.values * weights))
        port_volatility = float(np.sqrt(np.dot(weights.T, np.dot(cov.values, weights))))
        sharpe = (
            (port_return - settings.RISK_FREE_RATE) / port_volatility
            if port_volatility > 0
            else 0.0
        )
    except Exception:
        return result

    result["annual_return"] = round(port_return, 4)
    result["annual_volatility"] = round(port_volatility, 4)
    result["sharpe_ratio"] = round(sharpe, 2)
    result["market_data_tickers"] = available_tickers
    return result


def _volatility_flaws(annual_volatility: float, market_data_tickers: List[str], risk_tolerance: str) -> List[str]:
    # Only evaluate when we actually have a real volatility figure -- a
    # default of 0.0 from missing price data must not be misread as
    # "suspiciously low volatility".
    if not market_data_tickers:
        return []

    flaws = []
    if risk_tolerance == "Conservative" and annual_volatility > HIGH_VOLATILITY_THRESHOLD_CONSERVATIVE:
        flaws.append(
            f"Annualized volatility of {_pct(annual_volatility)} is too high for a "
            f"Conservative risk tolerance (threshold {_pct(HIGH_VOLATILITY_THRESHOLD_CONSERVATIVE)})"
        )
    if risk_tolerance == "Aggressive" and annual_volatility < LOW_VOLATILITY_THRESHOLD_AGGRESSIVE:
        flaws.append(
            f"Annualized volatility of {_pct(annual_volatility)} is suspiciously low for an "
            f"Aggressive risk tolerance (below {_pct(LOW_VOLATILITY_THRESHOLD_AGGRESSIVE)}) -- "
            "possible opportunity cost from being too conservatively positioned"
        )
    return flaws


def _diversification_score(concentration: Dict[str, float]) -> float:
    """Herfindahl-Hirschman Index over the FULL portfolio (including CASH),
    mapped to a 0-100 score where higher is better diversified. Using the
    full concentration breakdown (not just priced tickers) means this score
    stays meaningful even when price history can't be fetched."""
    if not concentration:
        return 0.0
    weights = np.array(list(concentration.values()), dtype=float)
    if weights.sum() <= 0:
        return 0.0
    weights = weights / weights.sum()
    hhi = float(np.sum(weights ** 2))
    return round(100 * (1 - hhi), 2)


def diagnostics_node(state: AgentState) -> dict:
    started_at = _now_iso()
    profile = state["client_profile"]
    holdings = profile.get("holdings") or {}

    flaws: List[str] = []
    error_detail = None

    try:
        total_value = sum(holdings.values())
        non_cash_holdings = {s: v for s, v in holdings.items() if s != "CASH" and v > 0}

        concentration = _compute_concentration(holdings, total_value)

        flaws += _concentration_flaws(concentration, profile.get("risk_tolerance", "Moderate"))
        flaws += _cash_flaws(concentration, profile)

        sector_exposure = _fetch_sector_exposure(non_cash_holdings)
        flaws += _sector_flaws(sector_exposure)

        price_metrics = _compute_price_metrics(non_cash_holdings)
        flaws += _volatility_flaws(
            price_metrics["annual_volatility"],
            price_metrics["market_data_tickers"],
            profile.get("risk_tolerance", "Moderate"),
        )

        diversification_score = _diversification_score(concentration)

        diagnostics: PortfolioDiagnostics = {
            "concentration": concentration,
            "sector_exposure": sector_exposure,
            "sharpe_ratio": price_metrics["sharpe_ratio"],
            "annual_return": price_metrics["annual_return"],
            "annual_volatility": price_metrics["annual_volatility"],
            "diversification_score": diversification_score,
            "flaws": flaws,
            "market_data_tickers": price_metrics["market_data_tickers"],
        }
        status = "success"
        summary = (
            f"Analyzed {len(holdings)} holdings (${total_value:,.0f} total); "
            f"found {len(flaws)} flaw(s); diversification score "
            f"{diversification_score:.1f}/100."
        )
    except Exception as exc:  # noqa: BLE001 -- node must never crash the graph
        status = "error"
        error_detail = repr(exc)
        summary = "Portfolio diagnostics failed; returning safe defaults."
        diagnostics = {
            "concentration": {},
            "sector_exposure": {},
            "sharpe_ratio": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "diversification_score": 0.0,
            "flaws": [f"Diagnostics could not be computed: {exc}"],
            "market_data_tickers": [],
        }

    record: AgentRunRecord = {
        "node_name": NODE_NAME,
        "started_at": started_at,
        "completed_at": _now_iso(),
        "status": status,
        "summary": summary,
        "error_detail": error_detail,
    }

    return {"portfolio_diagnostics": diagnostics, "audit_trail": [record]}


if __name__ == "__main__":
    import json
    import uuid

    from db import ClientProfile as ClientProfileRow
    from db import Holding, SessionLocal
    from services.market_data import get_current_prices

    db = SessionLocal()
    try:
        client_row = db.query(ClientProfileRow).filter(ClientProfileRow.id == 1).first()
        if client_row is None:
            raise SystemExit("client_id=1 not found -- run `python db.py` to seed the DB first.")

        holding_rows = db.query(Holding).filter(Holding.client_id == 1).all()

        holdings: Dict[str, float] = {}
        non_cash_symbols = [h.symbol for h in holding_rows if h.symbol != "CASH"]
        current_prices = get_current_prices(non_cash_symbols) if non_cash_symbols else {}

        for h in holding_rows:
            if h.symbol == "CASH":
                holdings["CASH"] = h.quantity  # dollar amount stored directly
            else:
                price = current_prices.get(h.symbol, 0.0)
                holdings[h.symbol] = h.quantity * price

        client_profile: ClientProfile = {
            "client_id": client_row.id,
            "age": client_row.age,
            "risk_tolerance": client_row.risk_tolerance,
            "time_horizon_years": client_row.time_horizon_years,
            "goals": client_row.goals or [],
            "holdings": holdings,
            "net_worth": client_row.net_worth,
        }
    finally:
        db.close()

    sample_state: AgentState = {
        "run_id": str(uuid.uuid4()),
        "client_profile": client_profile,
        "portfolio_diagnostics": None,
        "market_regime": None,
        "candidate_stocks": None,
        "suitability_result": None,
        "tax_assessment": None,
        "final_report": None,
        "audit_trail": [],
        "requires_human_approval": False,
        "human_approved": None,
        "research_attempts": 0,
        "needs_research_retry": False,
    }

    print("Client profile / holdings used for this smoke test:")
    print(json.dumps(client_profile, indent=2))
    print()

    result = diagnostics_node(sample_state)

    print("=== portfolio_diagnostics ===")
    print(json.dumps(result["portfolio_diagnostics"], indent=2))
    print()
    print("=== audit_trail ===")
    print(json.dumps(result["audit_trail"], indent=2))
