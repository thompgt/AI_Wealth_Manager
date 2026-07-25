"""
Suitability / Compliance Guardrail Agent.

Deterministic, rule-based compliance gate for candidate stock recommendations.
NO LLM calls anywhere in this file -- every check here must be auditable,
reproducible rule evaluation, because this is the gate between "the system
thinks this is a good idea" and "this becomes part of a client-facing
recommendation" for a system that (eventually) touches real portfolios.

Replaces the legacy `agents/compliance_critic.py`, which had exactly two
low-quality rules:
  1. A literal string match banning the ticker "DOGE" as its only "penny
     stock"/security-quality check.
  2. A flat 30%-of-net-worth position cap that ignored the client's risk
     tolerance, age, or time horizon entirely.

Both are replaced below with real, client-profile-driven logic backed by live
yfinance data.

Control-flow pattern KEPT from the legacy module: evaluate a list of proposed
changes (here, `candidate_stocks`), collect a list of human-readable violation
strings, and gate on whether that list is empty. The one deliberate behavioral
change: the legacy gate raised an exception on any violation, which is correct
for a live-trading gate where a bad trade must never execute. This system only
produces advisory reports/recommendations -- nothing here executes a trade --
so a per-candidate rejection is logged and the candidate is filtered out, not
treated as fatal. `approved` is True as long as at least one candidate
survives every check (not all-or-nothing).
"""

import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# When this file is run directly (`python agents/suitability.py`), Python
# puts only this file's own directory (agents/) on sys.path, not the repo
# root -- so the root-level `state.py` wouldn't be importable. Insert the
# repo root explicitly so both `python agents/suitability.py` and normal
# package imports (`from agents.suitability import suitability_node`) work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_setup import get_logger
from services.market_data import get_ticker_info
from state import AgentRunRecord, AgentState, Candidate, SuitabilityResult

logger = get_logger(__name__)

# --- Rule 2 thresholds -------------------------------------------------------
# Risk-tier position caps and the portfolio base they are measured against
# both live in agents/limits.py, shared with the Portfolio Diagnostics agent.
# See that module for why the denominator had to be unified, not just the
# percentages.
from agents.limits import (  # noqa: E402 -- after the sys.path bootstrap above
    DEFAULT_POSITION_CAP,
    RISK_TIER_POSITION_CAP,
    portfolio_base_value,
    position_cap_fraction,
)

# --- Rule 1 thresholds --------------------------------------------------------
MIN_MARKET_CAP = 2_000_000_000  # $2B small-cap/microcap threshold

# Exchange codes as ACTUALLY returned by yfinance's `Ticker.info["exchange"]`
# field. Verified live against real tickers while building this module:
#   NMS = Nasdaq Global Select Market  (AAPL, MSFT, SIRI all returned "NMS")
#   NYQ = New York Stock Exchange      (JNJ, KO, GME, IONQ all returned "NYQ")
#   PCX = NYSE Arca                    (SPY -- mostly ETFs)
#   NGM = Nasdaq Global Market, NCM = Nasdaq Capital Market, ASE = NYSE
#         American (formerly AMEX) -- not hit in the live sample above, but
#         documented Nasdaq/NYSE tiers, included as major exchanges.
# Confirmed OTC / pink-sheet codes that MUST be rejected (also verified live):
#   PNK = OTC Markets Pink Sheets      (SIRC, BBIG both returned "PNK",
#                                        fullExchangeName "OTC Markets OTCPK")
#   OQX = OTC Markets OTCQX            (BAYRY returned "OQX")
#   OID = OTC Markets OTCID (foreign ADRs) (NSRGY returned "OID")
#   OQB = OTC Markets OTCQB (documented OTC tier, not separately spot-checked)
MAJOR_EXCHANGES = {"NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "BATS"}

# --- Rule 3 thresholds --------------------------------------------------------
AGE_THRESHOLD = 60
HORIZON_THRESHOLD_YEARS = 5
MAX_BETA_NEAR_RETIREMENT = 1.5


def _fetch_info(ticker: str) -> Optional[Dict[str, Any]]:
    """Best-effort fetch of yfinance's `.info` dict for a ticker.

    Delegates to the shared, TTL-cached lookup in services.market_data so
    that a symbol already screened by Stock Research this run is not fetched
    over the network a second time. Returns None on any failure so callers
    can treat "couldn't verify" as its own explicit case.
    """
    return get_ticker_info(ticker)


def _check_security_quality(ticker: str, info: Optional[Dict[str, Any]]) -> Optional[str]:
    """Rule 1: reject micro/small caps and OTC/pink-sheet listings.

    Replaces the legacy hardcoded `asset == "DOGE"` string match. If we
    can't fetch verifiable quality data at all, that itself is a violation --
    we do not silently let unverifiable tickers through.
    """
    if info is None:
        return f"{ticker}: rejected -- could not fetch verifiable market data (yfinance info unavailable)."

    market_cap = info.get("marketCap")
    exchange = info.get("exchange")

    if market_cap is None:
        return f"{ticker}: rejected -- market cap unavailable, cannot verify security quality."
    if market_cap < MIN_MARKET_CAP:
        return (
            f"{ticker}: rejected -- market cap ${market_cap:,.0f} is below the "
            f"${MIN_MARKET_CAP:,.0f} small-cap/microcap threshold."
        )
    if exchange is None:
        return f"{ticker}: rejected -- exchange listing unavailable, cannot verify it trades on a major exchange."
    if exchange not in MAJOR_EXCHANGES:
        return (
            f"{ticker}: rejected -- listed on '{exchange}' "
            f"({info.get('fullExchangeName', 'unknown exchange')}), not a major exchange (NYSE/NASDAQ)."
        )
    return None


MIN_ALLOCATION_DOLLARS = 100.0


def _compute_allocations(
    candidates: List[Candidate],
    client_profile: Dict[str, Any],
) -> Dict[str, float]:
    """Rule 2 + sizing: decide how many actual dollars to put into each
    surviving candidate, replacing the legacy flat 30%-of-net-worth cap AND
    the old "does this cross a limit" yes/no check with a real portfolio-
    construction decision.

    Available CASH is distributed across candidates weighted by each
    candidate's confidence score (equal-weighted if every confidence is 0),
    with each ticker capped at the client's risk-tier position limit (as a
    fraction of the managed portfolio's value, net of any existing holding in
    that symbol). Cash that would overflow a capped-out ticker is
    redistributed across the remaining open tickers in further passes (simple
    water-filling), so available cash gets used even when confidence is
    skewed toward a name that's already near its cap.
    """
    holdings = client_profile.get("holdings", {})
    available_cash = holdings.get("CASH", 0.0)
    # Measured against the same base Portfolio Diagnostics uses for its
    # concentration flaws -- see agents/limits.py.
    base_value = portfolio_base_value(client_profile)
    cap_fraction = position_cap_fraction(client_profile.get("risk_tolerance", "Moderate"))

    tickers = [c["ticker"] for c in candidates]
    allocation = {t: 0.0 for t in tickers}
    if not tickers or available_cash <= 0 or base_value <= 0:
        return allocation

    cap_dollar = cap_fraction * base_value
    max_amount = {t: max(0.0, cap_dollar - holdings.get(t, 0.0)) for t in tickers}

    confidences = {c["ticker"]: max(c.get("confidence", 0.0), 0.0) for c in candidates}
    total_conf = sum(confidences.values())
    weight = (
        {t: v / total_conf for t, v in confidences.items()}
        if total_conf > 0
        else {t: 1.0 / len(tickers) for t in tickers}
    )

    remaining_cash = available_cash
    open_tickers = set(tickers)
    for _ in range(5):  # water-filling passes -- converges fast for <=5 candidates
        if remaining_cash <= 0 or not open_tickers:
            break
        open_weight_sum = sum(weight[t] for t in open_tickers)
        if open_weight_sum <= 0:
            break
        distributed = 0.0
        newly_capped = []
        for t in list(open_tickers):
            share = remaining_cash * (weight[t] / open_weight_sum)
            room = max_amount[t] - allocation[t]
            give = min(share, room)
            allocation[t] += give
            distributed += give
            if allocation[t] >= max_amount[t] - 1e-9:
                newly_capped.append(t)
        remaining_cash -= distributed
        for t in newly_capped:
            open_tickers.discard(t)
        if distributed <= 1e-9:
            break

    return allocation


def _check_age_horizon_volatility(
    ticker: str,
    info: Optional[Dict[str, Any]],
    client_profile: Dict[str, Any],
) -> Optional[str]:
    """Rule 3: near-retirement / short-horizon clients shouldn't get
    high-beta names.

    Beta is commonly missing from yfinance for many tickers (unlike market
    cap/exchange, which are near-universal) -- so unlike Rule 1, a missing
    beta here just skips this specific check for this candidate rather than
    being treated as its own violation.
    """
    age = client_profile.get("age", 0)
    horizon = client_profile.get("time_horizon_years", 99)

    if age < AGE_THRESHOLD and horizon > HORIZON_THRESHOLD_YEARS:
        return None  # rule doesn't apply to this client

    if info is None:
        return None  # can't verify beta at all; don't block solely for missing data here

    beta = info.get("beta")
    if beta is None:
        return None  # beta commonly unavailable on yfinance; skip rather than reject

    if beta > MAX_BETA_NEAR_RETIREMENT:
        return (
            f"{ticker}: rejected -- beta {beta:.2f} exceeds the {MAX_BETA_NEAR_RETIREMENT} max "
            f"for a client who is age {age} and/or has a {horizon}-year time horizon "
            f"(near-retirement/short-horizon volatility guardrail)."
        )
    return None


def suitability_node(state: AgentState) -> dict:
    """LangGraph node: evaluate `state['candidate_stocks']` against the
    client's profile via three deterministic rules and return a
    `SuitabilityResult` plus this node's own audit_trail entry.

    Returns ONLY {"suitability_result": ..., "audit_trail": [record]} --
    never the accumulated audit_trail list, since `AgentState.audit_trail`
    is `Annotated[..., operator.add]` and LangGraph concatenates it.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    logger.info(">>> [Suitability/Compliance Guardrail] Evaluating candidates against client profile...")

    candidates: List[Candidate] = state.get("candidate_stocks") or []
    client_profile = state["client_profile"]
    # portfolio_diagnostics is part of this node's documented input surface,
    # but none of the three deterministic rules below need to read it
    # directly -- risk tolerance/age/horizon/holdings on the client profile
    # fully drive the checks. Referenced here (read-only) for future rules
    # that may want to cross-check against diagnosed concentration flaws.
    _diagnostics = state.get("portfolio_diagnostics")

    violations: List[str] = []
    quality_passed: List[Candidate] = []

    try:
        for candidate in candidates:
            ticker = candidate["ticker"]
            info = _fetch_info(ticker)

            reason = _check_security_quality(ticker, info)
            if reason is None:
                reason = _check_age_horizon_volatility(ticker, info, client_profile)

            if reason is not None:
                logger.info("    [Suitability] REJECTED %s: %s", ticker, reason)
                violations.append(reason)
            else:
                quality_passed.append(candidate)

        base_value = portfolio_base_value(client_profile)
        allocations = _compute_allocations(quality_passed, client_profile)

        adjusted_recommendations: List[Candidate] = []
        for candidate in quality_passed:
            ticker = candidate["ticker"]
            amount = allocations.get(ticker, 0.0)
            if amount < MIN_ALLOCATION_DOLLARS:
                logger.info("    [Suitability] REJECTED %s: no available allocation room", ticker)
                violations.append(
                    f"{ticker}: rejected -- no meaningful room to add (${amount:,.0f} available) "
                    f"after risk-tier position caps and existing holdings were applied."
                )
                continue
            logger.info("    [Suitability] APPROVED %s: allocating $%s", ticker, f"{amount:,.0f}")
            adjusted_recommendations.append(
                {
                    **candidate,
                    "allocation_amount": round(amount, 2),
                    "allocation_pct": round(amount / base_value, 4) if base_value > 0 else 0.0,
                }
            )

        approved = len(adjusted_recommendations) > 0

        suitability_result: SuitabilityResult = {
            "approved": approved,
            "violations": violations,
            "adjusted_recommendations": adjusted_recommendations,
        }

        summary = (
            f"Evaluated {len(candidates)} candidate(s): {len(adjusted_recommendations)} approved "
            f"with a total of ${sum(c['allocation_amount'] for c in adjusted_recommendations):,.0f} "
            f"allocated, {len(violations)} violation(s) recorded."
        )
        logger.info("    [Suitability] %s", summary)

        record: AgentRunRecord = {
            "node_name": "suitability",
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "summary": summary,
            "error_detail": None,
        }

        return {"suitability_result": suitability_result, "audit_trail": [record]}

    except Exception as e:
        # Fail closed: an unexpected error means we can't vouch for any
        # candidate this run, so nothing is approved.
        record: AgentRunRecord = {
            "node_name": "suitability",
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "summary": "Suitability node failed with an unexpected error.",
            "error_detail": f"{e}\n{traceback.format_exc()}",
        }
        suitability_result: SuitabilityResult = {
            "approved": False,
            "violations": [f"Suitability node error: {e}"],
            "adjusted_recommendations": [],
        }
        return {"suitability_result": suitability_result, "audit_trail": [record]}


if __name__ == "__main__":
    import json

    from logging_setup import configure_logging

    configure_logging()

    # Realistic sample AgentState for a Conservative client.
    # Portfolio base = CASH 5,000 + AAPL 7,500 = $12,500 -- the managed
    # portfolio value, which is what position caps are measured against (see
    # agents/limits.py). Conservative cap = 15% = $1,875 per single position.
    #
    # Two candidates, $5,000 available CASH to deploy:
    #   - AAPL: real large-cap, major-exchange ticker that clears the
    #     security-quality screen (Rule 1) and the beta check (Rule 3, since
    #     this client is 45 / 20yr horizon so the rule doesn't even apply) --
    #     but the client ALREADY holds $7,500 of AAPL, exactly at the 15%
    #     Conservative cap, so there is zero room left to add more. Expect
    #     it to clear the quality/beta checks but get rejected at the sizing
    #     step for having no available allocation room.
    #   - JNJ: real large-cap, major-exchange ticker, no existing holding --
    #     should receive the full $5,000 of available cash (nothing left to
    #     compete for it once AAPL is capped out) and land in
    #     adjusted_recommendations with that allocation.
    sample_state: AgentState = {
        "run_id": "test-run-001",
        "client_profile": {
            "client_id": 1,
            "age": 45,
            "risk_tolerance": "Conservative",
            "time_horizon_years": 20,
            "goals": ["retirement"],
            "holdings": {
                "CASH": 5000.0,
                "AAPL": 7500.0,
            },
            "net_worth": 50000.0,
        },
        "portfolio_diagnostics": {
            "concentration": {"AAPL": 0.13},
            "sector_exposure": {"Technology": 0.13},
            "sharpe_ratio": 0.8,
            "annual_return": 0.09,
            "annual_volatility": 0.18,
            "diversification_score": 55.0,
            "flaws": ["13% concentrated in AAPL"],
            "market_data_tickers": ["AAPL"],
        },
        "market_regime": None,
        "candidate_stocks": [
            {
                "ticker": "AAPL",
                "valuation_metrics": {"pe_ratio": 30.0},
                "addresses_flaw": "13% concentrated in AAPL",  # deliberately bad pick, exercises the guardrail
                "regime_fit_rationale": "test candidate for position-limit violation",
                "confidence": 0.6,
            },
            {
                "ticker": "JNJ",
                "valuation_metrics": {"pe_ratio": 15.0},
                "addresses_flaw": "13% concentrated in AAPL",
                "regime_fit_rationale": "defensive healthcare diversifier",
                "confidence": 0.8,
            },
        ],
        "suitability_result": None,
        "tax_assessment": None,
        "final_report": None,
        "audit_trail": [],
        "requires_human_approval": False,
        "human_approved": None,
        "research_attempts": 0,
        "needs_research_retry": False,
    }

    result = suitability_node(sample_state)

    print("\n=== suitability_node output ===")
    print(json.dumps(result, indent=2, default=str))

    sr = result["suitability_result"]
    print("\n=== Assertions ===")
    approved_tickers = [c["ticker"] for c in sr["adjusted_recommendations"]]
    print("approved tickers:", approved_tickers)
    print("violations:", sr["violations"])
    assert "JNJ" in approved_tickers, "expected JNJ to be approved"
    assert any("AAPL" in v and "cap" in v for v in sr["violations"]), "expected AAPL position-limit violation"
    jnj = next(c for c in sr["adjusted_recommendations"] if c["ticker"] == "JNJ")
    # Portfolio base $12,500 x 15% Conservative cap = $1,875 max per position,
    # so JNJ is capped there rather than taking all $5,000 of available cash.
    assert jnj["allocation_amount"] == 1875.0, (
        f"expected JNJ to be capped at $1,875 (15% of the $12,500 portfolio), "
        f"got {jnj['allocation_amount']}"
    )
    print("\nAll assertions passed: clean candidate sized and capped, capped-out position rejected.")
