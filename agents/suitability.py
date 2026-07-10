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

import yfinance as yf

# When this file is run directly (`python agents/suitability.py`), Python
# puts only this file's own directory (agents/) on sys.path, not the repo
# root -- so the root-level `state.py` wouldn't be importable. Insert the
# repo root explicitly so both `python agents/suitability.py` and normal
# package imports (`from agents.suitability import suitability_node`) work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import AgentRunRecord, AgentState, Candidate, SuitabilityResult

# --- Rule 2 thresholds -------------------------------------------------------
# Risk-tier position-limit caps, expressed as the max fraction of net worth
# allowed in a single position. These EXACT numbers must match the thresholds
# used by the (separately-built) Portfolio Diagnostics agent so the two
# agents tell the client a consistent story.
RISK_TIER_POSITION_CAP = {
    "Conservative": 0.15,
    "Moderate": 0.25,
    "Aggressive": 0.35,
}
DEFAULT_POSITION_CAP = 0.25  # fallback if risk_tolerance is an unrecognized value

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

    Returns None on any failure (network error, delisted ticker, empty
    response, unexpected shape, etc.) so callers can treat "couldn't verify"
    as its own explicit case instead of crashing the whole node.
    """
    try:
        info = yf.Ticker(ticker).info
        if not info or not isinstance(info, dict):
            return None
        return info
    except Exception:
        return None


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


def _check_position_limit(
    ticker: str,
    client_profile: Dict[str, Any],
    num_candidates: int,
) -> Optional[str]:
    """Rule 2: risk-tier-scaled position limit.

    Replaces the legacy flat 30%-of-net-worth cap.

    SIMULATED SIZING ASSUMPTION: `Candidate` carries no proposed dollar trade
    size -- this system doesn't do live trade sizing yet, that's a future
    execution-layer concern. To still give this rule something concrete to
    check today, we simulate a "default" position size as the smaller of:
      (a) 5% of net worth -- a conservative default single-trade size, and
      (b) an even split of available cash across every candidate under
          consideration this run (available_cash / number_of_candidates).
    This is a deliberate, documented stand-in, not an oversight: it lets the
    guardrail catch obviously-oversized recommendations today without
    pretending to know a real trade size nothing upstream has decided on yet.
    The simulated size is added to any EXISTING holding in the same symbol
    before comparing to the risk-tier cap, since a client may already be
    partially positioned.
    """
    net_worth = client_profile.get("net_worth", 0.0)
    holdings = client_profile.get("holdings", {})
    available_cash = holdings.get("CASH", 0.0)
    risk_tolerance = client_profile.get("risk_tolerance", "Moderate")

    cap_fraction = RISK_TIER_POSITION_CAP.get(risk_tolerance, DEFAULT_POSITION_CAP)

    if num_candidates <= 0 or net_worth <= 0:
        return None  # nothing meaningful to check

    simulated_position = min(0.05 * net_worth, available_cash / num_candidates)
    existing_holding = holdings.get(ticker, 0.0)
    resulting_position = existing_holding + simulated_position
    resulting_fraction = resulting_position / net_worth

    if resulting_fraction > cap_fraction:
        return (
            f"{ticker}: rejected -- simulated position (existing ${existing_holding:,.0f} + "
            f"new ${simulated_position:,.0f} = ${resulting_position:,.0f}) would be "
            f"{resulting_fraction:.1%} of net worth (${net_worth:,.0f}), exceeding the "
            f"{risk_tolerance} client's {cap_fraction:.0%} single-position cap."
        )
    return None


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
    print(">>> [Suitability/Compliance Guardrail] Evaluating candidates against client profile...")

    candidates: List[Candidate] = state.get("candidate_stocks") or []
    client_profile = state["client_profile"]
    # portfolio_diagnostics is part of this node's documented input surface,
    # but none of the three deterministic rules below need to read it
    # directly -- risk tolerance/age/horizon/holdings on the client profile
    # fully drive the checks. Referenced here (read-only) for future rules
    # that may want to cross-check against diagnosed concentration flaws.
    _diagnostics = state.get("portfolio_diagnostics")

    violations: List[str] = []
    adjusted_recommendations: List[Candidate] = []
    num_candidates = len(candidates)

    try:
        for candidate in candidates:
            ticker = candidate["ticker"]
            info = _fetch_info(ticker)

            reason = _check_security_quality(ticker, info)
            if reason is None:
                reason = _check_position_limit(ticker, client_profile, num_candidates)
            if reason is None:
                reason = _check_age_horizon_volatility(ticker, info, client_profile)

            if reason is not None:
                print(f"    [Suitability] REJECTED {ticker}: {reason}")
                violations.append(reason)
            else:
                print(f"    [Suitability] APPROVED {ticker}")
                adjusted_recommendations.append(candidate)

        approved = len(adjusted_recommendations) > 0

        suitability_result: SuitabilityResult = {
            "approved": approved,
            "violations": violations,
            "adjusted_recommendations": adjusted_recommendations,
        }

        summary = (
            f"Evaluated {num_candidates} candidate(s): {len(adjusted_recommendations)} approved, "
            f"{len(violations)} violation(s) recorded."
        )
        print(f"    [Suitability] {summary}")

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

    # Realistic sample AgentState for a Conservative client.
    # net_worth = 50,000; Conservative cap = 15% = $7,500 per single position.
    #
    # Two candidates:
    #   - AAPL: real large-cap, major-exchange ticker that clears the
    #     security-quality screen (Rule 1) and the beta check (Rule 3, since
    #     this client is 45 / 20yr horizon so the rule doesn't even apply) --
    #     but the client ALREADY holds $6,500 of AAPL, and with cash split
    #     evenly across the 2 candidates the simulated new position pushes
    #     the total to $9,000 = 18% of net worth, over the 15% cap. This is
    #     the constructed, real, triggered position-limit violation.
    #   - JNJ: real large-cap, major-exchange ticker, no existing holding,
    #     simulated position is well under the cap -- should sail through
    #     and land in adjusted_recommendations.
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
                "AAPL": 6500.0,
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
    print("\nAll assertions passed: clean candidate approved, oversized position rejected.")
