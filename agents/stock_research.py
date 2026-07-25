"""
Stock Research Agent.

Finds candidate stocks that specifically address the flaws diagnosed by the
Portfolio Diagnostics agent, while fitting the client's risk profile and the
current market regime. This is deliberately NOT a generic "find cheap
stocks" screener -- every candidate must be traceable back to (a) a specific
flaw it addresses and (b) a rationale for why it fits the current regime.

LIMITATION (documented, not a hidden shortcut): there is no live/paid stock
screener API wired into this project. Instead we maintain a small curated
universe of well-known, liquid, large/mid-cap tickers spanning multiple
sectors, and pull real fundamentals for that universe from yfinance at
research time. This means the agent can only ever recommend a stock that is
already in CANDIDATE_UNIVERSE below -- it cannot discover names outside that
list. That's a real constraint on recommendation breadth, called out here
and in the module's run report.
"""

import os
import statistics
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from logging_setup import get_logger
from services.llm import get_chat_model, invoke_with_retry
from services.market_data import get_ticker_info
from state import AgentState, AgentRunRecord, Candidate

logger = get_logger(__name__)

NODE_NAME = "stock_research"

# ---------------------------------------------------------------------------
# Curated candidate universe.
#
# ~30 well-known, liquid, large/mid-cap US tickers spanning the sectors that
# matter for diversification/flaw-remediation purposes: Technology,
# Healthcare, Financials, Consumer Staples, Consumer Cyclical, Energy,
# Industrials, Utilities, Materials, Communication Services, Real Estate.
# Chosen for name-recognition and liquidity (not for being "hot picks") so
# that fundamentals data is reliably available via yfinance and so the
# universe reads as a defensible, if limited, cross-section of the market
# rather than a curated stock-tip list.
# ---------------------------------------------------------------------------
CANDIDATE_UNIVERSE: List[str] = [
    # Technology
    "MSFT", "AAPL", "ORCL", "CSCO", "IBM",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK",
    # Financials
    "JPM", "BRK-B", "V", "MA", "BAC",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST",
    # Consumer Cyclical
    "HD", "MCD", "NKE",
    # Energy
    "XOM", "CVX",
    # Industrials
    "HON", "UNP", "CAT",
    # Utilities
    "NEE", "DUK",
    # Materials
    "LIN", "SHW",
    # Communication Services
    "VZ",
    # Real Estate
    "O",
]

# Loose keyword -> GICS-ish sector name map used to interpret flaw strings.
SECTOR_KEYWORDS: Dict[str, str] = {
    "technology": "Technology",
    "tech": "Technology",
    "healthcare": "Healthcare",
    "health care": "Healthcare",
    "financial": "Financial Services",
    "financials": "Financial Services",
    "consumer staples": "Consumer Defensive",
    "consumer defensive": "Consumer Defensive",
    "consumer cyclical": "Consumer Cyclical",
    "consumer discretionary": "Consumer Cyclical",
    "energy": "Energy",
    "industrial": "Industrials",
    "industrials": "Industrials",
    "utilities": "Utilities",
    "utility": "Utilities",
    "materials": "Basic Materials",
    "communication": "Communication Services",
    "real estate": "Real Estate",
}


class RankedCandidate(BaseModel):
    """Structured-output schema for the LLM ranking step."""

    ticker: str = Field(description="Ticker symbol, must be one of the provided candidates")
    addresses_flaw: str = Field(description="The exact flaw string (from the provided list) this pick addresses")
    regime_fit_rationale: str = Field(description="Why this pick fits the current market regime")
    confidence: float = Field(description="Confidence in this recommendation, 0-1", ge=0.0, le=1.0)


class RankedCandidateList(BaseModel):
    picks: List[RankedCandidate]


def _fetch_universe_fundamentals(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Pulls real fundamentals for the curated universe via yfinance.Ticker.info.
    Per-ticker loop (acceptable for this universe size, ~30 names). Field
    availability varies across tickers/yfinance versions, so every field is
    fetched defensively with .get() and simply omitted downstream if absent.
    """
    fundamentals: Dict[str, Dict[str, Any]] = {}
    for ticker in tickers:
        # Shared, TTL-cached lookup. Previously each agent fetched `.info`
        # independently and every research retry refetched the whole
        # universe from scratch -- ~120 HTTP round trips per run for the same
        # ~30 symbols, and the dominant source of run latency.
        info = get_ticker_info(ticker) or {}

        raw: Dict[str, Any] = {
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            # field name for PEG varies across yfinance versions
            "peg_ratio": info.get("pegRatio") or info.get("trailingPegRatio"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "market_cap": info.get("marketCap"),
        }
        fundamentals[ticker] = {
            "sector": info.get("sector"),
            "metrics": {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))},
        }
    return fundamentals


def _flaws_mention_sector(flaws: List[str]) -> List[str]:
    """Return the list of sector names (our vocabulary) flagged as overweight in the flaws."""
    overweight_sectors = []
    joined = " ".join(f.lower() for f in flaws)
    for keyword, sector in SECTOR_KEYWORDS.items():
        if keyword in joined and sector not in overweight_sectors:
            overweight_sectors.append(sector)
    return overweight_sectors


def _flaws_mention_cash_heavy(flaws: List[str]) -> bool:
    joined = " ".join(f.lower() for f in flaws)
    return "cash" in joined and ("too much" in joined or "excess" in joined or "underinvested" in joined or "idle" in joined)


def _flaws_mention_high_volatility(flaws: List[str]) -> bool:
    joined = " ".join(f.lower() for f in flaws)
    return "volatil" in joined or "risk" in joined or "beta" in joined


def _filter_by_flaws(
    fundamentals: Dict[str, Dict[str, Any]],
    flaws: List[str],
    client_profile: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Narrow the universe based on simple keyword heuristics over the flaw
    strings (see module docstring: loose keyword matching is intentional,
    not full NLP).
    """
    filtered = dict(fundamentals)

    # 1. Overweight-sector flaws -> prefer OTHER sectors.
    overweight_sectors = _flaws_mention_sector(flaws)
    if overweight_sectors:
        diversifying = {
            t: d for t, d in filtered.items()
            if d.get("sector") and d["sector"] not in overweight_sectors
        }
        if diversifying:
            filtered = diversifying

    # 2. Too-much-cash flaw -> prefer higher-yield defensive names.
    # NOTE: yfinance's `dividendYield` field is already expressed in percent
    # units in the installed version (e.g. 6.7 means 6.7%, not 0.067), so
    # the threshold below is a percent, not a fraction. Verified empirically
    # against known real dividend yields (VZ=6.7, JPM=1.79, AAPL=0.34, ...).
    if _flaws_mention_cash_heavy(flaws):
        yielding = {
            t: d for t, d in filtered.items()
            if d["metrics"].get("dividend_yield", 0.0) and d["metrics"]["dividend_yield"] > 1.5
        }
        if yielding:
            filtered = yielding

    # 3. High-volatility flaw (esp. for a Conservative client) -> prefer lower-beta names.
    risk_tolerance = str(client_profile.get("risk_tolerance", "")).lower()
    if _flaws_mention_high_volatility(flaws) or risk_tolerance == "conservative":
        low_beta = {
            t: d for t, d in filtered.items()
            if d["metrics"].get("beta") is not None and d["metrics"]["beta"] < 1.1
        }
        if low_beta:
            filtered = low_beta

    return filtered


def _filter_by_cheapness(fundamentals: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Prefer candidates with below-median P/E OR below-median P/B within the
    filtered subset (relative valuation, no absolute magic number). Falls
    back to the full filtered set if too few names have usable valuation
    data to compute a median.
    """
    pes = [d["metrics"]["pe_ratio"] for d in fundamentals.values() if "pe_ratio" in d["metrics"]]
    pbs = [d["metrics"]["price_to_book"] for d in fundamentals.values() if "price_to_book" in d["metrics"]]

    median_pe = statistics.median(pes) if len(pes) >= 3 else None
    median_pb = statistics.median(pbs) if len(pbs) >= 3 else None

    if median_pe is None and median_pb is None:
        return list(fundamentals.keys())

    cheap: List[str] = []
    for ticker, d in fundamentals.items():
        pe = d["metrics"].get("pe_ratio")
        pb = d["metrics"].get("price_to_book")
        below_pe = median_pe is not None and pe is not None and pe <= median_pe
        below_pb = median_pb is not None and pb is not None and pb <= median_pb
        if below_pe or below_pb:
            cheap.append(ticker)

    return cheap if cheap else list(fundamentals.keys())


def _valuation_score(metrics: Dict[str, float]) -> float:
    """Lower is 'cheaper'. Used only to rank the deterministic fallback."""
    score = 0.0
    count = 0
    if "pe_ratio" in metrics:
        score += metrics["pe_ratio"]
        count += 1
    if "price_to_book" in metrics:
        score += metrics["price_to_book"] * 3  # roughly normalize scale vs P/E
        count += 1
    if "peg_ratio" in metrics:
        score += metrics["peg_ratio"] * 10
        count += 1
    return score / count if count else float("inf")


def _pick_driving_flaw(flaws: List[str]) -> str:
    """Best-effort: pick the flaw that most plausibly drove the filtering, for fallback labeling."""
    if not flaws:
        return "No specific flaw identified; general diversification improvement."
    for flaw in flaws:
        lower = flaw.lower()
        if any(k in lower for k in SECTOR_KEYWORDS) or "concentrat" in lower or "cash" in lower or "volatil" in lower:
            return flaw
    return flaws[0]


def _deterministic_fallback(
    filtered_fundamentals: Dict[str, Dict[str, Any]],
    cheap_tickers: List[str],
    flaws: List[str],
    market_regime: Dict[str, Any],
) -> List[Candidate]:
    ranked = sorted(
        cheap_tickers,
        key=lambda t: _valuation_score(filtered_fundamentals[t]["metrics"]),
    )[:5]

    driving_flaw = _pick_driving_flaw(flaws)
    regime_label = market_regime.get("regime_label", "Unknown") if market_regime else "Unknown"

    candidates: List[Candidate] = []
    for ticker in ranked:
        candidates.append(
            Candidate(
                ticker=ticker,
                valuation_metrics=dict(filtered_fundamentals[ticker]["metrics"]),
                addresses_flaw=driving_flaw,
                regime_fit_rationale=(
                    f"LLM ranking step failed (see audit_trail error_detail); this is an "
                    f"unranked deterministic fallback selected purely on relative valuation "
                    f"(below-median P/E or P/B within the flaw-filtered universe). No regime-fit "
                    f"analysis was performed for the '{regime_label}' regime."
                ),
                confidence=0.0,
            )
        )
    return candidates


def _llm_rank(
    filtered_fundamentals: Dict[str, Dict[str, Any]],
    cheap_tickers: List[str],
    flaws: List[str],
    client_profile: Dict[str, Any],
    market_regime: Dict[str, Any],
    guardrail_feedback: Optional[List[str]] = None,
) -> List[Candidate]:
    """
    Ask the LLM to pick the top 3-5 from the filtered+cheap candidate set and
    write addresses_flaw / regime_fit_rationale / confidence for each.
    Raises on any failure -- caller is responsible for catching and falling
    back to _deterministic_fallback.

    On a retry pass, `guardrail_feedback` carries the reasons the downstream
    compliance and tax agents rejected the previous picks, so the model has a
    reason to choose differently instead of deterministically reproducing the
    same rejected list.
    """
    llm = get_chat_model(temperature=0.2)
    structured_llm = llm.with_structured_output(RankedCandidateList)

    candidate_lines = []
    for ticker in cheap_tickers:
        d = filtered_fundamentals[ticker]
        candidate_lines.append(f"- {ticker} (sector={d.get('sector')}, metrics={d['metrics']})")

    feedback_block = ""
    if guardrail_feedback:
        feedback_block = (
            "\nPREVIOUS ATTEMPT REJECTED -- the compliance/suitability and tax agents "
            "rejected your earlier picks for these reasons. Those tickers have already "
            "been removed from the list above. Do not repeat the same kind of mistake:\n"
            + "\n".join(f"- {reason}" for reason in guardrail_feedback)
            + "\n"
        )

    prompt = f"""You are the Stock Research Agent inside a multi-agent wealth-management system.
{feedback_block}

Client profile:
- risk_tolerance: {client_profile.get('risk_tolerance')}
- age: {client_profile.get('age')}
- time_horizon_years: {client_profile.get('time_horizon_years')}
- goals: {client_profile.get('goals')}

Diagnosed portfolio flaws (plain language, from the Portfolio Diagnostics agent):
{chr(10).join(f"- {f}" for f in flaws) if flaws else "- (none identified)"}

Current market regime:
- regime_label: {market_regime.get('regime_label') if market_regime else 'Unknown'}
- narrative: {market_regime.get('narrative') if market_regime else 'N/A'}

Candidate stocks (already pre-filtered for sector diversification / defensiveness / cheapness
relative to this candidate set -- do not consider stocks outside this list):
{chr(10).join(candidate_lines)}

Select the 3 to 5 candidates that BEST address the diagnosed flaws while fitting the client's
risk profile and the current market regime. For each pick, set:
- ticker: must be exactly one of the tickers listed above
- addresses_flaw: copy the exact flaw string above that this pick most directly remediates
- regime_fit_rationale: 1-2 sentences on why this pick suits the current regime
- confidence: your confidence in this recommendation, 0 to 1
"""

    result = invoke_with_retry(
        lambda: structured_llm.invoke(prompt), what="Stock Research ranking"
    )
    picks = result.picks if isinstance(result, RankedCandidateList) else result["picks"]

    candidates: List[Candidate] = []
    for pick in picks:
        ticker = pick.ticker if hasattr(pick, "ticker") else pick["ticker"]
        if ticker not in filtered_fundamentals:
            # LLM hallucinated a ticker outside the provided set; skip it
            continue
        addresses_flaw = pick.addresses_flaw if hasattr(pick, "addresses_flaw") else pick["addresses_flaw"]
        rationale = pick.regime_fit_rationale if hasattr(pick, "regime_fit_rationale") else pick["regime_fit_rationale"]
        confidence = pick.confidence if hasattr(pick, "confidence") else pick["confidence"]
        candidates.append(
            Candidate(
                ticker=ticker,
                valuation_metrics=dict(filtered_fundamentals[ticker]["metrics"]),
                addresses_flaw=addresses_flaw,
                regime_fit_rationale=rationale,
                confidence=float(confidence),
            )
        )

    if not candidates:
        raise ValueError("LLM returned no usable picks from the provided candidate set")

    return candidates[:5]


def stock_research_node(state: AgentState) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    attempt = state.get("research_attempts", 0) + 1
    logger.info(
        ">>> [Stock Research] Screening candidate universe against diagnosed flaws "
        "(attempt %d)...", attempt,
    )

    diagnostics = state.get("portfolio_diagnostics") or {}
    flaws: List[str] = diagnostics.get("flaws", []) or []
    client_profile = state.get("client_profile") or {}
    market_regime = state.get("market_regime") or {}

    # Retry feedback from the guardrail gate. On the first pass both are
    # empty; on a retry these are what make the second pass differ from the
    # first. Without them the retry edge just re-derives the same answer.
    excluded: set = set(state.get("excluded_tickers") or [])
    guardrail_feedback: List[str] = list(state.get("guardrail_feedback") or [])

    fundamentals = _fetch_universe_fundamentals(CANDIDATE_UNIVERSE)
    have_data = {t: d for t, d in fundamentals.items() if d["metrics"]}
    logger.info(
        "    [Stock Research] Fetched fundamentals for %d/%d tickers",
        len(have_data), len(CANDIDATE_UNIVERSE),
    )

    if excluded:
        before = len(have_data)
        have_data = {t: d for t, d in have_data.items() if t not in excluded}
        logger.info(
            "    [Stock Research] Excluding %d previously-rejected ticker(s) %s (%d -> %d)",
            before - len(have_data), sorted(excluded), before, len(have_data),
        )

    filtered_by_flaws = _filter_by_flaws(have_data, flaws, client_profile)
    logger.info(
        "    [Stock Research] After flaw/risk filtering: %s", sorted(filtered_by_flaws.keys())
    )

    cheap_tickers = _filter_by_cheapness(filtered_by_flaws)
    logger.info("    [Stock Research] After cheapness filtering: %s", sorted(cheap_tickers))

    status = "success"
    error_detail: Optional[str] = None

    if not cheap_tickers:
        # Universe fully filtered out (e.g. no fundamentals data at all, or
        # everything excluded by prior rejections) -- relax back to whatever
        # is left that hasn't been explicitly rejected, so we still return
        # something rather than an empty recommendation set.
        relaxed = {t: d for t, d in (have_data or fundamentals).items() if t not in excluded}
        cheap_tickers = list(relaxed.keys())
        filtered_by_flaws = relaxed

    if not cheap_tickers:
        summary = (
            "No candidates remain: every ticker in the curated universe has already been "
            "rejected by the compliance/tax guardrails on a previous attempt."
        )
        logger.warning("    [Stock Research] %s", summary)
        return {
            "candidate_stocks": [],
            "audit_trail": [
                AgentRunRecord(
                    node_name=NODE_NAME,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    status="success",
                    summary=summary,
                    error_detail=None,
                )
            ],
        }

    try:
        candidates = _llm_rank(
            filtered_by_flaws, cheap_tickers, flaws, client_profile, market_regime,
            guardrail_feedback=guardrail_feedback,
        )
        summary = (
            f"LLM-ranked {len(candidates)} candidates from a flaw/cheapness-filtered pool "
            f"of {len(cheap_tickers)} (attempt {attempt})."
        )
    except Exception as e:
        status = "error"
        error_detail = f"LLM ranking failed, used deterministic fallback: {e}"
        logger.warning("    [Stock Research] %s", error_detail)
        candidates = _deterministic_fallback(filtered_by_flaws, cheap_tickers, flaws, market_regime)
        summary = (
            f"LLM ranking failed; deterministic fallback selected {len(candidates)} "
            f"candidates by relative valuation (attempt {attempt})."
        )

    # Belt-and-braces: never emit an excluded ticker even if the ranking step
    # somehow surfaced one. This is the last point before the guardrails.
    candidates = [c for c in candidates if c["ticker"] not in excluded]

    logger.info("    [Stock Research] Final candidates: %s", [c["ticker"] for c in candidates])

    audit_record = AgentRunRecord(
        node_name=NODE_NAME,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        summary=summary,
        error_detail=error_detail,
    )

    return {
        "candidate_stocks": candidates,
        "audit_trail": [audit_record],
    }


if __name__ == "__main__":
    sample_state: AgentState = {
        "run_id": "test-run-1",
        "client_profile": {
            "client_id": 1,
            "age": 45,
            "risk_tolerance": "Moderate",
            "time_horizon_years": 20,
            "goals": ["retirement", "growth"],
            "holdings": {"TSLA": 45000.0, "CASH": 5000.0, "GM": 5000.0},
            "net_worth": 55000.0,
        },
        "portfolio_diagnostics": {
            "concentration": {"TSLA": 0.818},
            "sector_exposure": {"Consumer Cyclical": 1.0},
            "sharpe_ratio": 0.4,
            "annual_return": 0.12,
            "annual_volatility": 0.55,
            "diversification_score": 15.0,
            "flaws": [
                "TSLA is 45% of the portfolio -- exceeds the 25% concentration limit for a Moderate risk tolerance",
                "100% of non-cash holdings are in the Consumer Cyclical sector",
            ],
            "market_data_tickers": ["TSLA", "GM"],
        },
        "market_regime": {
            "regime_label": "Bull",
            "confidence": 0.7,
            "supporting_signals": {"SPY_200dma_trend": "up"},
            "narrative": "Broad market indices are in a sustained uptrend with low volatility.",
        },
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

    result = stock_research_node(sample_state)

    print("\n=== FINAL RESULT ===")
    import json
    print(json.dumps(result, indent=2, default=str))
