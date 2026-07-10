"""
Market Regime Agent (HYBRID node).

Replaces the legacy agents/macro_sentinel.py, which asked Gemini to guess a
market condition off a single hardcoded prompt with zero real data, and
silently defaulted to "Bull" -- the most dangerous possible fail-open -- on
any API error.

This node instead:
  1. Deterministically fetches real price history for a basket of macro
     indicator tickers/ratios and computes auditable trend signals from them
     (services/market_data.py, real yfinance data, no LLM involved).
  2. Fetches recent macro news headlines (services/news_service.py).
  3. Feeds both to a Gemini LLM (langchain_google_genai) to synthesize a
     regime label / confidence / narrative, explicitly instructed to ground
     its call in the provided deterministic signals rather than free-associate.
  4. On ANY LLM failure (auth error, timeout, malformed output, etc.), falls
     back to regime_label="Volatile", confidence=0.0, and an explanatory
     narrative -- never a fixed optimistic default like "Bull". The failure
     is recorded as status="error" with error_detail in the AgentRunRecord,
     and the exception is swallowed rather than propagated, so it can never
     crash the graph.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Allow running this file directly (`python agents/market_regime.py`) by
# ensuring the repo root is importable, since that invocation sets
# sys.path[0] to agents/ rather than the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from pydantic import BaseModel, Field

from config import settings
from services.market_data import fetch_historical_prices
from services.news_service import search_financial_news
from state import AgentRunRecord, AgentState, MarketRegime

NODE_NAME = "market_regime"

# --------------------------------------------------------------------------
# Indicator basket
# --------------------------------------------------------------------------

# Every raw ticker we need to pull from yfinance in one batch.
ALL_TICKERS: List[str] = [
    "SPY", "IWM",       # small-cap/large-cap
    "HYG", "LQD",       # high-yield vs IG credit
    "SHY", "TLT",       # short vs long duration
    "^VIX",             # fear gauge
    "DX-Y.NYB",         # dollar strength
    "HG=F",             # copper
    "SMH",              # semis
    "XLU", "XLP",       # defensives
]

# Ratio pairs to compute a trend for: (name, numerator, denominator-or-avg-of)
RATIO_SIGNALS: List[Tuple[str, str, List[str]]] = [
    ("small_cap_vs_large_cap (IWM/SPY)", "IWM", ["SPY"]),
    ("high_yield_vs_ig_credit (HYG/LQD)", "HYG", ["LQD"]),
    ("short_vs_long_duration (SHY/TLT)", "SHY", ["TLT"]),
    ("semis_vs_broad_market (SMH/SPY)", "SMH", ["SPY"]),
    ("defensives_vs_broad_market (avg(XLU,XLP)/SPY)", "__DEFENSIVES_AVG__", ["SPY"]),
]

# Single-ticker level/pct-change signals (not ratios).
SOLO_SIGNALS: List[str] = ["^VIX", "DX-Y.NYB", "HG=F", "SPY", "IWM", "HYG", "LQD", "SHY", "TLT", "SMH", "XLU", "XLP"]

NEWS_KEYWORDS = ["Federal Reserve", "market outlook", "recession"]

MACRO_LOOKBACK_YEARS = 0.5  # ~6 months


# --------------------------------------------------------------------------
# Deterministic signal extraction
# --------------------------------------------------------------------------

def _pct_change(series: pd.Series) -> Optional[float]:
    """Percent change from first to last non-null value in the window."""
    clean = series.dropna()
    if len(clean) < 2:
        return None
    first, last = float(clean.iloc[0]), float(clean.iloc[-1])
    if first == 0:
        return None
    return (last - first) / first * 100.0


def _ratio_series(prices: pd.DataFrame, numerator: str, denom_cols: List[str]) -> Optional[pd.Series]:
    """Build a ratio series: numerator / mean(denom_cols), aligned & dropna."""
    if numerator == "__DEFENSIVES_AVG__":
        needed = denom_cols + ["XLU", "XLP"]
    else:
        needed = denom_cols + [numerator]
    for col in needed:
        if col not in prices.columns:
            return None

    df = prices[needed].dropna()
    if df.empty:
        return None

    if numerator == "__DEFENSIVES_AVG__":
        num = df[["XLU", "XLP"]].mean(axis=1)
    else:
        num = df[numerator]
    denom = df[denom_cols].mean(axis=1)
    denom = denom.replace(0, pd.NA)
    ratio = num / denom
    return ratio.dropna()


def compute_supporting_signals(prices: pd.DataFrame, requested_tickers: List[str]) -> Dict[str, Any]:
    """
    Deterministically compute, for every ticker and ratio pair, the pct
    change over the fetched window plus (for ratios) a rising/falling trend
    call. This dict is the auditable evidence trail -- a human should be
    able to see exactly why the regime call was made from this alone.
    """
    signals: Dict[str, Any] = {}

    available = list(prices.columns) if prices is not None and not prices.empty else []
    missing = [t for t in requested_tickers if t not in available]
    if missing:
        signals["missing_tickers"] = missing
        signals["missing_tickers_note"] = (
            "These tickers returned no usable data from yfinance (index/future "
            "symbols like ^VIX, DX-Y.NYB, HG=F sometimes have gaps or are "
            "unavailable) and were excluded from the signals below."
        )

    if prices is None or prices.empty:
        signals["error"] = "No price data returned for any requested ticker."
        return signals

    window_start = str(prices.index.min().date()) if len(prices.index) else None
    window_end = str(prices.index.max().date()) if len(prices.index) else None
    signals["window"] = {"start": window_start, "end": window_end, "trading_days": int(len(prices))}

    # Solo ticker levels + pct change
    solo: Dict[str, Any] = {}
    for ticker in SOLO_SIGNALS:
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna()
        if series.empty:
            continue
        pct = _pct_change(series)
        solo[ticker] = {
            "start_price": round(float(series.iloc[0]), 4),
            "end_price": round(float(series.iloc[-1]), 4),
            "pct_change": round(pct, 2) if pct is not None else None,
        }
    signals["ticker_pct_change"] = solo

    # Ratio trends
    ratios: Dict[str, Any] = {}
    for name, numerator, denom_cols in RATIO_SIGNALS:
        ratio_series = _ratio_series(prices, numerator, denom_cols)
        if ratio_series is None or len(ratio_series) < 2:
            ratios[name] = {"available": False}
            continue
        pct = _pct_change(ratio_series)
        trend = "flat"
        if pct is not None:
            if pct > 1.0:
                trend = "rising"
            elif pct < -1.0:
                trend = "falling"
        ratios[name] = {
            "available": True,
            "start_ratio": round(float(ratio_series.iloc[0]), 5),
            "end_ratio": round(float(ratio_series.iloc[-1]), 5),
            "pct_change": round(pct, 2) if pct is not None else None,
            "trend": trend,
        }
    signals["ratio_trends"] = ratios

    return signals


# --------------------------------------------------------------------------
# LLM synthesis
# --------------------------------------------------------------------------

VALID_LABELS = {"Bull", "Bear", "Volatile", "Late-cycle", "Early-cycle"}


class RegimeAssessment(BaseModel):
    regime_label: str = Field(
        description="Exactly one of: Bull, Bear, Volatile, Late-cycle, Early-cycle"
    )
    confidence: float = Field(description="0-1 confidence in this call", ge=0.0, le=1.0)
    narrative: str = Field(description="Short paragraph explaining the call, grounded in the supplied signals")


def _build_prompt(supporting_signals: Dict[str, Any], news_items: List[Dict[str, str]]) -> str:
    news_text = "\n".join(
        f"- {item.get('title', '')}: {item.get('snippet', '')}" for item in news_items
    ) or "No recent macro news retrieved."

    return f"""You are a macro market-regime analyst for a wealth-management firm. Your
call gates real client trades, so it must be grounded strictly in the
evidence below -- do NOT free-associate or rely on generic priors about
"the economy is usually fine."

DETERMINISTIC MARKET SIGNALS (computed directly from real price data, over
the trailing ~6 months; these are the ground truth you must reason from):
{json.dumps(supporting_signals, indent=2, default=str)}

Interpretation hints (use your judgment, weigh signals together):
- Rising IWM/SPY (small caps outperforming) -> risk-on / early-cycle tilt.
- Falling IWM/SPY -> risk-off.
- Falling HYG/LQD (high yield underperforming IG) -> credit stress, risk-off.
- Rising SHY/TLT (short duration outperforming long) can reflect a
  flattening/inverting curve worry or Fed-hike expectations; falling can
  reflect a dash-for-duration / flight to safety in long bonds.
- ^VIX level and pct_change: elevated/rising VIX -> fear/volatility.
- DX-Y.NYB rising -> tightening global financial conditions (headwind for
  risk assets, especially EM/commodities).
- HG=F (copper) falling -> weakening global industrial demand.
- Rising SMH/SPY -> tech/industrial cycle strength (leads by 1-2 quarters).
- Rising (XLU+XLP)/SPY -> defensive rotation, institutional de-risking.

RECENT MACRO NEWS HEADLINES:
{news_text}

Based ONLY on the signals and news above, classify the current market
regime as exactly one of: Bull, Bear, Volatile, Late-cycle, Early-cycle.
Provide a confidence between 0 and 1, and a short narrative (3-5 sentences)
that explicitly cites the specific signal values that drove your call.
"""


def _invoke_llm(supporting_signals: Dict[str, Any], news_items: List[Dict[str, str]]) -> RegimeAssessment:
    """
    Raises on any failure -- caller is responsible for the fallback.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.0,
    )

    prompt = _build_prompt(supporting_signals, news_items)

    try:
        structured_llm = llm.with_structured_output(RegimeAssessment)
        result = structured_llm.invoke(prompt)
        if isinstance(result, RegimeAssessment):
            assessment = result
        else:
            assessment = RegimeAssessment(**dict(result))
    except Exception:
        # Fall back to manual JSON parsing in case with_structured_output
        # isn't well-supported for this model/version combo.
        raw = llm.invoke(prompt)
        text = getattr(raw, "content", str(raw))
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
        assessment = RegimeAssessment(**parsed)

    if assessment.regime_label not in VALID_LABELS:
        raise ValueError(f"LLM returned invalid regime_label: {assessment.regime_label!r}")

    return assessment


# --------------------------------------------------------------------------
# Node entry point
# --------------------------------------------------------------------------

def market_regime_node(state: AgentState) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    print(">>> [Market Regime] Fetching indicator basket + macro news...")

    supporting_signals: Dict[str, Any] = {}

    # 1. Deterministic price fetch + signal computation. Any failure here is
    #    a real infra failure (not an LLM issue) but must still not crash
    #    the graph -- treat it the same as an LLM failure: fail safe.
    try:
        prices = fetch_historical_prices(ALL_TICKERS, years=MACRO_LOOKBACK_YEARS)
        supporting_signals = compute_supporting_signals(prices, ALL_TICKERS)
    except Exception as fetch_exc:
        supporting_signals = {
            "error": f"Failed to fetch/compute price signals: {fetch_exc}",
        }

    # 2. Macro news (best-effort; empty list on failure, never raises per
    #    services/news_service.py's own try/except).
    try:
        news_items = search_financial_news(NEWS_KEYWORDS, max_results=5)
    except Exception:
        news_items = []
    supporting_signals["news_headlines_used"] = [item.get("title") for item in news_items]

    # 3. LLM synthesis, with an explicit, honest fallback on ANY failure.
    #    Never default to an optimistic label like "Bull" -- that is exactly
    #    the fail-open bug in the legacy macro_sentinel.py this replaces.
    status = "success"
    error_detail: Optional[str] = None
    try:
        assessment = _invoke_llm(supporting_signals, news_items)
        market_regime: MarketRegime = {
            "regime_label": assessment.regime_label,
            "confidence": float(assessment.confidence),
            "supporting_signals": supporting_signals,
            "narrative": assessment.narrative,
        }
        summary = f"Regime={assessment.regime_label} (confidence={assessment.confidence:.2f})"
        print(f"    [Market Regime] {summary}")
    except Exception as llm_exc:
        status = "error"
        error_detail = f"{type(llm_exc).__name__}: {llm_exc}"
        print(f"    [Market Regime] LLM synthesis failed, falling back to safe default: {error_detail}")
        market_regime = {
            "regime_label": "Volatile",
            "confidence": 0.0,
            "supporting_signals": supporting_signals,
            "narrative": (
                "LLM synthesis failed ("
                + error_detail
                + "). Falling back to the safe default of 'Volatile' with "
                "confidence=0.0 -- this is NOT a real market assessment, it is "
                "a deliberate fail-safe so that downstream agents do not act on "
                "an unverified or fabricated regime call. See supporting_signals "
                "for the real, deterministically-computed price data that was "
                "gathered before the LLM call failed."
            ),
        }
        summary = f"LLM call failed; fell back to Volatile/0.0. Error: {error_detail}"

    completed_at = datetime.now(timezone.utc).isoformat()
    audit_record: AgentRunRecord = {
        "node_name": NODE_NAME,
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "summary": summary,
        "error_detail": error_detail,
    }

    return {
        "market_regime": market_regime,
        "audit_trail": [audit_record],
    }


# --------------------------------------------------------------------------
# Manual verification entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    sample_state: AgentState = {
        "run_id": "manual-test-run",
        "client_profile": {
            "client_id": 1,
            "age": 45,
            "risk_tolerance": "Moderate",
            "time_horizon_years": 20,
            "goals": ["retirement"],
            "holdings": {"CASH": 400000.0, "TSLA": 100000.0},
            "net_worth": 500000.0,
        },
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

    result = market_regime_node(sample_state)

    print("\n=== market_regime ===")
    print(json.dumps(result["market_regime"], indent=2, default=str))
    print("\n=== audit_trail ===")
    print(json.dumps(result["audit_trail"], indent=2, default=str))
