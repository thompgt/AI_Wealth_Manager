"""Finance Report Agent.

The client-facing "expert" narrator. This is the terminal node of the graph:
it does NOT call any tools or fetch any of its own data. Instead it reads the
FULL AgentState -- everything every upstream agent has already produced --
and synthesizes it into one coherent, professional report.

This is the natural evolution of the legacy
backend/app/agents/agent_workflow.py's report-writing prompt (same
"Portfolio Health Check / Market Context / Recommendations" spirit) but
restructured to consume the richer multi-agent state up front rather than
calling tools ad hoc mid-conversation.

Grounding requirement: the LLM prompt explicitly forbids inventing tickers,
numbers, or news not present in the structured data it is given. If the LLM
call fails for any reason (e.g. no real GEMINI_API_KEY is configured in this
environment), a deterministic, template-based fallback report is assembled
directly from the structured state so the graph always produces a usable,
non-empty final_report.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from state import AgentRunRecord, AgentState

NODE_NAME = "finance_report"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct(x: float) -> str:
    try:
        return f"{x * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Finance Report Agent for a wealth-management platform. \
You write the final, client-facing report that synthesizes the output of several \
upstream specialist agents (portfolio diagnostics, market regime analysis, \
stock research + suitability/compliance review, and tax-awareness screening).

CRITICAL GROUNDING RULES -- follow these strictly:
- Only state facts, numbers, tickers, and findings that are explicitly present \
in the structured data provided to you below. Never invent, estimate, or \
hallucinate a ticker, price, metric, or news item that is not given to you.
- If a section of data is empty, say so plainly instead of making something up.
- Do not give individualized investment advice framed as a directive to act; \
this is an informational/educational report only.
- Quote the client profile, portfolio diagnostics, market regime, suitability \
result, and tax assessment data as your only source of truth.

Write a professional, clear, client-friendly report with EXACTLY these six \
sections, using these headings:

1. Portfolio Health Check -- cite the actual sharpe_ratio, annual_return, \
annual_volatility, and diversification_score, and explain the specific flaws \
in plain language.
2. Market Context -- cite the regime_label, confidence, and narrative.
3. Recommendations -- for each approved candidate, explain what flaw it \
addresses, why it fits the current market regime, cite its valuation \
metrics, and state the specific recommended dollar allocation \
(allocation_amount) and what fraction of net worth it represents \
(allocation_pct).
4. What We Filtered Out and Why -- plainly summarize any suitability \
violations that caused candidates or actions to be blocked/adjusted, so the \
client understands the system applies real guardrails rather than blindly \
recommending everything.
5. Tax Notes -- summarize any wash-sale flags and tax efficiency notes in \
plain language.
6. Caveats & Disclaimers -- explicitly state this is an informational/ \
educational report, not executed trades or individualized financial advice, \
and that a human advisor should review it before the client acts on it.
"""

HUMAN_PROMPT = """Here is the full structured state for this client, produced by the \
upstream agents. Use ONLY this data.

CLIENT PROFILE:
{client_profile}

PORTFOLIO DIAGNOSTICS:
{portfolio_diagnostics}

MARKET REGIME:
{market_regime}

SUITABILITY RESULT (approved candidates in adjusted_recommendations, filtered \
items explained in violations):
{suitability_result}

TAX ASSESSMENT:
{tax_assessment}

Write the six-section client report now.
"""


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )


def _serialize_state_for_prompt(state: AgentState) -> Dict[str, str]:
    import json

    def _dump(value: Any) -> str:
        if value is None:
            return "null (not available)"
        return json.dumps(value, indent=2, default=str)

    return {
        "client_profile": _dump(state.get("client_profile")),
        "portfolio_diagnostics": _dump(state.get("portfolio_diagnostics")),
        "market_regime": _dump(state.get("market_regime")),
        "suitability_result": _dump(state.get("suitability_result")),
        "tax_assessment": _dump(state.get("tax_assessment")),
    }


def _call_llm(state: AgentState) -> str:
    """Invoke the Gemini model. Raises on any failure -- caller handles the
    fallback."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.4,
    )
    prompt = _build_prompt()
    chain = prompt | llm
    inputs = _serialize_state_for_prompt(state)
    response = chain.invoke(inputs)
    text = getattr(response, "content", None) or str(response)
    if not text or not text.strip():
        raise ValueError("LLM returned an empty report")
    return text.strip()


# --------------------------------------------------------------------------
# Deterministic fallback -- no LLM required. Always succeeds given any
# (possibly partial) state, so the graph never emits an empty final_report.
# --------------------------------------------------------------------------

def _fallback_health_check(diagnostics: Dict[str, Any]) -> str:
    if not diagnostics:
        return (
            "Portfolio diagnostics are not available for this run, so a "
            "detailed health check could not be produced."
        )
    lines = [
        f"- Sharpe ratio: {diagnostics.get('sharpe_ratio', 'N/A')}",
        f"- Annualized return: {_pct(diagnostics.get('annual_return', 0.0))}",
        f"- Annualized volatility: {_pct(diagnostics.get('annual_volatility', 0.0))}",
        f"- Diversification score: {diagnostics.get('diversification_score', 'N/A')}/100",
    ]
    flaws = diagnostics.get("flaws") or []
    if flaws:
        lines.append("")
        lines.append("Specific issues identified in your current portfolio:")
        for flaw in flaws:
            lines.append(f"  * {flaw}")
    else:
        lines.append("")
        lines.append("No specific portfolio flaws were flagged in this analysis.")
    return "\n".join(lines)


def _fallback_market_context(market_regime: Dict[str, Any]) -> str:
    if not market_regime:
        return "Market regime data is not available for this run."
    label = market_regime.get("regime_label", "Unknown")
    confidence = market_regime.get("confidence")
    confidence_str = _pct(confidence) if isinstance(confidence, (int, float)) else "N/A"
    narrative = market_regime.get("narrative", "")
    lines = [
        f"- Current regime classification: {label} (confidence: {confidence_str})",
    ]
    if narrative:
        lines.append(f"- Narrative: {narrative}")
    return "\n".join(lines)


def _fallback_recommendations(suitability_result: Dict[str, Any]) -> str:
    if not suitability_result:
        return "No suitability-reviewed recommendations are available for this run."
    candidates = suitability_result.get("adjusted_recommendations") or []
    if not candidates:
        return (
            "No candidates cleared suitability review this cycle, so there are "
            "no new recommendations to present."
        )
    lines = []
    for c in candidates:
        ticker = c.get("ticker", "UNKNOWN")
        addresses = c.get("addresses_flaw", "not specified")
        rationale = c.get("regime_fit_rationale", "not specified")
        metrics = c.get("valuation_metrics") or {}
        metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items()) or "no metrics provided"
        confidence = c.get("confidence")
        confidence_str = _pct(confidence) if isinstance(confidence, (int, float)) else "N/A"
        allocation_amount = c.get("allocation_amount", 0.0)
        allocation_pct = c.get("allocation_pct", 0.0)
        lines.append(f"- {ticker}: recommended allocation ${allocation_amount:,.0f} ({_pct(allocation_pct)} of net worth, confidence: {confidence_str})")
        lines.append(f"    Addresses: {addresses}")
        lines.append(f"    Why it fits the current regime: {rationale}")
        lines.append(f"    Valuation metrics: {metrics_str}")
    return "\n".join(lines)


def _fallback_filtered(suitability_result: Dict[str, Any]) -> str:
    if not suitability_result:
        return "No suitability review data is available for this run."
    violations = suitability_result.get("violations") or []
    approved = suitability_result.get("approved")
    lines = [f"- Overall suitability approval: {'Yes' if approved else 'No'}"]
    if violations:
        lines.append("- The following items were filtered out or adjusted by the compliance/suitability guardrail:")
        for v in violations:
            lines.append(f"    * {v}")
    else:
        lines.append("- No candidates or actions were filtered out this cycle.")
    return "\n".join(lines)


def _fallback_tax_notes(tax_assessment: Dict[str, Any]) -> str:
    if not tax_assessment:
        return "No tax assessment data is available for this run."
    wash_sale_flags = tax_assessment.get("wash_sale_flags") or []
    notes = tax_assessment.get("tax_efficiency_notes") or []
    lines = []
    if wash_sale_flags:
        lines.append(
            "- The following tickers are currently blocked from repurchase under "
            "the 30-day wash-sale rule: " + ", ".join(wash_sale_flags)
        )
    else:
        lines.append("- No wash-sale restrictions currently apply.")
    if notes:
        lines.append("- Additional tax efficiency notes:")
        for n in notes:
            lines.append(f"    * {n}")
    return "\n".join(lines)


DISCLAIMER_TEXT = (
    "This report is generated for informational and educational purposes only. "
    "It does not constitute executed trades, a solicitation to buy or sell any "
    "security, or individualized financial advice. All figures are drawn "
    "directly from the automated analysis above and may not reflect real-time "
    "market conditions. A qualified human financial advisor should review this "
    "report before you act on any of it."
)


def _assemble_fallback_report(state: AgentState, error: Exception) -> str:
    profile = state.get("client_profile") or {}
    client_id = profile.get("client_id", "unknown")
    risk_tolerance = profile.get("risk_tolerance", "unknown")
    horizon = profile.get("time_horizon_years", "unknown")

    diagnostics = state.get("portfolio_diagnostics") or {}
    market_regime = state.get("market_regime") or {}
    suitability_result = state.get("suitability_result") or {}
    tax_assessment = state.get("tax_assessment") or {}

    sections = [
        f"WEALTH REPORT -- Client #{client_id}",
        f"(Risk tolerance: {risk_tolerance}; time horizon: {horizon} years)",
        "",
        "[Note: this report was generated by the deterministic fallback "
        f"template because the LLM call failed: {error!r}]",
        "",
        "1. PORTFOLIO HEALTH CHECK",
        _fallback_health_check(diagnostics),
        "",
        "2. MARKET CONTEXT",
        _fallback_market_context(market_regime),
        "",
        "3. RECOMMENDATIONS",
        _fallback_recommendations(suitability_result),
        "",
        "4. WHAT WE FILTERED OUT AND WHY",
        _fallback_filtered(suitability_result),
        "",
        "5. TAX NOTES",
        _fallback_tax_notes(tax_assessment),
        "",
        "6. CAVEATS & DISCLAIMERS",
        DISCLAIMER_TEXT,
    ]
    return "\n".join(sections)


# --------------------------------------------------------------------------
# Node entry point
# --------------------------------------------------------------------------

def finance_report_node(state: AgentState) -> dict:
    started_at = _now_iso()
    error_detail = None

    try:
        report = _call_llm(state)
        status = "success"
        summary = "Generated client-facing report via Gemini LLM."
    except Exception as exc:  # noqa: BLE001 -- node must never crash the graph
        status = "error"
        error_detail = repr(exc)
        summary = (
            "Gemini LLM call failed (likely missing/invalid GEMINI_API_KEY); "
            "used deterministic fallback template instead."
        )
        report = _assemble_fallback_report(state, exc)

    record: AgentRunRecord = {
        "node_name": NODE_NAME,
        "started_at": started_at,
        "completed_at": _now_iso(),
        "status": status,
        "summary": summary,
        "error_detail": error_detail,
    }

    return {"final_report": report, "audit_trail": [record]}


if __name__ == "__main__":
    import json

    sample_state: AgentState = {
        "run_id": "smoke-test-run-001",
        "client_profile": {
            "client_id": 42,
            "age": 57,
            "risk_tolerance": "Moderate",
            "time_horizon_years": 10,
            "goals": ["Retirement", "College funding"],
            "holdings": {
                "AAPL": 420000.0,
                "MSFT": 85000.0,
                "CASH": 45000.0,
            },
            "net_worth": 550000.0,
        },
        "portfolio_diagnostics": {
            "concentration": {"AAPL": 0.7636, "MSFT": 0.1545, "CASH": 0.0818},
            "sector_exposure": {"Technology": 0.9998, "Unknown": 0.0002},
            "sharpe_ratio": 0.42,
            "annual_return": 0.081,
            "annual_volatility": 0.29,
            "diversification_score": 24.3,
            "flaws": [
                "AAPL is 76.4% of the portfolio -- exceeds the 25.0% concentration limit for a Moderate risk tolerance",
                "Technology sector is 100.0% of non-cash holdings -- exceeds the 50.0% sector concentration threshold",
            ],
            "market_data_tickers": ["AAPL", "MSFT"],
        },
        "market_regime": {
            "regime_label": "Late-cycle",
            "confidence": 0.68,
            "supporting_signals": {"10y2y_spread": -0.12, "vix": 22.4},
            "narrative": (
                "Yield curve inversion and elevated volatility suggest we are "
                "late in the economic cycle; defensive positioning and "
                "diversification are favored over further concentration in "
                "high-beta growth names."
            ),
        },
        "candidate_stocks": [
            {
                "ticker": "JNJ",
                "valuation_metrics": {"pe_ratio": 15.2, "peg_ratio": 2.1, "dividend_yield": 0.031},
                "addresses_flaw": "AAPL is 76.4% of the portfolio -- exceeds the 25.0% concentration limit for a Moderate risk tolerance",
                "regime_fit_rationale": "Defensive healthcare name with stable cash flows, appropriate for late-cycle positioning.",
                "confidence": 0.74,
            },
            {
                "ticker": "SOXL",
                "valuation_metrics": {"pe_ratio": 41.0, "peg_ratio": 3.4},
                "addresses_flaw": "AAPL is 76.4% of the portfolio -- exceeds the 25.0% concentration limit for a Moderate risk tolerance",
                "regime_fit_rationale": "Leveraged semiconductor exposure -- high risk, does not fit late-cycle defensive posture.",
                "confidence": 0.31,
            },
        ],
        "suitability_result": {
            "approved": True,
            "violations": [
                "SOXL rejected: 3x leveraged ETF exceeds volatility limits for a Moderate risk tolerance client",
            ],
            "adjusted_recommendations": [
                {
                    "ticker": "JNJ",
                    "valuation_metrics": {"pe_ratio": 15.2, "peg_ratio": 2.1, "dividend_yield": 0.031},
                    "addresses_flaw": "AAPL is 76.4% of the portfolio -- exceeds the 25.0% concentration limit for a Moderate risk tolerance",
                    "regime_fit_rationale": "Defensive healthcare name with stable cash flows, appropriate for late-cycle positioning.",
                    "confidence": 0.74,
                },
            ],
        },
        "tax_assessment": {
            "wash_sale_flags": ["MSFT"],
            "tax_efficiency_notes": [
                "MSFT was sold at a loss 18 days ago; repurchasing now would trigger a wash-sale disallowance -- wait until the 30-day window closes.",
                "AAPL has a large unrealized long-term gain; trimming the position would be more tax-efficient after 12 months of holding than a short-term sale.",
            ],
        },
        "final_report": None,
        "audit_trail": [],
        "requires_human_approval": False,
        "human_approved": None,
        "research_attempts": 0,
        "needs_research_retry": False,
    }

    print("Running finance_report_node against a fully-populated sample AgentState...")
    print(f"(GEMINI_API_KEY configured: {settings.GEMINI_API_KEY!r} -- dummy key expected to fail the live call)")
    print()

    result = finance_report_node(sample_state)

    print("=== audit_trail ===")
    print(json.dumps(result["audit_trail"], indent=2))
    print()
    print("=== final_report ===")
    print(result["final_report"])
