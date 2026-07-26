"""Finance Report: the client-facing narrative.

This node reads state and makes no tool calls, which is deliberate — the
report must describe what the analysis found, never go looking for new facts
to describe. Every number it can cite has already been computed, checked and
recorded upstream.

Two properties matter more than prose quality:

**It cannot silently substitute a template for analysis.** When the LLM is
unavailable, the deterministic report is produced *and says so*, in a banner
that distinguishes "no model is configured" from "the model call failed". A
system that quietly serves template output as AI analysis is lying to the
person relying on it, and the whole point of this project is the opposite.

**It must disclose what was withheld and what degraded.** The interesting
behaviour of this system is not that it recommends things; it is where it
refuses to. A report that lists five buys and omits the two that were blocked
on a wash sale, or that reads confidently while the regime call came from a
fallback, is the failure mode all the upstream guardrails exist to prevent.
"""

import json
from typing import Any, Dict, List, Optional

from config import settings
from db import utcnow
from logging_setup import get_logger
from services.llm import LLMUnavailable, classify_failure, get_chat_model, invoke_tracked
from state import AgentState

from agents.runtime import finish, node_run, summarize_degradations

logger = get_logger(__name__)

NODE_NAME = "finance_report"
PROMPT_VERSION = "report-v2"
DISCLOSURE_VERSION = "2026-07"


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "not available"
    try:
        return f"{value * 100:.1f}%"
    except (TypeError, ValueError):
        return "not available"


def _money(value: Optional[float]) -> str:
    if value is None:
        return "not available"
    try:
        return f"${value:,.0f}"
    except (TypeError, ValueError):
        return "not available"


SYSTEM_PROMPT = """You are the report writer for a wealth-management platform. You write the
final client-facing report synthesizing the output of several specialist
agents: portfolio diagnostics, market regime analysis, security research, a
suitability guardrail, a tax-awareness check and a rebalancing planner.

GROUNDING RULES -- these are absolute:
- State only facts, numbers, tickers and findings explicitly present in the
  data below. Never invent, estimate or infer a ticker, price, metric or news
  item that is not given to you.
- If a section of data is empty, say so plainly rather than filling the space.
- Where the data marks something as degraded or unavailable, say so in the
  report. Do not write around a gap to make the analysis sound more complete
  than it was.
- Do not restate a recommendation as a directive. This is an advisory report
  that a human adviser reviews before the client acts.

TONE:
- Write for an intelligent client who is not a finance professional. Explain
  what a number means, not just what it is.
- Be direct about problems. A client whose portfolio is 50% in one stock is
  better served by a clear sentence than a hedged one.
- Do not congratulate the client, and do not editorialise about the market
  beyond the regime data supplied.

Produce EXACTLY these sections, with these headings:

## 1. Summary
Three or four sentences: what was found, what is recommended, what was
withheld. Lead with the most consequential finding.

## 2. Portfolio Health
Cite the actual total value, concentration, sector and asset-class exposure,
volatility, drawdown, correlation and diversification figures, and explain the
specific findings in plain language.

## 3. Market Context
Cite the regime label, the confidence, and the specific signals that drove it.
If confidence is low, say why that matters for the recommendations below.

## 4. Recommended Changes
For each proposed trade, state the action, the security, the dollar amount,
which specific problem it addresses, and its estimated tax cost where given.
Explain sells and trims as carefully as buys.

## 5. What Was Withheld, and Why
Every blocked recommendation, every deferred trade and every guardrail
violation, with the reason for each. This section is the most important one in
the report. Do not summarize it away or soften it.

## 6. Tax Notes
Wash-sale findings, harvesting opportunities, realized gains year to date.

## 7. Limitations of This Analysis
Anything the run could not do: unpriced positions, missing data, degraded
steps. If nothing degraded, say the analysis ran on complete data.
"""


def _build_payload(state: AgentState) -> Dict[str, Any]:
    """Everything the report may cite, and nothing else."""
    return {
        "client_profile": state.get("client_profile"),
        "policy": state.get("policy"),
        "portfolio_diagnostics": state.get("portfolio_diagnostics"),
        "market_regime": state.get("market_regime"),
        "suitability_result": state.get("suitability_result"),
        "tax_assessment": state.get("tax_assessment"),
        "rebalance_plan": state.get("rebalance_plan"),
        "tax_blocked_recommendations": state.get("tax_blocked_recommendations"),
        "guardrail_feedback": state.get("guardrail_feedback"),
        "degradations": state.get("degradations"),
        "data_as_of": state.get("data_as_of"),
    }


DISCLAIMER = """---

**Important.** This report is produced by an automated analysis system for
informational purposes. It is not individualized investment advice, and no
trade described here has been executed. Any proposed transaction requires
review and approval by a qualified human adviser before it is acted upon.
Market data may be delayed. Tax figures are estimates computed from the
holdings and transactions recorded in this system, are not tax advice, and do
not account for your full tax circumstances -- consult a tax professional.
Past performance does not predict future results, and all investing involves
the risk of loss."""


def _fallback_report(state: AgentState, reason: str, detail: str) -> str:
    """The deterministic report. Complete, and honest about what it is.

    This is not a degraded stub: it contains every finding, every
    recommendation, every withheld item and every tax note the model version
    would have covered. What it lacks is the synthesis -- and the banner says
    exactly that, distinguishing a missing API key from a failed call, because
    those need different responses from whoever is reading.
    """
    profile = state.get("client_profile") or {}
    diagnostics = state.get("portfolio_diagnostics") or {}
    regime = state.get("market_regime") or {}
    suitability = state.get("suitability_result") or {}
    tax = state.get("tax_assessment") or {}
    plan = state.get("rebalance_plan") or {}
    policy = state.get("policy") or {}
    blocked = state.get("tax_blocked_recommendations") or []
    degradations = state.get("degradations") or []

    if reason == "no_api_key":
        banner = (
            "> **This report was generated without a language model.** No API key is "
            "configured on this deployment, so the narrative below is assembled directly "
            "from the analysis rather than written by a model. Every number, finding and "
            "guardrail decision is real and was produced by the same deterministic "
            "analysis either way -- what is missing is the written synthesis, not the "
            "substance."
        )
    else:
        banner = (
            f"> **This report was generated without a language model.** The model call "
            f"failed ({detail}), so the narrative below is assembled directly from the "
            f"analysis. Every number, finding and guardrail decision is real; the written "
            f"synthesis is absent. This is worth investigating -- it means the AI layer is "
            f"not currently working."
        )

    lines: List[str] = [
        "# Portfolio Analysis",
        "",
        banner,
        "",
        f"Prepared for **{profile.get('name', 'this client')}** on "
        f"{utcnow().strftime('%d %B %Y')}.",
        "",
    ]

    recommendations = suitability.get("adjusted_recommendations") or []
    proposals = plan.get("proposals") or []
    flaws = diagnostics.get("flaws") or []

    # --- 1. Summary ---
    lines += ["## 1. Summary", ""]
    summary_bits = []
    if flaws:
        summary_bits.append(f"The analysis identified {len(flaws)} finding(s), the most "
                            f"significant being: {flaws[0]}")
    else:
        summary_bits.append("No policy breaches or material findings were identified.")
    if proposals:
        sells = plan.get("sell_count", 0)
        buys = plan.get("buy_count", 0)
        summary_bits.append(
            f"{sells} sale(s) and {buys} purchase(s) are proposed, totalling "
            f"{_money(plan.get('gross_notional'))} of trading."
        )
    else:
        summary_bits.append("No trades are proposed.")
    if blocked:
        summary_bits.append(
            f"{len(blocked)} recommendation(s) were withheld on tax grounds: "
            f"{', '.join(blocked)}."
        )
    lines += [" ".join(summary_bits), ""]

    # --- 2. Portfolio Health ---
    lines += [
        "## 2. Portfolio Health",
        "",
        f"- Total value: {_money(diagnostics.get('total_value'))} "
        f"({_money(diagnostics.get('cash'))} in cash)",
        f"- Annualised return: {_pct(diagnostics.get('annual_return'))}",
        f"- Annualised volatility: {_pct(diagnostics.get('annual_volatility'))}",
        f"- Sharpe ratio: {diagnostics.get('sharpe_ratio') if diagnostics.get('sharpe_ratio') is not None else 'not available'}",
        f"- Largest peak-to-trough decline: {_pct(diagnostics.get('max_drawdown'))}",
        f"- Portfolio beta: {diagnostics.get('portfolio_beta') if diagnostics.get('portfolio_beta') is not None else 'not available'}",
        f"- Diversification score: {diagnostics.get('diversification_score', 0):.0f}/100"
        + (
            f", behaving like roughly {diagnostics['effective_positions']:.1f} equally-sized positions"
            if diagnostics.get("effective_positions")
            else ""
        ),
    ]
    if diagnostics.get("average_correlation") is not None:
        lines.append(
            f"- Average correlation between holdings: {diagnostics['average_correlation']:.2f}"
        )
    lines.append("")

    if flaws:
        lines += ["**Findings:**", ""]
        lines += [f"{index}. {flaw}" for index, flaw in enumerate(flaws, start=1)]
    else:
        lines.append("No findings: the portfolio is within every limit in the client's policy.")
    lines.append("")

    # --- 3. Market Context ---
    lines += [
        "## 3. Market Context",
        "",
        f"**Regime: {regime.get('regime_label', 'unknown')}** "
        f"(confidence {regime.get('confidence', 0) * 100:.0f}%)",
        "",
        regime.get("narrative", "No market regime assessment was produced."),
        "",
    ]
    if (regime.get("confidence") or 0) < 0.3:
        lines += [
            "Confidence in this assessment is low, which means the recommendations below "
            "rest mainly on portfolio construction and valuation rather than on a market "
            "view. That is the intended behaviour when the signals do not agree.",
            "",
        ]

    # --- 4. Recommended Changes ---
    lines += ["## 4. Recommended Changes", ""]
    if proposals:
        for proposal in sorted(proposals, key=lambda p: p.get("sequence", 0)):
            tax_cost = proposal.get("estimated_tax_cost")
            lines.append(
                f"### {proposal['side']} {proposal['symbol']} — "
                f"{_money(proposal.get('notional'))}"
            )
            lines.append("")
            lines.append(proposal.get("rationale", ""))
            if proposal.get("addresses_flaw"):
                lines.append("")
                lines.append(f"*Addresses:* {proposal['addresses_flaw']}")
            if tax_cost:
                lines.append("")
                lines.append(f"*Estimated tax cost:* {_money(tax_cost)}")
            lines.append("")
    elif recommendations:
        for recommendation in recommendations:
            lines.append(
                f"### {recommendation['ticker']} — "
                f"{_money(recommendation.get('allocation_amount'))} "
                f"({_pct(recommendation.get('allocation_pct'))} of the portfolio)"
            )
            lines.append("")
            lines.append(f"*Addresses:* {recommendation.get('addresses_flaw', 'not stated')}")
            lines.append("")
            lines.append(recommendation.get("regime_fit_rationale", ""))
            lines.append("")
    else:
        lines += [
            "No changes are recommended. Either the portfolio is already within its policy, "
            "or every candidate considered was withheld by a guardrail -- see the next "
            "section.",
            "",
        ]

    # --- 5. What Was Withheld ---
    lines += ["## 5. What Was Withheld, and Why", ""]
    withheld_any = False

    if blocked:
        withheld_any = True
        lines += [
            "**Blocked on tax grounds.** These securities passed every suitability check "
            "and were still not recommended:",
            "",
        ]
        for detail in tax.get("wash_sale_detail") or []:
            lines.append(f"- **{detail['symbol']}**: {detail['reason']}")
        for ticker in blocked:
            if not any(d.get("symbol") == ticker for d in (tax.get("wash_sale_detail") or [])):
                lines.append(f"- **{ticker}**: withheld by the wash-sale guardrail.")
        lines += [
            "",
            "The money that would have gone into these positions has not been quietly "
            "moved elsewhere. This run deploys less, and says so.",
            "",
        ]

    violations = suitability.get("violations") or []
    if violations:
        withheld_any = True
        lines += ["**Rejected by the suitability guardrail:**", ""]
        lines += [f"- {violation}" for violation in violations]
        lines.append("")

    deferred = plan.get("deferred") or []
    if deferred:
        withheld_any = True
        lines += ["**Deferred for human judgement:**", ""]
        for entry in deferred:
            label = entry.get("symbol") or entry.get("asset_class") or "position"
            lines.append(f"- **{label}**: {entry.get('reason')}")
        lines.append("")

    for note in plan.get("notes") or []:
        withheld_any = True
        lines.append(f"- {note}")
    if plan.get("notes"):
        lines.append("")

    if not withheld_any:
        lines += ["Nothing was withheld. Every candidate considered passed every guardrail.", ""]

    # --- 6. Tax Notes ---
    lines += ["## 6. Tax Notes", ""]
    for note in tax.get("tax_efficiency_notes") or ["No tax observations for this client."]:
        lines.append(f"- {note}")
    lines.append("")

    # --- 7. Limitations ---
    lines += ["## 7. Limitations of This Analysis", ""]
    degradation_text = summarize_degradations(degradations)
    if degradation_text:
        lines += [degradation_text, ""]
    else:
        lines += ["This analysis ran on complete data with no degraded steps.", ""]

    if policy.get("source") != "policy":
        lines += [
            "This client has no approved Investment Policy Statement on file. The limits "
            f"applied above are the system's built-in defaults for a "
            f"{policy.get('risk_tier', 'Moderate')} mandate and have not been reviewed or "
            "agreed. Recording an approved policy would make these limits an explicit, "
            "auditable agreement rather than an assumption.",
            "",
        ]

    if state.get("data_as_of"):
        lines += [f"Market data as of {state['data_as_of']}.", ""]

    lines.append(DISCLAIMER)
    return "\n".join(lines)


def finance_report_node(state: AgentState) -> dict:
    """Write the client-facing report."""
    report: Optional[str] = None

    with node_run(NODE_NAME, state) as ctx:
        ctx.prompt_version = PROMPT_VERSION
        ctx.temperature = 0.4

        payload = _build_payload(state)

        try:
            llm = get_chat_model(temperature=0.4)
            prompt = (
                SYSTEM_PROMPT
                + "\n\nHere is the complete analysis. Use only this data.\n\n"
                + json.dumps(payload, indent=2, default=str)
            )
            response, usage = invoke_tracked(
                lambda: llm.invoke(prompt), node=NODE_NAME
            )
            ctx.record_usage(usage)
            ctx.model_used = settings.GEMINI_MODEL
            text = getattr(response, "content", str(response)).strip()
            if not text:
                raise ValueError("The model returned an empty report.")

            degradation_text = summarize_degradations(state.get("degradations") or [])
            if degradation_text:
                # Appended deterministically rather than trusted to the model.
                # A model asked to disclose its own limitations sometimes
                # softens them, and this is the section that must not be soft.
                text += "\n\n---\n\n### Data limitations\n\n" + degradation_text
            report = text + "\n\n" + DISCLAIMER

        except LLMUnavailable as exc:
            report = _fallback_report(state, "no_api_key", str(exc))
            ctx.degrade(
                reason="no_api_key",
                detail=str(exc),
                impact=(
                    "This report was assembled from the analysis rather than written by a "
                    "language model, because none is configured. The findings and figures "
                    "are unaffected."
                ),
            )
        except Exception as exc:  # noqa: BLE001 -- a report must always be produced
            reason = classify_failure(exc)
            report = _fallback_report(state, reason, f"{type(exc).__name__}: {exc}")
            ctx.degrade(
                reason=reason,
                detail=f"{type(exc).__name__}: {exc}",
                impact=(
                    "This report was assembled from the analysis rather than written by a "
                    "language model, because the model call failed. The findings and "
                    "figures are unaffected."
                ),
            )

        ctx.output_snapshot = {
            "characters": len(report or ""),
            "llm_written": not ctx.degraded,
            "disclosure_version": DISCLOSURE_VERSION,
        }
        ctx.summary = (
            f"{'Model-written' if not ctx.degraded else 'Deterministic'} report, "
            f"{len(report or '')} characters."
        )
        logger.info("[Report] %s", ctx.summary)

    if report is None:
        report = _fallback_report(state, "node_exception", "the report node failed")

    return finish(ctx, {"final_report": report})
