"""Rebalance: turn approved recommendations into an executable trade list.

This node is the one the previous system had no equivalent of, and its absence
was the largest gap in the product. Recommendations were dollar amounts to buy;
nothing ever proposed selling the over-concentrated position that was the
headline finding, so the system diagnosed a problem it structurally could not
fix.

The planning logic lives in `services/rebalance.py`. This node's job is to
assemble its inputs -- the live portfolio, the resolved policy, the approved
candidates -- and to supply a per-asset-class candidate provider so an
underweight sleeve can be filled even when the global screen ranked its
instruments below the shortlist cut-off.

Nothing here executes. It produces proposals; the human approval gate decides
whether they become orders.
"""

from typing import Any, Dict, List, Optional

from db import ClientProfile as ClientProfileModel
from db import SessionLocal
from logging_setup import get_logger
from services.market_data import get_quotes
from services.policy import resolve as resolve_policy
from services.portfolio import load_portfolio
from services.rebalance import plan_rebalance
from services.screener import ScreenCriteria, screen
from state import AgentState, RebalancePlanState

from agents.runtime import finish, node_run

logger = get_logger(__name__)

NODE_NAME = "rebalance"


def _candidates_from_recommendations(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Approved recommendations, in the shape the planner expects."""
    payload = []
    for recommendation in recommendations:
        payload.append(
            {
                "ticker": recommendation.get("ticker"),
                "asset_class": recommendation.get("asset_class"),
                "price": recommendation.get("price"),
                "confidence": recommendation.get("confidence"),
                "rationale": recommendation.get("regime_fit_rationale")
                or recommendation.get("addresses_flaw"),
            }
        )
    return payload


def rebalance_node(state: AgentState) -> dict:
    """Build the trade plan that moves this portfolio toward its policy."""
    plan_state: Optional[RebalancePlanState] = None

    with node_run(NODE_NAME, state) as ctx:
        profile = state.get("client_profile") or {}
        client_id = profile.get("client_id")
        suitability = state.get("suitability_result") or {}
        recommendations = list(suitability.get("adjusted_recommendations") or [])

        db = SessionLocal()
        try:
            client = (
                db.query(ClientProfileModel).filter(ClientProfileModel.id == client_id).first()
            )
            if client is None:
                raise ValueError(f"No client profile with id {client_id}")

            policy = resolve_policy(db, client)
            view = load_portfolio(db, client)
            ctx.policy_version = policy.version

            held = sorted({h.symbol for h in view.holdings})

            def provider(asset_class: str) -> List[Dict[str, Any]]:
                """Best available instruments for one underweight sleeve.

                A single global screen ranks the whole universe, so a small
                sleeve such as emerging markets can fall below the cut-off and
                go unfilled for want of a *surfaced* candidate rather than a
                suitable one. Asking per asset class fixes that.
                """
                report = screen(
                    db,
                    ScreenCriteria(
                        asset_classes=[asset_class],
                        min_market_cap=policy.min_market_cap,
                        min_avg_dollar_volume=policy.min_avg_dollar_volume,
                        excluded_tickers=list(policy.excluded_tickers)
                        + list(state.get("excluded_tickers") or []),
                        excluded_sectors=list(policy.excluded_sectors),
                        held_tickers=held,
                    ),
                    regime_label=(state.get("market_regime") or {}).get("regime_label"),
                    risk_tier=policy.risk_tier,
                    limit=3,
                )
                if not report.results:
                    return []
                quotes = get_quotes([r.ticker for r in report.results])
                return [
                    {
                        "ticker": r.ticker,
                        "asset_class": r.asset_class,
                        "price": quotes[r.ticker].price,
                        "confidence": 0.4,
                        "rationale": (
                            f"Fills the {asset_class.replace('_', ' ')} allocation, which is "
                            f"below its policy target. Best available on the screen "
                            f"(composite {r.composite_score:+.2f})."
                        ),
                    }
                    for r in report.results
                    if r.ticker in quotes
                ]

            plan = plan_rebalance(
                db,
                client,
                view,
                policy,
                candidates=_candidates_from_recommendations(recommendations),
                candidate_provider=provider,
            )

            ctx.input_snapshot = {
                "approved_recommendations": [r.get("ticker") for r in recommendations],
                "total_value": round(view.total_value, 2),
                "spendable_cash": round(view.spendable_cash, 2),
                "policy_version": policy.version,
            }

            plan_state = RebalancePlanState(**plan.to_dict())

            if not plan.proposals:
                reason = plan.notes[0] if plan.notes else "the portfolio is already within policy"
                ctx.summary = f"No trades proposed: {reason}"
                logger.info("[Rebalance] %s", ctx.summary)
            else:
                ctx.summary = (
                    f"{len(plan.sells)} sell(s) and {len(plan.buys)} buy(s), "
                    f"${plan.gross_notional():,.0f} gross, "
                    f"${plan.estimated_tax():,.0f} estimated tax cost."
                )
                logger.info("[Rebalance] %s", ctx.summary)
                for proposal in sorted(plan.proposals, key=lambda p: p.sequence):
                    logger.info(
                        "    %s %s $%s",
                        proposal.side, proposal.symbol, f"{proposal.notional:,.0f}",
                    )

            if plan.deferred:
                ctx.degrade(
                    reason="deferred_trades",
                    detail=f"{len(plan.deferred)} proposed action(s) were held back.",
                    impact=(
                        f"{len(plan.deferred)} adjustment(s) this portfolio needs were not "
                        f"proposed, because acting on them would breach another limit or "
                        f"cost more in tax than the problem they solve. They are listed "
                        f"individually in the report with the reason for each."
                    ),
                )

            ctx.output_snapshot = {
                "sells": len(plan.sells),
                "buys": len(plan.buys),
                "gross_notional": round(plan.gross_notional(), 2),
                "deferred": len(plan.deferred),
            }
        finally:
            db.close()

    if plan_state is None:
        plan_state = RebalancePlanState(
            proposals=[], deferred=[], sell_count=0, buy_count=0,
            gross_notional=0.0, estimated_tax_cost=0.0,
            notes=[
                "The rebalancing step failed, so no trades are proposed in this run. The "
                "diagnostics and recommendations above stand, but nothing has been turned "
                "into an executable plan."
            ],
        )

    return finish(ctx, {"rebalance_plan": plan_state})
