"""Tax-Awareness: what a proposed trade costs after tax.

Deterministic. The real work lives in `services/tax_lots.py`; this node applies
it to the run's candidates and produces the assessment the guardrail gate
enforces.

Two changes from the previous version are worth stating plainly, because they
change outcomes rather than implementation:

* **The wash-sale check no longer fires on profitable sales.** It used to flag
  any sale of a symbol in the last 30 days, so a client who took a *gain* was
  blocked from re-buying that name for a month, and the retry loop spent its
  budget working around a restriction that did not exist. It now requires an
  actual realized loss, in an actual taxable account.

* **It fails closed.** The old node caught every exception and returned empty
  wash-sale flags, so a transient database error silently disabled the only
  hard tax guardrail while the run reported success. A tax check that cannot
  run must block rather than wave through, so a failure here produces an
  explicit "could not verify" state the guardrail gate treats as blocking.
"""

from typing import Dict, List, Optional

from db import ClientProfile as ClientProfileModel
from db import SessionLocal
from logging_setup import get_logger
from services import tax_lots
from services.policy import resolve as resolve_policy
from services.portfolio import load_portfolio
from state import AgentState, TaxAssessment

from agents.runtime import finish, node_run

logger = get_logger(__name__)

NODE_NAME = "tax_awareness"

# Losses smaller than this are not worth trading: the spread and the
# administrative cost of harvesting exceed the value of the deduction.
MIN_HARVEST_LOSS = 500.0


def tax_awareness_node(state: AgentState) -> dict:
    """Assess the tax consequences of this run's candidates."""
    assessment: Optional[TaxAssessment] = None

    with node_run(NODE_NAME, state) as ctx:
        candidates = list(state.get("candidate_stocks") or [])
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
            ctx.policy_version = policy.version

            accounts = list(client.accounts)
            taxable = [a for a in accounts if a.tax_treatment == "taxable"]
            ctx.input_snapshot = {
                "candidates": [c.get("ticker") for c in candidates],
                "accounts": len(accounts),
                "taxable_accounts": len(taxable),
                "lot_method": policy.lot_selection_method,
            }

            notes: List[str] = []
            wash_sale_flags: List[str] = []
            wash_sale_detail: List[Dict[str, object]] = []

            if not taxable:
                notes.append(
                    "This client holds no taxable accounts, so the wash-sale rule and loss "
                    "harvesting do not apply. Gains and losses inside tax-deferred and "
                    "tax-exempt accounts have no current tax consequence."
                )
            else:
                for candidate in candidates:
                    ticker = str(candidate.get("ticker") or "").upper()
                    if not ticker:
                        continue
                    target_account = None
                    account_id = candidate.get("account_id")
                    if account_id:
                        target_account = next((a for a in accounts if a.id == account_id), None)

                    finding = tax_lots.check_wash_sale(
                        db, client.id, ticker, target_account=target_account
                    )
                    if finding is not None:
                        wash_sale_flags.append(ticker)
                        wash_sale_detail.append(finding.to_dict())
                        logger.info("[Tax] wash-sale flag on %s: %s", ticker, finding.reason)

            # Harvesting opportunities are independent of the candidates: they
            # concern what the client already holds.
            view = load_portfolio(db, client)
            prices = {h.symbol: h.price for h in view.holdings}
            harvest: List[Dict[str, object]] = []
            if policy.harvest_losses and taxable:
                harvest = tax_lots.harvestable_losses(
                    db, client, prices, min_loss=MIN_HARVEST_LOSS
                )
                for entry in harvest[:5]:
                    conflict = entry.get("wash_sale_conflict")
                    tail = (
                        f", but note: {conflict}"
                        if conflict
                        else ", and the exposure could be maintained with a similar but not "
                             "substantially identical holding."
                    )
                    notes.append(
                        f"{entry['symbol']} in {entry['account_name']} is "
                        f"${abs(entry['unrealized_loss']):,.0f} below its cost basis. "
                        f"Realizing that loss would offset gains elsewhere{tail}"
                    )

            realized = tax_lots.realized_gains(db, client)
            if realized.get("total"):
                disallowed = realized.get("wash_sale_disallowed") or 0
                notes.append(
                    f"Realized year to date: ${realized['short_term']:,.0f} short-term and "
                    f"${realized['long_term']:,.0f} long-term."
                    + (
                        f" ${disallowed:,.0f} of losses is currently disallowed under the "
                        f"wash-sale rule and deferred into the basis of replacement shares."
                        if disallowed
                        else ""
                    )
                )

            # Short-term gain budget: a rebalance that would push realized
            # short-term gains past the client's stated tolerance should say so
            # before the trade, not after it.
            if policy.max_short_term_gain_budget is not None:
                remaining = policy.max_short_term_gain_budget - realized.get("short_term", 0.0)
                if remaining <= 0:
                    notes.append(
                        f"This client's ${policy.max_short_term_gain_budget:,.0f} short-term "
                        f"gain budget for the year is exhausted. Prefer long-held lots, or "
                        f"defer discretionary sales into the next tax year."
                    )
                else:
                    notes.append(
                        f"${remaining:,.0f} of the client's short-term gain budget remains "
                        f"for this tax year."
                    )

            if not notes:
                notes.append(
                    "No wash-sale conflicts, harvesting opportunities or realized gains to "
                    "report for this client."
                )

            assessment = TaxAssessment(
                wash_sale_flags=sorted(set(wash_sale_flags)),
                wash_sale_detail=wash_sale_detail,
                harvest_candidates=harvest,
                realized_ytd=realized,
                estimated_tax_cost=0.0,
                tax_efficiency_notes=notes,
                verification_failed=False,
            )

            ctx.output_snapshot = {
                "wash_sale_flags": assessment["wash_sale_flags"],
                "harvest_candidates": len(harvest),
                "realized_total": realized.get("total", 0.0),
            }
            ctx.summary = (
                f"{len(assessment['wash_sale_flags'])} wash-sale flag(s), "
                f"{len(harvest)} harvesting opportunity(ies), "
                f"${realized.get('total', 0.0):,.0f} realized year to date."
            )
            logger.info("[Tax] %s", ctx.summary)
        finally:
            db.close()

    if assessment is None:
        # Fail closed. The previous version returned empty flags on any
        # exception, so a database blip silently disabled the wash-sale
        # guardrail while the run still reported success. An unverifiable tax
        # position is not a clean one.
        assessment = TaxAssessment(
            wash_sale_flags=[],
            wash_sale_detail=[],
            harvest_candidates=[],
            realized_ytd={},
            estimated_tax_cost=0.0,
            tax_efficiency_notes=[
                "The tax assessment could not be completed, so no purchase in this run has "
                "been checked against the wash-sale rule. Every proposed purchase is "
                "unverified for tax purposes until this is resolved."
            ],
            verification_failed=True,
        )

    return finish(ctx, {"tax_assessment": assessment})
