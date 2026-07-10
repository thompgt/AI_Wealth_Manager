"""Tax-Awareness Agent.

A deterministic LangGraph node -- NO LLM calls anywhere in this module. It
checks a real, precise tax rule (the 30-day wash-sale rule) with plain
date-math and DB queries, and layers on lightweight tax-efficiency
observations (embedded gain/loss awareness) that downstream
report-generation can surface to the client.

Ported from the legacy agents/tax_architect.py's wash-sale check
(SELLs in the last 30 days -> flag any proposed BUY of the same symbol),
generalized to:
  - the new db.py schema (TransactionLog.client_id, the Holding model,
    ClientProfile instead of UserProfile)
  - produce a structured TaxAssessment (see state.py) that the orchestrator
    routes on, instead of raising/forcing a rebuild directly
  - embedded gain/loss awareness across the client's current Holding rows,
    using live prices from services/market_data.get_current_prices
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set

# Allow running this file directly (`python agents/tax_awareness.py`) as well
# as importing it as part of the `agents` package -- both need the repo
# root on sys.path so `import db`, `import state`, etc. resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from db import Holding, SessionLocal, TransactionLog
from services.market_data import get_current_prices
from state import AgentRunRecord, AgentState, Candidate, TaxAssessment

NODE_NAME = "tax_awareness"

# IRS wash-sale rule: cannot claim a loss on a security sold if a
# "substantially identical" security is bought within 30 days before or
# after the sale. We only have visibility into our own transaction log, so
# this checks the 30-day-after side (a proposed BUY following a recent SELL).
WASH_SALE_WINDOW_DAYS = 30

# Unrealized gain, as a fraction of cost basis, above which we flag the
# position as carrying a meaningful embedded tax liability worth planning
# around before trimming it.
SIGNIFICANT_GAIN_THRESHOLD = 0.20


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_recent_sold_symbols(db, client_id: int) -> Set[str]:
    """Symbols sold by this client within the wash-sale window, using the
    same naive-UTC convention as the rest of the codebase (db.py's
    TransactionLog.timestamp and seed data are naive UTC)."""
    cutoff = datetime.utcnow() - timedelta(days=WASH_SALE_WINDOW_DAYS)
    recent_sells = (
        db.query(TransactionLog)
        .filter(
            TransactionLog.client_id == client_id,
            TransactionLog.transaction_type == "SELL",
            TransactionLog.timestamp >= cutoff,
        )
        .all()
    )
    return {log.asset_symbol for log in recent_sells}


def _wash_sale_check(candidates: List[Candidate], sold_symbols: Set[str]):
    flags: List[str] = []
    notes: List[str] = []
    for candidate in candidates:
        ticker = candidate["ticker"]
        if ticker in sold_symbols and ticker not in flags:
            flags.append(ticker)
            notes.append(
                f"{ticker} was sold within the last {WASH_SALE_WINDOW_DAYS} days -- "
                f"buying it back now would be a wash-sale violation and disallow any "
                f"realized loss on the original sale. Consider waiting out the "
                f"{WASH_SALE_WINDOW_DAYS}-day window before repurchasing, or evaluate "
                f"a correlated-but-not-identical substitute security instead (specific "
                f"substitute selection is Stock Research's job, not this check's)."
            )
    return flags, notes


def _embedded_gain_loss_notes(holdings: List[Holding], prices: Dict[str, float]) -> List[str]:
    notes: List[str] = []
    for holding in holdings:
        if holding.symbol == "CASH":
            continue  # no meaningful cost-basis gain/loss on the cash sleeve

        price = prices.get(holding.symbol)
        if price is None or price <= 0 or holding.cost_basis <= 0:
            continue  # can't compute a meaningful gain/loss without real data

        cost_total = holding.quantity * holding.cost_basis
        market_value = holding.quantity * price
        gain = market_value - cost_total
        pct = gain / cost_total if cost_total else 0.0

        if gain < 0:
            notes.append(
                f"{holding.symbol} is sitting at an unrealized loss of "
                f"${abs(gain):,.2f} ({pct * 100:.1f}%) -- a potential tax-loss-"
                f"harvesting candidate."
            )
        elif pct > SIGNIFICANT_GAIN_THRESHOLD:
            notes.append(
                f"{holding.symbol} carries an unrealized gain of ${gain:,.2f} "
                f"({pct * 100:.1f}%) -- if Portfolio Diagnostics flags this position "
                f"as over-concentrated, trimming it would trigger a taxable event "
                f"worth planning around (e.g. tax-lot selection, spreading the sale "
                f"across tax years)."
            )
    return notes


def tax_awareness_node(state: AgentState) -> dict:
    started_at = _now_iso()
    client_id = state["client_profile"]["client_id"]
    candidates: List[Candidate] = state.get("candidate_stocks") or []

    error_detail = None
    try:
        db = SessionLocal()
        try:
            sold_symbols = _get_recent_sold_symbols(db, client_id)
            wash_sale_flags, wash_notes = _wash_sale_check(candidates, sold_symbols)

            holdings = db.query(Holding).filter(Holding.client_id == client_id).all()
            non_cash_symbols = [h.symbol for h in holdings if h.symbol != "CASH"]
            prices = get_current_prices(non_cash_symbols) if non_cash_symbols else {}
            gain_loss_notes = _embedded_gain_loss_notes(holdings, prices)
        finally:
            db.close()

        tax_efficiency_notes = wash_notes + gain_loss_notes

        tax_assessment: TaxAssessment = {
            "wash_sale_flags": wash_sale_flags,
            "tax_efficiency_notes": tax_efficiency_notes,
        }
        status = "success"
        summary = (
            f"Checked {len(candidates)} candidate(s) against {len(sold_symbols)} "
            f"recent sell(s) in the {WASH_SALE_WINDOW_DAYS}-day window; "
            f"{len(wash_sale_flags)} wash-sale flag(s); {len(gain_loss_notes)} "
            f"embedded gain/loss note(s)."
        )
    except Exception as exc:  # noqa: BLE001 -- node must never crash the graph
        status = "error"
        error_detail = repr(exc)
        summary = "Tax-awareness assessment failed; returning safe defaults."
        tax_assessment = {
            "wash_sale_flags": [],
            "tax_efficiency_notes": [f"Tax assessment could not be computed: {exc}"],
        }

    record: AgentRunRecord = {
        "node_name": NODE_NAME,
        "started_at": started_at,
        "completed_at": _now_iso(),
        "status": status,
        "summary": summary,
        "error_detail": error_detail,
    }

    return {"tax_assessment": tax_assessment, "audit_trail": [record]}


if __name__ == "__main__":
    import json
    import uuid

    from db import ClientProfile as ClientProfileRow
    from db import Holding as HoldingRow
    from services.market_data import get_current_prices as _get_current_prices
    from state import ClientProfile

    db = SessionLocal()
    try:
        client_row = db.query(ClientProfileRow).filter(ClientProfileRow.id == 1).first()
        if client_row is None:
            raise SystemExit("client_id=1 not found -- run `python db.py` to seed the DB first.")

        holding_rows = db.query(HoldingRow).filter(HoldingRow.client_id == 1).all()

        holdings: Dict[str, float] = {}
        non_cash_symbols = [h.symbol for h in holding_rows if h.symbol != "CASH"]
        current_prices = _get_current_prices(non_cash_symbols) if non_cash_symbols else {}

        for h in holding_rows:
            if h.symbol == "CASH":
                holdings["CASH"] = h.quantity
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

    # The seeded DB has a TSLA SELL dated 15 days ago for client_id=1 -- a
    # proposed TSLA BUY should get wash-sale-flagged. AAPL has no recent
    # sell on record and should NOT be flagged.
    candidate_stocks: List[Candidate] = [
        {
            "ticker": "TSLA",
            "valuation_metrics": {"pe_ratio": 60.0},
            "addresses_flaw": "smoke-test candidate",
            "regime_fit_rationale": "smoke-test candidate",
            "confidence": 0.5,
        },
        {
            "ticker": "AAPL",
            "valuation_metrics": {"pe_ratio": 28.0},
            "addresses_flaw": "smoke-test candidate",
            "regime_fit_rationale": "smoke-test candidate",
            "confidence": 0.5,
        },
    ]

    sample_state: AgentState = {
        "run_id": str(uuid.uuid4()),
        "client_profile": client_profile,
        "portfolio_diagnostics": None,
        "market_regime": None,
        "candidate_stocks": candidate_stocks,
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
    print("Candidate stocks: TSLA (expect wash-sale flag), AAPL (expect no flag)")
    print()

    result = tax_awareness_node(sample_state)

    print("=== tax_assessment ===")
    print(json.dumps(result["tax_assessment"], indent=2))
    print()
    print("=== audit_trail ===")
    print(json.dumps(result["audit_trail"], indent=2))

    flags = result["tax_assessment"]["wash_sale_flags"]
    assert "TSLA" in flags, f"expected TSLA to be wash-sale flagged, got {flags}"
    assert "AAPL" not in flags, f"expected AAPL to NOT be wash-sale flagged, got {flags}"
    print()
    print("PASS: TSLA flagged, AAPL not flagged.")
