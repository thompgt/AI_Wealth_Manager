"""
End-to-end test of the full LangGraph orchestrator against the seeded dev
client (client_id=1), replacing the old broken integration test that
imported the now-deleted backend/app package.

All network and LLM calls are stubbed via the `patched_external_calls`
fixture in conftest.py so this test is fast, deterministic, and does not
require a live GEMINI_API_KEY or internet access -- it exercises real graph
control flow (fan-out/fan-in, the human-approval interrupt, resume) against
fake data, not real market/LLM behavior (that's what the agents' own
`__main__` smoke tests and the manual verification steps in PLAN.md cover).
"""

import db
import orchestrator


def _ensure_seeded():
    db.init_db()
    db.seed_db()  # no-ops if already seeded


def test_full_graph_run_reaches_approval_gate_and_produces_a_report(patched_external_calls):
    _ensure_seeded()

    run = orchestrator.run_client_graph(1)
    result = run["result"]

    # The seeded client's real macro signals are stubbed out (confidence=0.0
    # via the forced LLM failure), so the run should pause at the
    # human-approval interrupt rather than completing straight through.
    assert "__interrupt__" in result
    interrupt_obj = result["__interrupt__"][0]
    assert interrupt_obj.value["reason"] in {"low_confidence_regime", "suitability_exhausted_retries"}

    resumed = orchestrator.resume_client_graph(run["run_id"], approved=True)
    final_state = resumed["result"]

    assert final_state.get("portfolio_diagnostics") is not None
    assert final_state.get("market_regime") is not None
    assert final_state.get("candidate_stocks") is not None
    assert final_state.get("suitability_result") is not None
    assert final_state.get("tax_assessment") is not None
    assert final_state.get("final_report")
    assert final_state.get("human_approved") is True

    node_sequence = [r["node_name"] for r in final_state.get("audit_trail", [])]
    for expected_node in [
        "portfolio_diagnostics",
        "market_regime",
        "stock_research",
        "suitability",
        "tax_awareness",
        "finance_report",
    ]:
        assert expected_node in node_sequence


def test_wash_sale_seed_scenario_is_still_caught(patched_external_calls):
    """
    The seeded client has a TSLA SELL logged 15 days ago (see db.seed_db).
    If stock_research happens to propose TSLA again this run, tax_awareness
    must flag it as a wash-sale candidate.
    """
    _ensure_seeded()

    run = orchestrator.run_client_graph(1)
    result = run["result"]
    if "__interrupt__" in result:
        resumed = orchestrator.resume_client_graph(run["run_id"], approved=True)
        result = resumed["result"]

    candidates = result.get("candidate_stocks") or []
    tax_assessment = result.get("tax_assessment") or {}
    if any(c["ticker"] == "TSLA" for c in candidates):
        assert "TSLA" in tax_assessment.get("wash_sale_flags", [])
