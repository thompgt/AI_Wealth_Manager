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


def test_a_wash_sale_flag_removes_the_ticker_from_the_final_recommendations(
    patched_external_calls, monkeypatch
):
    """End-to-end proof of the compliance fix.

    Forces Stock Research to propose TSLA -- which the seeded client sold at
    a loss 15 days ago -- and asserts it does not appear in the final
    recommendations, with the reason stated. Previously the flag was raised
    and the ticker was recommended anyway.
    """
    import agents.stock_research as stock_research

    _ensure_seeded()

    def only_tsla(state):
        return {
            "candidate_stocks": [
                {
                    "ticker": "TSLA",
                    "valuation_metrics": {"pe_ratio": 60.0},
                    "addresses_flaw": "forced test candidate",
                    "regime_fit_rationale": "forced test candidate",
                    "confidence": 0.9,
                }
            ],
            "audit_trail": [
                {
                    "node_name": "stock_research",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "completed_at": "2026-01-01T00:00:01+00:00",
                    "status": "success",
                    "summary": "forced single-candidate stub",
                    "error_detail": None,
                }
            ],
        }

    monkeypatch.setattr(stock_research, "stock_research_node", only_tsla)
    monkeypatch.setattr(orchestrator, "stock_research_node", only_tsla)
    orchestrator.reset_workflow()
    try:
        run = orchestrator.run_client_graph(1)
        result = run["result"]
        if "__interrupt__" in result:
            result = orchestrator.resume_client_graph(run["run_id"], approved=True)["result"]

        assert "TSLA" in (result.get("tax_assessment") or {}).get("wash_sale_flags", [])

        recommended = [
            r["ticker"]
            for r in (result.get("suitability_result") or {}).get("adjusted_recommendations", [])
        ]
        assert "TSLA" not in recommended, "a wash-sale-flagged ticker must never be recommended"
        assert "TSLA" in (result.get("tax_blocked_recommendations") or [])

        violations = (result.get("suitability_result") or {}).get("violations", [])
        assert any("TSLA" in v and "wash-sale" in v for v in violations)

        # And the client-facing report must say so rather than silently
        # dropping the name.
        assert "TSLA" in (result.get("final_report") or "")
    finally:
        orchestrator.reset_workflow()
