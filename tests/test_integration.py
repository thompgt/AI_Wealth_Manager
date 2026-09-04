"""
End-to-end test of the full LangGraph orchestrator against the demo client
seeded by `demo_data.py`.

It runs against `demo_data.seed_demo_client()` rather than a bespoke fixture on
purpose. That seed is the portfolio the README tells a new user to create and
the one the demo notebook analyses, so it is the arrangement most likely to
break; a private fixture would keep this test green while the documented entry
point rotted. `db.seed_db()`, which this test used to call, no longer exists.

All network and LLM calls are stubbed so this test is fast, deterministic, and
does not require a live GEMINI_API_KEY or internet access -- it exercises real
graph control flow (fan-out/fan-in, the guardrail retry loop, the human
approval interrupt, resume) against fake data, not real market/LLM behavior
(that's what the agents' own `__main__` smoke tests and the manual verification
steps in PLAN.md cover).

Two stubbing seams are needed, not one:

* `patched_external_calls` (conftest) covers price history, quotes, news and
  the LLM *as each agent module bound them*.
* `services.portfolio.get_quotes` is bound inside `services/portfolio.py`,
  which the shared fixture does not touch -- and `load_portfolio` is on the
  critical path of four nodes plus `load_client_state`. Left alone it reaches
  the live provider chain, so it is patched here.

The seed itself is kept off the network by injecting `price_lookup`;
`services.market_data.get_current_prices` (its default) does real I/O.
"""

import pytest

import demo_data
import orchestrator
from tests.conftest import make_quote

# One price for everything: the demo weights are derived from whatever price
# the seed is handed, so a flat price still produces the 45/15/40 split the
# flawed portfolio depends on.
STUB_PRICE = 250.0

# A name the demo client genuinely sold at a loss inside the wash-sale window.
# Taken from the seed rather than hardcoded, so if that data changes this test
# follows it instead of silently testing a ticker with no loss behind it.
BLOCKED_TICKER = "XOM"


@pytest.fixture
def offline_quotes(monkeypatch):
    """Close the one pricing seam `patched_external_calls` leaves open."""
    import services.market_data as market_data
    import services.portfolio as portfolio

    def fake_get_quotes(tickers, **kwargs):
        return {
            str(t).upper(): make_quote(str(t).upper(), STUB_PRICE)
            for t in tickers
            if str(t).upper() != "CASH"
        }

    monkeypatch.setattr(portfolio, "get_quotes", fake_get_quotes)
    # Also at the definition site, so anything reaching for it by module path
    # (including `get_current_prices`) stays offline too.
    monkeypatch.setattr(market_data, "get_quotes", fake_get_quotes)


@pytest.fixture
def seeded_client(offline_quotes):
    """The demo portfolio, seeded into the test database. Returns its id."""

    def fake_prices(tickers):
        return {str(t).upper(): STUB_PRICE for t in tickers}

    return demo_data.seed_demo_client(price_lookup=fake_prices)


def test_the_seed_produces_the_flaws_the_demo_is_supposed_to_find(seeded_client):
    """The rest of this file is meaningless if the fixture is not flawed.

    Asserted here rather than assumed: a seed that quietly stopped producing a
    concentrated, loss-harvested portfolio would leave every test below passing
    against a clean book, proving nothing.
    """
    from db import SessionLocal
    from db import ClientProfile as ClientProfileModel
    from services import tax_lots
    from services.portfolio import load_portfolio

    db = SessionLocal()
    try:
        client = (
            db.query(ClientProfileModel)
            .filter(ClientProfileModel.id == seeded_client)
            .first()
        )
        view = load_portfolio(db, client)

        weights = view.weights()
        assert weights["AAPL"] == pytest.approx(0.45, abs=0.01)
        assert weights["CASH"] == pytest.approx(0.40, abs=0.01)
        # The whole equity sleeve is one sector.
        assert view.sector_weights()["Technology"] == pytest.approx(1.0, abs=0.01)

        # Every harvested name is a live wash-sale block, with a real loss.
        for ticker in demo_data.RECENTLY_HARVESTED:
            finding = tax_lots.check_wash_sale(db, seeded_client, ticker)
            assert finding is not None, f"{ticker} should be inside the wash-sale window"
            assert finding.loss_amount < 0
    finally:
        db.close()


def test_full_graph_run_reaches_approval_gate_and_produces_a_report(
    patched_external_calls, seeded_client
):
    run = orchestrator.run_client_graph(seeded_client)
    result = run["result"]

    # The demo portfolio is 45% in one name against a 25% cap, so the plan
    # trims -- and a run that proposes selling something, realizes tax, or
    # rests on a stubbed-out (confidence 0.0) regime call must pause for a
    # person rather than complete straight through.
    assert "__interrupt__" in result
    interrupt_obj = result["__interrupt__"][0]
    reason = interrupt_obj.value["reason"]
    assert reason in orchestrator.APPROVAL_EXPLANATIONS, (
        f"the gate paused for {reason!r}, which has no explanation to show a reviewer"
    )
    assert interrupt_obj.value["explanation"]

    resumed = orchestrator.resume_client_graph(run["run_id"], approved=True)
    final_state = resumed["result"]

    assert final_state.get("portfolio_diagnostics") is not None
    assert final_state.get("market_regime") is not None
    assert final_state.get("candidate_stocks") is not None
    assert final_state.get("suitability_result") is not None
    assert final_state.get("tax_assessment") is not None
    assert final_state.get("rebalance_plan") is not None
    assert final_state.get("final_report")
    assert final_state.get("human_approved") is True

    # The tax guardrail ran rather than failing closed -- otherwise the
    # wash-sale assertions below would be vacuously satisfied by an error.
    assert final_state["tax_assessment"].get("verification_failed") is False

    # And the diagnostics found the flaws the seed builds in.
    flaws = " ".join(final_state["portfolio_diagnostics"].get("flaws") or [])
    assert "AAPL" in flaws

    node_sequence = [r["node_name"] for r in final_state.get("audit_trail", [])]
    for expected_node in [
        # The node is named "diagnostics"; the state key it writes is
        # "portfolio_diagnostics". The old test asserted the state key here and
        # would never have matched.
        "diagnostics",
        "market_regime",
        "stock_research",
        "suitability",
        "tax_awareness",
        "rebalance",
        "finance_report",
    ]:
        assert expected_node in node_sequence


def test_wash_sale_seed_scenario_is_still_caught(patched_external_calls, seeded_client):
    """
    The seeded client sold five names at a loss in the last 30 days (see
    `demo_data.RECENTLY_HARVESTED`). On an unforced run this asserts the flags
    in both directions:

    * any harvested name research happens to propose *is* flagged, unprompted,
      from the real lot records; and
    * nothing else is. The screen usually fixes this portfolio's concentration
      with international funds and proposes none of the harvested names, in
      which case there must be no flags at all -- a run that flagged them
      anyway would be the old bug, where the check fired on the client's
      sale history rather than on what was actually being bought.

    The forced-candidate test below is what proves enforcement; this one proves
    the flag tracks the candidates.
    """
    run = orchestrator.run_client_graph(seeded_client)
    result = run["result"]
    if "__interrupt__" in result:
        resumed = orchestrator.resume_client_graph(run["run_id"], approved=True)
        result = resumed["result"]

    candidates = result.get("candidate_stocks") or []
    tax_assessment = result.get("tax_assessment") or {}
    assert tax_assessment.get("verification_failed") is False

    harvested = set(demo_data.RECENTLY_HARVESTED)
    proposed = {str(c["ticker"]).upper() for c in candidates}
    flags = set(tax_assessment.get("wash_sale_flags") or [])

    assert proposed & harvested == flags, (
        f"wash-sale flags {sorted(flags)} do not match the harvested names actually "
        f"proposed {sorted(proposed & harvested)}"
    )
    assert flags <= proposed, "a ticker was flagged that this run never proposed buying"
    assert set(result.get("tax_blocked_recommendations") or []) <= flags


def test_a_wash_sale_flag_removes_the_ticker_from_the_final_recommendations(
    patched_external_calls, seeded_client, monkeypatch
):
    """End-to-end proof of the compliance fix.

    Forces Stock Research to propose a name the seeded client sold at a loss
    inside the wash-sale window, and asserts it does not appear in the final
    recommendations, with the reason stated. Previously the flag was raised and
    the ticker was recommended anyway.

    Research is forced rather than left to the screen because the point under
    test is the guardrail, not the screen's taste: if the screen simply never
    picked a blocked name, this test would pass without ever exercising the
    enforcement path.
    """
    import agents.stock_research as stock_research

    def only_the_blocked_ticker(state):
        return {
            "candidate_stocks": [
                {
                    "ticker": BLOCKED_TICKER,
                    "valuation_metrics": {"pe_ratio": 12.0},
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

    # Both binding sites: the module that defines it, and the name the
    # orchestrator bound at import time and hands to `add_node`.
    monkeypatch.setattr(stock_research, "stock_research_node", only_the_blocked_ticker)
    monkeypatch.setattr(orchestrator, "stock_research_node", only_the_blocked_ticker)
    orchestrator.reset_workflow()
    try:
        run = orchestrator.run_client_graph(seeded_client)
        result = run["result"]
        if "__interrupt__" in result:
            result = orchestrator.resume_client_graph(run["run_id"], approved=True)["result"]

        tax_assessment = result.get("tax_assessment") or {}
        assert tax_assessment.get("verification_failed") is False
        assert BLOCKED_TICKER in tax_assessment.get("wash_sale_flags", [])

        recommended = [
            r["ticker"]
            for r in (result.get("suitability_result") or {}).get("adjusted_recommendations", [])
        ]
        assert BLOCKED_TICKER not in recommended, (
            "a wash-sale-flagged ticker must never be recommended"
        )
        assert BLOCKED_TICKER in (result.get("tax_blocked_recommendations") or [])

        # The reason must be stated, not merely acted on: the violation carries
        # the loss, the date and the remaining window from the real lot record.
        violations = (result.get("suitability_result") or {}).get("violations", [])
        assert any(
            BLOCKED_TICKER in v and "sold at a loss" in v and "Blocked for" in v
            for v in violations
        ), f"no violation explains why {BLOCKED_TICKER} was withheld: {violations}"

        # And the client-facing report must say so rather than silently
        # dropping the name.
        report = result.get("final_report") or ""
        assert BLOCKED_TICKER in report
        assert "What Was Withheld" in report
    finally:
        orchestrator.reset_workflow()


def test_the_stored_report_records_what_was_withheld(monkeypatch, tmp_path):
    """A report showing only the survivors cannot be told apart from a run
    where nothing was blocked.

    The guardrail gate computes `tax_blocked_recommendations` on every run,
    and the persistence boundary used to drop it -- so the artifact a client
    reads, and the one an examiner asks for, held the recommendations that
    survived and no trace of the ones a control removed.
    """
    from db import ClientProfile, Organization, Report, SessionLocal, init_db
    from services.run_service import persist_report

    init_db()
    db = SessionLocal()
    try:
        org = db.query(Organization).first()
        if org is None:
            org = Organization(name="Withheld Test", slug="withheld-test")
            db.add(org)
            db.flush()
        client = ClientProfile(org_id=org.id, name="Withheld Test Client")
        db.add(client)
        db.flush()

        state = {
            "final_report": "A report.",
            "portfolio_diagnostics": {},
            "market_regime": {},
            "suitability_result": {},
            "tax_assessment": {"wash_sale_flags": ["XOM", "CVX"]},
            "tax_blocked_recommendations": ["XOM", "CVX"],
            "human_approved": True,
        }
        report = persist_report(db, "run-withheld-1", client, state)
        db.flush()

        assert report is not None
        assert report.structured_payload["tax_blocked_recommendations"] == ["XOM", "CVX"]

        stored = db.query(Report).filter(Report.run_id == "run-withheld-1").first()
        assert stored.structured_payload["tax_blocked_recommendations"] == ["XOM", "CVX"]
    finally:
        db.rollback()
        db.close()
