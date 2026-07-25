"""Tests for the guardrail gate -- the node that reconciles the Suitability
and Tax-Awareness agents into one decision.

This is the highest-stakes piece of control flow in the system: it is the
only place where a compliance finding turns into a recommendation being
withheld. Before these fixes it did neither of the two things it exists to
do (enforce the tax assessment, and make a retry differ from the first pass).
"""

from orchestrator import (
    MAX_RESEARCH_ATTEMPTS,
    approval_gate_node,
    guardrail_gate_node,
    route_after_guardrails,
)


def _rec(ticker, amount=1000.0):
    return {
        "ticker": ticker,
        "valuation_metrics": {},
        "addresses_flaw": "test",
        "regime_fit_rationale": "test",
        "confidence": 0.7,
        "allocation_amount": amount,
        "allocation_pct": 0.01,
    }


def _state(**overrides):
    base = {
        "suitability_result": {"approved": True, "violations": [], "adjusted_recommendations": []},
        "tax_assessment": {"wash_sale_flags": [], "tax_efficiency_notes": []},
        "market_regime": {"regime_label": "Bull", "confidence": 0.8},
        "research_attempts": 0,
        "excluded_tickers": [],
        "guardrail_feedback": [],
        "candidate_stocks": [],
    }
    base.update(overrides)
    return base


# --- tax enforcement --------------------------------------------------------

def test_wash_sale_flagged_recommendation_is_removed():
    """The core compliance bug: a wash-sale-flagged ticker used to sail
    straight through to the client report with a dollar amount attached."""
    state = _state(
        suitability_result={
            "approved": True,
            "violations": [],
            "adjusted_recommendations": [_rec("TSLA"), _rec("JNJ")],
        },
        tax_assessment={"wash_sale_flags": ["TSLA"], "tax_efficiency_notes": []},
    )
    update = guardrail_gate_node(state)

    tickers = [r["ticker"] for r in update["suitability_result"]["adjusted_recommendations"]]
    assert tickers == ["JNJ"], "TSLA must not survive a wash-sale flag"
    assert update["tax_blocked_recommendations"] == ["TSLA"]
    assert any("TSLA" in v and "wash-sale" in v for v in update["suitability_result"]["violations"])


def test_tax_block_is_explained_not_silent():
    """A withheld recommendation the client can't see the reason for is
    indistinguishable from the system just not finding anything."""
    state = _state(
        suitability_result={
            "approved": True,
            "violations": [],
            "adjusted_recommendations": [_rec("TSLA")],
        },
        tax_assessment={"wash_sale_flags": ["TSLA"], "tax_efficiency_notes": []},
    )
    update = guardrail_gate_node(state)
    violations = update["suitability_result"]["violations"]
    assert violations and "wash-sale" in violations[0]


def test_approval_flips_to_false_when_the_tax_block_empties_the_list():
    state = _state(
        suitability_result={
            "approved": True,
            "violations": [],
            "adjusted_recommendations": [_rec("TSLA")],
        },
        tax_assessment={"wash_sale_flags": ["TSLA"], "tax_efficiency_notes": []},
    )
    update = guardrail_gate_node(state)
    assert update["suitability_result"]["approved"] is False


def test_unflagged_recommendations_pass_through_untouched():
    state = _state(
        suitability_result={
            "approved": True,
            "violations": [],
            "adjusted_recommendations": [_rec("JNJ"), _rec("KO")],
        },
    )
    update = guardrail_gate_node(state)
    tickers = [r["ticker"] for r in update["suitability_result"]["adjusted_recommendations"]]
    assert tickers == ["JNJ", "KO"]
    assert update["tax_blocked_recommendations"] == []
    assert update["needs_research_retry"] is False


# --- retry feedback ---------------------------------------------------------

def test_rejected_tickers_are_accumulated_for_the_retry_pass():
    """Without this, a retry re-screens the same universe with the same
    inputs and returns the same rejected picks."""
    state = _state(
        suitability_result={
            "approved": False,
            "violations": [
                "SIRC: rejected -- market cap $80,000,000 is below the threshold.",
                "BBIG: rejected -- listed on 'PNK', not a major exchange.",
            ],
            "adjusted_recommendations": [],
        },
    )
    update = guardrail_gate_node(state)

    assert set(update["excluded_tickers"]) == {"SIRC", "BBIG"}
    assert update["needs_research_retry"] is True
    assert route_after_guardrails(update) == "retry"
    assert len(update["guardrail_feedback"]) == 2


def test_tax_blocked_tickers_are_also_excluded_from_the_retry():
    state = _state(
        suitability_result={
            "approved": True,
            "violations": [],
            "adjusted_recommendations": [_rec("TSLA")],
        },
        tax_assessment={"wash_sale_flags": ["TSLA"], "tax_efficiency_notes": []},
    )
    update = guardrail_gate_node(state)
    assert "TSLA" in update["excluded_tickers"]


def test_exclusions_carry_forward_across_attempts():
    state = _state(
        excluded_tickers=["OLD"],
        guardrail_feedback=["OLD: rejected earlier"],
        suitability_result={
            "approved": False,
            "violations": ["NEW: rejected -- market cap too small."],
            "adjusted_recommendations": [],
        },
    )
    update = guardrail_gate_node(state)
    assert set(update["excluded_tickers"]) == {"OLD", "NEW"}


def test_retry_budget_is_respected():
    state = _state(
        research_attempts=MAX_RESEARCH_ATTEMPTS,
        suitability_result={"approved": False, "violations": [], "adjusted_recommendations": []},
    )
    update = guardrail_gate_node(state)
    assert update["needs_research_retry"] is False
    assert route_after_guardrails(update) == "proceed"


# --- approval gate reasons --------------------------------------------------

def test_clean_high_confidence_run_is_auto_approved():
    state = _state(
        suitability_result={"approved": True, "violations": [], "adjusted_recommendations": [_rec("JNJ")]},
    )
    update = approval_gate_node(state)
    assert update["requires_human_approval"] is False
    assert update["human_approved"] is True


def test_first_pass_rejection_is_not_mislabelled_as_exhausted_retries(monkeypatch):
    """The gate used to report every no-approval outcome as
    'suitability_exhausted_retries', including ones where no retry was ever
    spent -- misleading whoever reviews the paused run."""
    captured = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return True

    monkeypatch.setattr("orchestrator.interrupt", fake_interrupt)

    state = _state(
        research_attempts=0,
        suitability_result={"approved": False, "violations": [], "adjusted_recommendations": []},
    )
    approval_gate_node(state)
    assert captured["reason"] == "no_suitable_candidates"


def test_exhausted_retries_are_labelled_as_such(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "orchestrator.interrupt", lambda payload: captured.update(payload) or True
    )

    state = _state(
        research_attempts=MAX_RESEARCH_ATTEMPTS,
        suitability_result={"approved": False, "violations": [], "adjusted_recommendations": []},
    )
    approval_gate_node(state)
    assert captured["reason"] == "suitability_exhausted_retries"


def test_low_confidence_regime_pauses_the_run(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "orchestrator.interrupt", lambda payload: captured.update(payload) or True
    )

    state = _state(
        market_regime={"regime_label": "Volatile", "confidence": 0.0},
        suitability_result={"approved": True, "violations": [], "adjusted_recommendations": [_rec("JNJ")]},
    )
    update = approval_gate_node(state)
    assert captured["reason"] == "low_confidence_regime"
    assert update["requires_human_approval"] is True
