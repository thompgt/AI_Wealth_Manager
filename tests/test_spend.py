"""The per-organisation daily model budget.

`RunBudget` caps one run. That is the wrong unit for the bill: a thousand runs
each finishing under their own cap is a thousand times the cap, and the run
rate limit does not help because a count times an unknown per-run cost is an
unknown.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from db import AgentRun, ClientProfile, Organization, SessionLocal, init_db
from services import spend


@pytest.fixture
def org_with_runs():
    """An org and a helper that records model spend against it."""
    init_db()
    db = SessionLocal()
    org = Organization(name="Spend Test", slug=f"spend-{datetime.now().timestamp()}")
    db.add(org)
    db.flush()
    client = ClientProfile(org_id=org.id, name="Spend Client")
    db.add(client)
    db.flush()

    def record(cost, when=None):
        db.add(
            AgentRun(
                org_id=org.id,
                client_id=client.id,
                run_id=f"run-{cost}-{when}",
                node_name="market_regime",
                started_at=when or datetime.now(timezone.utc).replace(tzinfo=None),
                cost_usd=Decimal(str(cost)),
            )
        )
        db.flush()

    try:
        yield db, org, record
    finally:
        db.rollback()
        db.close()


def test_an_org_with_no_runs_has_spent_nothing(org_with_runs):
    db, org, _ = org_with_runs
    assert spend.spend_today(db, org.id) == 0.0


def test_spend_sums_todays_runs(org_with_runs):
    db, org, record = org_with_runs
    record(0.25)
    record(0.40)
    assert spend.spend_today(db, org.id) == pytest.approx(0.65)


def test_yesterdays_spend_does_not_count_against_today(org_with_runs):
    """The window is a calendar day, so it has to actually roll over."""
    db, org, record = org_with_runs
    record(5.00, when=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1))
    record(0.10)
    assert spend.spend_today(db, org.id) == pytest.approx(0.10)


def test_one_orgs_spend_is_invisible_to_another(org_with_runs, monkeypatch):
    """Tenancy applies to the budget too.

    A shared counter would let a busy firm lock out a quiet one, which is a
    denial of service one customer can inflict on another.
    """
    db, org, record = org_with_runs
    record(10.00)

    other = Organization(name="Other", slug=f"other-{datetime.now().timestamp()}")
    db.add(other)
    db.flush()
    assert spend.spend_today(db, other.id) == 0.0


def test_under_the_limit_check_passes(org_with_runs, monkeypatch):
    db, org, record = org_with_runs
    monkeypatch.setattr(spend.settings, "LLM_ORG_DAILY_BUDGET_USD", 25.0)
    record(1.00)
    spend.check(db, org.id)  # must not raise


def test_at_the_limit_the_next_run_is_refused(org_with_runs, monkeypatch):
    db, org, record = org_with_runs
    monkeypatch.setattr(spend.settings, "LLM_ORG_DAILY_BUDGET_USD", 2.0)
    record(2.00)
    with pytest.raises(spend.DailyBudgetExceeded) as caught:
        spend.check(db, org.id)
    message = str(caught.value)
    # The message has to say when it resets. "Refused" with no horizon turns
    # into a support ticket every time.
    assert "resets at" in message
    assert "$2.00" in message


def test_a_non_positive_limit_means_unlimited(org_with_runs, monkeypatch):
    """Matching how RunBudget reads its own limit, so the two are learned once."""
    db, org, record = org_with_runs
    monkeypatch.setattr(spend.settings, "LLM_ORG_DAILY_BUDGET_USD", 0.0)
    record(1000.00)
    spend.check(db, org.id)
    assert spend.status(db, org.id)["unlimited"] is True


def test_status_reports_headroom_and_the_window(org_with_runs, monkeypatch):
    db, org, record = org_with_runs
    monkeypatch.setattr(spend.settings, "LLM_ORG_DAILY_BUDGET_USD", 10.0)
    record(2.50)
    result = spend.status(db, org.id)
    assert result["spent_usd"] == pytest.approx(2.50)
    assert result["limit_usd"] == 10.0
    assert result["remaining_usd"] == pytest.approx(7.50)
    assert result["window_end"] > result["window_start"]


def test_headroom_never_goes_negative(org_with_runs, monkeypatch):
    """An overshoot is possible -- the check is against recorded spend, so a
    run in flight has not contributed yet. Reporting negative headroom would
    render as a nonsense number in a dashboard."""
    db, org, record = org_with_runs
    monkeypatch.setattr(spend.settings, "LLM_ORG_DAILY_BUDGET_USD", 1.0)
    record(3.00)
    assert spend.status(db, org.id)["remaining_usd"] == 0.0
