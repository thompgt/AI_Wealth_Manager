"""Tax-Awareness node: what it flags, what it no longer flags, and what it
does when it cannot answer.

The helpers this file used to test (`_wash_sale_check`,
`_embedded_gain_loss_notes`, `SIGNIFICANT_GAIN_THRESHOLD`) are gone. Tax
handling is now lot-level and account-aware, and the node is a thin layer over
`services/tax_lots.py`, so the behaviour worth pinning down is the node's
output for a client whose lots are known: which candidates are blocked, which
losses are worth harvesting, and what an unverifiable tax position produces.

Everything below runs against the temp SQLite database conftest.py points at.
Prices are supplied directly, so nothing here touches the network.
"""

from decimal import Decimal
from datetime import timedelta
from uuid import uuid4

import pytest

import services.portfolio as portfolio_service
from agents.tax_awareness import MIN_HARVEST_LOSS, tax_awareness_node
from db import Account, ClientProfile, Organization, SessionLocal, TaxLot, init_db, utcnow
from services import tax_lots
from tests.conftest import make_quote


@pytest.fixture
def db():
    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def prices(monkeypatch):
    """Offline pricing for `load_portfolio`.

    `services.portfolio` looks `get_quotes` up on its own module at call time,
    so patching it there covers every caller. Tests populate the returned dict;
    a symbol left out of it is priced by nothing, which is exactly how an
    unpriced holding behaves in production.
    """
    quoted = {}

    def fake_get_quotes(symbols, **kwargs):
        return {s: make_quote(s, price=quoted[s]) for s in symbols if s in quoted}

    monkeypatch.setattr(portfolio_service, "get_quotes", fake_get_quotes)
    return quoted


def _make_client(db, accounts=(("Brokerage", "taxable", "individual"),)):
    """A client with the given (name, tax_treatment, account_type) accounts."""
    org = Organization(name="Tax Test Firm", slug=f"tax-test-{uuid4().hex[:10]}")
    db.add(org)
    db.flush()
    client = ClientProfile(
        org_id=org.id, name="Tax Test Client", risk_tolerance="Moderate", net_worth=Decimal("500000")
    )
    db.add(client)
    db.flush()
    created = []
    for name, treatment, account_type in accounts:
        account = Account(
            org_id=org.id,
            client_id=client.id,
            name=name,
            account_type=account_type,
            tax_treatment=treatment,
            cash_balance=Decimal("10000"),
        )
        db.add(account)
        created.append(account)
    db.flush()
    db.commit()
    return client, created


def _add_disposed_lot(db, account, symbol, *, realized_gain, days_ago, term="short", quantity=100):
    """A lot that has already been sold, with a known realized result."""
    disposed_at = utcnow() - timedelta(days=days_ago)
    lot = TaxLot(
        org_id=account.org_id,
        account_id=account.id,
        symbol=symbol,
        quantity=Decimal(str(quantity)),
        remaining_quantity=Decimal("0"),
        cost_per_share=Decimal("100"),
        acquired_at=disposed_at - timedelta(days=60),
        disposed_at=disposed_at,
        proceeds=Decimal("10000"),
        realized_gain=Decimal(str(realized_gain)),
        term=term,
    )
    db.add(lot)
    db.commit()
    return lot


def _run(client_id, candidates=()):
    state = {
        "run_id": f"tax-test-{uuid4().hex[:8]}",
        "client_profile": {"client_id": client_id},
        "candidate_stocks": list(candidates),
    }
    return tax_awareness_node(state)


def _assessment(result):
    return result["tax_assessment"]


# --- wash sales ---------------------------------------------------------------

def test_a_recent_loss_sale_blocks_repurchasing_the_same_symbol(db, prices):
    client, (brokerage,) = _make_client(db)
    _add_disposed_lot(db, brokerage, "TSLA", realized_gain=-4000, days_ago=10)

    assessment = _assessment(_run(client.id, [{"ticker": "TSLA"}, {"ticker": "AAPL"}]))

    assert assessment["wash_sale_flags"] == ["TSLA"]
    detail = assessment["wash_sale_detail"][0]
    assert detail["symbol"] == "TSLA"
    assert detail["blocked"] is True
    assert detail["days_remaining"] == 20
    assert "sold at a loss" in detail["reason"]


def test_a_recent_sale_at_a_gain_does_not_block_repurchasing(db, prices):
    """The rule applies to losses only.

    The previous version flagged any sale in the last 30 days, so a client who
    took a profit was barred from re-entering the name for a month and the
    research retry loop spent its budget routing around a restriction that
    does not exist.
    """
    client, (brokerage,) = _make_client(db)
    _add_disposed_lot(db, brokerage, "TSLA", realized_gain=4000, days_ago=10)

    assessment = _assessment(_run(client.id, [{"ticker": "TSLA"}]))

    assert assessment["wash_sale_flags"] == []


def test_a_loss_sale_older_than_the_window_does_not_block(db, prices):
    client, (brokerage,) = _make_client(db)
    _add_disposed_lot(db, brokerage, "TSLA", realized_gain=-4000, days_ago=45)

    assert _assessment(_run(client.id, [{"ticker": "TSLA"}]))["wash_sale_flags"] == []


def test_a_loss_realized_in_a_retirement_account_does_not_block(db, prices):
    """There is no deductible loss in a tax-exempt account, so there is
    nothing for the wash-sale rule to disallow."""
    client, (roth,) = _make_client(db, accounts=(("Roth", "tax_exempt", "roth_ira"),))
    _add_disposed_lot(db, roth, "TSLA", realized_gain=-4000, days_ago=5)

    assessment = _assessment(_run(client.id, [{"ticker": "TSLA"}]))

    assert assessment["wash_sale_flags"] == []
    assert any("no taxable accounts" in note for note in assessment["tax_efficiency_notes"])


def test_repurchasing_into_a_retirement_account_is_flagged_as_permanent(db, prices):
    """The most expensive version of the mistake: the loss is disallowed with
    no basis adjustment to recover it later."""
    client, (brokerage, ira) = _make_client(
        db,
        accounts=(
            ("Brokerage", "taxable", "individual"),
            ("Rollover IRA", "tax_deferred", "traditional_ira"),
        ),
    )
    _add_disposed_lot(db, brokerage, "TSLA", realized_gain=-4000, days_ago=5)

    assessment = _assessment(_run(client.id, [{"ticker": "TSLA", "account_id": ira.id}]))

    assert assessment["wash_sale_flags"] == ["TSLA"]
    detail = assessment["wash_sale_detail"][0]
    assert detail["permanent"] is True
    assert "permanently" in detail["reason"]


def test_the_same_symbol_proposed_twice_is_flagged_once(db, prices):
    client, (brokerage,) = _make_client(db)
    _add_disposed_lot(db, brokerage, "TSLA", realized_gain=-4000, days_ago=5)

    assessment = _assessment(_run(client.id, [{"ticker": "TSLA"}, {"ticker": "tsla"}]))

    assert assessment["wash_sale_flags"] == ["TSLA"]


# --- loss harvesting ----------------------------------------------------------

def test_a_position_well_below_its_basis_is_reported_as_a_harvest_candidate(db, prices):
    client, (brokerage,) = _make_client(db)
    tax_lots.seed_lots_from_positions(db, brokerage, [("AAPL", 100.0, 200.0, None)])
    db.commit()
    prices["AAPL"] = 100.0  # a $10,000 unrealized loss

    assessment = _assessment(_run(client.id))

    candidates = assessment["harvest_candidates"]
    assert [c["symbol"] for c in candidates] == ["AAPL"]
    assert candidates[0]["unrealized_loss"] == pytest.approx(-10000.0)
    assert candidates[0]["account_name"] == "Brokerage"
    assert any("below its cost basis" in note for note in assessment["tax_efficiency_notes"])


def test_a_loss_smaller_than_the_harvest_minimum_is_left_alone(db, prices):
    """Below this the spread and the paperwork cost more than the deduction
    is worth."""
    client, (brokerage,) = _make_client(db)
    tax_lots.seed_lots_from_positions(db, brokerage, [("AAPL", 100.0, 200.0, None)])
    db.commit()
    prices["AAPL"] = 200.0 - (MIN_HARVEST_LOSS / 100.0) / 2  # half the minimum loss

    assert _assessment(_run(client.id))["harvest_candidates"] == []


def test_losses_inside_a_retirement_account_are_not_harvest_candidates(db, prices):
    """Selling at a loss in an IRA produces no deduction, so it is not an
    opportunity however large the loss."""
    client, (ira,) = _make_client(db, accounts=(("IRA", "tax_deferred", "traditional_ira"),))
    tax_lots.seed_lots_from_positions(db, ira, [("AAPL", 100.0, 200.0, None)])
    db.commit()
    prices["AAPL"] = 100.0

    assert _assessment(_run(client.id))["harvest_candidates"] == []


# --- realized gains -----------------------------------------------------------

def test_realized_gains_year_to_date_are_reported_and_split_by_term(db, prices):
    client, (brokerage,) = _make_client(db)
    _add_disposed_lot(db, brokerage, "MSFT", realized_gain=3000, days_ago=0, term="long")
    _add_disposed_lot(db, brokerage, "NVDA", realized_gain=1000, days_ago=0, term="short")

    assessment = _assessment(_run(client.id))

    realized = assessment["realized_ytd"]
    assert realized["long_term"] == pytest.approx(3000.0)
    assert realized["short_term"] == pytest.approx(1000.0)
    assert realized["total"] == pytest.approx(4000.0)
    assert any("Realized year to date" in note for note in assessment["tax_efficiency_notes"])


def test_a_clean_tax_position_says_so_rather_than_returning_nothing(db, prices):
    client, _ = _make_client(db)

    assessment = _assessment(_run(client.id, [{"ticker": "AAPL"}]))

    assert assessment["wash_sale_flags"] == []
    assert assessment["harvest_candidates"] == []
    assert assessment["tax_efficiency_notes"] == [
        "No wash-sale conflicts, harvesting opportunities or realized gains to "
        "report for this client."
    ]


# --- failing closed -----------------------------------------------------------

def test_an_assessment_that_cannot_run_blocks_rather_than_waving_through(db, prices):
    """Regression guard for the most dangerous shape of the old node.

    It caught every exception and returned empty wash-sale flags, so a
    database error silently disabled the only hard tax guardrail while the run
    reported success. An unverifiable tax position must be explicit.
    """
    result = _run(999_999, [{"ticker": "TSLA"}])
    assessment = _assessment(result)

    assert assessment["verification_failed"] is True
    assert assessment["wash_sale_flags"] == []
    assert any("unverified" in note for note in assessment["tax_efficiency_notes"])
    # And the failure is on the record rather than swallowed. `node_run`
    # records the exception and then degrades, so the final status is
    # "degraded"; `error_detail` is what names the cause.
    record = result["audit_trail"][0]
    assert record["degraded"] is True
    assert "No client profile with id 999999" in record["error_detail"]
    assert result["degradations"]


def test_a_successful_assessment_is_not_marked_unverified(db, prices):
    client, _ = _make_client(db)

    result = _run(client.id, [{"ticker": "AAPL"}])

    assert _assessment(result)["verification_failed"] is False
    assert result["audit_trail"][0]["status"] == "success"
