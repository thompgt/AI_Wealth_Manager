"""Tests for the deterministic screens and sizing in agents/suitability.py.

Every limit now resolves from the client's ResolvedPolicy rather than a module
constant, so the tests build a policy object directly. Reference data is read
through `get_security_info`, which is patched on the suitability module -- the
name it bound at import time -- so nothing here touches a provider or a DB.
"""

import pytest

import agents.suitability as suitability
from agents.suitability import (
    _check_policy_exclusions,
    _check_risk_fit,
    _check_security_quality,
    _compute_allocations,
)
from services.policy import ResolvedPolicy
from services.portfolio import HoldingView, PortfolioView
from tests.conftest import make_security_info


# --- fixtures-by-hand -------------------------------------------------------


def _policy(**overrides):
    """A ResolvedPolicy built directly, so no DB or IPS row is needed."""
    fields = dict(
        client_id=1,
        version=3,
        source="policy",
        risk_tier="Moderate",
        max_position_pct=0.25,
        max_sector_pct=0.35,
        max_asset_class_pct=0.75,
        min_cash_pct=0.03,
        max_cash_pct=0.20,
        max_position_beta=1.40,
        max_portfolio_beta=1.05,
        max_portfolio_volatility=0.18,
        min_market_cap=2_000_000_000.0,
        min_avg_dollar_volume=10_000_000.0,
        min_position_notional=1_000.0,
        target_allocation={"us_equity": 0.60, "fixed_income": 0.37, "cash": 0.03},
        drift_bands={},
        allowed_asset_classes=[],
        excluded_tickers=[],
        excluded_sectors=[],
        lot_selection_method="HIFO",
        harvest_losses=True,
        max_short_term_gain_budget=None,
        benchmark_ticker="SPY",
        rebalance_frequency_days=90,
    )
    fields.update(overrides)
    return ResolvedPolicy(**fields)


def _candidate(ticker, **overrides):
    candidate = {
        "ticker": ticker,
        "valuation_metrics": {},
        "addresses_flaw": "test",
        "regime_fit_rationale": "test",
        "confidence": 0.5,
    }
    candidate.update(overrides)
    return candidate


def _holding(symbol, market_value, *, sector="Technology", security_type="equity"):
    return HoldingView(
        symbol=symbol,
        account_id=1,
        account_name="Taxable",
        tax_treatment="taxable",
        quantity=1.0,
        price=market_value,
        market_value=market_value,
        cost_basis=market_value,
        asset_class="us_equity",
        sector=sector,
        security_type=security_type,
        beta=1.0,
    )


def _view(holdings=(), cash=0.0, held=0.0):
    return PortfolioView(
        client_id=1,
        holdings=list(holdings),
        cash_by_account={1: cash},
        held_by_account={1: held},
    )


def _serve(monkeypatch, info):
    """Make the provider chain return `info` for every lookup."""
    monkeypatch.setattr(suitability, "get_security_info", lambda ticker: info)


# --- _check_security_quality ------------------------------------------------


def test_security_quality_passes_a_large_cap_on_a_major_exchange(monkeypatch):
    _serve(monkeypatch, make_security_info("AAPL"))
    assert _check_security_quality(_candidate("AAPL"), _policy()) is None


def test_security_quality_fails_closed_when_no_provider_knows_the_security(monkeypatch):
    """An unverifiable security is declined, not waved through hopefully."""
    _serve(monkeypatch, None)
    reason = _check_security_quality(_candidate("XYZ"), _policy())
    assert reason is not None
    assert reason.startswith("XYZ: rejected -- ")
    assert "verifiable" in reason


def test_security_quality_fails_closed_on_a_missing_market_cap(monkeypatch):
    _serve(monkeypatch, make_security_info("SMALL", market_cap=None))
    reason = _check_security_quality(_candidate("SMALL"), _policy())
    assert reason is not None
    assert "market capitalisation is unavailable" in reason


def test_security_quality_rejects_a_market_cap_below_the_clients_minimum(monkeypatch):
    _serve(monkeypatch, make_security_info("SMALL", market_cap=500_000_000.0))
    reason = _check_security_quality(_candidate("SMALL"), _policy(min_market_cap=2_000_000_000.0))
    assert reason is not None
    assert "$500,000,000" in reason and "$2,000,000,000" in reason


def test_security_quality_reads_the_minimum_from_the_policy_not_a_constant(monkeypatch):
    """The same security is eligible or not depending on whose policy applies."""
    _serve(monkeypatch, make_security_info("MIDCAP", market_cap=3_000_000_000.0))
    assert _check_security_quality(_candidate("MIDCAP"), _policy(min_market_cap=2e9)) is None
    assert _check_security_quality(_candidate("MIDCAP"), _policy(min_market_cap=5e9)) is not None


def test_security_quality_exempts_funds_from_the_market_cap_screen(monkeypatch):
    # An ETF has no market capitalisation to report, and requiring one would
    # reject every fund a diversified mandate needs.
    _serve(monkeypatch, make_security_info("VTI", quote_type="ETF", market_cap=None))
    assert _check_security_quality(_candidate("VTI"), _policy()) is None


def test_security_quality_fails_closed_on_a_missing_listing_venue(monkeypatch):
    _serve(monkeypatch, make_security_info("GHOST", exchange=None))
    reason = _check_security_quality(_candidate("GHOST"), _policy())
    assert reason is not None
    assert "listing venue is unavailable" in reason


def test_security_quality_rejects_an_over_the_counter_listing(monkeypatch):
    _serve(monkeypatch, make_security_info("PENNY", exchange="PNK"))
    reason = _check_security_quality(_candidate("PENNY"), _policy())
    assert reason is not None
    assert "'PNK'" in reason and "major US exchange" in reason


def test_security_quality_rejects_a_position_that_could_not_be_exited(monkeypatch):
    _serve(monkeypatch, make_security_info("THIN", avg_dollar_volume=100_000.0))
    reason = _check_security_quality(
        _candidate("THIN"), _policy(min_avg_dollar_volume=10_000_000.0)
    )
    assert reason is not None
    assert "turnover" in reason


def test_security_quality_skips_the_liquidity_screen_when_turnover_is_unknown(monkeypatch):
    # Unlike market cap, an absent turnover figure does not reject the name.
    _serve(monkeypatch, make_security_info("QUIET", avg_dollar_volume=None))
    assert _check_security_quality(_candidate("QUIET"), _policy()) is None


# --- _check_policy_exclusions -----------------------------------------------


def test_policy_exclusions_pass_a_candidate_nothing_prohibits():
    candidate = _candidate("AAPL", sector="Technology", asset_class="us_equity")
    assert _check_policy_exclusions(candidate, _policy()) is None


def test_excluded_ticker_is_rejected_regardless_of_case():
    reason = _check_policy_exclusions(
        _candidate("xom"), _policy(excluded_tickers=["XOM"])
    )
    assert reason is not None
    assert "exclusion list" in reason


def test_excluded_sector_is_rejected():
    candidate = _candidate("XOM", sector="Energy")
    reason = _check_policy_exclusions(candidate, _policy(excluded_sectors=["Energy"]))
    assert reason is not None
    assert "Energy sector is excluded" in reason


def test_asset_class_outside_the_permitted_list_is_rejected():
    candidate = _candidate("GLD", asset_class="commodity")
    reason = _check_policy_exclusions(
        candidate, _policy(allowed_asset_classes=["us_equity", "fixed_income"])
    )
    assert reason is not None
    assert "commodity asset class is not permitted" in reason


def test_an_empty_allowed_asset_class_list_permits_everything():
    # No list on file means "unrestricted", not "nothing is allowed".
    candidate = _candidate("GLD", asset_class="commodity")
    assert _check_policy_exclusions(candidate, _policy(allowed_asset_classes=[])) is None


# --- _check_risk_fit --------------------------------------------------------


def test_risk_fit_fails_open_when_beta_is_unavailable(monkeypatch):
    """Failing closed here would exclude most bond funds from every
    conservative portfolio -- the opposite of the intent."""
    _serve(monkeypatch, make_security_info("BND", beta=None))
    profile = {"age": 70, "time_horizon_years": 2}
    assert _check_risk_fit(_candidate("BND"), _policy(), profile) is None


def test_risk_fit_passes_a_beta_under_the_policy_ceiling(monkeypatch):
    _serve(monkeypatch, make_security_info("AAPL", beta=1.20))
    assert _check_risk_fit(_candidate("AAPL"), _policy(max_position_beta=1.40), {}) is None


def test_beta_above_the_policy_ceiling_is_rejected(monkeypatch):
    _serve(monkeypatch, make_security_info("TSLA", beta=2.10))
    reason = _check_risk_fit(_candidate("TSLA"), _policy(max_position_beta=1.40), {})
    assert reason is not None
    assert "2.10" in reason and "1.40" in reason and "Moderate" in reason


def test_high_beta_is_rejected_for_a_near_retirement_client(monkeypatch):
    # The policy ceiling here is permissive (2.5); it is the client's age that
    # makes a beta of 1.8 unsuitable.
    _serve(monkeypatch, make_security_info("TSLA", beta=1.80))
    profile = {"age": 65, "time_horizon_years": 20}
    reason = _check_risk_fit(_candidate("TSLA"), _policy(max_position_beta=2.5), profile)
    assert reason is not None
    assert "aged 65" in reason


def test_high_beta_is_rejected_for_a_short_horizon_client(monkeypatch):
    _serve(monkeypatch, make_security_info("TSLA", beta=1.80))
    profile = {"age": 35, "time_horizon_years": 3}
    reason = _check_risk_fit(_candidate("TSLA"), _policy(max_position_beta=2.5), profile)
    assert reason is not None
    assert "3-year horizon" in reason


def test_the_horizon_rule_does_not_apply_to_a_young_long_horizon_client(monkeypatch):
    _serve(monkeypatch, make_security_info("TSLA", beta=1.80))
    profile = {"age": 30, "time_horizon_years": 30}
    assert _check_risk_fit(_candidate("TSLA"), _policy(max_position_beta=2.5), profile) is None


# --- _compute_allocations ---------------------------------------------------


def test_no_candidates_allocates_nothing_and_says_nothing():
    allocations, notes = _compute_allocations([], _view(cash=100_000.0), _policy())
    assert allocations == {}
    assert notes == []


def test_investable_cash_is_split_by_confidence():
    # $100k portfolio with $10k cash and no floor; the 25% position cap
    # ($25k) binds on neither name, so the split is purely by confidence.
    view = _view([_holding("VTI", 90_000.0)], cash=10_000.0)
    allocations, notes = _compute_allocations(
        [_candidate("A", confidence=0.75), _candidate("B", confidence=0.25)],
        view,
        _policy(min_cash_pct=0.0),
    )
    assert allocations["A"] == pytest.approx(7_500.0)
    assert allocations["B"] == pytest.approx(2_500.0)
    assert notes == []


def test_candidates_are_equally_weighted_when_nothing_carries_a_confidence():
    # The deterministic research path emits no confidences; equal weight is
    # the honest response to having no ranking signal.
    view = _view([_holding("VTI", 90_000.0)], cash=10_000.0)
    allocations, _ = _compute_allocations(
        [_candidate("A", confidence=0.0), _candidate("B", confidence=0.0)],
        view,
        _policy(min_cash_pct=0.0),
    )
    assert allocations["A"] == pytest.approx(5_000.0)
    assert allocations["B"] == pytest.approx(5_000.0)


def test_cash_that_overflows_a_capped_name_is_redistributed_not_abandoned():
    # $45k portfolio, 15% cap = $6,750. AAPL already holds $40k so it has no
    # headroom; all $5k of cash must end up in JNJ rather than going unspent.
    view = _view([_holding("AAPL", 40_000.0)], cash=5_000.0)
    allocations, notes = _compute_allocations(
        [_candidate("AAPL"), _candidate("JNJ")],
        view,
        _policy(max_position_pct=0.15, min_cash_pct=0.0),
    )
    assert allocations["AAPL"] == 0.0
    assert allocations["JNJ"] == pytest.approx(5_000.0)
    assert notes == []


def test_the_cash_floor_is_reserved_before_anything_is_deployed():
    # Deploying the whole balance would create the liquidity flaw diagnostics
    # warns about on the very next run.
    view = _view([_holding("VTI", 90_000.0)], cash=10_000.0)
    allocations, _ = _compute_allocations(
        [_candidate("A", confidence=1.0)], view, _policy(min_cash_pct=0.05)
    )
    assert allocations["A"] == pytest.approx(5_000.0)


def test_unsettled_cash_cannot_fund_a_purchase():
    # $10k of cash but $8k is committed to unsettled trades: only $2k is
    # spendable today, even though all $10k is still owned.
    view = _view([_holding("VTI", 90_000.0)], cash=10_000.0, held=8_000.0)
    allocations, _ = _compute_allocations(
        [_candidate("A", confidence=1.0)], view, _policy(min_cash_pct=0.0)
    )
    assert allocations["A"] == pytest.approx(2_000.0)


def test_nothing_is_allocated_when_investable_cash_is_below_the_minimum_size():
    view = _view([_holding("VTI", 90_000.0)], cash=1_000.0)
    allocations, notes = _compute_allocations(
        [_candidate("A")], view, _policy(min_cash_pct=0.0, min_position_notional=5_000.0)
    )
    assert allocations == {"A": 0.0}
    assert len(notes) == 1
    assert "minimum position size" in notes[0]


def test_the_sector_cap_binds_as_the_basket_is_assembled():
    # $90k invested, 44% of it already Technology, against a 48% sector cap.
    # Only the amount that keeps the sector at the cap may be deployed.
    view = _view(
        [_holding("TECH", 40_000.0, sector="Technology"),
         _holding("HLTH", 50_000.0, sector="Healthcare")],
        cash=10_000.0,
    )
    candidate = _candidate("TCAND", confidence=1.0, sector="Technology", security_type="equity")
    allocations, notes = _compute_allocations(
        [candidate], view, _policy(min_cash_pct=0.0, max_sector_pct=0.48, max_position_pct=0.35)
    )
    bought = allocations["TCAND"]
    assert 0 < bought < 10_000.0
    # The technology sleeve lands exactly on its limit, not over it.
    assert (40_000.0 + bought) / (90_000.0 + bought) == pytest.approx(0.48, abs=1e-6)
    assert len(notes) == 1
    assert "undeployed" in notes[0]


def test_a_broad_fund_is_not_charged_against_a_sector_budget():
    # The same portfolio and the same sector cap, but an ETF carries no
    # single-sector exposure, so the whole $10k can be deployed.
    view = _view(
        [_holding("TECH", 40_000.0, sector="Technology"),
         _holding("HLTH", 50_000.0, sector="Healthcare")],
        cash=10_000.0,
    )
    candidate = _candidate("VTI", confidence=1.0, sector="Technology", security_type="etf")
    allocations, notes = _compute_allocations(
        [candidate], view, _policy(min_cash_pct=0.0, max_sector_pct=0.48, max_position_pct=0.35)
    )
    assert allocations["VTI"] == pytest.approx(10_000.0)
    assert notes == []


def test_cash_left_over_after_every_name_caps_out_is_reported_not_forced_in():
    # A 2% position cap on a $100k portfolio leaves $2k of room per name, so
    # two candidates absorb $4k of the $100k and the rest must stay in cash.
    view = _view(cash=100_000.0)
    allocations, notes = _compute_allocations(
        [_candidate("A"), _candidate("B")],
        view,
        _policy(min_cash_pct=0.0, max_position_pct=0.02),
    )
    assert allocations["A"] == pytest.approx(2_000.0)
    assert allocations["B"] == pytest.approx(2_000.0)
    assert len(notes) == 1
    assert "$96,000" in notes[0] and "policy limit" in notes[0]
