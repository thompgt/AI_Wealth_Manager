"""Tests for the deterministic measurement helpers in agents/diagnostics.py.

Everything here is offline: the statistics helpers take a DataFrame, and
`_build_flaws` takes an already-loaded PortfolioView plus a ResolvedPolicy, so
neither a database nor a market-data provider is involved.
"""

import pandas as pd
import pytest

from agents.diagnostics import (
    HIGH_CORRELATION,
    _annualized_stats,
    _build_flaws,
    _correlation_analysis,
    _diversification_score,
)
from services.policy import ResolvedPolicy
from services.portfolio import DriftEntry, HoldingView, PortfolioView
from tests.conftest import make_price_frame


# --- fixtures-by-hand -------------------------------------------------------


def _policy(**overrides):
    """A ResolvedPolicy built directly, so no DB or IPS row is needed.

    Limits are the built-in Moderate defaults unless a test overrides the one
    it is exercising.
    """
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


def _holding(symbol, market_value, *, sector="Technology", security_type="equity",
             beta=None, asset_class="us_equity"):
    return HoldingView(
        symbol=symbol,
        account_id=1,
        account_name="Taxable",
        tax_treatment="taxable",
        quantity=1.0,
        price=market_value,
        market_value=market_value,
        cost_basis=market_value,
        asset_class=asset_class,
        sector=sector,
        security_type=security_type,
        beta=beta,
    )


def _view(holdings=(), cash=0.0):
    return PortfolioView(client_id=1, holdings=list(holdings), cash_by_account={1: cash})


def _frame(columns, days=140):
    """A price frame from explicit per-column price lists."""
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(columns, index=index)


def _wiggle(days=140, seed=0, start=100.0):
    """A deterministic price path with real day-to-day variance.

    make_price_frame's constant-drift series have zero *return* variance, so
    their volatility is 0 and their correlation is undefined. The correlation
    and Sharpe assertions need a path that actually moves.
    """
    prices = [start]
    value = float(seed * 7 + 1)
    for _ in range(days - 1):
        # A simple deterministic recurrence: reproducible without numpy's RNG
        # and different enough per seed to be effectively uncorrelated.
        value = (value * 1103515245 + 12345) % 2147483648
        shock = (value / 2147483648.0) - 0.5
        prices.append(prices[-1] * (1.0 + 0.02 * shock))
    return prices


# --- _annualized_stats ------------------------------------------------------


def test_annualized_stats_returns_nothing_for_an_empty_price_frame():
    stats = _annualized_stats(pd.DataFrame(), {"AAPL": 1.0}, 0.02)
    assert stats == {
        "annual_return": None,
        "annual_volatility": None,
        "sharpe_ratio": None,
        "max_drawdown": None,
    }


def test_annualized_stats_returns_nothing_without_weights():
    prices = make_price_frame(["AAPL"])
    assert _annualized_stats(prices, {}, 0.02)["annual_volatility"] is None


def test_annualized_stats_returns_nothing_when_no_weighted_column_is_priced():
    # The client holds MSFT, but the frame only has AAPL: there is nothing to
    # compute the portfolio's statistics from.
    prices = make_price_frame(["AAPL"])
    assert _annualized_stats(prices, {"MSFT": 1.0}, 0.02)["annual_return"] is None


def test_annualized_stats_refuses_a_window_too_short_to_be_meaningful():
    # 20 business days is 19 returns, below MIN_RETURN_OBSERVATIONS.
    prices = make_price_frame(["AAPL"], days=20)
    assert _annualized_stats(prices, {"AAPL": 1.0}, 0.02)["annual_volatility"] is None


def test_annualized_stats_annualizes_a_constant_daily_drift():
    # make_price_frame compounds a fixed 0.1%/day, so the annualized return is
    # 0.001 * 252 and there is no variance and therefore no drawdown.
    prices = make_price_frame(["AAPL"], days=140, daily_drift=0.001)
    stats = _annualized_stats(prices, {"AAPL": 1.0}, 0.02)
    assert stats["annual_return"] == pytest.approx(0.252, abs=1e-3)
    assert stats["annual_volatility"] == pytest.approx(0.0, abs=1e-3)
    assert stats["max_drawdown"] == pytest.approx(0.0, abs=1e-6)


def test_annualized_stats_sharpe_is_excess_return_over_volatility():
    prices = _frame({"AAPL": _wiggle(seed=1)})
    stats = _annualized_stats(prices, {"AAPL": 1.0}, 0.02)
    expected = (stats["annual_return"] - 0.02) / stats["annual_volatility"]
    assert stats["sharpe_ratio"] == pytest.approx(expected, abs=1e-3)


def test_annualized_stats_uses_actual_weights_not_an_equal_weighted_basket():
    """A portfolio mostly in the calm name must not be described by the wild one."""
    calm = [100.0 * (1.001 ** day) for day in range(140)]
    wild = _wiggle(seed=3)
    prices = _frame({"CALM": calm, "WILD": wild})

    mostly_calm = _annualized_stats(prices, {"CALM": 0.99, "WILD": 0.01}, 0.02)
    mostly_wild = _annualized_stats(prices, {"CALM": 0.01, "WILD": 0.99}, 0.02)
    assert mostly_calm["annual_volatility"] < mostly_wild["annual_volatility"]


def test_annualized_stats_ignores_zero_weight_positions():
    prices = _frame({"AAPL": _wiggle(seed=2), "MSFT": _wiggle(seed=5)})
    held_only = _annualized_stats(prices, {"AAPL": 1.0}, 0.02)
    with_zero = _annualized_stats(prices, {"AAPL": 1.0, "MSFT": 0.0}, 0.02)
    assert held_only == with_zero


def test_annualized_stats_measures_peak_to_trough_decline():
    # Straight up 100 -> 200, then straight down to 140: a 30% drawdown.
    up = [100.0 + day * (100.0 / 69) for day in range(70)]
    down = [200.0 - day * (60.0 / 69) for day in range(70)]
    prices = _frame({"AAPL": up + down})
    stats = _annualized_stats(prices, {"AAPL": 1.0}, 0.02)
    assert stats["max_drawdown"] == pytest.approx(-0.30, abs=0.01)


# --- _correlation_analysis --------------------------------------------------


def test_correlation_analysis_needs_at_least_two_positions():
    prices = make_price_frame(["AAPL"])
    assert _correlation_analysis(prices, {"AAPL": 1.0}) == (None, [])


def test_correlation_analysis_refuses_a_window_shorter_than_six_months():
    # 100 business days is below MIN_CORRELATION_OBSERVATIONS (120), where the
    # estimate's confidence interval is too wide to act on.
    prices = _frame({"A": _wiggle(days=100, seed=1), "B": _wiggle(days=100, seed=2)}, days=100)
    assert _correlation_analysis(prices, {"A": 0.5, "B": 0.5}) == (None, [])


def test_correlation_analysis_returns_nothing_when_prices_never_move():
    # Flat prices give zero-variance returns, so every correlation is NaN --
    # an undefined estimate must not be reported as a number.
    prices = _frame({"A": [100.0] * 140, "B": [50.0] * 140})
    assert _correlation_analysis(prices, {"A": 0.5, "B": 0.5}) == (None, [])


def test_correlation_analysis_clusters_holdings_that_move_identically():
    # B is A scaled, so their daily returns are identical: correlation 1.0.
    base = _wiggle(seed=4)
    prices = _frame({"A": base, "B": [p * 2 for p in base]})
    average, clusters = _correlation_analysis(prices, {"A": 0.5, "B": 0.5})
    assert average == pytest.approx(1.0, abs=1e-6)
    assert clusters == [["A", "B"]]


def test_correlation_analysis_leaves_unrelated_holdings_unclustered():
    prices = _frame({"A": _wiggle(seed=11), "B": _wiggle(seed=29)})
    average, clusters = _correlation_analysis(prices, {"A": 0.5, "B": 0.5})
    assert clusters == []
    assert average < HIGH_CORRELATION


def test_correlation_analysis_weights_pairs_by_position_size():
    """A long tail of tiny holdings must not disguise a correlated core.

    A and B are identical (correlation 1.0) and C is unrelated. Growing C from
    a rounding-error position to half the portfolio must pull the reported
    average down, because pairs are weighted by the dollars involved.
    """
    base = _wiggle(seed=6)
    prices = _frame({"A": base, "B": [p * 3 for p in base], "C": _wiggle(seed=41)})

    tiny_c, _ = _correlation_analysis(prices, {"A": 0.495, "B": 0.495, "C": 0.01})
    large_c, _ = _correlation_analysis(prices, {"A": 0.25, "B": 0.25, "C": 0.50})
    assert tiny_c > large_c


# --- _diversification_score -------------------------------------------------


def test_diversification_score_and_effective_count_for_equal_weights():
    # 4 equal positions -> HHI = 0.25 -> score 75, effective count 4.
    assert _diversification_score({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}) == (75.0, 4.0)


def test_diversification_score_is_zero_for_a_single_position():
    assert _diversification_score({"AAPL": 1.0}) == (0.0, 1.0)


def test_diversification_score_handles_an_empty_portfolio():
    assert _diversification_score({}) == (0.0, None)


def test_diversification_score_ignores_zero_weight_entries():
    assert _diversification_score({"A": 0.5, "B": 0.5, "C": 0.0}) == (50.0, 2.0)


def test_diversification_score_normalizes_raw_dollar_weights():
    # Dollars rather than fractions must give the same answer as the fractions.
    assert _diversification_score({"A": 30_000.0, "B": 30_000.0}) == (50.0, 2.0)


def test_effective_positions_is_far_below_the_holding_count_when_lopsided():
    """The number a client understands: 4 holdings that behave like ~1."""
    _, effective = _diversification_score({"A": 0.91, "B": 0.03, "C": 0.03, "D": 0.03})
    assert effective < 2.0


# --- _build_flaws -----------------------------------------------------------


_NO_STATS = {
    "annual_return": None,
    "annual_volatility": None,
    "sharpe_ratio": None,
    "max_drawdown": None,
}


def _flaws(view=None, policy=None, stats=None, drift=(), breaches=(),
           average_correlation=None, clusters=(), effective=5.0):
    return _build_flaws(
        view if view is not None else _view([_holding("AAPL", 50_000.0, beta=1.0)], cash=5_000.0),
        policy if policy is not None else _policy(),
        stats if stats is not None else dict(_NO_STATS),
        list(drift),
        list(breaches),
        average_correlation,
        list(clusters),
        effective,
    )


def test_build_flaws_is_empty_for_a_portfolio_inside_every_limit():
    assert _flaws() == []


def test_position_breach_names_the_ticker_the_limit_and_the_dollars_over():
    breach = {
        "kind": "position", "key": "TSLA", "weight": 0.40,
        "limit": 0.25, "excess_value": 15_000.0,
    }
    flaws = _flaws(breaches=[breach])
    assert len(flaws) == 1
    assert "TSLA" in flaws[0]
    assert "40%" in flaws[0] and "25%" in flaws[0] and "$15,000" in flaws[0]


def test_sector_breach_is_reported_against_invested_assets():
    breach = {
        "kind": "sector", "key": "Technology", "weight": 0.62,
        "limit": 0.35, "excess_value": 27_000.0,
    }
    flaws = _flaws(breaches=[breach])
    assert len(flaws) == 1
    assert "Technology" in flaws[0] and "invested assets" in flaws[0]


def test_cash_above_the_ceiling_is_described_as_uninvested_money():
    breach = {
        "kind": "cash_high", "key": "CASH", "weight": 0.55,
        "limit": 0.20, "excess_value": 35_000.0,
    }
    flaws = _flaws(breaches=[breach])
    assert len(flaws) == 1
    assert "uninvested" in flaws[0] and "$35,000" in flaws[0]


def test_cash_below_the_floor_is_described_as_a_liquidity_risk():
    breach = {
        "kind": "cash_low", "key": "CASH", "weight": 0.01,
        "limit": 0.05, "excess_value": 4_000.0,
    }
    flaws = _flaws(breaches=[breach])
    assert len(flaws) == 1
    assert "liquidity floor" in flaws[0]


def test_breached_drift_uses_the_client_facing_asset_class_label():
    entry = DriftEntry(
        asset_class="fixed_income", current_weight=0.20, target_weight=0.48,
        drift=-0.28, band=0.05, breached=True, dollar_gap=280_000.0,
    )
    flaws = _flaws(drift=[entry])
    assert len(flaws) == 1
    # "em equity" is not a phrase to put in front of a client.
    assert flaws[0].startswith("Fixed income")
    assert "below" in flaws[0] and "$280,000" in flaws[0]


def test_drift_inside_its_tolerance_band_is_not_a_flaw():
    entry = DriftEntry(
        asset_class="us_equity", current_weight=0.62, target_weight=0.60,
        drift=0.02, band=0.05, breached=False, dollar_gap=-20_000.0,
    )
    assert _flaws(drift=[entry]) == []


def test_volatility_above_the_mandate_ceiling_is_flagged():
    stats = {**_NO_STATS, "annual_volatility": 0.31}
    flaws = _flaws(stats=stats, policy=_policy(max_portfolio_volatility=0.18))
    assert len(flaws) == 1
    assert "31%" in flaws[0] and "18%" in flaws[0] and "Moderate" in flaws[0]


def test_volatility_at_or_below_the_ceiling_is_not_flagged():
    stats = {**_NO_STATS, "annual_volatility": 0.18}
    assert _flaws(stats=stats, policy=_policy(max_portfolio_volatility=0.18)) == []


def test_volatility_is_not_flagged_when_the_policy_sets_no_ceiling():
    stats = {**_NO_STATS, "annual_volatility": 0.90}
    assert _flaws(stats=stats, policy=_policy(max_portfolio_volatility=None)) == []


def test_a_drawdown_deeper_than_twenty_percent_is_flagged():
    flaws = _flaws(stats={**_NO_STATS, "max_drawdown": -0.34})
    assert len(flaws) == 1
    assert "34%" in flaws[0] and "peak-to-trough" in flaws[0]


def test_a_shallow_drawdown_is_not_flagged():
    assert _flaws(stats={**_NO_STATS, "max_drawdown": -0.15}) == []


def test_portfolio_beta_above_the_policy_ceiling_is_flagged():
    view = _view([_holding("TQQQ", 100_000.0, beta=1.80)])
    flaws = _flaws(view=view, policy=_policy(max_portfolio_beta=1.05))
    assert len(flaws) == 1
    assert "1.80" in flaws[0] and "1.05" in flaws[0]


def test_portfolio_beta_is_skipped_when_no_holding_reports_one():
    # Beta is genuinely missing for many instruments; an absent value is not
    # a breach.
    view = _view([_holding("MYSTERY", 100_000.0, beta=None)])
    assert _flaws(view=view, policy=_policy(max_portfolio_beta=0.10)) == []


def test_high_average_correlation_is_flagged_as_hidden_concentration():
    flaws = _flaws(average_correlation=0.88)
    assert len(flaws) == 1
    assert "0.88" in flaws[0] and "less diversified" in flaws[0]


def test_moderate_average_correlation_is_not_flagged():
    assert _flaws(average_correlation=0.60) == []


def test_a_cluster_of_three_near_identical_holdings_is_flagged():
    flaws = _flaws(clusters=[["GOOG", "GOOGL", "META"]])
    assert len(flaws) == 1
    assert "GOOG, GOOGL, META" in flaws[0]


def test_a_pair_is_not_reported_as_a_cluster():
    # Two share classes of one company are legitimately correlated; only a
    # cluster of three or more is a finding.
    assert _flaws(clusters=[["GOOG", "GOOGL"]]) == []


def test_few_effective_positions_are_reported_against_the_holding_count():
    view = _view([
        _holding("AAPL", 91_000.0),
        _holding("MSFT", 3_000.0),
        _holding("KO", 3_000.0),
        _holding("JNJ", 3_000.0),
    ])
    flaws = _flaws(view=view, effective=1.2)
    assert len(flaws) == 1
    assert "4 holdings" in flaws[0] and "1.2" in flaws[0]


def test_effective_positions_are_not_reported_when_they_match_the_count():
    # 3 holdings behaving like 3 positions is not a finding, even though 3 < 4.
    view = _view([
        _holding("AAPL", 10_000.0),
        _holding("MSFT", 10_000.0),
        _holding("KO", 10_000.0),
    ])
    assert _flaws(view=view, effective=3.0) == []


def test_an_entirely_uninvested_portfolio_is_a_flaw_in_itself():
    flaws = _flaws(view=_view(cash=250_000.0), effective=None)
    assert len(flaws) == 1
    assert "$250,000" in flaws[0] and "cash" in flaws[0]
