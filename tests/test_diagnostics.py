from agents.diagnostics import (
    CASH_DRAG_THRESHOLD,
    CONCENTRATION_LIMITS,
    _cash_flaws,
    _compute_concentration,
    _concentration_flaws,
    _diversification_score,
    _sector_flaws,
    _volatility_flaws,
)


def test_compute_concentration_splits_by_total_value():
    holdings = {"CASH": 250.0, "AAPL": 750.0}
    concentration = _compute_concentration(holdings, total_value=1000.0)
    assert concentration == {"CASH": 0.25, "AAPL": 0.75}


def test_compute_concentration_handles_zero_total():
    holdings = {"CASH": 0.0}
    assert _compute_concentration(holdings, total_value=0.0) == {"CASH": 0.0}


def test_concentration_flaws_flags_over_limit_non_cash_position():
    limit = CONCENTRATION_LIMITS["Moderate"]
    concentration = {"CASH": 0.5, "TSLA": limit + 0.05}
    flaws = _concentration_flaws(concentration, "Moderate")
    assert len(flaws) == 1
    assert "TSLA" in flaws[0]
    assert "CASH" not in " ".join(flaws)


def test_concentration_flaws_ignores_position_under_limit():
    concentration = {"AAPL": 0.10}
    assert _concentration_flaws(concentration, "Aggressive") == []


def test_cash_flaws_flags_drag_for_growth_oriented_client():
    concentration = {"CASH": CASH_DRAG_THRESHOLD + 0.1}
    profile = {"risk_tolerance": "Moderate", "goals": [], "time_horizon_years": 20}
    flaws = _cash_flaws(concentration, profile)
    assert any("cash drag" in f for f in flaws)


def test_cash_flaws_does_not_flag_drag_for_conservative_client():
    concentration = {"CASH": CASH_DRAG_THRESHOLD + 0.1}
    profile = {"risk_tolerance": "Conservative", "goals": [], "time_horizon_years": 20}
    assert _cash_flaws(concentration, profile) == []


def test_cash_flaws_flags_liquidity_risk_for_near_term_goal():
    concentration = {"CASH": 0.01}
    profile = {"risk_tolerance": "Moderate", "goals": ["home purchase"], "time_horizon_years": 2}
    flaws = _cash_flaws(concentration, profile)
    assert any("liquidity" in f for f in flaws)


def test_sector_flaws_flags_over_concentrated_sector():
    flaws = _sector_flaws({"Technology": 0.9, "Healthcare": 0.1})
    assert len(flaws) == 1
    assert "Technology" in flaws[0]


def test_sector_flaws_empty_when_balanced():
    assert _sector_flaws({"Technology": 0.5, "Healthcare": 0.5}) == []


def test_volatility_flaws_flags_high_volatility_for_conservative_client():
    flaws = _volatility_flaws(0.40, market_data_tickers=["AAPL"], risk_tolerance="Conservative")
    assert len(flaws) == 1


def test_volatility_flaws_skipped_without_market_data():
    # A default 0.0 volatility from missing price data must not be
    # misread as "suspiciously low" for an Aggressive client.
    assert _volatility_flaws(0.0, market_data_tickers=[], risk_tolerance="Aggressive") == []


def test_diversification_score_perfect_for_equal_weights():
    # 4 equal positions -> HHI = 0.25 -> score = 100 * (1 - 0.25) = 75
    concentration = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    assert _diversification_score(concentration) == 75.0


def test_diversification_score_zero_for_single_position():
    assert _diversification_score({"AAPL": 1.0}) == 0.0


def test_diversification_score_handles_empty_portfolio():
    assert _diversification_score({}) == 0.0
