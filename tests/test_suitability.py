from agents.suitability import (
    MIN_MARKET_CAP,
    _check_age_horizon_volatility,
    _check_security_quality,
    _compute_allocations,
)


def _candidate(ticker, confidence=0.5):
    return {
        "ticker": ticker,
        "valuation_metrics": {},
        "addresses_flaw": "test",
        "regime_fit_rationale": "test",
        "confidence": confidence,
    }


def test_security_quality_rejects_missing_info():
    assert _check_security_quality("XYZ", None) is not None


def test_security_quality_rejects_small_cap():
    info = {"marketCap": MIN_MARKET_CAP - 1, "exchange": "NMS"}
    reason = _check_security_quality("SMALL", info)
    assert reason is not None
    assert "market cap" in reason


def test_security_quality_rejects_otc_exchange():
    info = {"marketCap": MIN_MARKET_CAP * 10, "exchange": "PNK", "fullExchangeName": "OTC Markets Pink Sheets"}
    reason = _check_security_quality("PENNY", info)
    assert reason is not None
    assert "major exchange" in reason


def test_security_quality_passes_large_cap_major_exchange():
    info = {"marketCap": MIN_MARKET_CAP * 10, "exchange": "NMS"}
    assert _check_security_quality("AAPL", info) is None


def test_compute_allocations_caps_a_ticker_already_at_its_limit():
    # Conservative cap = 15% of $50,000 = $7,500; AAPL already holds exactly
    # that, so it should get $0 of the available $5,000 cash.
    profile = {
        "net_worth": 50000.0,
        "holdings": {"CASH": 5000.0, "AAPL": 7500.0},
        "risk_tolerance": "Conservative",
    }
    allocations = _compute_allocations([_candidate("AAPL"), _candidate("JNJ")], profile)
    assert allocations["AAPL"] == 0.0
    assert allocations["JNJ"] == 5000.0


def test_compute_allocations_splits_cash_by_confidence():
    profile = {
        "net_worth": 1000000.0,  # cap is high enough that no ticker gets capped out
        "holdings": {"CASH": 10000.0},
        "risk_tolerance": "Aggressive",
    }
    allocations = _compute_allocations(
        [_candidate("A", confidence=0.75), _candidate("B", confidence=0.25)], profile
    )
    assert abs(allocations["A"] - 7500.0) < 1e-6
    assert abs(allocations["B"] - 2500.0) < 1e-6


def test_compute_allocations_equal_weight_when_all_confidences_zero():
    profile = {
        "net_worth": 1000000.0,
        "holdings": {"CASH": 10000.0},
        "risk_tolerance": "Aggressive",
    }
    allocations = _compute_allocations(
        [_candidate("A", confidence=0.0), _candidate("B", confidence=0.0)], profile
    )
    assert abs(allocations["A"] - 5000.0) < 1e-6
    assert abs(allocations["B"] - 5000.0) < 1e-6


def test_compute_allocations_empty_with_no_cash_or_net_worth():
    profile = {"net_worth": 0.0, "holdings": {}, "risk_tolerance": "Moderate"}
    assert _compute_allocations([_candidate("AAPL")], profile) == {"AAPL": 0.0}


def test_age_horizon_volatility_rule_does_not_apply_to_young_long_horizon_client():
    profile = {"age": 30, "time_horizon_years": 30}
    info = {"beta": 3.0}
    assert _check_age_horizon_volatility("TSLA", info, profile) is None


def test_age_horizon_volatility_rejects_high_beta_for_near_retirement_client():
    profile = {"age": 65, "time_horizon_years": 3}
    info = {"beta": 2.5}
    reason = _check_age_horizon_volatility("TSLA", info, profile)
    assert reason is not None
    assert "beta" in reason


def test_age_horizon_volatility_skips_when_beta_unavailable():
    profile = {"age": 65, "time_horizon_years": 3}
    info = {"beta": None}
    assert _check_age_horizon_volatility("TSLA", info, profile) is None
