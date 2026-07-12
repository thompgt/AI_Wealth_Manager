from agents.suitability import (
    MIN_MARKET_CAP,
    _check_age_horizon_volatility,
    _check_position_limit,
    _check_security_quality,
)


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


def test_position_limit_rejects_oversized_position():
    profile = {
        "net_worth": 50000.0,
        "holdings": {"CASH": 5000.0, "AAPL": 6500.0},
        "risk_tolerance": "Conservative",
    }
    # Matches the scenario documented in agents/suitability.py's own __main__
    # smoke test: existing $6,500 AAPL + simulated new position pushes the
    # client over the 15% Conservative cap.
    reason = _check_position_limit("AAPL", profile, num_candidates=2)
    assert reason is not None
    assert "cap" in reason


def test_position_limit_allows_small_new_position():
    profile = {
        "net_worth": 50000.0,
        "holdings": {"CASH": 5000.0},
        "risk_tolerance": "Conservative",
    }
    assert _check_position_limit("JNJ", profile, num_candidates=2) is None


def test_position_limit_skipped_with_no_candidates_or_net_worth():
    assert _check_position_limit("AAPL", {"net_worth": 0.0, "holdings": {}}, num_candidates=0) is None


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
