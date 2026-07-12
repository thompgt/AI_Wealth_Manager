from agents.stock_research import (
    _filter_by_cheapness,
    _flaws_mention_cash_heavy,
    _flaws_mention_high_volatility,
    _flaws_mention_sector,
    _pick_driving_flaw,
    _valuation_score,
)


def test_flaws_mention_sector_detects_known_keyword():
    flaws = ["Technology sector is 90.0% of non-cash holdings -- exceeds the 50.0% threshold"]
    assert _flaws_mention_sector(flaws) == ["Technology"]


def test_flaws_mention_sector_empty_when_no_keyword_present():
    assert _flaws_mention_sector(["AAPL is 40% of the portfolio"]) == []


def test_flaws_mention_cash_heavy_detects_drag_language():
    flaws = ["CASH is 71.0% of the portfolio -- too much cash drag for a Moderate risk tolerance"]
    assert _flaws_mention_cash_heavy(flaws) is True


def test_flaws_mention_cash_heavy_false_for_low_cash_liquidity_flaw():
    flaws = ["CASH is only 2.0% of the portfolio -- insufficient liquidity buffer"]
    assert _flaws_mention_cash_heavy(flaws) is False


def test_flaws_mention_high_volatility_detects_keyword():
    assert _flaws_mention_high_volatility(["Annualized volatility of 44.0% is too high"]) is True
    assert _flaws_mention_high_volatility(["AAPL is 40% of the portfolio"]) is False


def test_filter_by_cheapness_prefers_below_median_valuation():
    fundamentals = {
        "CHEAP": {"sector": "Technology", "metrics": {"pe_ratio": 5.0, "price_to_book": 1.0}},
        "MID": {"sector": "Technology", "metrics": {"pe_ratio": 15.0, "price_to_book": 3.0}},
        "EXPENSIVE": {"sector": "Technology", "metrics": {"pe_ratio": 50.0, "price_to_book": 10.0}},
    }
    cheap = _filter_by_cheapness(fundamentals)
    assert "CHEAP" in cheap
    assert "EXPENSIVE" not in cheap


def test_filter_by_cheapness_falls_back_when_too_few_have_valuation_data():
    fundamentals = {"A": {"sector": "Technology", "metrics": {}}}
    assert _filter_by_cheapness(fundamentals) == ["A"]


def test_valuation_score_lower_for_cheaper_stock():
    cheap_score = _valuation_score({"pe_ratio": 10.0, "price_to_book": 1.0})
    expensive_score = _valuation_score({"pe_ratio": 50.0, "price_to_book": 8.0})
    assert cheap_score < expensive_score


def test_valuation_score_infinite_when_no_metrics():
    assert _valuation_score({}) == float("inf")


def test_pick_driving_flaw_prefers_concentration_language():
    flaws = ["Some unrelated note", "AAPL is 40% of the portfolio -- exceeds concentration limit"]
    assert "concentrat" in _pick_driving_flaw(flaws).lower()


def test_pick_driving_flaw_default_when_no_flaws():
    assert "No specific flaw" in _pick_driving_flaw([])
