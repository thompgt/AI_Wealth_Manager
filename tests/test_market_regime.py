import pandas as pd

from agents.market_regime import _pct_change, _ratio_series, compute_supporting_signals
from tests.conftest import make_price_frame


def test_pct_change_computes_first_to_last():
    series = pd.Series([100.0, 110.0, 121.0])
    assert abs(_pct_change(series) - 21.0) < 1e-6


def test_pct_change_none_with_fewer_than_two_points():
    assert _pct_change(pd.Series([100.0])) is None


def test_ratio_series_computes_ratio_of_numerator_to_denominator():
    prices = make_price_frame(["IWM", "SPY"], days=10)
    ratio = _ratio_series(prices, "IWM", ["SPY"])
    assert ratio is not None
    assert len(ratio) == 10


def test_ratio_series_none_when_column_missing():
    prices = make_price_frame(["SPY"], days=10)
    assert _ratio_series(prices, "IWM", ["SPY"]) is None


def test_compute_supporting_signals_reports_missing_tickers():
    prices = make_price_frame(["SPY"], days=10)
    signals = compute_supporting_signals(prices, requested_tickers=["SPY", "IWM"])
    assert signals["missing_tickers"] == ["IWM"]


def test_compute_supporting_signals_handles_empty_price_frame():
    signals = compute_supporting_signals(pd.DataFrame(), requested_tickers=["SPY"])
    assert "error" in signals


def test_compute_supporting_signals_detects_rising_ratio_trend():
    # IWM is given a higher drift than SPY in make_price_frame, so IWM/SPY
    # should trend upward over the window.
    prices = make_price_frame(["SPY", "IWM"], days=130)
    signals = compute_supporting_signals(prices, requested_tickers=["SPY", "IWM"])
    ratio_info = signals["ratio_trends"]["small_cap_vs_large_cap (IWM/SPY)"]
    assert ratio_info["available"] is True
    assert ratio_info["trend"] == "rising"
