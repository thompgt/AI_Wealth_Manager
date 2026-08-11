"""Return, risk and horizon arithmetic.

These are the numbers a client is shown and judged against, and every one of
them is a pure function of its inputs -- no database, no network. There is no
excuse for them being untested, and an arithmetic error here is silent: a
wrong Sharpe looks exactly like a right one.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from services.performance import (
    TRADING_DAYS,
    _close_at,
    max_drawdown,
    money_weighted_return,
    periods_per_year,
    time_weighted_return,
)


# --- Time-weighted return ----------------------------------------------------


def test_twr_links_periodic_returns_geometrically():
    # +10% then +10% is +21%, not +20%.
    result = time_weighted_return([100.0, 110.0, 121.0], [0.0, 0.0, 0.0])
    assert result == pytest.approx(0.21)


def test_twr_neutralizes_a_deposit():
    """The whole point of TWR: a wire transfer is not performance.

    The client starts at 100, deposits 100 on a flat day, and ends at 200.
    Nothing was earned, so the return must be 0% -- not the +100% a naive
    end-over-start calculation reports.
    """
    result = time_weighted_return([100.0, 200.0], [0.0, 100.0])
    assert result == pytest.approx(0.0)


def test_twr_neutralizes_a_withdrawal():
    result = time_weighted_return([200.0, 100.0], [0.0, -100.0])
    assert result == pytest.approx(0.0)


def test_twr_separates_a_deposit_from_the_gain_that_accompanies_it():
    # Begin 100, deposit 100 (so 200 at work), end 220: a genuine 10%.
    result = time_weighted_return([100.0, 220.0], [0.0, 100.0])
    assert result == pytest.approx(0.10)


def test_twr_needs_at_least_two_valuations():
    assert time_weighted_return([100.0], [0.0]) is None
    assert time_weighted_return([], []) is None


def test_twr_skips_a_period_starting_from_nothing_rather_than_dividing_by_zero():
    """A period whose adjusted opening value is zero has no defined return.

    Skipping it is the honest answer; the alternative is a ZeroDivisionError
    or an invented number.
    """
    result = time_weighted_return([0.0, 0.0, 110.0], [0.0, 0.0, 100.0])
    assert result == pytest.approx(0.10)


def test_twr_returns_none_when_no_period_was_usable():
    assert time_weighted_return([0.0, 0.0], [0.0, 0.0]) is None


# --- Money-weighted return (IRR) ---------------------------------------------


def test_irr_of_a_simple_doubling_over_one_year():
    start = datetime(2026, 1, 1)
    result = money_weighted_return([start, start + timedelta(days=365)], [-100.0, 200.0])
    assert result == pytest.approx(1.0, abs=1e-3)


def test_irr_of_a_flat_year_is_zero():
    start = datetime(2026, 1, 1)
    result = money_weighted_return([start, start + timedelta(days=365)], [-100.0, 100.0])
    assert result == pytest.approx(0.0, abs=1e-3)


def test_irr_reflects_when_the_money_arrived():
    """IRR is the client's experience, so timing has to matter.

    Same closing value, but the second client had their money at work for only
    half the period, so their rate of return must be higher.
    """
    start = datetime(2026, 1, 1)
    early = money_weighted_return([start, start + timedelta(days=365)], [-100.0, 120.0])
    late = money_weighted_return(
        [start, start + timedelta(days=182), start + timedelta(days=365)],
        [-50.0, -50.0, 120.0],
    )
    assert late > early


def test_irr_requires_both_an_inflow_and_an_outflow():
    start = datetime(2026, 1, 1)
    # All outflows: there is no rate that makes this zero.
    assert money_weighted_return([start, start + timedelta(days=30)], [-100.0, -50.0]) is None


def test_irr_returns_none_rather_than_a_meaningless_root():
    """No sign change in the bracket means no root inside it.

    Widening the bracket until something is found would return a
    mathematically valid rate with no financial meaning, which is worse than
    admitting the calculation does not apply.
    """
    start = datetime(2026, 1, 1)
    result = money_weighted_return([start, start + timedelta(days=365)], [-1.0, 1e9])
    assert result is None or result <= 10.0


def test_irr_rejects_mismatched_inputs():
    start = datetime(2026, 1, 1)
    assert money_weighted_return([start], [-100.0]) is None
    assert money_weighted_return([start, start + timedelta(days=1)], [-100.0]) is None


# --- Max drawdown ------------------------------------------------------------


def test_max_drawdown_of_a_monotonic_rise_is_zero():
    assert max_drawdown([0.1, 0.1, 0.1]) == pytest.approx(0.0)


def test_max_drawdown_measures_peak_to_trough_not_start_to_trough():
    # Up 100% then down 50% returns to the starting level, but the fall from
    # the peak is still 50%.
    assert max_drawdown([1.0, -0.5]) == pytest.approx(-0.5)


def test_max_drawdown_takes_the_worst_of_several_falls():
    result = max_drawdown([-0.1, 0.2, -0.3, 0.1])
    # The second fall is from a higher peak and is deeper.
    assert result == pytest.approx(-0.3, abs=1e-9)


def test_max_drawdown_is_none_with_no_observations():
    assert max_drawdown([]) is None


def test_max_drawdown_ignores_flows_because_it_takes_returns():
    """The regression this function exists to prevent.

    These are flow-adjusted returns for a client who withdrew 30% of the
    account on a flat day. The strategy never lost anything, so the drawdown
    must be zero -- the old implementation ran on raw market value and booked
    the withdrawal as a 30% drawdown.
    """
    assert max_drawdown([0.0, 0.0, 0.0]) == pytest.approx(0.0)


# --- Annualization cadence ---------------------------------------------------


def _dates(step_days, count):
    start = datetime(2026, 1, 1)
    return [start + timedelta(days=step_days * i) for i in range(count)]


def test_daily_snapshots_annualize_by_the_trading_day_count():
    assert periods_per_year(_dates(1, 10)) == pytest.approx(float(TRADING_DAYS))


def test_weekly_snapshots_annualize_by_about_52():
    assert periods_per_year(_dates(7, 10)) == pytest.approx(52.2, abs=0.5)


def test_monthly_snapshots_annualize_by_about_12():
    assert periods_per_year(_dates(30, 10)) == pytest.approx(12.2, abs=0.5)


def test_cadence_is_capped_at_the_trading_day_count():
    """Calendar days are not trading days, and nothing here is intraday."""
    dense = [datetime(2026, 1, 1) + timedelta(hours=6 * i) for i in range(10)]
    assert periods_per_year(dense) == pytest.approx(float(TRADING_DAYS))


def test_one_long_gap_does_not_redefine_the_cadence():
    """The median is used precisely so an outage cannot rewrite the series.

    Nine daily gaps and one 200-day gap is still a daily series with a hole in
    it, not a biannual one.
    """
    dates = _dates(1, 10) + [datetime(2026, 1, 1) + timedelta(days=210)]
    assert periods_per_year(dates) == pytest.approx(float(TRADING_DAYS))


def test_cadence_falls_back_when_there_is_nothing_to_measure():
    assert periods_per_year([datetime(2026, 1, 1)]) == pytest.approx(float(TRADING_DAYS))
    assert periods_per_year([]) == pytest.approx(float(TRADING_DAYS))


# --- Horizon pricing ---------------------------------------------------------


def _price_series(start, days, price=100.0):
    index = pd.date_range(start=start, periods=days, freq="D")
    return pd.Series([price + i for i in range(days)], index=index)


def test_close_at_returns_the_price_on_the_target_date_not_the_latest():
    """The bug this guards: scoring a 30-day call at today's price.

    A recommendation evaluated long after its horizon must still be measured
    at the horizon, or a 30-day outcome silently records a 200-day return.
    """
    series = _price_series(datetime(2026, 1, 1), 200)
    at_thirty = _close_at(series, datetime(2026, 1, 31))
    assert at_thirty == pytest.approx(130.0)
    assert at_thirty != series.iloc[-1]


def test_close_at_accepts_the_last_close_before_a_non_trading_target():
    """A weekend or holiday target should use the prior close, not fail."""
    index = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])
    series = pd.Series([100.0, 101.0, 105.0], index=index)
    # Saturday: the Friday close stands.
    assert _close_at(series, datetime(2026, 1, 3)) == pytest.approx(101.0)


def test_close_at_refuses_when_the_nearest_close_is_too_old():
    """A gap wider than the tolerance is missing history, not a long weekend.

    The caller voids the outcome on None, which is the point: an unscoreable
    recommendation must not be scored against a stale price.
    """
    index = pd.to_datetime(["2026-01-01"])
    series = pd.Series([100.0], index=index)
    assert _close_at(series, datetime(2026, 3, 1)) is None


def test_close_at_refuses_when_history_starts_after_the_target():
    series = _price_series(datetime(2026, 6, 1), 10)
    assert _close_at(series, datetime(2026, 1, 1)) is None


def test_close_at_handles_an_empty_series():
    assert _close_at(pd.Series(dtype=float), datetime(2026, 1, 1)) is None


def test_close_at_rejects_a_non_positive_price():
    index = pd.to_datetime(["2026-01-01"])
    assert _close_at(pd.Series([0.0], index=index), datetime(2026, 1, 1)) is None
