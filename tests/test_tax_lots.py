"""Lot selection, holding-period classification and tax estimation.

Lot selection decides which shares a sale consumes, which sets the realized
gain, which sets the tax, which gates whether the rebalancer proposes the
trade at all. An arithmetic error here costs the client money quietly, and
every function exercised below is pure -- it needs no database.
"""

from datetime import datetime, timedelta

import pytest

from services.tax_lots import (
    LONG_TERM_DAYS,
    SaleEstimate,
    TOP_CAPITAL_GAINS_RATE,
    TOP_MARGINAL_RATE,
    select_lots,
    term_for,
)


class FakeLot:
    """A tax lot without a database.

    `select_lots` only reads four attributes, so a stand-in keeps these tests
    pure. Using the real model would need a session, a client and an account
    to test arithmetic that involves none of them.
    """

    def __init__(self, lot_id, quantity, cost, acquired_days_ago):
        self.id = lot_id
        self.remaining_quantity = quantity
        self.cost_per_share = cost
        self.acquired_at = datetime(2026, 6, 1) - timedelta(days=acquired_days_ago)


AS_OF = datetime(2026, 6, 1)


# --- Holding period ----------------------------------------------------------


def test_exactly_one_year_is_still_short_term():
    """IRC 1222 says "more than one year"; 365 days is not more than a year.

    Off by one here changes the rate applied to the entire gain, which for a
    large lot is the difference between 37% and 20%.
    """
    acquired = AS_OF - timedelta(days=LONG_TERM_DAYS)
    assert term_for(acquired, AS_OF) == "short"


def test_one_day_past_a_year_is_long_term():
    acquired = AS_OF - timedelta(days=LONG_TERM_DAYS + 1)
    assert term_for(acquired, AS_OF) == "long"


# --- Lot selection methods ---------------------------------------------------


def _lots():
    return [
        FakeLot(1, 10.0, 50.0, 800),   # cheap, long-term  -> big gain
        FakeLot(2, 10.0, 150.0, 800),  # expensive, long-term -> loss at 120
        FakeLot(3, 10.0, 100.0, 10),   # mid, short-term
    ]


def test_hifo_sells_the_highest_cost_shares_first():
    selections, shortfall = select_lots(_lots(), 10.0, 120.0, method="HIFO", as_of=AS_OF)
    assert shortfall == 0.0
    assert [s.lot_id for s in selections] == [2]
    # Highest basis means the smallest gain -- here a loss, which is the point.
    assert selections[0].gain == pytest.approx(-300.0)


def test_fifo_sells_the_oldest_shares_first():
    selections, _ = select_lots(_lots(), 10.0, 120.0, method="FIFO", as_of=AS_OF)
    assert [s.lot_id for s in selections] == [1]


def test_lifo_sells_the_newest_shares_first():
    selections, _ = select_lots(_lots(), 10.0, 120.0, method="LIFO", as_of=AS_OF)
    assert [s.lot_id for s in selections] == [3]


def test_mintax_takes_losses_before_gains():
    selections, _ = select_lots(_lots(), 10.0, 120.0, method="MINTAX", as_of=AS_OF)
    assert selections[0].lot_id == 2
    assert selections[0].gain < 0


def test_mintax_prefers_a_long_term_gain_over_a_short_term_one():
    """The rate difference usually dominates the basis difference.

    Both lots show the same $200 gain, so HIFO would be indifferent; MINTAX
    takes the long-term one because it is taxed at 20% rather than 37%.
    """
    lots = [
        FakeLot(1, 10.0, 100.0, 800),  # long-term, $200 gain at 120
        FakeLot(2, 10.0, 100.0, 10),   # short-term, same $200 gain
    ]
    selections, _ = select_lots(lots, 10.0, 120.0, method="MINTAX", as_of=AS_OF)
    assert selections[0].lot_id == 1
    assert selections[0].term == "long"


def test_an_unknown_method_falls_back_to_hifo_rather_than_raising():
    selections, _ = select_lots(_lots(), 10.0, 120.0, method="NONSENSE", as_of=AS_OF)
    assert [s.lot_id for s in selections] == [2]


# --- Quantities and shortfall ------------------------------------------------


def test_a_sale_spans_as_many_lots_as_it_needs():
    selections, shortfall = select_lots(_lots(), 25.0, 120.0, method="FIFO", as_of=AS_OF)
    assert shortfall == 0.0
    assert sum(s.quantity for s in selections) == pytest.approx(25.0)
    assert [s.lot_id for s in selections] == [1, 2, 3]
    # The final lot is only partially consumed.
    assert selections[-1].quantity == pytest.approx(5.0)


def test_selling_more_than_is_held_reports_the_shortfall():
    """Never silently sells what the client does not own."""
    selections, shortfall = select_lots(_lots(), 100.0, 120.0, method="HIFO", as_of=AS_OF)
    assert sum(s.quantity for s in selections) == pytest.approx(30.0)
    assert shortfall == pytest.approx(70.0)


def test_lots_with_nothing_remaining_are_ignored():
    lots = [FakeLot(1, 0.0, 50.0, 800), FakeLot(2, 10.0, 100.0, 800)]
    selections, shortfall = select_lots(lots, 5.0, 120.0, method="FIFO", as_of=AS_OF)
    assert [s.lot_id for s in selections] == [2]
    assert shortfall == 0.0


def test_no_lots_means_the_whole_quantity_is_short():
    selections, shortfall = select_lots([], 10.0, 120.0, as_of=AS_OF)
    assert selections == []
    assert shortfall == pytest.approx(10.0)


def test_gain_and_proceeds_are_consistent_per_selection():
    selections, _ = select_lots(_lots(), 30.0, 120.0, method="FIFO", as_of=AS_OF)
    for selection in selections:
        expected = selection.proceeds - selection.quantity * selection.cost_per_share
        assert selection.gain == pytest.approx(expected)


# --- Tax estimation ----------------------------------------------------------


def _estimate(short_gain=0.0, long_gain=0.0):
    return SaleEstimate(
        symbol="AAPL",
        quantity=10.0,
        proceeds=1000.0,
        cost_basis=1000.0 - short_gain - long_gain,
        realized_gain=short_gain + long_gain,
        short_term_gain=short_gain,
        long_term_gain=long_gain,
    )


def test_estimated_tax_uses_the_supplied_rates():
    estimate = _estimate(short_gain=1000.0, long_gain=2000.0)
    assert estimate.estimated_tax(0.22, 0.15) == pytest.approx(1000 * 0.22 + 2000 * 0.15)


def test_estimated_tax_falls_back_to_top_brackets_when_no_rate_is_known():
    estimate = _estimate(short_gain=1000.0, long_gain=2000.0)
    expected = 1000 * TOP_MARGINAL_RATE + 2000 * TOP_CAPITAL_GAINS_RATE
    assert estimate.estimated_tax() == pytest.approx(expected)


def test_the_top_bracket_default_materially_overstates_a_lower_bracket_client():
    """Why the rate had to become a policy field rather than a constant.

    The rebalancer withholds a trim whose tax exceeds 25% of the trade, so
    this gap is not cosmetic -- it decides whether a concentration fix is
    proposed at all.
    """
    estimate = _estimate(short_gain=10_000.0)
    assumed = estimate.estimated_tax()
    actual = estimate.estimated_tax(0.22, 0.15)
    assert assumed > actual * 1.6


def test_a_loss_produces_a_negative_tax_meaning_a_benefit():
    estimate = _estimate(short_gain=-5000.0)
    assert estimate.estimated_tax(0.22, 0.15) < 0


def test_a_zero_rate_client_owes_nothing():
    estimate = _estimate(short_gain=1000.0, long_gain=2000.0)
    assert estimate.estimated_tax(0.0, 0.0) == pytest.approx(0.0)
