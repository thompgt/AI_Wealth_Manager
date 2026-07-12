from types import SimpleNamespace

from agents.tax_awareness import SIGNIFICANT_GAIN_THRESHOLD, _embedded_gain_loss_notes, _wash_sale_check


def _candidate(ticker):
    return {
        "ticker": ticker,
        "valuation_metrics": {},
        "addresses_flaw": "test",
        "regime_fit_rationale": "test",
        "confidence": 0.5,
    }


def test_wash_sale_check_flags_recently_sold_symbol():
    candidates = [_candidate("TSLA"), _candidate("AAPL")]
    flags, notes = _wash_sale_check(candidates, sold_symbols={"TSLA"})
    assert flags == ["TSLA"]
    assert len(notes) == 1
    assert "wash-sale" in notes[0]


def test_wash_sale_check_no_flags_when_nothing_recently_sold():
    candidates = [_candidate("AAPL")]
    flags, notes = _wash_sale_check(candidates, sold_symbols=set())
    assert flags == []
    assert notes == []


def test_wash_sale_check_does_not_duplicate_flags():
    candidates = [_candidate("TSLA"), _candidate("TSLA")]
    flags, _ = _wash_sale_check(candidates, sold_symbols={"TSLA"})
    assert flags == ["TSLA"]


def _holding(symbol, quantity, cost_basis):
    return SimpleNamespace(symbol=symbol, quantity=quantity, cost_basis=cost_basis)


def test_embedded_gain_loss_notes_flags_unrealized_loss():
    holdings = [_holding("AAPL", 10, 200.0)]
    notes = _embedded_gain_loss_notes(holdings, prices={"AAPL": 150.0})
    assert len(notes) == 1
    assert "loss" in notes[0]


def test_embedded_gain_loss_notes_flags_significant_gain():
    cost = 100.0
    price = cost * (1 + SIGNIFICANT_GAIN_THRESHOLD + 0.1)
    holdings = [_holding("TSLA", 10, cost)]
    notes = _embedded_gain_loss_notes(holdings, prices={"TSLA": price})
    assert len(notes) == 1
    assert "gain" in notes[0]


def test_embedded_gain_loss_notes_ignores_cash_and_small_gains():
    holdings = [
        _holding("CASH", 10000, 1.0),
        _holding("AAPL", 10, 100.0),
    ]
    # AAPL priced at only a 5% gain -- below SIGNIFICANT_GAIN_THRESHOLD.
    notes = _embedded_gain_loss_notes(holdings, prices={"AAPL": 105.0})
    assert notes == []


def test_embedded_gain_loss_notes_skips_holdings_without_price_data():
    holdings = [_holding("UNKNOWN", 10, 100.0)]
    assert _embedded_gain_loss_notes(holdings, prices={}) == []
