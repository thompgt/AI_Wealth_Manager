"""Unit tests for resilience (circuit breaker, retries) and market data provider logic."""

import time
import pytest
from services.resilience import CircuitBreaker, CircuitOpenError, retry_call, classify, ProviderError
from services import market_data


def test_classify_permanent_vs_transient_errors():
    """Permanent error substrings should return False for retries; others True."""
    assert classify(ValueError("Invalid API key provided")) is False
    assert classify(RuntimeError("404 Not Found")) is False
    assert classify(Exception("connection reset by peer")) is True
    assert classify(ProviderError("Network timeout", transient=True)) is True
    assert classify(ProviderError("Bad request", transient=False)) is False


def test_circuit_breaker_tripping_and_recovery():
    """Circuit breaker trips after failure_threshold and recovers after success."""
    cb = CircuitBreaker("test-provider", failure_threshold=2, reset_timeout=0.2)

    assert cb.is_open is False
    cb.before_call()

    # Record 1 failure - still closed
    cb.record_failure()
    assert cb.is_open is False

    # Record 2nd failure - trips circuit
    cb.record_failure()
    assert cb.is_open is True

    # Immediate call raises CircuitOpenError
    with pytest.raises(CircuitOpenError):
        cb.before_call()

    # Wait for reset_timeout
    time.sleep(0.25)

    # Allowed probe in half-open state
    cb.before_call()

    # Probe succeeds -> closed
    cb.record_success()
    assert cb.is_open is False


def test_retry_call_success_after_transient():
    """retry_call retries transient failures and returns success if within attempt budget."""
    attempts_made = 0

    def flaky_func():
        nonlocal attempts_made
        attempts_made += 1
        if attempts_made < 2:
            raise ConnectionError("temporary blip")
        return "ok"

    result = retry_call(flaky_func, attempts=3, base_delay=0.01, max_delay=0.05)
    assert result == "ok"
    assert attempts_made == 2


def test_market_data_cache_clear_and_synthetic_security_info():
    """Verify security info cache clearing and security info fetching behavior."""
    market_data.clear_ticker_info_cache()
    info = market_data.get_security_info("AAPL")
    assert info is not None
    assert info.symbol == "AAPL"
    # Second call returns cached
    info2 = market_data.get_security_info("AAPL")
    assert info2 is info
    market_data.clear_ticker_info_cache()
