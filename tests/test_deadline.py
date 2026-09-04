"""The wall-clock ceiling on a run.

Every individual call was already bounded; the total was not. The tests here
pin the two properties that make a deadline different from a suggestion: it
refuses to *start* new outbound work, and it shortens the timeout of work it
does allow, so one late call cannot overrun the budget by a full timeout.
"""

import time

import pytest

from services import deadline
from services.deadline import DeadlineExceeded, RunDeadline, run_deadline


def expired_deadline():
    """A deadline whose time is already spent."""
    return RunDeadline(limit_seconds=10.0, started_at=time.monotonic() - 11.0)


# --- The primitive -----------------------------------------------------------


def test_an_unbound_deadline_never_blocks_anything():
    """The services stay usable from a script, a notebook or a test.

    A ceiling that must be set up before market data can be fetched at all
    would push every caller into ceremony, and the ones that skipped it would
    be the ones running unbounded.
    """
    assert deadline.current_deadline() is None
    deadline.check("anything")  # must not raise
    assert deadline.expired() is False
    assert deadline.timeout_for(15.0) == 15.0


def test_a_non_positive_limit_means_unlimited():
    """Matching how the LLM spend budget reads its own limit."""
    with run_deadline(RunDeadline(limit_seconds=0)) as active:
        assert active.unlimited
        assert not active.expired()
        assert active.remaining() == float("inf")
        deadline.check("a long backfill")
        assert deadline.timeout_for(15.0) == 15.0


def test_an_expired_deadline_refuses_to_start_work():
    with run_deadline(expired_deadline()):
        assert deadline.expired()
        with pytest.raises(DeadlineExceeded) as caught:
            deadline.check("fetching AAPL")
        # The message has to say what did not happen, because it reaches the
        # audit trail and then the report.
        assert "fetching AAPL" in str(caught.value)


def test_a_call_is_shortened_to_fit_the_remaining_budget():
    """The difference between a deadline and a suggestion.

    A 15-second call started with 2 seconds left otherwise overruns by 13,
    and thirty such calls overrun by minutes -- which is how a bounded run
    takes an unbounded amount of time while every individual limit is met.
    """
    with run_deadline(RunDeadline(limit_seconds=10.0, started_at=time.monotonic() - 8.0)):
        shortened = deadline.timeout_for(15.0)
        assert 1.0 < shortened <= 2.0


def test_shortening_never_produces_a_non_positive_timeout():
    """Zero means 'no timeout' to most HTTP clients.

    Handing one to a client at the exact moment the run is out of time would
    turn the deadline into an unbounded call -- the precise failure it exists
    to prevent.
    """
    with run_deadline(expired_deadline()):
        assert deadline.timeout_for(15.0) > 0


def test_a_shorter_request_is_not_lengthened():
    """It only ever shortens; a 1s call stays a 1s call."""
    with run_deadline(RunDeadline(limit_seconds=600.0)):
        assert deadline.timeout_for(1.0) == 1.0


def test_the_binding_is_restored_on_exit():
    outer = RunDeadline(limit_seconds=600.0)
    with run_deadline(outer):
        with run_deadline(RunDeadline(limit_seconds=1.0)):
            assert deadline.current_deadline().limit_seconds == 1.0
        assert deadline.current_deadline() is outer
    assert deadline.current_deadline() is None


# --- Enforcement in the retry loop -------------------------------------------


def test_retry_call_does_not_open_a_connection_when_out_of_time():
    from services.resilience import retry_call

    calls = []

    def fn():
        calls.append(1)
        return "value"

    with run_deadline(expired_deadline()):
        with pytest.raises(DeadlineExceeded):
            retry_call(fn, label="quote fetch")

    assert calls == [], "the deadline must prevent the call, not just its retries"


def test_retry_call_stops_retrying_once_the_deadline_passes():
    """Retrying is the most expensive thing the loop can do.

    The error raised is the provider's, not the deadline's: the caller's
    fallback path is keyed to *why* the provider failed, and replacing that
    with "out of time" would lose the reason.
    """
    from services.resilience import retry_call

    active = RunDeadline(limit_seconds=0.05)
    calls = []

    def flaky():
        calls.append(1)
        # Spend the run's remaining time inside the call, which is what a
        # slow provider actually does. Without this the loop retries faster
        # than the clock moves and the test proves nothing.
        time.sleep(0.06)
        # Something classify() treats as retryable, so the loop would
        # otherwise use all three attempts.
        raise TimeoutError("provider timed out")

    with run_deadline(active):
        with pytest.raises(TimeoutError):
            retry_call(flaky, attempts=3, base_delay=0.01, label="quote fetch")

    assert len(calls) < 3, "the loop should have given up before its attempt budget"


def test_a_healthy_deadline_leaves_the_retry_loop_alone():
    from services.resilience import retry_call

    with run_deadline(RunDeadline(limit_seconds=600.0)):
        assert retry_call(lambda: "ok", label="quote fetch") == "ok"


# --- Enforcement in the LLM path ---------------------------------------------


def test_the_llm_is_not_called_when_the_run_is_out_of_time():
    """An LLM call is the most expensive thing a node does in wall clock.

    Sixty seconds times three attempts with backoff between them. A run past
    its deadline must take the deterministic fallback, which is instant --
    exactly what a run out of time should produce.
    """
    from services.llm import invoke_tracked

    called = []

    with run_deadline(expired_deadline()):
        with pytest.raises(DeadlineExceeded):
            invoke_tracked(lambda: called.append(1), node="market_regime")

    assert called == []


def test_a_deadline_failure_is_classified_apart_from_a_timeout():
    """A timeout means the model did not answer; this means we never asked.

    The strings are written to the audit table and queried later, so an
    operator tuning the deadline has to be able to tell them apart.
    """
    from services.llm import classify_failure

    assert classify_failure(DeadlineExceeded("out of time")) == "run_deadline"
    assert classify_failure(TimeoutError("deadline exceeded")) == "timeout"
