"""A wall-clock ceiling on a single graph run.

Every individual outbound call is already bounded: the LLM has a timeout and a
retry cap, market data has a timeout, a retry cap, a circuit breaker and a rate
limiter. What none of that bounds is the *total*.

A run screens a catalogue of roughly 170 names. At a 15-second market-data
timeout with three attempts, a provider that accepts connections and never
answers costs 45 seconds per ticker, and the circuit breaker only helps once
five consecutive calls have actually failed -- which, against a hanging
provider rather than a refusing one, takes almost four minutes on its own. The
per-call limits are all satisfied while the run takes hours.

The consequences are worse than slowness. Each worker thread is a slot; two
hung runs consume the entire default pool, and every queued client's analysis
stops. The job timeout eventually reclaims the row, but at 15 minutes, after
the damage, and it fails the run outright rather than returning the partial
analysis that had already been computed.

A deadline changes the failure from "hangs, then fails" to "stops reaching out,
degrades, and reports what it has". It is bound per thread rather than carried
on graph state, for the same reason the LLM budget is: LangGraph checkpoints
its state, and neither a lock nor a monotonic timestamp survives that round
trip meaningfully.

The contract is deliberately narrow. A deadline stops *new outbound work*; it
never interrupts a call in flight, because cancelling a request mid-flight
gives no guarantee about what the other side did with it, and for an order
placement that distinction is the whole game. Purely local computation is never
blocked -- a node that has its data should always be allowed to finish
producing its answer.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic
from typing import Iterator, Optional

from config import settings
from logging_setup import get_logger

logger = get_logger(__name__)


class DeadlineExceeded(Exception):
    """The run is out of time; this outbound call was not attempted.

    Distinct from a timeout on purpose. A timeout means the remote side did
    not answer in time and the request may well have been received; this means
    nothing was sent. An audit trail that conflates the two cannot answer
    whether an order reached a venue.
    """


@dataclass
class RunDeadline:
    """The remaining wall clock for one run.

    Monotonic rather than wall time: a run must not gain or lose its budget
    because NTP stepped the clock, and on a container that has just started
    that step is routine.
    """

    limit_seconds: float = field(
        default_factory=lambda: float(settings.RUN_DEADLINE_SECONDS)
    )
    started_at: float = field(default_factory=monotonic)

    @property
    def unlimited(self) -> bool:
        # A non-positive limit means no ceiling, matching how the LLM budget
        # reads its own limit. Useful for the notebook and for a deliberate
        # long-running backfill.
        return self.limit_seconds <= 0

    def elapsed(self) -> float:
        return monotonic() - self.started_at

    def remaining(self) -> float:
        if self.unlimited:
            return float("inf")
        return max(0.0, self.limit_seconds - self.elapsed())

    def expired(self) -> bool:
        return not self.unlimited and self.remaining() <= 0

    def check(self, operation: str) -> None:
        """Raise if there is no time left to start `operation`."""
        if self.expired():
            raise DeadlineExceeded(
                f"The run exceeded its {self.limit_seconds:.0f}s wall-clock budget "
                f"before {operation} could be attempted. The analysis continues with "
                f"the data already gathered."
            )

    def budget_for(self, requested_timeout: float) -> float:
        """The timeout to actually use for one call.

        Clamping each call to the remaining run time is the difference between
        a deadline and a suggestion: a 15-second call started with 2 seconds
        left otherwise overruns the run budget by 13 seconds, and thirty such
        calls overrun it by minutes.

        Never returns zero or less -- a non-positive timeout is "no timeout" to
        most clients, which is the exact opposite of what is wanted here.
        Callers check `expired()` first; this only ever shortens.
        """
        if self.unlimited:
            return requested_timeout
        return max(0.1, min(requested_timeout, self.remaining()))


# Thread-local for the same reason as the LLM budget: the graph and its nodes
# run synchronously on job-worker threads, where a contextvar set by the parent
# does not reliably propagate into the pool.
_local = threading.local()


@contextmanager
def run_deadline(deadline: Optional[RunDeadline] = None) -> Iterator[RunDeadline]:
    """Bind a wall-clock ceiling for everything invoked inside this block."""
    previous = getattr(_local, "deadline", None)
    active = deadline if deadline is not None else RunDeadline()
    _local.deadline = active
    try:
        yield active
    finally:
        _local.deadline = previous


def current_deadline() -> Optional[RunDeadline]:
    return getattr(_local, "deadline", None)


def check(operation: str) -> None:
    """Raise DeadlineExceeded if the bound run is out of time.

    A no-op when nothing is bound, so the services stay usable from a script,
    a notebook or a test without ceremony.
    """
    active = current_deadline()
    if active is not None:
        active.check(operation)


def expired() -> bool:
    active = current_deadline()
    return bool(active and active.expired())


def timeout_for(requested_timeout: float) -> float:
    """Shorten a call's timeout to fit inside the run's remaining budget."""
    active = current_deadline()
    if active is None:
        return requested_timeout
    return active.budget_for(requested_timeout)


def remaining() -> Optional[float]:
    active = current_deadline()
    return None if active is None else active.remaining()
