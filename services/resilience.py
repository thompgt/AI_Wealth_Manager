"""Retry, circuit breaking and rate limiting for outbound calls.

The previous system had a careful retry policy for the LLM and none at all for
market data — which is backwards in terms of consequence. An LLM failure
degrades a narrative; a market-data failure silently removes a holding from
the analysis, and every concentration percentage, every flaw and every
position size downstream is then computed against a portfolio that is missing
a position. Nothing in the output says so.

Three primitives, deliberately separate:

* **Retry** handles the blip — a 429, a reset connection, a 503.
* **The circuit breaker** handles the outage. Retrying a provider that is down
  is not resilience; it multiplies one failure by the number of tickers in the
  run and turns a degraded run into a timed-out one. After enough consecutive
  failures the provider is taken out of rotation and callers fail over
  immediately, with one probe allowed through after a cooldown.
* **The rate limiter** handles us being the problem. An unauthenticated data
  source will ban an IP that fans out hundreds of requests with no pacing, and
  that ban outlives the run that caused it.
"""

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Tuple, Type, TypeVar

from logging_setup import get_logger
from services import deadline

logger = get_logger(__name__)

T = TypeVar("T")


class ProviderError(RuntimeError):
    """A provider call failed. Carries whether retrying could plausibly help."""

    def __init__(self, message: str, *, transient: bool = True, provider: str = ""):
        super().__init__(message)
        self.transient = transient
        self.provider = provider


class CircuitOpenError(ProviderError):
    """The provider is out of rotation; fail over without calling it."""

    def __init__(self, provider: str, retry_in: float):
        super().__init__(
            f"{provider} circuit is open for another {retry_in:.0f}s after repeated failures",
            transient=True,
            provider=provider,
        )
        self.retry_in = retry_in


# Substrings that mean "this will fail again the same way". Matching on text
# is unlovely, but HTTP clients, SDKs and DNS layers all raise different
# exception types for the same underlying condition, and the alternative --
# retrying everything -- turns a bad API key into a slow bad API key.
_PERMANENT_MARKERS: Tuple[str, ...] = (
    "invalid api key",
    "unauthorized",
    "forbidden",
    "not found",
    "no data found",
    "delisted",
    "unknown symbol",
    "invalid ticker",
    "400 client error",
    "401",
    "403",
    "404",
)


def classify(exc: BaseException) -> bool:
    """True when retrying is plausibly worthwhile."""
    if isinstance(exc, ProviderError):
        return exc.transient
    text = f"{type(exc).__name__}: {exc}".lower()
    return not any(marker in text for marker in _PERMANENT_MARKERS)


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.75,
    max_delay: float = 8.0,
    label: str = "call",
    retry_on: Optional[Iterable[Type[BaseException]]] = None,
) -> T:
    """Call `fn`, retrying transient failures with jittered exponential backoff.

    The jitter is not cosmetic: a run fetches many tickers, and without it
    every failed request in a batch retries at the same instant, reproducing
    the thundering herd that caused the rate limit in the first place.
    """
    retry_types = tuple(retry_on) if retry_on else (Exception,)
    last: Optional[BaseException] = None

    # Checked before the first attempt as well as between them. A run that is
    # already out of time should not open a new connection at all -- and the
    # first attempt is where most of the time goes, so checking only between
    # retries would let one final call overrun the budget by a full timeout.
    deadline.check(label)

    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fn()
        except retry_types as exc:  # noqa: PERF203 -- the retry is the point
            last = exc
            if not classify(exc) or attempt >= attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 0.5 + random.random()  # noqa: S311 -- jitter, not crypto
            if deadline.expired():
                # Retrying is the most expensive thing this function can do,
                # and a retry that starts after the run is over cannot help
                # the run. Fail with the underlying error rather than the
                # deadline: the caller's fallback path is keyed to *why* the
                # provider failed, and "out of time" would lose that.
                logger.debug("%s: out of run time; not retrying.", label)
                break
            logger.debug(
                "%s failed (attempt %d/%d): %s; retrying in %.2fs",
                label, attempt, attempts, exc, delay,
            )
            time.sleep(delay)

    assert last is not None
    raise last


@dataclass
class CircuitBreaker:
    """Per-provider failure gate.

    Deliberately simple: consecutive failures trip it, one success closes it.
    A rolling error-rate window would be more precise, but this is guarding a
    handful of providers with clear up/down behaviour, and a breaker whose
    state is hard to reason about is a breaker nobody trusts enough to leave
    enabled.
    """

    name: str
    failure_threshold: int = 5
    reset_timeout: float = 120.0

    _failures: int = field(default=0, init=False)
    _opened_at: Optional[float] = field(default=None, init=False)
    _half_open: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_at is not None and not self._half_open

    def before_call(self) -> None:
        """Raise CircuitOpenError if the provider is out of rotation."""
        with self._lock:
            if self._opened_at is None:
                return
            elapsed = time.monotonic() - self._opened_at
            if elapsed < self.reset_timeout:
                raise CircuitOpenError(self.name, self.reset_timeout - elapsed)
            # Cooldown elapsed: let exactly one call through to probe.
            self._half_open = True

    def record_success(self) -> None:
        with self._lock:
            if self._opened_at is not None:
                logger.info("%s circuit closed after a successful probe", self.name)
            self._failures = 0
            self._opened_at = None
            self._half_open = False

    def record_failure(self) -> None:
        with self._lock:
            if self._half_open:
                # The probe failed. Restart the full cooldown rather than
                # letting probes through on every subsequent call.
                self._opened_at = time.monotonic()
                self._half_open = False
                return
            self._failures += 1
            if self._failures >= self.failure_threshold and self._opened_at is None:
                self._opened_at = time.monotonic()
                logger.warning(
                    "%s circuit opened after %d consecutive failures; failing over for %.0fs",
                    self.name, self._failures, self.reset_timeout,
                )

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._half_open = False


class RateLimiter:
    """Token bucket, shared across threads.

    Paces outbound calls so a wide screen does not get the process IP banned.
    Blocking rather than rejecting is correct here: the caller genuinely wants
    the data and a short wait is better than a failed run.
    """

    def __init__(self, rate_per_second: float, burst: Optional[int] = None):
        self.rate = max(0.1, rate_per_second)
        self.capacity = float(burst if burst is not None else max(1.0, rate_per_second))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait = (tokens - self._tokens) / self.rate
            # Sleep outside the lock so waiting threads do not serialize on it.
            time.sleep(min(wait, 1.0))
