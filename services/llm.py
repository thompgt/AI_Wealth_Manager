"""Shared Gemini client factory with retry, budget and cost accounting.

Why this module exists
----------------------
Each LLM-backed agent used to construct its own `ChatGoogleGenerativeAI` with
`gemini-1.5-pro` hardcoded inline, no timeout, and no retry. That had three
production consequences:

1.  When Google retired the 1.5 series, *every* LLM call in the system began
    failing. Because each agent has a deterministic fallback, the system kept
    returning 200s and producing plausible-looking reports -- while the entire
    AI layer was dead. Silent, total capability loss with no alarm.
2.  A single transient 429/503 dropped a run into fallback mode for good.
3.  A model change required editing three files.

What this version adds
----------------------
**Cost attribution.** Token counts and dollar cost are captured per call and
carried back to the agent, which writes them to `agent_runs`. Without this,
"how much does a run cost?" and "which client is expensive?" have no answer,
and a retry loop that misbehaves is invisible until the bill arrives.

**A per-run budget.** `RunBudget` caps spend across a whole graph run. The
retry edge can drive several research passes, and an unbounded loop against a
metered API is the kind of bug that is cheap to write and expensive to own.
Exceeding the budget degrades to the deterministic path -- the same failure
mode the system already handles everywhere -- rather than raising.

**Honest degradation reporting.** Every failure is classified and returned as
a typed reason, so an agent records *why* it fell back: no key configured, a
rate limit, a retired model, or a budget stop. "Degraded" alone does not tell
an operator what to fix.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar

from config import settings
from logging_setup import get_logger
from metrics import (
    llm_budget_headroom,
    llm_calls,
    llm_cost,
    llm_latency,
    llm_tokens,
    timed,
)

logger = get_logger(__name__)

T = TypeVar("T")


class LLMUnavailable(RuntimeError):
    """Raised when no usable LLM is configured at all.

    Distinct from a call failure: agents use it to record 'no API key was
    configured' rather than misreporting a configuration gap as a model error.
    """


class BudgetExceeded(RuntimeError):
    """The run has spent its LLM budget. Further calls are refused."""


# Substrings of exception reprs that indicate a permanent failure -- retrying
# an invalid API key or a retired model id just wastes wall-clock and quota.
_PERMANENT_MARKERS = (
    "api key not valid",
    "api_key_invalid",
    "permission_denied",
    "invalid argument",
    "not found for api version",
    "is not supported",
    "unauthorized",
)

# Substrings that indicate a genuinely transient failure worth retrying.
_TRANSIENT_MARKERS = (
    "429",
    "rate limit",
    "resource_exhausted",
    "quota",
    "503",
    "service unavailable",
    "unavailable",
    "500",
    "internal error",
    "deadline",
    "timeout",
    "connection",
)


def _is_transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _PERMANENT_MARKERS):
        return False
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def classify_failure(exc: BaseException) -> str:
    """A short, stable reason string for the audit record.

    Stable because it is written to the database and queried later: an
    operator asking "how often are we falling back, and why?" needs a
    vocabulary, not free-form exception text that changes with the SDK.
    """
    if isinstance(exc, LLMUnavailable):
        return "no_api_key"
    if isinstance(exc, BudgetExceeded):
        return "budget_exceeded"
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(m in text for m in ("api key not valid", "api_key_invalid", "unauthorized")):
        return "invalid_api_key"
    if any(m in text for m in ("not found for api version", "is not supported")):
        return "model_unavailable"
    if any(m in text for m in ("429", "rate limit", "resource_exhausted", "quota")):
        return "rate_limited"
    if any(m in text for m in ("timeout", "deadline")):
        return "timeout"
    if any(m in text for m in ("503", "500", "service unavailable", "internal error")):
        return "provider_error"
    if "connection" in text:
        return "network_error"
    return "unknown_error"


# --- Usage and budget --------------------------------------------------------


@dataclass
class LLMUsage:
    """Tokens and cost for one call, or accumulated across many."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return round(
            self.prompt_tokens / 1_000_000 * settings.LLM_INPUT_COST_PER_MTOK
            + self.completion_tokens / 1_000_000 * settings.LLM_OUTPUT_COST_PER_MTOK,
            6,
        )

    def add(self, other: "LLMUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
            "cost_usd": self.cost_usd,
            "model": settings.GEMINI_MODEL,
        }


@dataclass
class RunBudget:
    """Spend ceiling for a single graph run.

    Thread-safe because the graph fans out: two agents can be mid-call at the
    same moment, and an unsynchronized check-then-spend lets both pass a
    budget only one of them fits in.
    """

    limit_usd: float = field(default_factory=lambda: settings.LLM_RUN_BUDGET_USD)
    usage: LLMUsage = field(default_factory=LLMUsage)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.limit_usd - self.usage.cost_usd)

    def check(self) -> None:
        if self.limit_usd <= 0:
            return  # a non-positive limit means "unlimited"
        if self.remaining() <= 0:
            raise BudgetExceeded(
                f"This run has spent its ${self.limit_usd:.2f} LLM budget "
                f"(${self.usage.cost_usd:.4f} across {self.usage.calls} calls). "
                "Remaining agents will use their deterministic paths."
            )

    def record(self, usage: LLMUsage) -> None:
        with self._lock:
            self.usage.add(usage)


# The active run's budget, bound per thread rather than carried on graph
# state: LangGraph checkpoints its state, and a budget holds a lock, which is
# not serializable. Thread-local rather than a contextvar because the graph and
# its nodes run synchronously on worker threads, where a contextvar set in the
# parent does not reliably propagate.
_budget_local = threading.local()


@contextmanager
def run_budget(budget: Optional["RunBudget"] = None):
    """Bind a spend ceiling for everything invoked inside this block."""
    previous = getattr(_budget_local, "budget", None)
    _budget_local.budget = budget if budget is not None else RunBudget()
    # A gauge, not a per-run series: a run id label would add one time series
    # per run and never retire it. With more than one worker this reads as
    # "headroom on whichever run touched it last", which is enough to see
    # runs crowding the cap without unbounded cardinality.
    llm_budget_headroom.set(_budget_local.budget.remaining())
    try:
        yield _budget_local.budget
    finally:
        _budget_local.budget = previous


def current_budget() -> Optional["RunBudget"]:
    return getattr(_budget_local, "budget", None)


def extract_usage(response: Any) -> LLMUsage:
    """Pull token counts off a LangChain response.

    Providers report usage in several shapes and move it between them across
    versions. Missing counts are reported as zero rather than estimated:
    a fabricated token count that feeds a cost figure and a budget is worse
    than an honest zero, because it silently mis-attributes spend.
    """
    usage = LLMUsage(calls=1)
    metadata = None
    for attribute in ("usage_metadata", "response_metadata"):
        candidate = getattr(response, attribute, None)
        if isinstance(candidate, dict) and candidate:
            metadata = candidate
            break
    if metadata is None:
        return usage

    nested = metadata.get("usage_metadata") or metadata.get("token_usage") or metadata
    if not isinstance(nested, dict):
        return usage

    for key in ("input_tokens", "prompt_token_count", "prompt_tokens"):
        if isinstance(nested.get(key), int):
            usage.prompt_tokens = nested[key]
            break
    for key in ("output_tokens", "candidates_token_count", "completion_tokens"):
        if isinstance(nested.get(key), int):
            usage.completion_tokens = nested[key]
            break
    return usage


# --- Client ------------------------------------------------------------------


def get_chat_model(temperature: float | None = None, **kwargs: Any):
    """Build a configured Gemini chat model.

    Raises LLMUnavailable when no real API key is configured, so callers can
    skip the network round trip entirely and go straight to their
    deterministic fallback with an accurate reason.
    """
    if not settings.llm_configured:
        raise LLMUnavailable(
            "GEMINI_API_KEY is not configured (still the placeholder value). "
            "Agents will use their deterministic fallbacks."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=0,  # retry policy lives here, not in the SDK
        **kwargs,
    )


def invoke_with_retry(call: Callable[[], T], *, what: str = "LLM call") -> T:
    """Run `call`, retrying transient failures with exponential backoff.

    Re-raises the final exception so the calling agent can record it in its
    audit trail and fall back deterministically.
    """
    attempts = max(1, settings.LLM_MAX_ATTEMPTS)
    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 -- classified and re-raised below
            last_exc = exc
            if not _is_transient(exc) or attempt == attempts:
                logger.warning(
                    "%s failed permanently on attempt %d/%d: %s",
                    what, attempt, attempts, exc,
                )
                raise
            delay = settings.LLM_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "%s failed with a transient error on attempt %d/%d (%s); retrying in %.1fs",
                what, attempt, attempts, exc, delay,
            )
            time.sleep(delay)

    assert last_exc is not None  # unreachable; loop either returns or raises
    raise last_exc


def invoke_tracked(
    call: Callable[[], T],
    *,
    node: str,
    budget: Optional[RunBudget] = None,
) -> tuple[T, LLMUsage]:
    """Invoke with retry, budget enforcement and cost accounting.

    The single entry point agents should use. Returns the response together
    with its usage, so the caller can attach both to its audit record without
    reaching into the response object and guessing where the token counts live.
    """
    if budget is None:
        budget = current_budget()
    if budget is not None:
        # Checked before the call, not after: a budget that only stops
        # spending once it has already been exceeded is a report, not a limit.
        budget.check()

    try:
        with timed(llm_latency, node=node):
            response = invoke_with_retry(call, what=f"{node} LLM call")
    except Exception as exc:  # noqa: BLE001 -- classified for metrics, then re-raised
        llm_calls.labels(node, classify_failure(exc)).inc()
        raise

    usage = extract_usage(response)
    if budget is not None:
        budget.record(usage)
        # Published after every call rather than only at the end of a run: the
        # point is to watch headroom fall toward zero while there is still
        # time to notice, not to learn afterwards that it ran out.
        llm_budget_headroom.set(budget.remaining())

    llm_calls.labels(node, "success").inc()
    if usage.prompt_tokens:
        llm_tokens.labels(node, "prompt").inc(usage.prompt_tokens)
    if usage.completion_tokens:
        llm_tokens.labels(node, "completion").inc(usage.completion_tokens)
    if usage.cost_usd:
        llm_cost.labels(node).inc(usage.cost_usd)

    return response, usage
