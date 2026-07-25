"""Shared Gemini client factory with retry/backoff.

Why this module exists
----------------------
Each of the three LLM-backed agents (market regime, stock research, finance
report) used to construct its own `ChatGoogleGenerativeAI` with the model id
`gemini-1.5-pro` hardcoded inline, no timeout, and no retry. That had three
production consequences:

1.  When Google retired the 1.5 series, *every* LLM call in the system began
    failing. Because each agent has a deterministic fallback, the system kept
    returning 200s and producing plausible-looking reports -- while the entire
    AI layer was dead. Silent, total capability loss with no alarm.
2.  A single transient 429/503 dropped a run into fallback mode for good:
    regime becomes "Volatile"/confidence 0.0, research picks become
    confidence 0.0 (which collapses confidence-weighted position sizing to
    equal weight), and the client report becomes a template.
3.  A model change required editing three files.

`get_chat_model()` centralises the model id (config.GEMINI_MODEL) and
`invoke_with_retry()` retries genuinely transient failures with exponential
backoff while failing fast on permanent ones (bad key, bad model id, bad
request), so agents drop to their fallback immediately when retrying cannot
possibly help.
"""

from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

from config import settings
from logging_setup import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class LLMUnavailable(RuntimeError):
    """Raised when no usable LLM is configured at all.

    Distinct from a call failure: agents use it to record 'no API key was
    configured' rather than misreporting a configuration gap as a model error.
    """


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
