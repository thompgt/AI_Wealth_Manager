"""Provider registry and failover.

`call_with_failover` is the single path through which every outbound market
data request travels. It wraps each attempt in the three resilience
primitives, in this order and for these reasons:

  1. **Circuit check** first, so a provider known to be down costs nothing.
  2. **Rate limit** next, so we pace ourselves before spending a request.
  3. **Retry** innermost, so a blip is absorbed without consuming a failover
     step — moving to the next provider on a single 429 would discard a
     licensed source in favour of a free one for no reason.

A failure that exhausts one provider moves to the next; only when every
provider in the chain has failed does the caller see an error. The
alternative — a single provider, no retry, silent degradation — meant one
transient Yahoo hiccup removed a holding from the analysis, and every
concentration figure downstream was computed against a portfolio missing a
position, with nothing in the output saying so.
"""

import threading
from typing import Callable, Dict, List, Optional, TypeVar

from config import settings
from logging_setup import get_logger
from metrics import market_data_circuit_open, market_data_latency, market_data_requests, timed
from services.providers.base import MarketDataProvider, NewsItem, Quote, SecurityInfo
from services.providers.polygon_provider import PolygonProvider
from services.providers.tiingo_provider import TiingoProvider
from services.providers.yfinance_provider import YFinanceProvider
from services.resilience import CircuitBreaker, CircuitOpenError, ProviderError, RateLimiter, retry_call

logger = get_logger(__name__)

T = TypeVar("T")

_PROVIDER_TYPES = {
    "yfinance": YFinanceProvider,
    "polygon": PolygonProvider,
    "tiingo": TiingoProvider,
}

_lock = threading.Lock()
_providers: Dict[str, MarketDataProvider] = {}
_breakers: Dict[str, CircuitBreaker] = {}
_limiters: Dict[str, RateLimiter] = {}


class AllProvidersFailed(ProviderError):
    """Every provider in the chain failed. Carries each one's reason.

    Reporting only the last failure would hide the interesting case: the
    licensed provider rejected our key and we quietly fell back to the free
    one for the rest of the deployment.
    """

    def __init__(self, operation: str, failures: Dict[str, str]):
        detail = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
        super().__init__(f"All market data providers failed for {operation} -- {detail}")
        self.failures = failures


def _build(name: str) -> Optional[MarketDataProvider]:
    provider_type = _PROVIDER_TYPES.get(name)
    if provider_type is None:
        logger.warning("Unknown market data provider %r in MARKET_DATA_PROVIDERS; ignoring.", name)
        return None
    provider = provider_type()
    if not provider.is_available():
        return None
    return provider


def get_chain() -> List[MarketDataProvider]:
    """Configured providers, in priority order, instantiated once each."""
    with _lock:
        chain: List[MarketDataProvider] = []
        for name in settings.market_data_provider_chain:
            if name not in _providers:
                built = _build(name)
                if built is None:
                    continue
                _providers[name] = built
                _breakers[name] = CircuitBreaker(
                    name,
                    failure_threshold=settings.MARKET_DATA_CIRCUIT_FAIL_THRESHOLD,
                    reset_timeout=settings.MARKET_DATA_CIRCUIT_RESET_SECONDS,
                )
                _limiters[name] = RateLimiter(settings.MARKET_DATA_RATE_LIMIT_PER_SEC)
            chain.append(_providers[name])
        return chain


def reset_providers() -> None:
    """Drop cached providers and breaker state. For tests and config reloads."""
    with _lock:
        _providers.clear()
        _breakers.clear()
        _limiters.clear()


def call_with_failover(
    operation: str,
    fn: Callable[[MarketDataProvider], T],
    *,
    accept: Optional[Callable[[T], bool]] = None,
) -> T:
    """Run `fn` against each provider until one succeeds.

    `accept` lets a caller reject a technically-successful but useless result
    (an empty quote dict) and continue down the chain. Without it, a provider
    that returns `{}` for every symbol would satisfy the loop and the
    remaining providers would never be tried.
    """
    failures: Dict[str, str] = {}
    chain = get_chain()
    if not chain:
        raise AllProvidersFailed(operation, {"(none)": "no market data provider is configured"})

    for provider in chain:
        breaker = _breakers[provider.name]
        limiter = _limiters[provider.name]
        try:
            breaker.before_call()
        except CircuitOpenError as exc:
            market_data_requests.labels(provider.name, operation, "circuit_open").inc()
            failures[provider.name] = str(exc)
            continue
        finally:
            market_data_circuit_open.labels(provider.name).set(1 if breaker.is_open else 0)

        def attempt() -> T:
            limiter.acquire()
            return fn(provider)

        try:
            with timed(market_data_latency, provider=provider.name, kind=operation):
                result = retry_call(
                    attempt,
                    attempts=settings.MARKET_DATA_MAX_ATTEMPTS,
                    base_delay=settings.MARKET_DATA_BACKOFF_SECONDS,
                    label=f"{provider.name}.{operation}",
                )
        except Exception as exc:  # noqa: BLE001 -- failover is the handler
            breaker.record_failure()
            market_data_circuit_open.labels(provider.name).set(1 if breaker.is_open else 0)
            market_data_requests.labels(provider.name, operation, "error").inc()
            failures[provider.name] = str(exc)
            logger.warning(
                "Provider %s failed %s (%s); trying the next provider.",
                provider.name, operation, exc,
            )
            continue

        if accept is not None and not accept(result):
            # An empty answer is not a provider fault, so it must not count
            # towards opening the circuit -- a thinly-traded symbol nobody
            # covers would otherwise take a healthy provider offline.
            breaker.record_success()
            market_data_requests.labels(provider.name, operation, "empty").inc()
            failures[provider.name] = "returned no usable data"
            continue

        breaker.record_success()
        market_data_circuit_open.labels(provider.name).set(0)
        market_data_requests.labels(provider.name, operation, "ok").inc()
        return result

    raise AllProvidersFailed(operation, failures)


def provider_health() -> List[Dict[str, object]]:
    """Per-provider state for the health endpoint.

    `/health` previously reported `ok` while every market data call was
    failing, because it only checked the database. A green health check during
    a total data outage is worse than no health check.
    """
    get_chain()
    with _lock:
        return [
            {
                "provider": name,
                "available": True,
                "circuit_open": _breakers[name].is_open,
                "requires_credentials": _providers[name].requires_credentials,
            }
            for name in _providers
        ]


__all__ = [
    "AllProvidersFailed",
    "MarketDataProvider",
    "NewsItem",
    "Quote",
    "SecurityInfo",
    "call_with_failover",
    "get_chain",
    "provider_health",
    "reset_providers",
]
