"""Prometheus metrics.

The failure mode this exists to catch is the quiet one. Every agent in this
system is written to degrade rather than crash: a dead market-data provider, a
retired LLM model, or an exhausted quota all produce a completed run with a
200 response and deterministic fallback output. Nothing in the HTTP status
codes distinguishes that from a healthy run. `agent_degraded_total` and
`llm_calls_total{outcome="fallback"}` do.

Metric names follow Prometheus convention: `_total` for counters, base units
(seconds, dollars) with no unit prefix, and labels kept low-cardinality —
never a client id, run id or ticker, all of which would produce an unbounded
number of time series.
"""

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator, Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST  # noqa: F401

# A private registry rather than the global default: the default registry is
# process-wide and shared with anything else that imports prometheus_client,
# which makes duplicate-registration errors in tests both likely and
# confusing.
REGISTRY = CollectorRegistry()


# --- HTTP --------------------------------------------------------------------

http_requests = Counter(
    "http_requests_total",
    "HTTP requests served.",
    ["method", "path", "status"],
    registry=REGISTRY,
)
http_latency = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "path"],
    registry=REGISTRY,
)

# --- Graph runs --------------------------------------------------------------

runs_started = Counter("analysis_runs_started_total", "Analysis runs started.", registry=REGISTRY)
runs_finished = Counter(
    "analysis_runs_finished_total",
    "Analysis runs that reached a terminal state.",
    ["outcome"],  # completed | pending_approval | failed | cancelled | timed_out
    registry=REGISTRY,
)
run_duration = Histogram(
    "analysis_run_duration_seconds",
    "Wall-clock duration of a full graph run.",
    buckets=(5, 15, 30, 60, 120, 300, 600, 1200),
    registry=REGISTRY,
)
agent_duration = Histogram(
    "agent_node_duration_seconds",
    "Per-node execution time.",
    ["node"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
    registry=REGISTRY,
)
agent_degraded = Counter(
    "agent_degraded_total",
    "Node executions that fell back to a deterministic path instead of their "
    "intended one. The system returns a 200 either way, so this is the only "
    "signal that output quality dropped.",
    ["node", "reason"],
    registry=REGISTRY,
)
agent_errors = Counter(
    "agent_errors_total", "Node executions that raised.", ["node"], registry=REGISTRY
)

# --- Guardrails --------------------------------------------------------------

guardrail_blocks = Counter(
    "guardrail_blocks_total",
    "Recommendations withheld by a guardrail, by which rule stopped them.",
    ["rule"],
    registry=REGISTRY,
)
approvals_requested = Counter(
    "approvals_requested_total", "Runs paused for human approval.", ["reason"], registry=REGISTRY
)
approvals_decided = Counter(
    "approvals_decided_total", "Human decisions recorded.", ["decision"], registry=REGISTRY
)

# --- LLM ---------------------------------------------------------------------

llm_calls = Counter(
    "llm_calls_total",
    "LLM invocations by outcome.",
    ["node", "outcome"],  # success | retried | fallback | unavailable
    registry=REGISTRY,
)
llm_tokens = Counter(
    "llm_tokens_total", "Tokens consumed.", ["node", "kind"], registry=REGISTRY
)
llm_cost = Counter(
    "llm_cost_usd_total", "Attributed LLM spend in USD.", ["node"], registry=REGISTRY
)
llm_latency = Histogram(
    "llm_call_duration_seconds",
    "LLM call latency including retries.",
    ["node"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80),
    registry=REGISTRY,
)
llm_budget_headroom = Gauge(
    "llm_budget_headroom_usd",
    "Unspent LLM budget on the run that most recently spent, in USD. "
    "`llm_cost_usd_total` is cumulative across every run and so cannot say "
    "whether any single run is near its LLM_RUN_BUDGET_USD cap -- and hitting "
    "that cap drops the remaining agents to their deterministic paths, which "
    "is a 200 response with worse output. This is the leading indicator for "
    "the degradation that `agent_degraded_total` reports after the fact.",
    registry=REGISTRY,
)


# --- Market data -------------------------------------------------------------

market_data_requests = Counter(
    "market_data_requests_total",
    "Provider calls by outcome.",
    ["provider", "kind", "outcome"],  # ok | error | circuit_open | cache_hit
    registry=REGISTRY,
)
market_data_latency = Histogram(
    "market_data_duration_seconds",
    "Provider call latency.",
    ["provider", "kind"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
    registry=REGISTRY,
)
market_data_circuit_open = Gauge(
    "market_data_circuit_open",
    "1 while a provider is out of rotation after repeated failures.",
    ["provider"],
    registry=REGISTRY,
)
stale_quotes = Counter(
    "stale_quotes_total",
    "Quotes served past the staleness threshold and used for sizing anyway.",
    registry=REGISTRY,
)

# --- Jobs and trading --------------------------------------------------------

jobs_enqueued = Counter("jobs_enqueued_total", "Jobs queued.", ["job_type"], registry=REGISTRY)
jobs_finished = Counter(
    "jobs_finished_total", "Jobs reaching a terminal state.", ["job_type", "status"], registry=REGISTRY
)
job_queue_depth = Gauge(
    "job_queue_depth", "Jobs waiting to be claimed.", ["job_type"], registry=REGISTRY
)
job_wait_seconds = Histogram(
    "job_queue_wait_seconds",
    "Time between enqueue and claim.",
    buckets=(1, 5, 15, 30, 60, 300, 900),
    registry=REGISTRY,
)

orders_placed = Counter(
    "orders_placed_total", "Orders submitted.", ["side", "status"], registry=REGISTRY
)
order_notional = Counter(
    "order_notional_usd_total", "Gross notional traded.", ["side"], registry=REGISTRY
)


@contextmanager
def timed(histogram: Histogram, **labels: str) -> Iterator[None]:
    """Observe elapsed seconds even when the block raises.

    Timing only the success path is how a p99 latency chart ends up looking
    healthy during an outage: the slow calls all fail, so none of them are
    recorded.
    """
    start = perf_counter()
    try:
        yield
    finally:
        target = histogram.labels(**labels) if labels else histogram
        target.observe(perf_counter() - start)


def render() -> bytes:
    return generate_latest(REGISTRY)


def normalize_path(path: str, route_template: Optional[str] = None) -> str:
    """Label HTTP metrics with the route template, not the concrete path.

    `/api/v1/clients/{client_id}` is one time series; `/api/v1/clients/8123`
    is one time series per client, which is how a metrics backend gets taken
    down by its own instrumentation.
    """
    return route_template or path
