"""Shared node scaffolding: timing, audit records and degradation reporting.

Every agent previously hand-rolled its own try/except, its own audit-record
dict and its own idea of what "degraded" meant. The results diverged: some
nodes recorded a failure reason, some swallowed it, and none recorded which
model, prompt version or data produced the output. That last gap is the
important one — the audit trail could say `stock_research` ran and summarize
it in a sentence, but not what the agent saw, what it sent, what answered, or
what it cost. For a system that sizes positions with an LLM, that is not a
reproducible record.

`node_run` is a context manager wrapping each node. It:

* times the node and stamps the audit record,
* catches exceptions so a failed node degrades the run rather than killing it,
* requires a degradation to state a *reason* and an *impact*, so an operator
  reading the trail learns what to fix and a client reading the report learns
  what the analysis is missing,
* carries model, prompt version, token counts and cost through to persistence.

The one rule this enforces on callers: a node that falls back must call
`ctx.degrade(...)`. Silent fallback is what made total capability loss
invisible for the entire life of the previous system.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterator, List, Optional

from db import utcnow
from logging_setup import get_logger, log_context
from metrics import agent_degraded, agent_duration, agent_errors, timed
from state import AgentRunRecord, Degradation

logger = get_logger(__name__)


@dataclass
class NodeContext:
    """Mutable record a node fills in as it runs."""

    node_name: str
    run_id: str
    started_at: str
    summary: str = ""
    status: str = "success"
    error_detail: Optional[str] = None
    degradations: List[Degradation] = field(default_factory=list)

    model_used: Optional[str] = None
    prompt_version: Optional[str] = None
    temperature: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    policy_version: Optional[int] = None
    data_provenance: Optional[Dict[str, Any]] = None
    input_snapshot: Dict[str, Any] = field(default_factory=dict)
    output_snapshot: Dict[str, Any] = field(default_factory=dict)

    _elapsed_ms: Optional[int] = None

    def degrade(self, reason: str, detail: str, impact: str) -> None:
        """Record that this node produced fallback output.

        `impact` is mandatory and is written for a reader, not a developer:
        it is the sentence that appears in the client report explaining what
        the analysis is missing. "LLM call failed" is not an impact; "the
        market regime call is a placeholder and did not inform these picks"
        is.
        """
        self.status = "degraded"
        self.degradations.append(
            Degradation(node=self.node_name, reason=reason, detail=detail[:500], impact=impact)
        )
        agent_degraded.labels(self.node_name, reason).inc()
        logger.warning(
            "[%s] degraded (%s): %s", self.node_name, reason, detail,
            extra={"run_id": self.run_id, "node": self.node_name},
        )

    def record_usage(self, usage) -> None:
        """Attach LLM token counts and cost from a tracked invocation."""
        if usage is None:
            return
        self.prompt_tokens = (self.prompt_tokens or 0) + usage.prompt_tokens
        self.completion_tokens = (self.completion_tokens or 0) + usage.completion_tokens
        self.cost_usd = round((self.cost_usd or 0.0) + usage.cost_usd, 6)

    @property
    def degraded(self) -> bool:
        return bool(self.degradations)

    def to_record(self) -> AgentRunRecord:
        return AgentRunRecord(
            node_name=self.node_name,
            started_at=self.started_at,
            completed_at=utcnow().isoformat(),
            status=self.status,
            summary=self.summary,
            error_detail=self.error_detail,
            degraded=self.degraded,
            model_used=self.model_used,
            prompt_version=self.prompt_version,
            temperature=self.temperature,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cost_usd=self.cost_usd,
            duration_ms=self._elapsed_ms,
            policy_version=self.policy_version,
            data_provenance=self.data_provenance,
            input_snapshot=self.input_snapshot or None,
            output_snapshot=self.output_snapshot or None,
        )


@contextmanager
def node_run(node_name: str, state: Dict[str, Any]) -> Iterator[NodeContext]:
    """Wrap a node body. Never raises; a failure becomes a degraded record.

    Nodes must not crash the graph -- a failed diagnostics pass should still
    produce a report that says diagnostics failed, not a 500 with nothing to
    show. But the old blanket `except Exception: return safe_defaults` also
    hid the failure completely. Here the exception is caught, recorded with
    its type and message, counted, and surfaced as a degradation the report
    is obliged to mention.
    """
    context = NodeContext(
        node_name=node_name,
        run_id=str(state.get("run_id") or "-"),
        started_at=utcnow().isoformat(),
    )
    policy = state.get("policy") or {}
    context.policy_version = policy.get("version")

    start = perf_counter()
    with log_context(run_id=context.run_id, node=node_name,
                     client_id=(state.get("client_profile") or {}).get("client_id")):
        try:
            with timed(agent_duration, node=node_name):
                yield context
        except Exception as exc:  # noqa: BLE001 -- a node must not kill the graph
            agent_errors.labels(node_name).inc()
            context.status = "error"
            context.error_detail = f"{type(exc).__name__}: {exc}"[:2000]
            context.degrade(
                reason="node_exception",
                detail=context.error_detail,
                impact=(
                    f"The {node_name.replace('_', ' ')} step failed entirely, so its "
                    "findings are absent from this analysis rather than merely reduced."
                ),
            )
            if not context.summary:
                context.summary = f"{node_name} failed: {type(exc).__name__}"
            logger.exception("[%s] raised; the run continues in degraded form.", node_name)
        finally:
            context._elapsed_ms = int((perf_counter() - start) * 1000)


def finish(context: NodeContext, updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the state update a node returns.

    Always includes this node's own audit record and any degradations. Nodes
    return only their own entries -- the reducers concatenate -- because
    returning the accumulated list duplicates every prior record on each pass
    through the retry loop.
    """
    payload: Dict[str, Any] = dict(updates or {})
    payload["audit_trail"] = [context.to_record()]
    if context.degradations:
        payload["degradations"] = list(context.degradations)
    return payload


def summarize_degradations(degradations: List[Degradation]) -> str:
    """One paragraph a client can read, or empty when the run was clean."""
    if not degradations:
        return ""
    lines = [
        "This analysis ran with reduced information. Specifically:",
    ]
    seen = set()
    for degradation in degradations:
        impact = degradation.get("impact") or degradation.get("detail") or ""
        if impact in seen:
            continue
        seen.add(impact)
        lines.append(f"- {impact}")
    return "\n".join(lines)
