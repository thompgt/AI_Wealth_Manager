"""Central logging configuration and run correlation.

Two problems this solves beyond "we have logs".

**Correlation.** A single analysis run emits log lines from six agents, three
service layers and a background worker, interleaved with every other run in
flight. Without a run id stamped on every record you cannot reconstruct one
run from a production log — you can only grep for a UUID that individual call
sites happened to remember to include. `log_context()` binds run/client/user
ids to the current thread and a filter attaches them to every record emitted
inside that scope, including records from libraries that know nothing about
this system.

**Machine readability.** Text formatting is right for a terminal and wrong for
a log aggregator. The format follows the environment (`LOG_FORMAT=auto`), so
local work stays readable and deployments emit JSON without a code change.

`configure_logging()` is idempotent and is called from the API server, the job
worker, the orchestrator's __main__ block, and the demo notebook.
"""

import json
import logging
import sys
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from config import settings

_CONFIGURED = False

_TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)-28s [%(run_id)s] %(message)s"

# Thread-local rather than contextvars: the job worker and FastAPI's sync
# endpoints both run on threads, and the graph itself is synchronous. A
# contextvar would not propagate into the threadpool that actually executes
# the agents.
_local = threading.local()

# Attributes LogRecord always has. Anything else on a record was put there by
# a caller passing `extra=`, and belongs in the structured payload.
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


def current_context() -> Dict[str, Any]:
    return dict(getattr(_local, "context", {}) or {})


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Bind fields to every log record emitted on this thread inside the block.

    Nested scopes merge rather than replace, so a node can add `node=` on top
    of the run-level `run_id=` without discarding it.
    """
    previous = current_context()
    merged = {**previous, **{k: v for k, v in fields.items() if v is not None}}
    _local.context = merged
    try:
        yield
    finally:
        _local.context = previous


class _ContextFilter(logging.Filter):
    """Attaches the bound context to each record.

    Also guarantees `run_id` exists, because the text formatter references it
    and a missing attribute would raise inside logging itself — turning a log
    line into a crash.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = current_context()
        for key, value in ctx.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        return True


class _JsonFormatter(logging.Formatter):
    """One JSON object per line.

    Exceptions are rendered into the `exception` field rather than trailing
    newlines, because a multi-line log record is two records to most
    collectors and the traceback ends up orphaned from its message.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _RedactingFilter(logging.Filter):
    """Last line of defence against secrets in logs.

    Not a substitute for not logging them — it only catches the obvious
    shapes — but an API key pasted into an exception message is a real and
    recurring way credentials end up in a log aggregator with a long
    retention policy.
    """

    _SENSITIVE_KEYS = ("password", "secret", "token", "api_key", "authorization", "key_hash")

    def filter(self, record: logging.LogRecord) -> bool:
        for key in list(record.__dict__):
            if any(marker in key.lower() for marker in self._SENSITIVE_KEYS):
                record.__dict__[key] = "***redacted***"
        return True


def configure_logging(level: Optional[str] = None, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved = getattr(logging, (level or settings.LOG_LEVEL).upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    if settings.effective_log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    handler.addFilter(_ContextFilter())
    handler.addFilter(_RedactingFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(resolved)

    # The HTTP stack is extremely chatty at INFO and drowns out the agent
    # narration that actually matters here.
    for noisy in ("peewee", "urllib3", "httpx", "httpcore", "alembic", "primp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # yfinance logs a multi-line ERROR block for every symbol it cannot
    # resolve. Screening a wide universe legitimately touches names it has no
    # data for, and the provider layer already reports those outcomes with the
    # context of which run and which provider. Left at ERROR, a normal screen
    # buries the real failures under hundreds of expected ones.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    _CONFIGURED = True


def new_run_id() -> str:
    return str(uuid.uuid4())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
