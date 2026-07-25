"""Central logging configuration.

The agents previously narrated themselves with bare `print()` calls, which
means their output cannot be levelled, filtered, routed, or correlated with a
run in a deployed environment. Every module now takes a logger from
`get_logger(__name__)` instead.

`configure_logging()` is idempotent and is called from the API server, the
orchestrator's __main__ block, and the demo notebook.
"""

import logging
import sys

from config import settings

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-8s %(name)-28s %(message)s"


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, (level or settings.LOG_LEVEL).upper(), logging.INFO),
        format=_FORMAT,
        stream=sys.stdout,
    )
    # yfinance and its HTTP stack are extremely chatty at INFO and drown out
    # the agent narration that actually matters here.
    for noisy in ("yfinance", "peewee", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
