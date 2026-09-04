"""LangGraph checkpointer selection.

The graph pauses at `approval_gate` via `interrupt()` and is resumed by a
*separate* HTTP request (`POST /api/v1/runs/{run_id}/approve`), possibly
minutes or hours later. The checkpoint is what carries the paused run across
that gap.

The previous `MemorySaver()` therefore had two failure modes that only show
up in a real deployment:

  * Any process restart (deploy, crash, autoscale event) silently discarded
    every pending approval. The resume call then fails with an opaque error
    and the client's run is unrecoverable.
  * Under `uvicorn --workers N`, the approve request is load-balanced to an
    arbitrary worker, which almost certainly is not the one holding the
    paused run in its private memory. Approvals fail ~(N-1)/N of the time.

This module returns a durable, shared checkpointer instead: Postgres when the
app is pointed at Postgres, SQLite otherwise. It falls back to MemorySaver
only if no durable option can be constructed, and logs loudly when it does.
"""

from __future__ import annotations

from logging_setup import get_logger

logger = get_logger(__name__)

_checkpointer = None


def _build_checkpointer():
    from config import settings

    db_url = settings.NEON_DATABASE_URL

    if db_url.startswith(("postgres://", "postgresql://")):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            normalized = db_url.replace("postgres://", "postgresql://", 1)
            saver = PostgresSaver.from_conn_string(normalized).__enter__()
            saver.setup()
            logger.info("Using PostgresSaver for LangGraph checkpoints.")
            return saver
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Postgres checkpointer unavailable (%s); falling back to SQLite at %s",
                exc, settings.CHECKPOINT_DB_PATH,
            )

    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        # check_same_thread=False because FastAPI serves sync endpoints from a
        # threadpool, so the connection is used from more than one thread.
        conn = sqlite3.connect(settings.CHECKPOINT_DB_PATH, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        logger.info("Using SqliteSaver for LangGraph checkpoints (%s).", settings.CHECKPOINT_DB_PATH)
        return saver
    except Exception as exc:  # noqa: BLE001
        from langgraph.checkpoint.memory import MemorySaver

        logger.error(
            "No durable checkpointer could be created (%s). Falling back to in-memory "
            "checkpoints: PENDING HUMAN APPROVALS WILL BE LOST ON RESTART and will not "
            "work across multiple workers. This is not safe for production.",
            exc,
        )
        return MemorySaver()


def get_checkpointer():
    """Process-wide singleton checkpointer."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = _build_checkpointer()
    return _checkpointer


def reset_checkpointer() -> None:
    """Used by tests to force a fresh checkpointer against a temp database."""
    global _checkpointer
    _checkpointer = None


def is_durable() -> bool:
    """True when the active checkpointer survives a process restart.

    The MemorySaver fallback logs an error when it is chosen, and a logged
    error in a deployment is something nobody reads until the incident. This
    turns the same fact into a value a readiness probe can fail on, because a
    replica whose paused approvals evaporate on the next deploy should not be
    taking traffic -- it will accept runs, pause them at the approval gate,
    and lose them.
    """
    saver = get_checkpointer()
    return type(saver).__name__ != "MemorySaver"


def probe() -> bool:
    """Can the checkpoint store actually be read right now?

    Constructing the saver is not proof it works: the SQLite file can be
    deleted or its volume unmounted, and a Postgres saver holds a connection
    that can be severed, both long after `setup()` succeeded. This performs a
    real read against the store.
    """
    try:
        saver = get_checkpointer()
        if type(saver).__name__ == "MemorySaver":
            return False
        # A thread id that will never exist. The point is to exercise the
        # store's read path and its connection, not to find a checkpoint --
        # an empty result is a healthy answer.
        saver.get_tuple({"configurable": {"thread_id": "__readiness_probe__"}})
        return True
    except Exception:  # noqa: BLE001 -- a probe reports, it does not raise
        logger.exception("Checkpointer readiness probe failed")
        return False
