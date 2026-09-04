"""Is this process's code running against the schema it expects?

A replica that starts with new code against a database still on the previous
migration is the most confusing failure a rollout produces. Every probe passes,
`/health` is green, and the process throws `UndefinedColumn` on the first
request that touches a new column -- so the symptom is a partial outage
localized to whichever endpoints happened to use the new field, and the cause
is three layers away from it.

Comparing the `alembic_version` row against the head revision in the migration
directory answers it in one query, before any traffic arrives. A readiness
probe that fails on a mismatch makes a rolling deploy stop after the first
batch instead of replacing every healthy replica with a broken one.

The comparison is against *head*, not "any known revision", and the direction
matters: a database ahead of the code is also a mismatch, and is the normal
state during a rollback. Reporting it lets the rollback be deliberate rather
than a surprise.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from logging_setup import get_logger

logger = get_logger(__name__)


def _script_directory():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = Path(__file__).resolve().parent
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    return ScriptDirectory.from_config(cfg)


@lru_cache(maxsize=1)
def _ancestry() -> Optional[dict]:
    """Every revision reachable from head, and the head set.

    Needed to tell *which side* is ahead. Comparing the two revision strings
    only says they differ; walking the chain says whether the applied revision
    is an ancestor of this build's head (a migration has not run) or unknown
    to this build entirely (the database is ahead, i.e. a rollback).
    """
    try:
        script = _script_directory()
        heads = list(script.get_heads())
        reachable = {rev.revision for rev in script.walk_revisions("base", "heads")}
        return {"heads": heads, "reachable": reachable}
    except Exception:  # noqa: BLE001
        logger.exception("Could not read the migration history")
        return None


@lru_cache(maxsize=1)
def head_revisions() -> Optional[List[str]]:
    """The head revision(s) of the migration directory shipped with this code.

    Cached: the answer is a property of the filesystem inside the image and
    cannot change while the process runs, and reading the whole migration
    directory on every readiness probe is a needless disk walk on a path that
    is polled every few seconds.

    Returns None -- not an empty list -- when the migration directory cannot
    be read at all, so callers can distinguish "no migrations" from "cannot
    tell", and decline to fail a probe on the second.
    """
    ancestry = _ancestry()
    return None if ancestry is None else list(ancestry["heads"])


def applied_revisions(db: Session) -> Optional[List[str]]:
    """The revision(s) recorded in the database's `alembic_version` table."""
    try:
        rows = db.execute(text("SELECT version_num FROM alembic_version")).fetchall()
        return sorted(row[0] for row in rows)
    except Exception:
        # The table is absent on a database created by `init_db()` rather than
        # by `alembic upgrade` -- which is the normal state for local
        # development and for the test suite. That is not a mismatch to fail
        # a probe on; it is a database that has never been under migration
        # control, and is reported as such.
        return None


def schema_status(db: Session) -> dict:
    """A structured comparison, for the readiness probe and the runbook."""
    head = head_revisions()
    applied = applied_revisions(db)

    if head is None:
        return {"state": "unknown", "detail": "migration directory unreadable"}
    if applied is None:
        return {
            "state": "unmanaged",
            "detail": "no alembic_version table; schema was created directly from the models",
            "head": sorted(head),
        }

    head_sorted = sorted(head)
    if applied == head_sorted:
        return {"state": "current", "head": head_sorted, "applied": applied}

    # Which side is ahead is worth naming, and it is an ancestry question, not
    # a string comparison: two revisions always differ, so "they are not equal"
    # says nothing about direction. An applied revision this build knows about
    # means a migration has not run yet; one it has never heard of means the
    # database is ahead, which is what a rollback looks like from here.
    ancestry = _ancestry()
    reachable = ancestry["reachable"] if ancestry else set()
    if all(rev in reachable for rev in applied):
        direction = "database_behind_code"
        detail = "run `alembic upgrade head` before serving traffic from this build"
    else:
        direction = "database_ahead_of_code"
        detail = (
            "the database carries a revision this build does not ship -- it was "
            "migrated by a newer deployment. Roll forward, or downgrade the "
            "database deliberately before serving from this build."
        )
    return {
        "state": direction,
        "head": head_sorted,
        "applied": applied,
        "detail": detail,
    }


def schema_is_current(db: Session) -> bool:
    """True when it is safe to serve traffic from this build.

    `unmanaged` counts as current on purpose. Local development and the test
    suite create their schema from the models, and failing readiness there
    would mean the probe is only usable in the one environment it is hardest
    to test in.
    """
    return schema_status(db)["state"] in ("current", "unmanaged")
