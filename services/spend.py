"""Per-organisation LLM spend, and the daily ceiling on it.

`RunBudget` in `services/llm.py` caps what one run may spend. That is the wrong
unit for the bill. Nothing stopped a firm from starting a thousand runs, each
one dutifully finishing under its own dollar cap, and the run rate limit does
not help: it is a count, and a count times an unknown per-run cost is an
unknown. The two limits answer different questions -- "can one run go haywire"
and "can the account" -- and only the first was answered.

The check is at *enqueue*, not mid-run. Stopping a run halfway leaves a client
with a half-finished analysis and a report that cannot be written, having
already spent most of the money; refusing to start is the only point where the
answer is both cheap and complete. The per-run budget already handles the
haywire case from the inside.

Spend is summed from `agent_runs.cost_usd`, which every node writes, rather
than kept in a counter. A counter is one process's view and resets when it
restarts; the audit rows are the same numbers the invoice can be reconciled
against, and they are per-org because the table is.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from config import settings
from db import AgentRun
from logging_setup import get_logger

logger = get_logger(__name__)


class DailyBudgetExceeded(Exception):
    """This organisation has spent its daily model budget."""


def _window_start(now: Optional[datetime] = None) -> datetime:
    """The start of the current UTC day.

    A rolling 24-hour window would be defensible too, but a calendar day is
    what an operator raising a limit is thinking in, and what a finance team
    reconciles against. UTC rather than a local zone so the boundary does not
    move twice a year.
    """
    now = now or datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def spend_today(db: Session, org_id: int, *, now: Optional[datetime] = None) -> float:
    """Total model spend attributed to this org since the start of the UTC day."""
    start = _window_start(now)
    # The stored timestamps are naive UTC (see db.utcnow), so the comparison
    # value is stripped of its tzinfo rather than the column being converted
    # -- converting in SQL would make the index on started_at unusable.
    total = (
        db.query(func.sum(AgentRun.cost_usd))
        .filter(AgentRun.org_id == org_id, AgentRun.started_at >= start.replace(tzinfo=None))
        .scalar()
    )
    return float(total or 0.0)


def daily_limit() -> float:
    return float(settings.LLM_ORG_DAILY_BUDGET_USD)


def status(db: Session, org_id: int, *, now: Optional[datetime] = None) -> Dict[str, object]:
    """Spend, limit and headroom -- for the API and for an operator."""
    limit = daily_limit()
    spent = spend_today(db, org_id, now=now)
    return {
        "spent_usd": round(spent, 4),
        "limit_usd": limit,
        "remaining_usd": None if limit <= 0 else round(max(0.0, limit - spent), 4),
        "unlimited": limit <= 0,
        "window_start": _window_start(now).isoformat(),
        "window_end": (_window_start(now) + timedelta(days=1)).isoformat(),
    }


def check(db: Session, org_id: int, *, now: Optional[datetime] = None) -> None:
    """Raise if this organisation may not start another run today.

    Checked against spend already *recorded*, so a run in flight has not yet
    contributed. That undercounts by at most one run's budget, which is the
    right way to be wrong here: the alternative is reserving spend up front
    and having to release it correctly on every failure path, and a
    reservation leaked by a crashed worker locks a firm out of its own
    account until someone notices.
    """
    limit = daily_limit()
    if limit <= 0:
        return  # non-positive means unlimited, matching RunBudget

    spent = spend_today(db, org_id, now=now)
    if spent < limit:
        return

    logger.warning(
        "Org %s has spent $%.2f against a $%.2f daily model budget; refusing new runs.",
        org_id, spent, limit,
    )
    raise DailyBudgetExceeded(
        f"This organisation has used ${spent:.2f} of its ${limit:.2f} daily model "
        f"budget. Runs already queued will finish. The budget resets at "
        f"{(_window_start(now) + timedelta(days=1)).strftime('%Y-%m-%d %H:%M UTC')}."
    )
