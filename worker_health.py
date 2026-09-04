"""Container healthcheck for the job worker.

The worker serves no HTTP, so there is no port to probe -- and "the process is
running" is not the property that matters. The failure this catches is the
worker that is alive but no longer working: every poll thread dead from an
error the loop swallowed, or a database connection that fails on every claim
attempt. Such a process satisfies a process-liveness check forever while the
queue grows behind it.

The question asked instead is whether this worker has *touched the database
recently*, which is the only externally visible evidence that its loop is
turning. Exit 0 for healthy, 1 for not -- the contract Docker and Kubernetes
`exec` probes expect.
"""

import socket
import sys

from config import settings
from db import Job, SessionLocal, utcnow

# Deliberately NOT services.jobs.WORKER_ID, which is `hostname-pid`. This
# process is a *different* pid in the same container, so matching on the full
# worker id finds nothing, reports "idle" forever, and gives a healthy answer
# for a container whose worker is dead -- a check that can only pass.
#
# The hostname is the container, and is what identifies "the worker running
# here" across processes. Every worker id this container produces starts with
# it.
_HOST_PREFIX = f"{socket.gethostname()}-"


def check() -> tuple[bool, str]:
    db = SessionLocal()
    try:
        # A successful query proves the pool is alive and the credentials
        # still work, which is most of what can break between a worker and
        # its queue.
        recent = (
            db.query(Job)
            .filter(
                Job.worker_id.like(f"{_HOST_PREFIX}%"),
                Job.status == "running",
            )
            .order_by(Job.heartbeat_at.desc())
            .first()
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"database unreachable: {exc}"
    finally:
        db.close()

    if recent is None:
        # No job in flight. That is the normal state of an idle worker and
        # must not fail the check -- a queue with nothing in it is not a
        # broken worker, and failing here would restart every idle worker on
        # a quiet afternoon.
        return True, "idle"

    if recent.heartbeat_at is None:
        return False, f"job {recent.job_id} claimed but never heartbeated"

    age = (utcnow() - recent.heartbeat_at).total_seconds()
    if age > settings.JOB_HEARTBEAT_STALE_SECONDS:
        return False, (
            f"job {recent.job_id} last heartbeated {age:.0f}s ago "
            f"(stale after {settings.JOB_HEARTBEAT_STALE_SECONDS}s)"
        )
    return True, f"running {recent.job_id}, heartbeat {age:.0f}s ago"


if __name__ == "__main__":
    healthy, detail = check()
    print(detail)
    sys.exit(0 if healthy else 1)
