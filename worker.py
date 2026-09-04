"""Job worker entrypoint.

`JOB_WORKER_ENABLED=true` runs the worker threads inside the API process,
which is the right shape for local development: one command, one process, no
orchestration. It is the wrong shape for a deployment.

A graph run is minutes of work holding the GIL between I/O waits, in the same
interpreter serving requests. Sharing a process means a burst of analysis runs
adds latency to every unrelated API call, the two cannot be scaled apart when
only one of them is the bottleneck, and a deploy that restarts the API to ship
a dashboard change also kills every run in flight.

Running this module instead gives the worker its own process and its own
lifecycle. It shares the image and the code with the API -- the difference is
the command, not the build.

    python worker.py

Shutdown is the part worth reading. SIGTERM is what an orchestrator sends
before SIGKILL, and the default disposition terminates immediately, abandoning
whatever job was mid-run to be reclaimed as stale minutes later. The handler
here stops claiming new work and waits for in-flight jobs, up to the grace
period the platform allows.
"""

import signal
import sys
import threading

from config import settings
from db import SessionLocal, init_db
from logging_setup import configure_logging, get_logger
from services import jobs

logger = get_logger(__name__)

# Set by the signal handler; the main thread waits on it. A handler must not
# do the draining itself: it runs on the main thread, interrupting arbitrary
# code, and joining worker threads from inside it can deadlock against a lock
# the interrupted frame already holds.
_stop = threading.Event()


def _handle_signal(signum, _frame):
    logger.info("Received %s; draining.", signal.Signals(signum).name)
    _stop.set()


def main() -> int:
    configure_logging()

    problems = settings.validate_for_environment()
    if problems:
        for problem in problems:
            logger.error("Unsafe configuration: %s", problem)
        # The same refusal the API makes. A worker is not a lesser process
        # here -- it is the one that actually runs the agents, reaches the
        # LLM and writes recommendations, so booting it with placeholder
        # secrets is worse, not better.
        logger.error("Refusing to start in ENVIRONMENT=%s.", settings.ENVIRONMENT)
        return 1

    if not settings.JOB_WORKER_ENABLED:
        logger.error(
            "JOB_WORKER_ENABLED is false, so this process would start and claim "
            "nothing. Set it true here and false on the API replicas."
        )
        return 1

    init_db()

    # Jobs whose worker died -- an OOM kill, a node eviction, a SIGKILL after
    # the grace period -- sit in `running` with a heartbeat that stopped.
    # Reclaiming them at startup is what makes a crash cost one restart rather
    # than one lost run.
    db = SessionLocal()
    try:
        reaped = jobs.reap_stale_jobs(db)
        if reaped:
            logger.warning("Failed %d job(s) that exceeded the timeout.", reaped)
    finally:
        db.close()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handle_signal)

    worker = jobs.start_worker()
    if worker is None:
        logger.error("Worker did not start.")
        return 1

    logger.info(
        "Worker up: %d thread(s), job timeout %ds, LLM %s.",
        settings.JOB_WORKER_COUNT,
        settings.JOB_TIMEOUT_SECONDS,
        "enabled" if settings.llm_configured else "DISABLED (deterministic fallbacks)",
    )

    _stop.wait()

    # The grace period is the platform's, and overrunning it means SIGKILL --
    # which is exactly the abandoned-job case this exists to avoid. Waiting
    # slightly less than the configured period leaves room to log the outcome.
    logger.info("Draining: waiting up to %ss for in-flight jobs.",
                settings.WORKER_SHUTDOWN_GRACE_SECONDS)
    jobs.stop_worker(timeout=settings.WORKER_SHUTDOWN_GRACE_SECONDS)
    logger.info("Worker stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
