"""Durable job queue and background worker.

A graph run takes minutes: dozens of market-data calls, a news search, and up
to four LLM invocations with backoff between retries. The previous version ran
all of it inline in the HTTP request. That meant a request thread held for the
whole duration, a client-side timeout that raced the work, no progress, no
cancellation, and a deploy mid-run losing it silently -- the checkpoint
survived, but nothing ever resumed an orphaned run.

The design is deliberately boring: a `jobs` table and a worker thread pool, no
Redis, no Celery. That keeps local development a single process and deployment
a single container, and the durability properties that matter come from the
database either way. The table *is* the queue, so a worker that dies mid-run
stops heartbeating and its job is reclaimed by another worker rather than
being lost.

Claiming is the one subtle part. Two workers polling the same table will
otherwise pick up the same row; the claim is a conditional UPDATE that only
succeeds for one of them, verified by the affected row count.
"""

import os
import socket
import threading
import time
import traceback
import uuid
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import and_, or_, update
from sqlalchemy.orm import Session

from config import settings
from db import Job, SessionLocal, utcnow
from logging_setup import get_logger, log_context
from metrics import job_queue_depth, job_wait_seconds, jobs_enqueued, jobs_finished

logger = get_logger(__name__)

WORKER_ID = f"{socket.gethostname()}-{os.getpid()}"

# job_type -> handler. Handlers take (db, job) and return a JSON-serializable
# result. Registered rather than imported directly so a handler can live
# beside the code it drives without this module importing the world.
_HANDLERS: Dict[str, Callable[[Session, Job], Dict[str, Any]]] = {}


class JobCancelled(Exception):
    """Raised inside a handler when the job has been asked to stop."""


def register(job_type: str):
    def decorator(fn: Callable[[Session, Job], Dict[str, Any]]):
        _HANDLERS[job_type] = fn
        return fn

    return decorator


# --- Enqueue -----------------------------------------------------------------


def enqueue(
    db: Session,
    *,
    job_type: str,
    org_id: int,
    client_id: Optional[int] = None,
    payload: Optional[Dict[str, Any]] = None,
    requested_by_user_id: Optional[int] = None,
    priority: int = 100,
    max_attempts: int = 1,
    dedupe: bool = True,
) -> Job:
    """Queue a job. Caller commits.

    `dedupe` prevents a second analysis run for a client that already has one
    queued or running. Two concurrent runs for the same client would read the
    same portfolio, propose overlapping trades, and between them double the
    intended position -- the failure is silent and expensive, so the default
    is to refuse.
    """
    if dedupe and client_id is not None:
        existing = (
            db.query(Job)
            .filter(
                Job.client_id == client_id,
                Job.job_type == job_type,
                Job.status.in_(("queued", "running")),
            )
            .first()
        )
        if existing is not None:
            logger.info(
                "Job %s for client %s already %s; returning it instead of queueing a second.",
                job_type, client_id, existing.status,
            )
            return existing

    job = Job(
        job_id=str(uuid.uuid4()),
        org_id=org_id,
        client_id=client_id,
        job_type=job_type,
        status="queued",
        priority=priority,
        payload=payload or {},
        max_attempts=max_attempts,
        requested_by_user_id=requested_by_user_id,
    )
    db.add(job)
    db.flush()
    jobs_enqueued.labels(job_type).inc()
    logger.info("Queued %s job %s for client %s", job_type, job.job_id, client_id)
    return job


def request_cancel(db: Session, job: Job) -> bool:
    """Ask a job to stop. Returns False if it has already finished.

    Cooperative rather than forced: the worker checks the flag between steps.
    Killing a thread mid-run would leave half-applied database state, which is
    worse than letting the current step finish.
    """
    if job.status in ("succeeded", "failed", "cancelled", "timed_out"):
        return False
    job.cancel_requested = True
    if job.status == "queued":
        # Not yet claimed, so it can be cancelled outright.
        job.status = "cancelled"
        job.finished_at = utcnow()
        jobs_finished.labels(job.job_type, "cancelled").inc()
    db.flush()
    return True


# --- Claiming ----------------------------------------------------------------


def claim_next(db: Session, worker_id: str = WORKER_ID) -> Optional[Job]:
    """Atomically claim one runnable job, or return None.

    The conditional UPDATE is what makes this safe with several workers: both
    may select the same candidate row, but only one UPDATE matches the
    still-queued predicate, and `rowcount` tells the loser to try again.
    """
    stale_before = utcnow() - timedelta(seconds=settings.JOB_HEARTBEAT_STALE_SECONDS)

    candidate = (
        db.query(Job)
        .filter(
            or_(
                Job.status == "queued",
                # Reclaim a job whose worker died: claimed, still marked
                # running, but no longer heartbeating.
                and_(
                    Job.status == "running",
                    Job.heartbeat_at.isnot(None),
                    Job.heartbeat_at < stale_before,
                ),
            )
        )
        .order_by(Job.priority.asc(), Job.queued_at.asc())
        .first()
    )
    if candidate is None:
        return None

    now = utcnow()
    result = db.execute(
        update(Job)
        .where(
            Job.id == candidate.id,
            or_(
                Job.status == "queued",
                and_(Job.status == "running", Job.heartbeat_at < stale_before),
            ),
        )
        .values(
            status="running",
            worker_id=worker_id,
            started_at=candidate.started_at or now,
            heartbeat_at=now,
            attempts=Job.attempts + 1,
        )
    )
    db.commit()

    if result.rowcount != 1:
        # Another worker won the race. Not an error; the next poll picks up
        # whatever is left.
        return None

    db.refresh(candidate)
    if candidate.queued_at:
        job_wait_seconds.observe(max(0.0, (now - candidate.queued_at).total_seconds()))
    return candidate


def heartbeat(db: Session, job: Job, *, step: Optional[str] = None,
              progress: Optional[float] = None) -> bool:
    """Record liveness and progress. Returns False if cancellation was asked for.

    Handlers call this between steps: it is both the "I am still alive" signal
    that stops another worker reclaiming the job, and the cancellation check.
    """
    job.heartbeat_at = utcnow()
    if step is not None:
        job.current_step = step[:120]
    if progress is not None:
        job.progress_pct = max(0.0, min(100.0, progress))
    db.commit()
    db.refresh(job)
    return not job.cancel_requested


# --- Worker ------------------------------------------------------------------


class JobWorker:
    """Polls for jobs and runs them on a small thread pool.

    Threads rather than processes because the work is overwhelmingly I/O
    bound -- HTTP to market data providers and to the LLM -- so the GIL is not
    the constraint, and threads share the connection pool and the in-process
    caches that make a run affordable.
    """

    def __init__(self, worker_count: Optional[int] = None, poll_interval: Optional[float] = None):
        self.worker_count = worker_count or settings.JOB_WORKER_COUNT
        self.poll_interval = poll_interval or settings.JOB_POLL_INTERVAL_SECONDS
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for index in range(self.worker_count):
            thread = threading.Thread(
                target=self._loop, name=f"job-worker-{index}", daemon=True
            )
            thread.start()
            self._threads.append(thread)
        logger.info("Started %d job worker thread(s) as %s", self.worker_count, WORKER_ID)

    def stop(self, timeout: Optional[float] = None) -> None:
        """Signal the workers to stop and wait for in-flight jobs.

        Daemon threads mean the process can still exit if a job overruns, but
        the wait gives an in-flight run a chance to finish and commit rather
        than being reclaimed as stale by the next deploy.

        The budget is for the *pool*, not per thread. Joining each thread with
        the full timeout meant N threads could take N x timeout to drain, so a
        two-thread worker with a 25s budget could overrun a 30s grace period
        and be killed anyway -- turning a deliberate drain into the abrupt
        termination it was meant to replace.
        """
        budget = timeout if timeout is not None else settings.WORKER_SHUTDOWN_GRACE_SECONDS
        self._stop.set()
        deadline = time.monotonic() + budget
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        still_running = [t.name for t in self._threads if t.is_alive()]
        self._threads.clear()
        if still_running:
            # Named, because the next thing that happens is a SIGKILL and
            # these jobs will reappear as stale reclaims minutes later. The
            # log line is what connects the two events.
            logger.warning(
                "Job worker(s) %s still running after %.0fs; their jobs will be "
                "reclaimed by another worker once the heartbeat goes stale.",
                ", ".join(still_running), budget,
            )
        else:
            logger.info("Job workers stopped cleanly.")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                claimed = self._claim_and_run()
            except Exception:  # noqa: BLE001 -- a worker loop must never die
                logger.exception("Job worker loop error; continuing.")
                claimed = False
            if not claimed:
                self._stop.wait(self.poll_interval)

    def _claim_and_run(self) -> bool:
        db = SessionLocal()
        try:
            job = claim_next(db)
            if job is None:
                self._report_depth(db)
                return False
            self._run(db, job)
            return True
        finally:
            db.close()

    @staticmethod
    def _report_depth(db: Session) -> None:
        try:
            rows = (
                db.query(Job.job_type, Job.id)
                .filter(Job.status == "queued")
                .all()
            )
            counts: Dict[str, int] = {}
            for job_type, _ in rows:
                counts[job_type] = counts.get(job_type, 0) + 1
            for job_type in set(list(counts) + list(_HANDLERS)):
                job_queue_depth.labels(job_type).set(counts.get(job_type, 0))
        except Exception:  # noqa: BLE001 -- metrics must never break the loop
            pass

    def _run(self, db: Session, job: Job) -> None:
        handler = _HANDLERS.get(job.job_type)
        started = time.monotonic()

        with log_context(job_id=job.job_id, client_id=job.client_id, run_id=job.run_id):
            if handler is None:
                job.status = "failed"
                job.error = f"No handler registered for job type {job.job_type!r}."
                job.finished_at = utcnow()
                db.commit()
                jobs_finished.labels(job.job_type, "failed").inc()
                logger.error("No handler for job type %r", job.job_type)
                return

            logger.info("Running job %s (%s)", job.job_id, job.job_type)
            try:
                result = handler(db, job)
                elapsed = time.monotonic() - started

                db.refresh(job)
                if job.cancel_requested:
                    # The work finished, but a cancellation was requested
                    # while it ran. Record it truthfully as cancelled with the
                    # result attached rather than pretending either that it
                    # stopped or that nobody asked.
                    job.status = "cancelled"
                    job.result = result
                    jobs_finished.labels(job.job_type, "cancelled").inc()
                else:
                    job.status = "succeeded"
                    job.result = result
                    job.progress_pct = 100.0
                    jobs_finished.labels(job.job_type, "succeeded").inc()
                job.finished_at = utcnow()
                db.commit()
                logger.info("Job %s finished in %.1fs", job.job_id, elapsed)

            except JobCancelled:
                db.rollback()
                job.status = "cancelled"
                job.finished_at = utcnow()
                job.error = "Cancelled by request."
                db.commit()
                jobs_finished.labels(job.job_type, "cancelled").inc()
                logger.info("Job %s cancelled.", job.job_id)

            except Exception as exc:  # noqa: BLE001 -- recorded, not raised
                db.rollback()
                db.refresh(job)
                # Retry only while attempts remain, and only for jobs
                # configured to allow it. Retrying an analysis run that failed
                # for a deterministic reason just burns money reproducing the
                # same failure.
                if job.attempts < job.max_attempts:
                    job.status = "queued"
                    job.error = f"{type(exc).__name__}: {exc}"[:2000]
                    job.heartbeat_at = None
                    logger.warning(
                        "Job %s failed (attempt %d/%d); requeued: %s",
                        job.job_id, job.attempts, job.max_attempts, exc,
                    )
                else:
                    job.status = "failed"
                    job.error = (
                        f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
                    )[:8000]
                    job.finished_at = utcnow()
                    jobs_finished.labels(job.job_type, "failed").inc()
                    logger.exception("Job %s failed permanently.", job.job_id)
                db.commit()


_worker: Optional[JobWorker] = None


def start_worker() -> Optional[JobWorker]:
    """Start the in-process worker unless this replica is API-only."""
    global _worker
    if not settings.JOB_WORKER_ENABLED:
        logger.info("Job worker disabled on this process (JOB_WORKER_ENABLED=false).")
        return None
    if _worker is None:
        _worker = JobWorker()
        _worker.start()
    return _worker


def stop_worker(timeout: Optional[float] = None) -> None:
    """Stop the in-process worker, waiting `timeout` for in-flight jobs.

    The default comes from settings rather than the 10s literal that used to
    be hardcoded in `JobWorker.stop`, because the right value is a property of
    the deployment's termination grace period, not of this module.
    """
    global _worker
    if _worker is not None:
        _worker.stop(timeout=timeout if timeout is not None
                     else settings.WORKER_SHUTDOWN_GRACE_SECONDS)
        _worker = None


def reap_stale_jobs(db: Session) -> int:
    """Fail jobs that exceeded the timeout without finishing.

    A job whose worker vanished is reclaimed by `claim_next`. This handles the
    other case: a job that is genuinely stuck, holding a slot forever. Without
    it, a hung provider call silently consumes the worker pool.
    """
    cutoff = utcnow() - timedelta(seconds=settings.JOB_TIMEOUT_SECONDS)
    stuck = (
        db.query(Job)
        .filter(Job.status == "running", Job.started_at.isnot(None), Job.started_at < cutoff)
        .all()
    )
    for job in stuck:
        job.status = "timed_out"
        job.finished_at = utcnow()
        job.error = (
            f"Exceeded the {settings.JOB_TIMEOUT_SECONDS}s job timeout without completing."
        )
        jobs_finished.labels(job.job_type, "timed_out").inc()
        logger.error("Job %s timed out.", job.job_id)
    if stuck:
        db.commit()
    return len(stuck)


def job_to_dict(job: Job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "job_type": job.job_type,
        "status": job.status,
        "client_id": job.client_id,
        "run_id": job.run_id,
        "progress_pct": round(job.progress_pct or 0.0, 1),
        "current_step": job.current_step,
        "attempts": job.attempts,
        "queued_at": job.queued_at.isoformat() if job.queued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "error": job.error,
        "result": job.result,
        "cancel_requested": bool(job.cancel_requested),
    }
