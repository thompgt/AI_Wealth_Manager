"""Unit tests for dead-worker job reclamation and heartbeat renewal."""

import time
from datetime import timedelta
import pytest
from db import Job, SessionLocal, utcnow
from config import settings
from services import jobs


def test_reclaim_orphaned_job_requeues_when_attempts_remain():
    """Stale running job with attempts < max_attempts is reclaimed back to queued."""
    db = SessionLocal()
    job_id = "test-reclaim-requeue"
    try:
        stale_time = utcnow() - timedelta(seconds=settings.JOB_HEARTBEAT_STALE_SECONDS + 30)
        job = Job(
            job_id=job_id,
            org_id=1,
            client_id=1,
            job_type="analysis",
            worker_id="dead-worker-1",
            status="running",
            attempts=1,
            max_attempts=3,
            started_at=stale_time,
            heartbeat_at=stale_time,
        )
        db.add(job)
        db.commit()

        reclaimed = jobs.reclaim_orphaned_jobs(db, max_stale_seconds=settings.JOB_HEARTBEAT_STALE_SECONDS)
        assert reclaimed == 1

        db.refresh(job)
        assert job.status == "queued"
        assert job.worker_id is None
        assert job.heartbeat_at is None
        assert "Reclaimed from dead worker" in (job.error or "")
    finally:
        db.query(Job).filter(Job.job_id == job_id).delete()
        db.commit()
        db.close()


def test_reclaim_orphaned_job_requeues_when_null_heartbeat():
    """Worker died immediately before first heartbeat: started_at is stale, heartbeat_at is None."""
    db = SessionLocal()
    job_id = "test-reclaim-null-hb"
    try:
        stale_time = utcnow() - timedelta(seconds=settings.JOB_HEARTBEAT_STALE_SECONDS + 30)
        job = Job(
            job_id=job_id,
            org_id=1,
            client_id=1,
            job_type="analysis",
            worker_id="dead-worker-2",
            status="running",
            attempts=1,
            max_attempts=3,
            started_at=stale_time,
            heartbeat_at=None,
        )
        db.add(job)
        db.commit()

        reclaimed = jobs.reclaim_orphaned_jobs(db, max_stale_seconds=settings.JOB_HEARTBEAT_STALE_SECONDS)
        assert reclaimed == 1

        db.refresh(job)
        assert job.status == "queued"
        assert job.worker_id is None
    finally:
        db.query(Job).filter(Job.job_id == job_id).delete()
        db.commit()
        db.close()


def test_reclaim_orphaned_job_fails_when_max_attempts_reached():
    """Stale running job with attempts >= max_attempts is failed permanently."""
    db = SessionLocal()
    job_id = "test-reclaim-fail-max-attempts"
    try:
        stale_time = utcnow() - timedelta(seconds=settings.JOB_HEARTBEAT_STALE_SECONDS + 30)
        job = Job(
            job_id=job_id,
            org_id=1,
            client_id=1,
            job_type="analysis",
            worker_id="dead-worker-3",
            status="running",
            attempts=3,
            max_attempts=3,
            started_at=stale_time,
            heartbeat_at=stale_time,
        )
        db.add(job)
        db.commit()

        reclaimed = jobs.reclaim_orphaned_jobs(db, max_stale_seconds=settings.JOB_HEARTBEAT_STALE_SECONDS)
        assert reclaimed == 1

        db.refresh(job)
        assert job.status == "failed"
        assert "exceeded max attempts" in (job.error or "")
        assert job.finished_at is not None
    finally:
        db.query(Job).filter(Job.job_id == job_id).delete()
        db.commit()
        db.close()


def test_reclaim_orphaned_job_ignores_fresh_jobs():
    """Active running job with recent heartbeat is NOT reclaimed."""
    db = SessionLocal()
    job_id = "test-reclaim-active-alive"
    try:
        job = Job(
            job_id=job_id,
            org_id=1,
            client_id=1,
            job_type="analysis",
            worker_id="live-worker-1",
            status="running",
            attempts=1,
            max_attempts=3,
            started_at=utcnow(),
            heartbeat_at=utcnow(),
        )
        db.add(job)
        db.commit()

        reclaimed = jobs.reclaim_orphaned_jobs(db, max_stale_seconds=settings.JOB_HEARTBEAT_STALE_SECONDS)
        assert reclaimed == 0

        db.refresh(job)
        assert job.status == "running"
        assert job.worker_id == "live-worker-1"
    finally:
        db.query(Job).filter(Job.job_id == job_id).delete()
        db.commit()
        db.close()


def test_job_heartbeat_runner_updates_db():
    """JobHeartbeatRunner periodically refreshes heartbeat_at while running."""
    db = SessionLocal()
    job_id = "test-hb-runner"
    try:
        initial_time = utcnow() - timedelta(seconds=10)
        job = Job(
            job_id=job_id,
            org_id=1,
            client_id=1,
            job_type="analysis",
            worker_id="test-runner-worker",
            status="running",
            heartbeat_at=initial_time,
        )
        db.add(job)
        db.commit()

        with jobs.JobHeartbeatRunner(job_id=job_id, interval=0.05):
            time.sleep(0.15)

        db.refresh(job)
        assert job.heartbeat_at is not None
        assert job.heartbeat_at > initial_time
    finally:
        db.query(Job).filter(Job.job_id == job_id).delete()
        db.commit()
        db.close()
