"""Unit tests for worker_health check probe."""

import socket
from datetime import timedelta
from db import Job, SessionLocal, utcnow
from config import settings
import worker_health


def test_worker_health_idle(monkeypatch):
    """When no jobs exist for the worker, check() returns True ('idle')."""
    db = SessionLocal()
    try:
        # Clear any existing jobs for this host
        db.query(Job).filter(Job.worker_id.like(f"{socket.gethostname()}-%")).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()

    healthy, detail = worker_health.check()
    assert healthy is True
    assert detail == "idle"


def test_worker_health_running_with_fresh_heartbeat():
    """A running job with a fresh heartbeat reports healthy."""
    db = SessionLocal()
    job_id = "test-job-healthy"
    try:
        job = Job(
            job_id=job_id,
            org_id=1,
            client_id=1,
            worker_id=f"{socket.gethostname()}-pid123",
            status="running",
            heartbeat_at=utcnow(),
        )
        db.add(job)
        db.commit()

        healthy, detail = worker_health.check()
        assert healthy is True
        assert "running test-job-healthy" in detail
    finally:
        db.query(Job).filter(Job.job_id == job_id).delete()
        db.commit()
        db.close()


def test_worker_health_running_stale_heartbeat():
    """A running job with heartbeat older than JOB_HEARTBEAT_STALE_SECONDS fails health check."""
    db = SessionLocal()
    job_id = "test-job-stale"
    try:
        stale_time = utcnow() - timedelta(seconds=settings.JOB_HEARTBEAT_STALE_SECONDS + 10)
        job = Job(
            job_id=job_id,
            org_id=1,
            client_id=1,
            worker_id=f"{socket.gethostname()}-pid456",
            status="running",
            heartbeat_at=stale_time,
        )
        db.add(job)
        db.commit()

        healthy, detail = worker_health.check()
        assert healthy is False
        assert "stale after" in detail
    finally:
        db.query(Job).filter(Job.job_id == job_id).delete()
        db.commit()
        db.close()


def test_worker_health_claimed_no_heartbeat():
    """A running job with null heartbeat fails health check."""
    db = SessionLocal()
    job_id = "test-job-null-hb"
    try:
        job = Job(
            job_id=job_id,
            org_id=1,
            client_id=1,
            worker_id=f"{socket.gethostname()}-pid789",
            status="running",
            heartbeat_at=None,
        )
        db.add(job)
        db.commit()

        healthy, detail = worker_health.check()
        assert healthy is False
        assert "never heartbeated" in detail
    finally:
        db.query(Job).filter(Job.job_id == job_id).delete()
        db.commit()
        db.close()
