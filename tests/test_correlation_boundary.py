"""Unit tests for correlation ID propagation across the request and job boundary."""

from db import Job, SessionLocal
from services import jobs
from logging_setup import current_context


def test_correlation_id_preserved_on_enqueue_and_dict():
    """Verify correlation_id is persisted on the Job model and rendered in job_to_dict."""
    db = SessionLocal()
    try:
        job = jobs.enqueue(
            db,
            job_type="test_correlation_job",
            org_id=1,
            client_id=1,
            correlation_id="req-test-uuid-42",
        )
        db.commit()

        assert job.correlation_id == "req-test-uuid-42"
        assert (job.payload or {}).get("correlation_id") == "req-test-uuid-42"

        data = jobs.job_to_dict(job)
        assert data.get("correlation_id") == "req-test-uuid-42"
    finally:
        db.query(Job).filter(Job.job_type == "test_correlation_job").delete()
        db.commit()
        db.close()


def test_correlation_id_bound_during_job_execution():
    """Worker execution binds correlation_id to thread-local log context."""
    db = SessionLocal()
    captured_context = {}

    @jobs.register("test_context_capture")
    def handler(db_sess, job_row):
        nonlocal captured_context
        captured_context = current_context()
        return {"ok": True}

    worker = jobs.JobWorker(worker_count=1)
    try:
        job = jobs.enqueue(
            db,
            job_type="test_context_capture",
            org_id=1,
            client_id=1,
            correlation_id="trace-corr-99",
        )
        db.commit()

        # Run job directly through worker's _run
        worker._run(db, job)

        assert captured_context.get("request_id") == "trace-corr-99"
        assert captured_context.get("correlation_id") == "trace-corr-99"
        assert captured_context.get("job_id") == job.job_id
    finally:
        db.query(Job).filter(Job.job_type == "test_context_capture").delete()
        db.commit()
        db.close()
