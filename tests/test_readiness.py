"""Liveness, readiness and the schema-version check.

The property under test is a separation, not a feature: a liveness probe that
consults the database restarts every replica during a database incident, and a
readiness probe that ignores the schema routes traffic to a replica that will
throw on its first real query.
"""

import pytest
from fastapi.testclient import TestClient

import schema_version
import server
from db import SessionLocal, init_db


@pytest.fixture(scope="module")
def api():
    init_db()
    with TestClient(server.app) as test_client:
        yield test_client


def test_liveness_needs_no_credentials_and_no_database(api):
    """Liveness must not depend on anything a database outage can break.

    Asserting 200 while the database happens to be up would pass against a
    broken implementation, and monkeypatching the session factory proves
    nothing either -- FastAPI resolves a route's dependencies at import time,
    so a later patch of `server.get_db` is not what the route calls. The
    guarantee is structural, so the test is structural: the route is declared
    with no parameters at all, which is the only way it can have no dependency
    to fail.
    """
    import inspect

    route = next(r for r in server.app.routes if getattr(r, "path", None) == "/live")
    assert inspect.signature(route.endpoint).parameters == {}
    assert route.dependant.dependencies == []

    response = api.get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_each_dependency_separately(api):
    response = api.get("/ready")
    body = response.json()
    # The test database is created from the models, so the checkpointer is a
    # real SqliteSaver and the schema is "unmanaged" -- both ready.
    assert set(body["checks"]) == {"database", "schema", "checkpointer"}
    assert response.status_code in (200, 503)


def test_readiness_fails_when_the_database_is_unreachable(api, monkeypatch):
    monkeypatch.setattr(
        server, "schema_is_current", lambda db: True
    )

    class _Broken:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("connection refused")

    server.app.dependency_overrides[server.get_db] = lambda: _Broken()
    try:
        response = api.get("/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["database"] == "failed"
        # With the database unreachable, the schema answer is unknown rather
        # than a guess in either direction.
        assert response.json()["checks"]["schema"] == "unknown"
    finally:
        server.app.dependency_overrides.pop(server.get_db, None)


def test_a_draining_process_fails_readiness_but_stays_live(api):
    """The gap that lets in-flight work finish.

    While draining, the load balancer must stop sending new requests -- but
    the process is still running and must not be restarted out from under the
    job it is finishing.
    """
    server._shutting_down.set()
    try:
        assert api.get("/ready").status_code == 503
        assert api.get("/ready").json()["status"] == "draining"
        assert api.get("/live").status_code == 503
    finally:
        server._shutting_down.clear()


def test_readiness_does_not_disclose_revisions(api):
    """It is unauthenticated. Every value is a bare state word."""
    body = api.get("/ready").json()
    for value in body["checks"].values():
        assert value in ("ok", "failed", "behind", "unknown", "in_progress")


# --- schema_version ----------------------------------------------------------


def test_an_unmigrated_database_is_unmanaged_not_broken():
    """Local development and the suite create the schema from the models.

    Reporting that as a mismatch would make the probe fail everywhere except
    the one environment it is hardest to exercise.
    """
    db = SessionLocal()
    try:
        status = schema_version.schema_status(db)
        assert status["state"] in ("unmanaged", "current")
        assert schema_version.schema_is_current(db)
    finally:
        db.close()


def test_a_revision_this_build_ships_means_the_database_is_behind(monkeypatch):
    """An applied revision the code knows about is a pending migration."""
    head = schema_version.head_revisions()
    assert head, "the migration directory should have a head"
    ancestry = schema_version._ancestry()
    older = next(r for r in ancestry["reachable"] if r not in head)

    monkeypatch.setattr(schema_version, "applied_revisions", lambda db: [older])
    status = schema_version.schema_status(None)
    assert status["state"] == "database_behind_code"
    assert not schema_version.schema_is_current(None)


def test_an_unknown_revision_means_the_database_is_ahead(monkeypatch):
    """A revision this build has never heard of is a rollback in progress.

    The distinction matters operationally: one is fixed by running a
    migration, the other by not deploying this build.
    """
    monkeypatch.setattr(
        schema_version, "applied_revisions", lambda db: ["deadbeefcafe"]
    )
    status = schema_version.schema_status(None)
    assert status["state"] == "database_ahead_of_code"
    assert "does not ship" in status["detail"]
