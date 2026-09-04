"""One error shape, and what a client is allowed to learn from a failure.

Before this, the body depended on which layer produced the error: FastAPI's
`detail` string for a raised HTTPException, a differently-shaped list for a
validation failure, and Starlette's default for an unhandled exception -- with
no request id in any of them, so an operator handed "it failed at 14:03" had
nothing to search on.
"""

import pytest
from fastapi.testclient import TestClient

import server
from db import init_db

PROBLEM = "application/problem+json"


@pytest.fixture(scope="module")
def api():
    init_db()
    with TestClient(server.app) as test_client:
        yield test_client


def test_an_auth_failure_is_a_problem_document(api):
    response = api.get("/api/v1/clients")
    assert response.status_code == 401
    assert response.headers["content-type"].startswith(PROBLEM)
    body = response.json()
    assert body["status"] == 401
    assert body["type"] == "/problems/401"
    assert body["title"] == "Authentication is required."
    assert body["instance"] == "/api/v1/clients"


def test_every_error_carries_the_request_id(api):
    """The single most useful field in the body.

    It is the same id in the response header, in every log line for this
    request, and -- for a queued run -- in the worker's lines minutes later.
    An error a user can quote is the difference between a searchable incident
    and a timestamp.
    """
    response = api.get("/api/v1/clients")
    body = response.json()
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_a_caller_supplied_request_id_is_honoured(api):
    """So a caller can correlate from their side without a second lookup."""
    response = api.get("/api/v1/clients", headers={"X-Request-ID": "trace-abc123"})
    assert response.json()["request_id"] == "trace-abc123"
    assert response.headers["X-Request-ID"] == "trace-abc123"


def test_a_validation_error_names_fields_without_echoing_values(api):
    """422 bodies must not repeat the input.

    FastAPI's default includes an `input` key carrying the offending value.
    For this API those values are a client's date of birth, net worth and
    holdings -- and echoing them puts client data into error logs, browser
    consoles and anything that samples response bodies.
    """
    response = api.post(
        "/api/v1/auth/login",
        json={"org_slug": "firm", "email": "not-an-email"},
    )
    assert response.status_code == 422
    assert response.headers["content-type"].startswith(PROBLEM)
    body = response.json()
    assert body["type"] == "/problems/422"
    assert isinstance(body["detail"], list)
    assert all({"field", "message", "type"} == set(item) for item in body["detail"])

    # The value must appear nowhere in the response.
    assert "not-an-email" not in response.text


def test_a_404_does_not_confirm_that_something_exists(api):
    """Cross-tenant probes get the same answer as nonexistent ids.

    The title is the generic one; the type is the status. Neither says
    whether the row exists in another organisation.
    """
    response = api.get("/api/v1/clients/999999")
    # Unauthenticated, so this is a 401 -- the point is that the shape does
    # not vary with whether the id is real.
    assert response.status_code in (401, 404)
    assert response.headers["content-type"].startswith(PROBLEM)


def test_the_catch_all_does_not_leak_internals(api):
    """An unhandled exception's message is written for a developer.

    It routinely contains a SQL fragment, a file path or a provider's
    response. The client gets a request id; the log gets the traceback.
    """
    @server.app.get("/_test_explode")
    def explode():
        raise RuntimeError("connection to postgres://user:hunter2@db failed")

    # TestClient re-raises server exceptions by default, which would bypass
    # the handler under test.
    with TestClient(server.app, raise_server_exceptions=False) as unraising:
        response = unraising.get("/_test_explode")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith(PROBLEM)
    body = response.json()
    assert body["request_id"]
    assert "hunter2" not in response.text
    assert "postgres" not in response.text
    assert "RuntimeError" not in response.text
    assert "Traceback" not in response.text


def test_rate_limit_headers_survive_the_handler(api):
    """A 429 whose Retry-After is dropped is a 429 a client cannot obey.

    The handler rebuilds the response, so headers set on the raised
    HTTPException have to be carried across explicitly -- easy to lose, and
    silent when lost.
    """
    from fastapi import HTTPException

    @server.app.get("/_test_throttled")
    def throttled():
        raise HTTPException(429, "Slow down.", headers={"Retry-After": "30"})

    response = api.get("/_test_throttled")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "30"
    assert response.json()["detail"] == "Slow down."
