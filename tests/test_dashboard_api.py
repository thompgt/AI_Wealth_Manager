"""The dashboard's transport layer, against a stub and against the real app.

The reason this file exists is the failure it would have caught: `app.py`
referenced a setting deleted months earlier and raised `AttributeError` on
import, and no test noticed because no test could import it. The last test
here imports the dashboard module, which is now the cheapest possible guard
against the same class of breakage.
"""

import httpx
import pytest

import dashboard_api as api


def stub(handler):
    """An httpx.Client whose transport answers from a function."""
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api")


def session(role="advisor"):
    return api.Session(base_url="http://api", token="t0ken", user={"role": role, "email": "a@b.c"})


# --- Errors ------------------------------------------------------------------


def test_a_201_is_success_not_an_error():
    """The bug this encodes: the old client compared status to 200 exactly.

    Creating a client answers 201, so every successful creation was shown to
    the operator as a failure -- while having succeeded on the server, which
    is the worst possible combination. They retry, and now there are two.
    """
    def handler(request):
        return httpx.Response(201, json={"id": 7, "name": "Ada"})

    with stub(handler) as http:
        created = api.create_client(session(), {"name": "Ada"}, client=http)
    assert created["id"] == 7


def test_a_string_detail_becomes_the_message():
    def handler(request):
        return httpx.Response(403, json={"detail": "Compliance role required."})

    with stub(handler) as http:
        with pytest.raises(api.ApiError) as caught:
            api.trigger_run(session(), 1, client=http)
    assert caught.value.message == "Compliance role required."
    assert caught.value.status_code == 403


def test_a_validation_detail_is_flattened_not_repr_ed():
    """FastAPI returns a list of field errors for a 422.

    Rendering it directly puts a Python list of dicts in front of an operator,
    which tells them nothing about which field they got wrong.
    """
    def handler(request):
        return httpx.Response(422, json={"detail": [
            {"loc": ["body", "net_worth"], "msg": "Input should be greater than or equal to 0"},
        ]})

    with stub(handler) as http:
        with pytest.raises(api.ApiError) as caught:
            api.create_client(session(), {}, client=http)
    assert caught.value.message == "net_worth: Input should be greater than or equal to 0"


def test_an_unreachable_api_says_so_rather_than_blaming_the_request():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with stub(handler) as http:
        with pytest.raises(api.ApiError) as caught:
            api.list_clients(session(), client=http)
    assert "Could not reach the API" in caught.value.message


def test_a_non_json_error_body_still_produces_a_message():
    """A 502 from a proxy is HTML, not JSON. It is still an error."""
    def handler(request):
        return httpx.Response(502, text="<html>Bad Gateway</html>")

    with stub(handler) as http:
        with pytest.raises(api.ApiError) as caught:
            api.list_clients(session(), client=http)
    assert caught.value.status_code == 502
    assert caught.value.message


# --- Auth and roles ----------------------------------------------------------


def test_login_carries_the_servers_view_of_the_user():
    """Not the token's claims -- /auth/me, which is what the checks use."""
    def handler(request):
        if request.url.path == "/api/v1/auth/login":
            return httpx.Response(200, json={"access_token": "abc", "token_type": "bearer"})
        assert request.headers["Authorization"] == "Bearer abc"
        return httpx.Response(200, json={"email": "c@firm.example", "role": "compliance"})

    with stub(handler) as http:
        result = api.login("firm", "c@firm.example", "pw", base_url="http://api", client=http)
    assert result.token == "abc"
    assert result.role == "compliance"


def test_only_compliance_and_admin_may_approve():
    """The UI must gate on the same fact the server enforces.

    An advisor approving their own run is not a human-in-the-loop, and a
    dashboard that offers the button anyway teaches the operator that the
    system is broken rather than that the control is working.
    """
    assert not session("viewer").may_approve
    assert not session("advisor").may_approve
    assert session("compliance").may_approve
    assert session("admin").may_approve

    assert not session("viewer").may_run
    assert session("advisor").may_run


def test_approve_does_not_send_a_decided_by_field():
    """An approver named in a request body is an approver anyone can name."""
    captured = {}

    def handler(request):
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"status": "completed", "report_id": 3})

    with stub(handler) as http:
        api.approve_run(session("compliance"), "run-1", True, note="ok", client=http)
    assert set(captured) == {"approved", "note"}
    assert "decided_by" not in captured


# --- Run lifecycle -----------------------------------------------------------


def test_triggering_a_run_returns_a_job_not_a_result():
    """The endpoint answers 202 with a job id; the graph takes minutes."""
    def handler(request):
        assert request.url.path == "/api/v1/clients/4/runs"
        return httpx.Response(202, json={"job_id": "job-1", "status": "queued",
                                         "poll": "/api/v1/jobs/job-1"})

    with stub(handler) as http:
        job = api.trigger_run(session(), 4, client=http)
    assert job["job_id"] == "job-1"
    assert job["status"] == "queued"


def test_terminal_states_cover_every_way_a_job_stops():
    """A state missing from this tuple is a UI that polls forever."""
    assert set(api.TERMINAL_JOB_STATES) == {"succeeded", "failed", "cancelled", "timed_out"}


def test_the_view_model_carries_what_was_withheld():
    """The withheld recommendations are the point of the system.

    A view model that flattens the survivors and drops the blocked list shows
    a run where a control fired as though nothing had been caught.
    """
    report = {
        "id": 12,
        "run_id": "run-9",
        "report_text": "prose",
        "llm_enabled": True,
        "approval_state": "approved",
        "structured_payload": {
            "portfolio_diagnostics": {"flaws": ["45% in one name"]},
            "suitability_result": {"adjusted_recommendations": [{"ticker": "VTI"}]},
            "tax_assessment": {"wash_sale_flags": ["XOM", "CVX"]},
            "tax_blocked_recommendations": ["XOM", "CVX"],
        },
    }
    view = api.view_model(report)
    assert view["status"] == "completed"
    assert view["final_report"] == "prose"
    assert view["tax_blocked_recommendations"] == ["XOM", "CVX"]
    assert view["portfolio_diagnostics"]["flaws"] == ["45% in one name"]


def test_the_view_model_survives_a_report_with_no_payload():
    """An older report, or one stored before a field existed."""
    view = api.view_model({"id": 1, "report_text": "prose"})
    assert view["final_report"] == "prose"
    assert view["tax_blocked_recommendations"] is None


# --- The guard that was missing ----------------------------------------------


def test_the_dashboard_module_imports():
    """The whole reason this file exists.

    `app.py` read `settings.API_AUTH_KEY` for however long it took someone to
    run the dashboard by hand after the auth rework deleted it. Importing the
    module is a two-line test and would have caught it on the same push.
    """
    import app

    assert hasattr(app, "Page")
    assert hasattr(app, "Layout")
