"""The dashboard's HTTP client.

Separated from `app.py` for one reason: nothing in the test suite could import
`app.py`. Solara components are not callable outside a render, so a module that
mixes UI and transport is untestable end to end -- which is how the dashboard
came to reference `settings.API_AUTH_KEY`, a setting deleted when authentication
moved to organisations and JWT sessions, and raise `AttributeError` on import
for as long as it took someone to run it by hand.

Everything here is a plain function over `httpx`, so the whole client is
exercised against a stub transport in CI.

Three things this client must get right, none of which the previous version
did:

**It logs in as a person.** The API separates proposing from approving: an
advisor may trigger a run, only a compliance user may approve one. A dashboard
holding a single shared key would either be unable to approve at all or would
let whoever holds the key approve their own work -- which is not a
human-in-the-loop, it is a rubber stamp with extra steps. The session carries
the operator's own role, so the UI can show the approve control only when their
role permits it rather than offering a button the server will refuse.

**A run is a job, not a request.** `POST /runs` returns 202 and a job id. The
graph takes minutes; the old client posted and waited on a 120-second timeout
that raced work which routinely outlasts it.

**Status codes are checked as ranges.** The old client compared against 200
exactly, so creating a client -- which correctly answers 201 -- was reported to
the operator as a failure while having succeeded on the server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_BASE_URL = os.environ.get("WEALTH_API_URL", "http://localhost:8000")

# The graph is slow but the individual calls are not; a run is polled, never
# waited on. The one genuinely slow call is the approval resume, which
# continues the graph inline.
DEFAULT_TIMEOUT = 20.0
APPROVE_TIMEOUT = 180.0


class ApiError(Exception):
    """An API call that failed, carrying a message fit to show an operator.

    FastAPI returns a `detail` that is a string for a raised HTTPException and
    a list of field errors for a validation failure. Rendering the list
    directly puts a Python repr in front of the user, so both shapes are
    flattened here rather than at each of a dozen call sites.
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    @classmethod
    def from_response(cls, response: httpx.Response) -> "ApiError":
        try:
            detail = response.json().get("detail")
        except Exception:  # noqa: BLE001 -- a non-JSON error body is still an error
            detail = None

        if isinstance(detail, list):
            parts = []
            for item in detail:
                if isinstance(item, dict):
                    location = ".".join(str(p) for p in item.get("loc", [])[1:])
                    msg = str(item.get("msg"))
                    parts.append(f"{location}: {msg}" if location else msg)
                else:
                    parts.append(str(item))
            message = "; ".join(parts)
        elif isinstance(detail, str):
            message = detail
        else:
            message = f"{response.status_code} {response.reason_phrase}".strip()
        return cls(message or "Request failed.", response.status_code)


@dataclass
class Session:
    """An authenticated operator session.

    Holds the bearer token and, importantly, the role -- so the UI decides
    what to *offer* using the same fact the server uses to decide what to
    allow.
    """

    base_url: str
    token: str
    user: Dict[str, Any] = field(default_factory=dict)

    @property
    def role(self) -> str:
        return str(self.user.get("role", "viewer"))

    @property
    def may_approve(self) -> bool:
        # Mirrors the server's `approval:decide` capability. A named property
        # rather than a comparison scattered through the UI, so the day a role
        # is added there is one place to change.
        return self.role in ("compliance", "admin")

    @property
    def may_run(self) -> bool:
        return self.role in ("advisor", "compliance", "admin")

    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


def _request(
    session: Optional[Session],
    method: str,
    path: str,
    *,
    base_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    client: Optional[httpx.Client] = None,
    **kwargs: Any,
) -> Any:
    """The one place an HTTP response becomes either data or an ApiError."""
    root = base_url or (session.base_url if session else DEFAULT_BASE_URL)
    headers = dict(kwargs.pop("headers", {}))
    if session is not None:
        headers.update(session.headers())

    url = f"{root.rstrip('/')}{path}"
    owned = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        response = http.request(method, url, headers=headers, timeout=timeout, **kwargs)
    except httpx.RequestError as exc:
        # A connection error and an API error are different problems with
        # different fixes -- "nothing is listening at this address" versus
        # "the service refused this". Collapsing them into one message sends
        # an operator to the wrong place.
        raise ApiError(f"Could not reach the API at {root}: {exc}") from exc
    finally:
        if owned:
            http.close()

    if response.status_code >= 400:
        raise ApiError.from_response(response)
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


# --- Auth --------------------------------------------------------------------


def login(
    org_slug: str,
    email: str,
    password: str,
    *,
    base_url: Optional[str] = None,
    client: Optional[httpx.Client] = None,
) -> Session:
    root = base_url or DEFAULT_BASE_URL
    payload = _request(
        None, "POST", "/api/v1/auth/login",
        base_url=root, client=client,
        json={"org_slug": org_slug, "email": email, "password": password},
    )
    session = Session(base_url=root, token=payload["access_token"])
    # The token's own claims are not used for display: /auth/me is the
    # server's view of who this is, and it is the view the authorization
    # checks are made against.
    session.user = _request(session, "GET", "/api/v1/auth/me", client=client) or {}
    return session


# --- Clients -----------------------------------------------------------------


def list_clients(session: Session, *, client: Optional[httpx.Client] = None) -> List[Dict[str, Any]]:
    return _request(session, "GET", "/api/v1/clients", client=client) or []


def create_client(
    session: Session, payload: Dict[str, Any], *, client: Optional[httpx.Client] = None
) -> Dict[str, Any]:
    return _request(session, "POST", "/api/v1/clients", json=payload, client=client)


def get_portfolio(
    session: Session, client_id: int, *, client: Optional[httpx.Client] = None
) -> Dict[str, Any]:
    return _request(session, "GET", f"/api/v1/clients/{client_id}/portfolio", client=client)


# --- Runs --------------------------------------------------------------------


def trigger_run(
    session: Session, client_id: int, *, client: Optional[httpx.Client] = None
) -> Dict[str, Any]:
    """Queue a run. Returns the job envelope, not a result."""
    return _request(session, "POST", f"/api/v1/clients/{client_id}/runs", client=client)


def get_job(
    session: Session, job_id: str, *, client: Optional[httpx.Client] = None
) -> Dict[str, Any]:
    return _request(session, "GET", f"/api/v1/jobs/{job_id}", client=client)


def get_report(
    session: Session, report_id: int, *, client: Optional[httpx.Client] = None
) -> Dict[str, Any]:
    return _request(session, "GET", f"/api/v1/reports/{report_id}", client=client)


def approve_run(
    session: Session,
    run_id: str,
    approved: bool,
    note: Optional[str] = None,
    *,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Resume a paused run.

    No `decided_by` in the body. The API takes the identity from the session,
    because an approver named in a request body is an approver anyone can
    name -- which is what made the previous audit trail forgeable by whoever
    could type a colleague's name.
    """
    return _request(
        session, "POST", f"/api/v1/runs/{run_id}/approve",
        json={"approved": approved, "note": note},
        timeout=APPROVE_TIMEOUT, client=client,
    )


# --- Presentation adapter ----------------------------------------------------

# A job in one of these states will never change again, so polling stops.
TERMINAL_JOB_STATES = ("succeeded", "failed", "cancelled", "timed_out")


def view_model(report: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a stored report into the shape the result panels read.

    The API nests the structured payload under `structured_payload` with the
    prose alongside it; the panels want one flat mapping. Reshaping here
    rather than in the components keeps the transport contract in one file, so
    an API field rename is a change to this function and not a hunt through
    the UI.
    """
    payload = report.get("structured_payload") or {}
    return {
        "status": "completed",
        "run_id": report.get("run_id"),
        "report_id": report.get("id"),
        "llm_enabled": report.get("llm_enabled"),
        "final_report": report.get("report_text"),
        "approval_state": report.get("approval_state"),
        "degraded": report.get("degraded"),
        "portfolio_diagnostics": payload.get("portfolio_diagnostics"),
        "market_regime": payload.get("market_regime"),
        "suitability_result": payload.get("suitability_result"),
        "tax_assessment": payload.get("tax_assessment"),
        "tax_blocked_recommendations": payload.get("tax_blocked_recommendations"),
        "rebalance_plan": payload.get("rebalance_plan"),
        "degradations": payload.get("degradations"),
        "policy": payload.get("policy"),
    }
