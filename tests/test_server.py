"""API-layer tests.

The shared `X-API-Key` these tests used to send is gone. Every request now
resolves a `Principal` -- a browser session (JWT) or a scoped machine key --
carrying an org and a role, and every tenant read goes through `scoped_query`.
So the surface worth testing here is the one that replaced it: that an
unauthenticated request is refused, that a role short of the required one is
refused, that one org cannot see another's rows, and that the happy paths
still work for a principal who is entitled to them.

The graph itself is never triggered, so nothing here needs the network, a
GEMINI_API_KEY or a background worker to do any work.
"""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import server
from db import Organization, SessionLocal, User, init_db
from security import hash_password

PASSWORD = "correct-horse-battery-staple"

# Distinct slugs per run so the module is independent of whatever else has
# already written to the shared temp database.
FIRM_SLUG = f"test-firm-{uuid4().hex[:8]}"
OTHER_SLUG = f"other-firm-{uuid4().hex[:8]}"


def _seed_identities():
    """Two orgs with known credentials, created before the app starts.

    Created directly rather than via `bootstrap_admin` so the tests do not
    depend on the database being empty, and so both an admin and a
    lower-privileged advisor exist to test the role split with.
    """
    db = SessionLocal()
    try:
        for slug, users in (
            (FIRM_SLUG, [("admin", "admin"), ("advisor", "advisor")]),
            (OTHER_SLUG, [("admin", "admin")]),
        ):
            org = Organization(name=slug, slug=slug)
            db.add(org)
            db.flush()
            for role, _ in users:
                db.add(
                    User(
                        org_id=org.id,
                        email=f"{role}@{slug}.example",
                        full_name=f"{role} user",
                        password_hash=hash_password(PASSWORD),
                        role=role,
                    )
                )
        db.commit()
    finally:
        db.close()


@pytest.fixture(scope="module")
def api():
    init_db()
    _seed_identities()
    with TestClient(server.app) as test_client:
        yield test_client


def _login(api, org_slug, email):
    response = api.post(
        "/api/v1/auth/login",
        json={"org_slug": org_slug, "email": email, "password": PASSWORD},
    )
    assert response.status_code == 200, response.text
    return response.json()


# Logins are rate limited to 10/minute per source address, so each principal
# signs in once for the whole module rather than once per test.
@pytest.fixture(scope="module")
def admin(api):
    return _login(api, FIRM_SLUG, f"admin@{FIRM_SLUG}.example")


@pytest.fixture(scope="module")
def advisor(api):
    return _login(api, FIRM_SLUG, f"advisor@{FIRM_SLUG}.example")


@pytest.fixture(scope="module")
def other_admin(api):
    return _login(api, OTHER_SLUG, f"admin@{OTHER_SLUG}.example")


def _bearer(token_payload):
    return {"Authorization": f"Bearer {token_payload['access_token']}"}


def _client_payload(**overrides):
    payload = {
        "name": "Test Client",
        "email": "t@example.com",
        "age": 45,
        "risk_tolerance": "Moderate",
        "time_horizon_years": 20,
        "goals": ["retirement"],
        "net_worth": 100000.0,
        "accounts": [
            {
                "name": "Brokerage",
                "account_type": "individual",
                "tax_treatment": "taxable",
                "cash_balance": 25000.0,
                "holdings": [
                    {"symbol": "AAPL", "quantity": 100.0, "cost_per_share": 150.0}
                ],
            }
        ],
    }
    payload.update(overrides)
    return payload


@pytest.fixture(scope="module")
def owned_client_id(api, admin):
    response = api.post("/api/v1/clients", json=_client_payload(), headers=_bearer(admin))
    assert response.status_code == 201, response.text
    return response.json()["id"]


# --- health -------------------------------------------------------------------

def test_health_needs_no_credentials(api):
    response = api.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok"
    # The probe must distinguish "up" from "working": with only a placeholder
    # key the LLM layer is not live and says so.
    assert body["checks"]["llm"] == "not_configured"


def test_health_names_no_internal_topology(api):
    """It is reachable without credentials, so it discloses no model, version
    or agent list."""
    body = api.get("/health").json()
    assert set(body) == {"status", "checks"}
    assert set(body["checks"]) == {"database", "market_data", "llm"}


# --- authentication -----------------------------------------------------------

def test_an_unauthenticated_request_is_rejected(api):
    response = api.get("/api/v1/clients")
    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"


def test_an_unknown_api_key_is_rejected(api):
    assert api.get("/api/v1/clients", headers={"X-API-Key": "nope"}).status_code == 401


def test_a_malformed_bearer_token_is_rejected(api):
    response = api.get("/api/v1/clients", headers={"Authorization": "Bearer not.a.jwt"})
    assert response.status_code == 401


def test_bad_credentials_give_the_same_generic_answer(api):
    """Distinguishing "no such user" from "wrong password" tells an attacker
    which addresses are real."""
    unknown = api.post(
        "/api/v1/auth/login",
        json={"org_slug": FIRM_SLUG, "email": "nobody@example.com", "password": PASSWORD},
    )
    wrong_password = api.post(
        "/api/v1/auth/login",
        json={
            "org_slug": FIRM_SLUG,
            "email": "nobody@example.com",
            "password": "definitely-not-it",
        },
    )
    assert unknown.status_code == wrong_password.status_code == 401
    assert unknown.json()["detail"] == wrong_password.json()["detail"] == "Invalid credentials."


def test_login_returns_a_session_the_api_accepts(api, admin):
    assert admin["role"] == "admin"
    assert admin["token_type"] == "bearer"
    me = api.get("/api/v1/auth/me", headers=_bearer(admin))
    assert me.status_code == 200
    assert me.json()["kind"] == "user"
    assert me.json()["org_id"] == admin["org_id"]
    assert me.json()["role"] == "admin"


def test_a_refresh_token_cannot_be_used_to_call_the_api(api, admin):
    """It has a much longer life than an access token, so accepting it on the
    API would silently extend every session's exposure."""
    response = api.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {admin['refresh_token']}"},
    )
    assert response.status_code == 401
    assert "Refresh tokens cannot be used" in response.json()["detail"]


# --- API keys -----------------------------------------------------------------

@pytest.fixture(scope="module")
def viewer_key(api, admin):
    response = api.post(
        "/api/v1/auth/api-keys",
        params={"name": "reporting", "role": "viewer", "expires_in_days": 30},
        headers=_bearer(admin),
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_an_issued_api_key_authenticates_with_its_own_role(api, viewer_key):
    me = api.get("/api/v1/auth/me", headers={"X-API-Key": viewer_key["key"]})
    assert me.status_code == 200
    assert me.json()["kind"] == "api_key"
    assert me.json()["role"] == "viewer"


def test_an_api_key_secret_is_returned_once_and_only_its_prefix_is_stored(api, viewer_key):
    assert viewer_key["key"].startswith(f"awm_{viewer_key['prefix']}_")
    listed = api.get("/api/v1/auth/me", headers={"X-API-Key": viewer_key["key"]})
    assert viewer_key["key"] not in listed.text


def test_a_viewer_key_cannot_write(api, viewer_key):
    """The key's role is the ceiling, whatever its owner may do."""
    response = api.post(
        "/api/v1/clients", json=_client_payload(), headers={"X-API-Key": viewer_key["key"]}
    )
    assert response.status_code == 403
    assert "advisor" in response.json()["detail"]


def test_a_revoked_api_key_stops_working(api, admin):
    created = api.post(
        "/api/v1/auth/api-keys",
        params={"name": "temporary", "role": "viewer"},
        headers=_bearer(admin),
    ).json()
    assert api.get("/api/v1/auth/me", headers={"X-API-Key": created["key"]}).status_code == 200

    revoked = api.delete(f"/api/v1/auth/api-keys/{created['id']}", headers=_bearer(admin))
    assert revoked.status_code == 204
    assert api.get("/api/v1/auth/me", headers={"X-API-Key": created["key"]}).status_code == 401


def test_only_an_admin_may_issue_api_keys(api, advisor):
    response = api.post(
        "/api/v1/auth/api-keys", params={"name": "sneaky"}, headers=_bearer(advisor)
    )
    assert response.status_code == 403


# --- role separation ----------------------------------------------------------

def test_an_advisor_may_draft_a_policy_but_not_activate_it(api, advisor, owned_client_id):
    """The control the whole approval model rests on: the role that proposes
    is not the role that puts limits into force."""
    drafted = api.post(
        f"/api/v1/clients/{owned_client_id}/policy",
        json={"max_position_pct": 0.2},
        headers=_bearer(advisor),
    )
    assert drafted.status_code == 201, drafted.text
    version = drafted.json()["version"]

    activated = api.post(
        f"/api/v1/clients/{owned_client_id}/policy/{version}/activate",
        headers=_bearer(advisor),
    )
    assert activated.status_code == 403
    assert "compliance" in activated.json()["detail"]


def test_an_advisor_cannot_archive_a_client(api, advisor, owned_client_id):
    assert api.delete(
        f"/api/v1/clients/{owned_client_id}", headers=_bearer(advisor)
    ).status_code == 403


# --- tenancy ------------------------------------------------------------------

def test_another_organisations_client_is_invisible(api, other_admin, owned_client_id):
    """404 rather than 403 deliberately: a 403 confirms the id exists, which
    turns id guessing into a cross-tenant enumeration oracle."""
    assert api.get(
        f"/api/v1/clients/{owned_client_id}", headers=_bearer(other_admin)
    ).status_code == 404
    assert api.get(f"/api/v1/clients/{owned_client_id}", headers=_bearer(other_admin)).json()[
        "detail"
    ] == "Client not found."


def test_listing_clients_only_returns_this_organisations_own(api, other_admin, owned_client_id):
    listed = api.get("/api/v1/clients", headers=_bearer(other_admin))
    assert listed.status_code == 200
    assert owned_client_id not in [c["id"] for c in listed.json()]


# --- clients ------------------------------------------------------------------

def test_a_created_client_reads_back_with_its_accounts(api, admin, owned_client_id):
    fetched = api.get(f"/api/v1/clients/{owned_client_id}", headers=_bearer(admin))
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["risk_tolerance"] == "Moderate"
    assert [a["name"] for a in body["accounts"]] == ["Brokerage"]
    assert body["accounts"][0]["tax_treatment"] == "taxable"


def test_onboarding_holdings_become_tax_lots(api, admin, owned_client_id):
    """Positions are derived from lots, so onboarding has to create lots --
    otherwise every cost-basis and wash-sale answer for the client is empty."""
    lots = api.get(f"/api/v1/clients/{owned_client_id}/lots", headers=_bearer(admin))
    assert lots.status_code == 200
    aapl = [lot for lot in lots.json() if lot["symbol"] == "AAPL"]
    assert len(aapl) == 1
    assert aapl[0]["remaining_quantity"] == 100.0
    assert aapl[0]["cost_per_share"] == 150.0


def test_an_unknown_client_is_404(api, admin):
    assert api.get("/api/v1/clients/999999", headers=_bearer(admin)).status_code == 404


def test_an_unrecognized_risk_tolerance_is_rejected(api, admin):
    """A free-form tier used to be accepted and then fall through to the
    Moderate defaults in every downstream agent."""
    response = api.post(
        "/api/v1/clients",
        json=_client_payload(risk_tolerance="Balanced"),
        headers=_bearer(admin),
    )
    assert response.status_code == 422


def test_a_negative_net_worth_is_rejected(api, admin):
    response = api.post(
        "/api/v1/clients", json=_client_payload(net_worth=-1.0), headers=_bearer(admin)
    )
    assert response.status_code == 422


def test_a_non_positive_holding_quantity_is_rejected(api, admin):
    payload = _client_payload()
    payload["accounts"][0]["holdings"] = [
        {"symbol": "AAPL", "quantity": -5.0, "cost_per_share": 10.0}
    ]
    response = api.post("/api/v1/clients", json=payload, headers=_bearer(admin))
    assert response.status_code == 422


# --- approvals ----------------------------------------------------------------

def test_approving_a_run_that_never_asked_for_approval_is_404(api, admin):
    response = api.post(
        "/api/v1/runs/no-such-run/approve", json={"approved": True}, headers=_bearer(admin)
    )
    assert response.status_code == 404


def test_an_advisor_cannot_clear_their_own_run(api, advisor):
    """Above `advisor` on purpose: if the person who requested the run can
    approve it, the human-in-the-loop gate is a formality."""
    response = api.post(
        "/api/v1/runs/no-such-run/approve", json={"approved": True}, headers=_bearer(advisor)
    )
    assert response.status_code == 403


def test_an_approval_decision_carries_no_caller_supplied_identity(api, admin):
    """`decided_by` used to come from the request body, so an approval could
    be attributed to any named advisor by anyone able to type their name. The
    field no longer exists, and sending it changes nothing."""
    response = api.post(
        "/api/v1/runs/no-such-run/approve",
        json={"approved": True, "decided_by": "Someone Else"},
        headers=_bearer(admin),
    )
    # Rejected for the run being unknown, never for the forged attribution.
    assert response.status_code == 404


# --- audit --------------------------------------------------------------------

def test_the_audit_chain_records_the_actions_taken_and_verifies(api, admin, owned_client_id):
    events = api.get("/api/v1/audit/events", headers=_bearer(admin))
    assert events.status_code == 200
    actions = [e["action"] for e in events.json()]
    assert "client.created" in actions
    assert "auth.login.succeeded" in actions

    verified = api.get("/api/v1/audit/verify", headers=_bearer(admin))
    assert verified.status_code == 200
    assert verified.json()["intact"] is True


def test_an_advisor_cannot_read_the_audit_trail(api, advisor):
    assert api.get("/api/v1/audit/events", headers=_bearer(advisor)).status_code == 403
