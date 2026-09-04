# Production readiness workplan

The system already has the things a demo usually lacks: tenancy enforced by
construction, a versioned investment policy, tax-lot accounting, a hash-chained
audit log, a durable job queue, deterministic guardrails with no LLM in the
decision path, and 231 offline tests. What it lacks is the operational surface
that stands between "runs on a laptop" and "runs for clients": a way to deploy
it, a way to tell whether it is healthy, bounded failure modes, protected client
data, and documentation that describes the system that actually exists.

Each item below is scoped to be independently shippable. Status is updated as
each lands.

| # | Item | Status |
|---|---|---|
| 1 | Container images and a compose stack that runs api + worker + Postgres | done |
| 2 | Split liveness from readiness; readiness checks deps and schema version | done |
| 3 | Request correlation ids threaded through logs and responses | todo |
| 4 | One error model (RFC 9457) across every endpoint, leaking nothing | todo |
| 5 | Graceful shutdown and reclaim of jobs orphaned by a dead worker | todo |
| 6 | Wall-clock budgets per node and per run | todo |
| 7 | Per-org daily LLM spend cap (a per-run one already exists) | todo |
| 8 | Postgres statement/lock timeouts and startup connect retry | todo |
| 9 | Field-level encryption for client PII at rest | todo |
| 10 | Client data export and retention-aware purge | todo |
| 11 | API key rotation and lifecycle visibility | todo |
| 12 | Prompt-injection red-team and golden-fixture agent evals | todo |
| 13 | CI: Postgres migrations, dependency audit, coverage floor | todo |
| 14 | Alert rules for the failure modes that return HTTP 200 | todo |
| 15 | Load and soak harness, with capacity numbers written down | todo |
| 16 | Operations runbook: deploy, rollback, restore, incident triage | todo |
| 17 | Threat model, security policy, licence | todo |
| 18 | README rewritten against the architecture that exists today | todo |
| 19 | Reconnect the dashboard: it does not import, and calls a removed endpoint | todo |

## Why these, and not features

Every item is a failure the system cannot currently *see* or *survive*, not a
capability it lacks:

* **1, 2, 16** — there is no artifact to deploy and no signal an orchestrator can
  read, so a rollout is a manual act with no automated abort.
* **3, 4, 14** — the agents are written to degrade rather than crash, which means
  a total loss of market data, a retired model or an exhausted quota all return
  200 with plausible output. Nothing in the HTTP surface distinguishes that from
  a healthy run, and nothing pages.
* **5, 6, 7** — the current failure modes are unbounded: a hung provider pins a
  worker forever, a deploy strands running jobs, and while a single run's LLM
  spend is capped, nothing caps the number of runs an org may start in a day.
* **8** — the Postgres path is pooled and pre-pinged, but a query with no
  statement timeout blocks a connection until someone notices, and a boot that
  races the database coming up fails permanently instead of retrying.
* **9, 10, 11, 17** — this stores names, emails, dates of birth, net worth and
  holdings. Plaintext at rest, no export, no purge and no written threat model
  is the wrong posture for that data regardless of scale.
* **12** — three agents take model output. Two prompts render text the firm did
  not write. There is fencing but no adversarial suite proving it holds, and no
  regression test that a prompt edit did not change what the system recommends.
* **13, 15** — the migration chain is only ever exercised on SQLite, dependencies
  are never audited, and nobody knows what this handles per host.
* **18** — the README documents a single shared `X-API-Key`, six agents and no
  accounts. The code has orgs, JWT sessions, four roles, accounts, tax lots,
  orders and a rebalance agent. A README that misdescribes the auth model is a
  production hazard, not a documentation nit.
* **19** — found while writing the compose file, not from the backlog: `app.py`
  reads `settings.API_AUTH_KEY`, which the auth rework deleted, so the dashboard
  raises `AttributeError` on import and never starts. It also posts to
  `/clients/{id}/run`, which became `/clients/{id}/runs` returning 202 and a job
  id to poll. The only client-facing surface in the system has been dead since
  the auth rework, and nothing caught it because nothing imports `app.py` in a
  test.

## Found along the way

Items added because the work surfaced them, rather than from the initial
review. Recorded here rather than folded silently into another item, since what
a review *missed* is worth as much as what it found:

* **19** — the dashboard does not start. A README screenshot is not a test.
