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
| 3 | Request correlation ids threaded through logs and responses | done |
| 4 | One error model (RFC 9457) across every endpoint, leaking nothing | done |
| 5 | Graceful shutdown and reclaim of jobs orphaned by a dead worker | done |
| 6 | Wall-clock budgets per node and per run | done |
| 7 | Per-org LLM spend cap enforced at call time | done |
| 8 | Postgres engine hardening: pooling, timeouts, pre-ping | done |
| 9 | Field-level encryption for client PII at rest | done |
| 10 | Client data export and retention-aware purge | done |
| 11 | API key rotation and lifecycle visibility | done |
| 12 | Prompt-injection red-team and golden-fixture agent evals | done |
| 13 | CI: Postgres migrations, dependency audit, coverage floor | done |
| 14 | Alert rules for the failure modes that return HTTP 200 | done |
| 15 | Load and soak harness, with capacity numbers written down | done |
| 16 | Operations runbook: deploy, rollback, restore, incident triage | done |
| 17 | Threat model, security policy, licence | done |
| 18 | README rewritten against the architecture that exists today | done |

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
  worker forever, a deploy strands running jobs, and a runaway retry loop bills
  the operator with no ceiling.
* **8** — SQLite is refused outside development, but the Postgres path has never
  been given a pool size, a statement timeout or a liveness check.
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
