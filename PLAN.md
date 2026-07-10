# AI Wealth Manager — Architecture Redesign & Implementation Plan

_Supersedes the original plan below. Original single-pipeline design is preserved for reference in [PLAN.legacy.md](PLAN.legacy.md)._

## Context

This repo currently contains **two disconnected, half-finished implementations** that never call
each other:

1. **Root-level LangGraph pipeline** (`workflow.py`, `state.py`, `agents/*.py`, `server.py`, `db.py`,
   `config.py`, `app.py`) — a 4-node deterministic graph (`macro_sentinel → quant_builder →
   tax_architect → compliance_critic`) with hardcoded trading logic (always force-buys $10k of TSLA,
   asset picks hardcoded per market regime, trade price hardcoded to `1.0`), a compliance gate that
   is a literal string match against `"DOGE"`, and zero concept of client risk tolerance, age, or
   goals. Exposed via one FastAPI endpoint and a Solara dashboard.
2. **`backend/app/` LangChain scaffold** (`main.py`, `agents/agent_workflow.py`,
   `analytics/risk_engine.py`, `services/market_data.py`, `services/news_service.py`,
   `core/memory.py`, `core/config.py`) — a single LangChain tool-agent with two *real, working*
   tools (PyPortfolioOpt risk metrics via `yfinance`, DuckDuckGo news search), but state is a flat
   unsynchronized JSON file, and a different, incompatible config/DB from implementation #1.

A third directory, `C:\Users\thoma\Code\AI_Wealth_Manger`, is a stale duplicate git clone with no
unique code (empty frontend/notebooks dirs) — disposable.

Additional flaws confirmed by direct inspection: no `.gitignore` anywhere (`backend/.env` and the
live `wealth_manager.db` are committed to git); no API auth; no persisted audit log (just `print()`);
`requirements.txt` gaps (root missing `solara`/`pandas`/`plotly`; `backend/requirements.txt` has many
unused heavy deps like `transformers`, `cvxpy`, `reportlab`); broken test import paths.

**Goal of this work**: replace both systems with one coherent multi-agent architecture that (a) takes
a client's risk tolerance, age, existing portfolio, and goals as first-class input via a frontend,
(b) diagnoses concrete flaws in the current portfolio, (c) assesses market regime from real
macro-indicator tickers + news, (d) researches undervalued stocks that specifically fix the diagnosed
flaws, (e) enforces suitability/compliance against the *client's* profile (not hardcoded constants),
and (f) produces a client-facing report from a finance-expert agent — while fixing the security/hygiene
flaws above along the way.

**Decisions locked in for this plan:**
- Deliver both a written architecture doc and the actual code implementation.
- Frontend: keep **Solara** (Python-native) for now, rewired to the new unified backend — not a
  React/Next rewrite.
- LLM provider: **keep Gemini**, standardized on `langchain-google-genai`'s `ChatGoogleGenerativeAI`
  instead of the current mix of raw `google-generativeai` SDK + LangChain.

---

## Target Architecture

### Agent roster

| Agent | Type | Responsibility | Key inputs | Key outputs |
|---|---|---|---|---|
| **Client Profile Intake** | form/validation (not LLM) | Capture & validate risk tolerance, age, horizon, goals, holdings | Solara form submission | `ClientProfile` row |
| **Data Ingestion Service** | shared utility (not an agent) | Single source of truth for prices/news, with caching | Ticker lists | Price DataFrames, quotes, news |
| **Portfolio Diagnostics Agent** | deterministic/quant | Compute concentration, sector exposure, correlation, Sharpe/volatility, drawdown risk vs the *client's* risk tolerance; produce a `flaws` list | `ClientProfile.holdings`, price history | `PortfolioDiagnostics` incl. `flaws: [...]` |
| **Market Regime Agent** | hybrid (quant signals → LLM synthesis) | Judge Bull/Bear/Volatile/cycle-stage from macro-indicator tickers + news | Indicator basket prices, news | `MarketRegime {regime_label, confidence, supporting_signals, narrative}` |
| **Stock Research Agent** | hybrid | Find undervalued candidates that specifically fix diagnosed flaws and fit client profile + regime | `PortfolioDiagnostics.flaws`, `ClientProfile`, `MarketRegime` | `CandidateList` with per-candidate rationale |
| **Suitability/Compliance Guardrail** | deterministic | Replace hardcoded 30%/"DOGE" rules with risk-tolerance-scaled position limits, age/horizon-appropriate asset-class checks, security-quality screen | `CandidateList`, `ClientProfile`, `PortfolioDiagnostics` | `{approved, violations, adjusted_recommendations}` |
| **Tax-Awareness Agent** | deterministic | Generalized wash-sale check + tax-lot-aware substitution suggestions | `CandidateList`, `TransactionLog` history | `{wash_sale_flags, tax_efficiency_notes}` |
| **Finance Report Agent** | LLM, grounded on structured state | Synthesize everything into a client-facing report: health check, flaws in plain language, macro context, recommendations, caveats | All upstream state | Persisted report (text + structured payload) |
| **Orchestrator** | LangGraph `StateGraph` (not a persona) | Own sequencing, parallel fan-out/fan-in, conditional loop-backs, human-approval interrupts | Full `AgentState` | End-to-end run |

### Market Regime indicator basket (with rationale)

| Ticker(s) | Signal | Rationale |
|---|---|---|
| `SPY` vs `IWM` | Small-cap/large-cap ratio | Small caps lead risk-on/off rotations |
| `HYG` vs `LQD` | High-yield vs IG credit spread proxy | Credit markets move ahead of equities into stress |
| `SHY`/`TLT` ratio (or `^TNX`/`^IRX`) | Yield-curve proxy | Classic macro-cycle signal |
| `^VIX` | Fear gauge | Market-implied risk premium |
| `DX-Y.NYB` (DXY) | Dollar strength | Tightens global financial conditions |
| `HG=F` / `CPER` | "Dr. Copper" | Real-time global industrial-activity proxy |
| `SMH` vs `SPY` | Semiconductors | Leads broader tech/industrial cycle by 1-2 quarters |
| `XLU`/`XLP` vs `SPY` | Defensives rotation | Signals institutional de-risking |

### Orchestration flow

```
Client Intake
      │
      ▼ fan-out
 Portfolio Diagnostics  ‖  Market Regime      (run in parallel — no data dependency)
      │                    │
      └────────┬───────────┘  fan-in (both required)
               ▼
        Stock Research
               │
      ┌────────┴────────┐  fan-out
      ▼                 ▼
 Suitability/       Tax-Awareness              (run in parallel)
 Compliance
      └────────┬────────┘  fan-in
               ▼
      [conditional: rejected w/ alternatives? → loop back to Stock Research]
               ▼
        Finance Report Agent
               │
               ▼
   [Human approval gate — required before anything becomes an "active
    recommendation"/trade; also triggered if Suitability rejects N-in-a-row
    or Market Regime confidence is low]
               ▼
      Deliver to client / persist
```

Implemented as **one** LangGraph `StateGraph` (replacing both `workflow.py` and
`backend/app/agents/agent_workflow.py`), using LangGraph's checkpointer for persisted, resumable
state — this also doubles as the audit trail.

### Shared state schema (extends `state.py`'s `AgentState` TypedDict)

```python
class ClientProfile(TypedDict):
    client_id: int
    age: int
    risk_tolerance: str        # "Conservative" | "Moderate" | "Aggressive" (derived from a short questionnaire, not free text)
    time_horizon_years: int
    goals: List[str]
    holdings: Dict[str, float]

class AgentState(TypedDict):
    client_profile: ClientProfile
    portfolio_diagnostics: Optional[PortfolioDiagnostics]
    market_regime: Optional[MarketRegime]
    candidate_stocks: Optional[List[Candidate]]
    tax_flags: Optional[List[str]]
    suitability_result: Optional[SuitabilityResult]
    final_report: Optional[str]
    audit_trail: List[AgentRunRecord]
    requires_human_approval: bool
    human_approved: Optional[bool]
```

### Frontend (Solara)

- Rewire the existing `app.py` shell to call the *new* unified FastAPI endpoints (today it's
  disconnected from the real analytics in `backend/app`).
- Client-profile form: numeric age; a short 3-5 question risk-tolerance questionnaire (loss
  tolerance, reaction to a 20% drawdown, income stability) mapped to a discrete tier — not a raw
  free-text/self-reported label; time horizon; multi-select goals; holdings entered via a manual
  symbol+value table (CSV import is a documented future extension, not in this pass).
- Results view: portfolio health summary (Sharpe/volatility/diversification, pie/treemap of
  allocation), a "flaws" panel in plain language, a market-regime banner showing which indicators
  drove the call, ranked recommended stocks (each tagged with the flaw it addresses), a
  compliance/suitability notes panel (what got filtered and why), and the full narrative report.
- All business logic stays behind FastAPI endpoints, never inlined into Solara callbacks — keeps a
  future React/Next swap to a presentation-layer-only change.

### Data & persistence (extends `db.py`, target Postgres/Neon, SQLite for local dev)

```
client_profiles   (id, name, email, age, risk_tolerance, time_horizon_years, goals JSON, net_worth, created_at, updated_at)
holdings          (id, client_id FK, symbol, quantity, cost_basis, acquired_at)
transaction_logs  (existing table, rename user_id -> client_id FK)
agent_runs        (id, client_id FK, run_id, node_name, started_at, completed_at, input_snapshot JSON, output_snapshot JSON, model_used, status, error_detail)
reports           (id, client_id FK, run_id FK, generated_at, report_text, structured_payload JSON, version)
market_data_cache (ticker, as_of_date, close_price, fetched_at)
approvals         (id, run_id FK, requested_at, decided_at, decided_by, decision, notes)
```

Keep `db.py`'s existing `postgres://`→`postgresql://` URL-normalization shim — it's correct.

---

## Migration Plan (what to keep, replace, delete)

**Keep & promote (real, working logic — reuse, don't rewrite):**
- `backend/app/analytics/risk_engine.py::calculate_portfolio_metrics` → core of Portfolio
  Diagnostics Agent (extend with sector/factor exposure).
- `backend/app/services/market_data.py::fetch_historical_prices`, `get_current_prices` → the shared
  Data Ingestion service, adding a `market_data_cache`-backed cache layer.
- `backend/app/services/news_service.py::search_financial_news`, `get_portfolio_news` → feeds Market
  Regime and Stock Research.
- `agents/tax_architect.py`'s wash-sale query pattern → generalized into Tax-Awareness Agent.
- `agents/compliance_critic.py`'s "raise on violation, gate before commit" control-flow pattern →
  keep the pattern, replace hardcoded rule bodies with client-profile-driven thresholds.
- `db.py`'s engine/URL-normalization logic; `server.py`'s FastAPI app as the base to extend.

**Replace/retire:**
- `workflow.py` (root) and `backend/app/agents/agent_workflow.py` — both superseded by the new
  unified `StateGraph`. Delete after the new graph is live.
- `agents/macro_sentinel.py` → Market Regime Agent (real ticker signals, not one ungrounded prompt).
- `agents/quant_builder.py` → Stock Research Agent; delete the hardcoded TSLA test-buy and
  hardcoded regime→asset map.
- `backend/app/core/memory.py` (JSON file) → replaced by the SQL tables above.

**Delete outright (no migration needed):**
- `C:\Users\thoma\Code\AI_Wealth_Manger` — stale duplicate clone, no unique code.
- `backend/app/api/`, `backend/app/models/` — empty scaffolding stubs.

**Layout consolidation**: flatten everything to **one top-level package** — a single `agents/`
directory at repo root (`intake.py`, `diagnostics.py`, `market_regime.py`, `stock_research.py`,
`suitability.py`, `tax_awareness.py`, `finance_report.py`, `orchestrator.py`) plus a top-level
`services/` for the shared ingestion layer, `db.py`, `config.py`, `server.py`, `app.py` — matching
the root pipeline's already-more-production-shaped layout rather than the `backend/app/` nesting.
Standardize on **one** `Settings` class (merge `config.py` + `backend/app/core/config.py`'s env vars
into one scheme) and **one** LLM client pattern (`langchain-google-genai`'s `ChatGoogleGenerativeAI`
everywhere, replacing the raw `google.generativeai` SDK use in `macro_sentinel.py`).

---

## Flaws-to-Fix Checklist (bundled into this migration)

1. Add `.gitignore` (`.env`, `*.db`, `__pycache__/`, `.pytest_cache/`); `git rm --cached` the
   committed `.env` and `wealth_manager.db`; rotate any key ever pushed.
2. `git rm --cached -r **/__pycache__`.
3. Remove hardcoded trade price (`1.0`) — route through Data Ingestion's `get_current_prices`.
4. Remove hardcoded forced-TSLA-buy test logic (goes away with `quant_builder.py` replacement).
5. Replace `"DOGE"` string-match rule and flat 30%-of-net-worth cap with real, risk-tolerance-scaled
   Suitability logic.
6. Add API auth (API-key or JWT via FastAPI `Depends`, scoped per `client_id`) — currently anyone can
   call `/api/v1/rebalance(user_id)` for any id.
7. Add persisted audit logging (`agent_runs` table + LangGraph checkpointer) — replace all `print()`.
8. Fix root `requirements.txt` (add `solara`, `pandas`, `plotly`); strip `backend/requirements.txt`
   of unused heavy deps (`transformers`, `sentence-transformers`, `cvxpy`, `duckdb`, `reportlab`,
   `weasyprint`, `jinja2`, `newspaper3k`).
9. Fix `tests/test_integration.py` broken import paths (resolved naturally by flattening to one
   package root + a proper `conftest.py`/`pyproject.toml`).

---

## Dependency-Ordering Review

Verified against the actual source of every legacy file (not just the summary above). This changed
the step order from the first draft in five concrete ways:

1. **`requirements.txt` merge moved from last to early (new step 3).** Root `requirements.txt`
   today has no `langchain-google-genai`, `PyPortfolioOpt`, `yfinance`, or `duckduckgo-search` — only
   `backend/requirements.txt` does. If the merge stays last (old step 11), steps 4-10 (building
   `services/`, the agent modules, `market_regime.py`'s `ChatGoogleGenerativeAI` call) would be
   written against packages that aren't installed yet. Doing the merge right after config
   consolidation means everything needed is installed before it's imported.
2. **`backend/app/core/config.py` is no longer deleted in the config-consolidation step.**
   `backend/app/analytics/risk_engine.py` does `from app.core.config import settings` (uses
   `settings.RISK_FREE_RATE`, `settings.LOOKBACK_PERIOD_YEARS`) and `backend/app/services/market_data.py`
   has the same import. Deleting `backend/app/core/config.py` before those two files are ported
   (old step 4) leaves them with a dangling import. Since nothing in steps 3-9 needs to *import*
   the old `backend/app` package (only *read* it as reference while porting logic), the low-risk fix
   is to leave `backend/app/core/config.py` in place and untouched until the whole `backend/`
   directory is torn down at the end (see point 4), rather than deleting it mid-sequence.
3. **`wealth_manager.db` must be deleted and regenerated, not altered in place.** SQLAlchemy's
   `Base.metadata.create_all()` does not add/rename columns on tables that already exist — it's a
   no-op for `transaction_logs` if that table is already present. Renaming `TransactionLog.user_id`
   to `client_id` in the model (per the target schema) would then desync from the actual SQLite file
   (`git ls-files` confirms `wealth_manager.db` is currently tracked with the old schema), producing
   `OperationalError: no such column: client_id` at runtime, not a clean migration. Since it's a
   3-row dev/seed DB (not shared production data), the correct fix is: delete the physical `.db` file
   as part of the `db.py` schema step and regenerate it from the updated `init_db()`/`seed_db()`.
   The new seed script must also seed `age`/`risk_tolerance`/`goals` on the test client profile (the
   current seed only sets `name`/`net_worth`/`portfolio`) — otherwise there's nothing for Portfolio
   Diagnostics, Suitability, or Stock Research to condition on during manual verification. Keep the
   existing 15-days-ago TSLA SELL row so the wash-sale scenario still exercises Tax-Awareness.
4. **Added an explicit final teardown of the entire `backend/` directory.** The original step 10
   only listed `backend/app/agents/agent_workflow.py`, `backend/app/core/memory.py`,
   `backend/app/api/`, `backend/app/models/` for deletion — it never named
   `backend/app/main.py` (the second, still-runnable FastAPI app), `backend/app/analytics/`,
   `backend/app/services/` (the old copies, once their logic is ported into the new top-level
   `services/`), `backend/app/core/config.py`, `backend/requirements.txt`, `backend/.env`, or the
   `backend/` directory itself. Leaving any of these in place directly contradicts the stated goal
   of "one coherent" backend — a second, dead-but-importable FastAPI app would still exist. This is
   now an explicit step (new step 11): delete `backend/` in full once every piece of real logic
   it contained has been ported and verified working in its new location.
5. **`server.py` and `tests/test_workflow.py` will be non-functional from the DB step through the
   `server.py` rewrite step** — `server.py` imports `UserProfile`/`TransactionLog` by their current
   names/columns and drives the old `workflow.py` graph, so once the DB schema and directory layout
   change, it breaks until it's rewritten (old step 8, now step 9). This is expected for a solo,
   local, big-bang rewrite (there's no concurrent user traffic to protect) — not a sign anything went
   wrong. Don't try to keep `pytest` green mid-sequence; run the test suite only after the `server.py`
   rewrite and `app.py` rewire are both done.
6. **Stale-clone deletion re-confirmed safe**: `C:\Users\thoma\Code\AI_Wealth_Manger` — `git status`
   clean, no unpushed commits (`dd97bcb`, `82ce415` only, both already on `origin/main`), and
   `origin/main` has already advanced past it (`dd97bcb..d0737f2`). No unique or unpushed work exists
   there.
7. **Minor**: pin a minimum `langgraph` version in the merged `requirements.txt` (currently unpinned)
   — the orchestrator step relies on checkpointer/interrupt APIs for the human-approval gate, which
   need a recent-enough release.

## Implementation Steps

1. **Repo hygiene first** (fast, low-risk): add `.gitignore`, untrack `.env`/`wealth_manager.db`/
   `__pycache__`, delete `C:\Users\thoma\Code\AI_Wealth_Manger`.
2. **Consolidate config**: merge `config.py` + `backend/app/core/config.py`'s env vars into one
   root `config.py` `Settings` (Gemini key, DB URL, risk-free rate, lookback years, etc.). Do **not**
   delete `backend/app/core/config.py` yet — leave it in place, unused, until step 11 (see
   Dependency-Ordering Review point 2).
3. **Merge `requirements.txt`** into one root file: add missing deps (`solara`, `pandas`, `plotly`,
   `langchain-google-genai`, `PyPortfolioOpt`, `yfinance`, `duckduckgo-search`, pinned `langgraph`),
   strip unused heavy deps (`transformers`, `sentence-transformers`, `cvxpy`, `duckdb`, `reportlab`,
   `weasyprint`, `jinja2`, `newspaper3k`). Doing this now (not last) means every later step can
   actually `pip install` and run what it writes.
4. **Extend `db.py`**: add `client_profiles`, `holdings`, `agent_runs`, `reports`,
   `market_data_cache`, `approvals` tables; migrate `user_profiles`/`transaction_logs` to the new
   shape (rename `user_id`→`client_id` FK). Delete the physical `wealth_manager.db` file and
   regenerate it via an updated `init_db()`/`seed_db()` that also seeds `age`/`risk_tolerance`/
   `goals` on the test client (see Dependency-Ordering Review point 3) — do not rely on
   `create_all()` to alter the existing file in place.
5. **Build the shared `services/` layer**: port `market_data.py` and `news_service.py` to a
   top-level `services/` (reading from `backend/app/services/` as reference, updating their config
   import to the new unified `config.py`), add caching against `market_data_cache`.
6. **Rewrite `state.py`**: new `AgentState`/`ClientProfile`/`PortfolioDiagnostics`/`MarketRegime`/
   `Candidate`/`SuitabilityResult` schemas as designed above.
7. **Build each agent module** under a unified `agents/`: `diagnostics.py` (port
   `risk_engine.py` logic + flaw summarization), `market_regime.py` (indicator basket fetch +
   `ChatGoogleGenerativeAI` synthesis), `stock_research.py`, `suitability.py` (port
   `compliance_critic.py` pattern), `tax_awareness.py` (port `tax_architect.py` pattern),
   `finance_report.py` (port/restructure `agent_workflow.py`'s report prompt to consume full state).
8. **Build `orchestrator.py`**: new `StateGraph` wiring the parallel fan-out/fan-in flow above, with
   conditional loop-back and human-approval interrupt via LangGraph checkpointer.
9. **Extend `server.py`**: endpoints for profile CRUD, triggering a graph run, fetching a report,
   and approvals; add API-key/JWT auth dependency. (`server.py` and `tests/test_workflow.py` are
   expected to be broken from step 4 until this step completes — see Dependency-Ordering Review
   point 5.)
10. **Rewire `app.py`** (Solara): new client-intake form (incl. risk questionnaire) and results view
    as designed above, calling only the new FastAPI endpoints.
11. **Delete superseded files, including the full `backend/` directory**: `workflow.py`,
    `agents/macro_sentinel.py`, `agents/quant_builder.py`, and all of `backend/` — `app/main.py`,
    `app/agents/agent_workflow.py`, `app/core/config.py`, `app/core/memory.py`, `app/analytics/`,
    `app/services/`, `app/api/`, `app/models/`, `requirements.txt`, `.env` — once every piece of
    real logic has been ported and verified working in its new location (see Dependency-Ordering
    Review point 4).
12. **Tests**: fix import paths (`conftest.py`/package layout), add unit tests per new agent
    (diagnostics math, suitability threshold logic, wash-sale generalization) and an E2E graph test
    replacing `tests/test_integration.py` / `tests/test_workflow.py`. Run the full suite now that
    steps 1-11 have landed.

## Verification

- `pytest` from repo root passes (fixes the broken-import-path flaw as part of the check).
- Fresh `pip install -r requirements.txt` in a clean venv succeeds (fixes the Solara dep-gap flaw).
- Run `uvicorn server:app --reload` + `solara run app.py`; walk through: submit a client profile with
  sample holdings → confirm Portfolio Diagnostics surfaces real flaws (e.g. concentration) → confirm
  Market Regime shows real indicator values, not a hardcoded guess → confirm Stock Research
  candidates reference the specific diagnosed flaws → confirm Suitability blocks a deliberately
  oversized position per the client's stated risk tier (not the old flat 30%) → confirm a wash-sale
  scenario (seed data: sale 15 days ago) is still caught → confirm the final report renders and cites
  the upstream numbers → confirm `agent_runs` rows are written (audit trail working) instead of only
  stdout prints.
- `git status`/`git ls-files` confirm `.env`, `wealth_manager.db`, and `__pycache__` are no longer
  tracked.
