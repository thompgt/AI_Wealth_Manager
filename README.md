# AI Wealth Manager

A multi-agent portfolio analysis system. Six specialist agents run as a LangGraph workflow
over a client's portfolio and live market data, and produce a sized, explained set of
recommendations — with hard compliance and tax guardrails that can and do **withhold**
recommendations the system would otherwise make.

The interesting behaviour is not that it recommends things. It's where it refuses to.

<p align="center">
  <img src="docs/screenshots/03-recommendations.png" alt="Recommended deployment, with two candidates withheld by the wash-sale guardrail" width="900">
</p>

In the run above, the research agent proposed XOM and CVX, the suitability agent cleared them
and sized them at $60,000 each — and the tax agent blocked both, because the client had sold
them at a loss inside the 30-day wash-sale window. The dollars are not quietly redistributed
to the survivors; the run deploys less, and says why.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env              # optional: add a GEMINI_API_KEY

python demo_data.py               # seed a demo client
jupyter lab demo.ipynb            # run the live demo
```

Or run the service and dashboard:

```bash
uvicorn server:app --reload       # API       → http://localhost:8000/docs
solara run app.py                 # dashboard → http://localhost:8765
```

**It works without an API key.** Three of the six agents use an LLM; the other three are
deliberately deterministic. Without `GEMINI_API_KEY` the LLM-backed agents fall back to
documented non-AI paths, and every run says so — in the report, in the dashboard, and on
`GET /health`. What it will not do is quietly pass template output off as model output.

## The live demo

**[`demo.ipynb`](demo.ipynb)** runs the real orchestrator against live yfinance data —
nothing mocked, nothing pre-recorded — and is committed with its outputs, so it reads
without running anything. It walks through each agent, shows the guardrails firing, and
resumes a genuinely paused run.

`demo_data.py` seeds a deliberately flawed portfolio: 45% in one technology name, 100% of
the equity sleeve in a single sector, 40% in cash, and recent loss-sales across the value
names the research screen tends to favour — so the tax guardrail has something real to catch.

### What the demo shows

Portfolio diagnostics scores each position against **this client's own risk tier**, not a
fixed threshold:

![Concentration against the client's own limit](docs/screenshots/01-diagnostics-concentration.png)

The market regime call is grounded in deterministic signals computed from real price history,
so the evidence trail exists whether or not the LLM is available:

![Macro ratio signals](docs/screenshots/02-market-regime-signals.png)

The effect of the recommended trades on the portfolio mix:

![Portfolio mix before and after](docs/screenshots/04-before-after.png)

Every node's execution is recorded — note diagnostics and market regime overlapping, and
suitability and tax-awareness running as a parallel pair:

![Agent execution timeline](docs/screenshots/05-audit-timeline.png)

### The dashboard

Select a client and run an analysis:

![Dashboard with a client selected](docs/screenshots/06-dashboard-client.png)

The run pauses at a real human-in-the-loop interrupt. Execution stopped, state was written to
a durable checkpoint, and resuming is a separate HTTP request — the process could restart
right now without losing it:

![Human-in-the-loop approval gate](docs/screenshots/07-dashboard-approval.png)

After approval, the portfolio health numbers — with the degraded-LLM state stated plainly at
the top rather than left to look like model output:

![Dashboard results](docs/screenshots/08-dashboard-results.png)

Below that, the diagnosed flaws, the regime call with its fail-safe reasoning spelled out, and
the sized recommendations — each tagged with the flaw it addresses:

![Diagnosed flaws and sized recommendations](docs/screenshots/09-dashboard-recommendations.png)

And at the bottom, the part that matters: the guardrails reporting what they withheld. XOM and
CVX were researched, cleared by suitability, sized at $60,000 each — and then blocked, with the
reason shown to the client rather than silently dropped:

![Suitability and tax notes showing two withheld recommendations](docs/screenshots/10-dashboard-tax.png)

---

## Architecture

```
             ┌──────────────────────┐   ┌──────────────────┐
   START ───▶│ Portfolio Diagnostics│   │  Market Regime   │◀─── START
             │   (deterministic)    │   │  (hybrid + LLM)  │
             └──────────┬───────────┘   └────────┬─────────┘
                        └───────────┬────────────┘
                                    ▼
                          ┌───────────────────┐
                     ┌───▶│  Stock Research   │
                     │    │  (screen + LLM)   │
                     │    └─────────┬─────────┘
                     │       ┌──────┴──────┐
                     │       ▼             ▼
                     │ ┌───────────┐ ┌──────────────┐
                     │ │Suitability│ │Tax-Awareness │
                     │ │ (rules)   │ │ (wash-sale)  │
                     │ └─────┬─────┘ └──────┬───────┘
                     │       └──────┬───────┘
                     │              ▼
                     │     ┌─────────────────┐
                     └─────┤  Guardrail Gate │   retry (≤3) if nothing survives
                    retry  └────────┬────────┘
                                    ▼ proceed
                          ┌───────────────────┐
                          │  Finance Report   │
                          └─────────┬─────────┘
                                    ▼
                          ┌───────────────────┐
                          │   Approval Gate   │  ← human-in-the-loop interrupt
                          └─────────┬─────────┘
                                    ▼ END
```

| Agent | LLM? | What it does |
|---|---|---|
| **Portfolio Diagnostics** | No | Values holdings at live prices; computes concentration, sector exposure, and Sharpe/return/volatility via PyPortfolioOpt. Scores all of it against the client's own risk tier. |
| **Market Regime** | Hybrid | Computes auditable trend signals from 12 macro tickers and 5 ratio pairs, then has an LLM classify the regime *grounded in those signals*. Fails to `Volatile`/confidence 0.0 — never an optimistic default. |
| **Stock Research** | Yes | Screens a curated 34-name universe driven by the *diagnosed flaws*, filters on relative valuation, and ranks. Every candidate must name the flaw it addresses. |
| **Suitability** | **No** | Security quality (≥ $2B cap, major exchange, verifiable data), a near-retirement beta ceiling, and risk-tier position caps. Then sizes survivors from available cash by confidence weight, water-filling around each cap. |
| **Tax-Awareness** | **No** | 30-day wash-sale check against the transaction log; embedded gain/loss notes on current holdings. |
| **Finance Report** | Yes | Synthesises everything into a six-section client report under strict grounding rules, with a deterministic template fallback. |

The three agents that decide **what reaches the client contain no LLM calls at all.**
Compliance decisions have to be reproducible and auditable, and a model that is right 97% of
the time is not a control.

### The guardrail gate

Suitability and Tax-Awareness run in parallel and neither can see the other's verdict, so
Suitability can approve and size a position that Tax-Awareness is simultaneously flagging. The
**guardrail gate** is the single place those two are reconciled: it strips flagged
recommendations, records why in the client-visible violations, and — if nothing survives —
sends Stock Research back for another pass with the rejected tickers excluded and the reasons
attached to the prompt.

## Configuration

Every setting has a working default (see [`.env.example`](.env.example)), so it runs out of
the box locally. Several of those defaults are deliberately unsafe for a real deployment, and
**the server refuses to start** with them when `ENVIRONMENT` is anything but `development`: a
placeholder API key, a placeholder Gemini key, or a SQLite URL behind multiple workers.

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | Anything else enforces the startup safety checks |
| `NEON_DATABASE_URL` | SQLite file | Postgres required for multi-worker deployments |
| `GEMINI_API_KEY` | placeholder | Absent → deterministic fallbacks, clearly labelled |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Central, so a model deprecation is a config change |
| `API_AUTH_KEY` | dev-only value | `X-API-Key` header, compared with `compare_digest` |
| `CHECKPOINT_DB_PATH` | `./checkpoints.sqlite` | Durable store for paused human-approval runs |

## API

All endpoints except `/health` require an `X-API-Key` header.

| Method | Path | |
|---|---|---|
| `GET` | `/health` | Liveness, DB reachability, and whether the LLM layer is actually live |
| `POST` | `/api/v1/clients` | Create a client with holdings |
| `GET` | `/api/v1/clients` · `/api/v1/clients/{id}` | List / fetch |
| `PUT` | `/api/v1/clients/{id}` | Update |
| `POST` | `/api/v1/clients/{id}/run` | Run the graph; returns results or a pending approval |
| `POST` | `/api/v1/runs/{run_id}/approve` | Resume a paused run |
| `GET` | `/api/v1/clients/{id}/reports` · `/api/v1/reports/{id}` | Reports |

Interactive docs at `http://localhost:8000/docs`.

## Auditability

Every node writes a record — start, end, status, and a one-line summary — persisted to the
`agent_runs` table. Reports persist their full structured payload alongside the prose, so any
recommendation can be reconstructed after the fact: what the portfolio looked like, what the
regime call was, what was proposed, and what was withheld.

## Tests

```bash
pytest              # 78 tests, no network or API key required
```

The suite runs against a temporary database, never the local one, and stubs every network and
LLM call. It covers each agent's pure logic, the guardrail gate's reconciliation and retry
accumulation, the HTTP surface, and an end-to-end proof that a wash-sale-flagged ticker cannot
reach the final recommendations.

## Layout

```
orchestrator.py       LangGraph graph, guardrail gate, approval gate
state.py              The AgentState passed between nodes
checkpointer.py       Durable checkpoint selection (Postgres → SQLite → memory)
config.py             Settings + environment safety validation
db.py                 SQLAlchemy models, init, dev seed
server.py             FastAPI app
app.py                Solara dashboard
agents/               The six agents, plus shared risk limits
services/             Market data, news, and the shared LLM client
demo_data.py          Demo client seeding
demo.ipynb            The live demo
```

## Scope and limitations

- **Advisory only.** Nothing here executes a trade. Reports are informational and carry an
  explicit disclaimer; a human should review before acting.
- **The stock universe is a fixed 34-name list.** There is no paid screener API wired in, so
  the research agent cannot discover names outside `CANDIDATE_UNIVERSE` in
  `agents/stock_research.py`. This is a real constraint on recommendation breadth.
- **The wash-sale check sees only this system's transaction log**, and only the
  30-days-*after* side of the rule. It cannot see the client's other brokerage accounts or a
  spouse's, which the actual IRS rule covers.
- **Market data is best-effort.** Prices that cannot be resolved are excluded from the
  analysis and logged, rather than valued at zero.
