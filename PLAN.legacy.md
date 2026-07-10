# AI Wealth Manager Engine - Architecture & Workflow Plan (Legacy)

> Superseded by [PLAN.md](PLAN.md), which redesigns this into a unified multi-agent architecture
> driven by client risk tolerance/age/goals. Kept here for historical reference.

## 1. System Architecture & File Layout
The system is designed as a fully autonomous, modular backend orchestrating multiple specialized AI and deterministic agents. The directory structure is as follows:

```
ai-wealth-manager/
├── requirements.txt       # Project dependencies
├── config.py              # Pydantic BaseSettings for env vars
├── db.py                  # Neon Postgres database engine & SQLAlchemy models
├── state.py               # Shared state schema for the agent network
├── server.py              # FastAPI application & endpoints
├── agents/                # Agent Modules
│   ├── __init__.py
│   ├── macro_sentinel.py  # Gemini-powered macro/market condition analysis
│   ├── quant_builder.py   # Deterministic portfolio optimization (Mean-Variance)
│   ├── tax_architect.py   # Wash-sale rule checker against TransactionLogs
│   └── compliance_critic.py # Strict, deterministic validation firewall
└── tests/                 # Pytest test suite
    ├── __init__.py
    └── test_workflow.py   # E2E and unit tests
```

## 2. Core Workflow Definition
The system operates as a sequential multi-agent loop using LangGraph/state-machine architecture:
1. **Macro Sentinel**: Analyzes input and current market streams, determining conditions (Bull/Bear/Volatile).
2. **Quant Builder**: Processes the user's profile and market condition to draft a targeted buy/sell JSON payload.
3. **Tax Alpha Architect**: Inspects the drafted payload against historical `TransactionLogs` to prevent wash-sales. If a violation is caught, it flags it and routes back to the Quant Builder for an alternative.
4. **Compliance Critic**: Evaluates the final JSON payload against strict rules (e.g., no asset > 30% allocation, no penny stocks). Acts as the final deterministic gatekeeper. Commits the trade if valid, else throws a hard exception.

## 3. Implementation Steps
- **Step 1**: Setup workspace, generate `requirements.txt`, and install/verify dependencies.
- **Step 2**: Configure `db.py` for Neon Postgres. Create seed data ($500k net worth, trades from 15 days ago for wash-sale testing).
- **Step 3**: Develop state schemas (`state.py`) and specialized agents in `agents/`. Integrate Gemini 1.5 Pro for cognitive tasks.
- **Step 4**: Stitch agents using LangGraph/State-Machine in an orchestrator function. Ensure full auditability via stdout logging.
- **Step 5**: Build FastAPI endpoint (`server.py`), write `pytest` suite, and execute E2E validation against wash-sales and compliance bounds.
