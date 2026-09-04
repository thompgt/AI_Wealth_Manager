"""Solara dashboard.

An HTTP client of `server.py` and nothing else -- no database session, no
agent import, no business rule. Every decision it displays was made server
side, which is what lets the API be the single place a control is enforced.

The rewrite fixed four things, listed because each was invisible from the
screenshots in the README:

1. **It did not start.** `settings.API_AUTH_KEY` was deleted when
   authentication moved to organisations and JWT sessions. Importing the
   module raised `AttributeError`.
2. **It called an endpoint that no longer exists.** `POST /clients/{id}/run`
   became `POST /clients/{id}/runs`, returning 202 and a job id to poll,
   because a graph run outlasts a request.
3. **It scored risk itself.** A three-question quiz here produced a tier that
   was posted as a plain string, while the server derives a tier from a
   versioned, expiring questionnaire and treats the field as never
   free-typed. Two scoring systems, one of them unauditable.
4. **It dropped every holding.** It sent a flat `holdings` list; the API takes
   holdings nested inside accounts, because a lot's tax treatment depends on
   which account holds it. The extra key was ignored and clients were created
   empty.

Transport lives in `dashboard_api.py` so it can be tested without a browser.
"""

import asyncio
from typing import Any, Dict, List, Optional

import httpx  # noqa: F401 -- re-exported for tests that patch the transport
import pandas as pd
import plotly.express as px
import solara

import dashboard_api as api

# How often a queued run is polled, and for how long before the UI stops
# asking. The cap is not a timeout on the run -- the worker owns that, and the
# job survives this page being closed -- it only bounds the polling.
POLL_INTERVAL_SECONDS = 2.0
POLL_CEILING_SECONDS = 900.0


async def _call(fn, *args, **kwargs):
    """Run a synchronous API call without blocking the render loop.

    `dashboard_api` is deliberately synchronous: it is far easier to test that
    way, and it is the same code whether a script or this page calls it. But
    Solara's tasks are coroutines on one event loop, and a blocking request
    inside one freezes every other component on the page for its duration --
    which for the approval resume is minutes.
    """
    return await asyncio.to_thread(lambda: fn(*args, **kwargs))


# --- Login -------------------------------------------------------------------


@solara.component
def LoginForm(on_session):
    """Sign in as a person.

    Not a shared key. The API separates proposing a recommendation from
    approving one -- an advisor may run analysis, only a compliance user may
    clear it -- and a dashboard authenticating as one machine identity would
    either be unable to approve anything or would let whoever holds the key
    approve their own work.
    """
    org_slug, set_org_slug = solara.use_state("")
    email, set_email = solara.use_state("")
    password, set_password = solara.use_state("")
    error, set_error = solara.use_state(None)

    @solara.lab.use_task(dependencies=None)
    async def submit():
        set_error(None)
        try:
            session = await _call(api.login, org_slug.strip(), email.strip(), password)
        except api.ApiError as exc:
            # The server answers the same way for an unknown organisation, an
            # unknown user and a wrong password, and this must not undo that
            # by being more specific.
            set_error(exc.message)
            return
        on_session(session)

    with solara.Card("Sign in"):
        solara.InputText("Organisation", value=org_slug, on_value=set_org_slug)
        solara.InputText("Email", value=email, on_value=set_email)
        solara.InputText("Password", value=password, on_value=set_password, password=True)
        if error:
            solara.Error(error)
        solara.Button("Sign in", on_click=submit, color="primary", disabled=submit.pending)
        if submit.pending:
            solara.ProgressLinear()


# --- Risk questionnaire ------------------------------------------------------


@solara.component
def RiskQuestionnaire(schema, answers, set_answers):
    """Render the *server's* questionnaire.

    The questions, their scoring and the tier they imply are all server side
    and versioned, and an assessment expires. Restating them here would create
    a second scoring system that drifts from the one whose answers are stored
    against the client's file and shown to an examiner.
    """
    if not schema:
        solara.Info("Loading the risk questionnaire...")
        return

    solara.Markdown(
        f"**Risk tolerance questionnaire** (version {schema.get('version')}, "
        f"valid {schema.get('valid_days')} days)"
    )
    for question in schema.get("questions", []):
        qid = question["id"]
        label_to_id = {opt["label"]: opt["id"] for opt in question["options"]}
        id_to_label = {opt["id"]: opt["label"] for opt in question["options"]}

        def make_setter(question_id, mapping):
            def setter(label):
                updated = dict(answers)
                updated[question_id] = mapping.get(label)
                set_answers(updated)
            return setter

        solara.Select(
            label=question["prompt"],
            value=id_to_label.get(answers.get(qid)),
            values=list(label_to_id.keys()),
            on_value=make_setter(qid, label_to_id),
        )
        if question.get("help_text"):
            solara.Markdown(f"<small>{question['help_text']}</small>")


# --- New client --------------------------------------------------------------


@solara.component
def NewClientForm(session, on_created):
    name, set_name = solara.use_state("")
    email, set_email = solara.use_state("")
    age, set_age = solara.use_state(45)
    horizon, set_horizon = solara.use_state(15)
    goals_text, set_goals_text = solara.use_state("")
    cash, set_cash = solara.use_state(100000.0)
    account_type, set_account_type = solara.use_state("individual")
    holdings_rows, set_holdings_rows = solara.use_state(
        [{"symbol": "", "quantity": 0.0, "cost_per_share": 0.0}]
    )
    schema, set_schema = solara.use_state(None)
    answers, set_answers = solara.use_state({})
    error, set_error = solara.use_state(None)

    @solara.lab.use_task(dependencies=[])
    async def load_schema():
        try:
            set_schema(await _call(api.get_questionnaire, session))
        except api.ApiError as exc:
            set_error(exc.message)

    # Tax treatment follows from the account type rather than being a separate
    # question: a Roth IRA is tax-exempt by definition, and letting the two be
    # set independently invites a taxable Roth, which would silently disable
    # the wash-sale and gain-budget logic for that account.
    tax_treatment = {
        "individual": "taxable",
        "joint": "taxable",
        "trust": "taxable",
        "custodial": "taxable",
        "traditional_ira": "tax_deferred",
        "401k": "tax_deferred",
        "roth_ira": "tax_exempt",
    }[account_type]

    solara.Markdown("### New client")
    solara.InputText("Full name", value=name, on_value=set_name)
    solara.InputText("Email", value=email, on_value=set_email)
    solara.InputInt("Age", value=age, on_value=set_age)
    solara.InputInt("Time horizon (years)", value=horizon, on_value=set_horizon)
    solara.InputText("Goals (comma-separated)", value=goals_text, on_value=set_goals_text)

    RiskQuestionnaire(schema, answers, set_answers)

    solara.Markdown("**Account**")
    solara.Select(
        label="Account type",
        value=account_type,
        values=["individual", "joint", "traditional_ira", "roth_ira", "401k", "trust", "custodial"],
        on_value=set_account_type,
    )
    solara.Markdown(f"Tax treatment: **{tax_treatment}** (implied by the account type)")
    solara.InputFloat("Uninvested cash ($)", value=cash, on_value=set_cash)

    solara.Markdown("**Holdings** (leave empty to start in cash)")
    for idx, row in enumerate(holdings_rows):
        with solara.Row():
            def make_row_setter(i, field):
                def setter(value):
                    new_rows = [dict(r) for r in holdings_rows]
                    new_rows[i][field] = value
                    set_holdings_rows(new_rows)
                return setter

            solara.InputText("Symbol", value=row["symbol"], on_value=make_row_setter(idx, "symbol"))
            solara.InputFloat("Shares", value=row["quantity"], on_value=make_row_setter(idx, "quantity"))
            solara.InputFloat(
                "Cost per share ($)", value=row["cost_per_share"],
                on_value=make_row_setter(idx, "cost_per_share"),
            )

    solara.Button(
        "+ Add holding",
        on_click=lambda: set_holdings_rows(
            holdings_rows + [{"symbol": "", "quantity": 0.0, "cost_per_share": 0.0}]
        ),
    )

    @solara.lab.use_task(dependencies=None)
    async def submit():
        set_error(None)
        if not name.strip():
            set_error("A name is required.")
            return
        expected = {q["id"] for q in (schema or {}).get("questions", [])}
        if expected - {k for k, v in answers.items() if v}:
            set_error("Please answer every question in the risk questionnaire.")
            return

        # Holdings are nested inside the account, not attached to the client.
        # A lot's tax treatment is a property of the account holding it, and
        # the tax agent's answers are wrong without it.
        holdings = [
            {
                "symbol": row["symbol"].strip().upper(),
                "quantity": row["quantity"],
                "cost_per_share": row["cost_per_share"],
            }
            for row in holdings_rows
            if row["symbol"].strip() and row["quantity"] > 0
        ]

        payload = {
            "name": name.strip(),
            "email": email.strip() or None,
            "age": age,
            "time_horizon_years": horizon,
            "goals": [g.strip() for g in goals_text.split(",") if g.strip()],
            "net_worth": cash + sum(h["quantity"] * h["cost_per_share"] for h in holdings),
            "accounts": [
                {
                    "name": f"{name.strip()} {account_type.replace('_', ' ').title()}",
                    "account_type": account_type,
                    "tax_treatment": tax_treatment,
                    "cash_balance": cash,
                    "holdings": holdings,
                }
            ],
        }

        try:
            created = await _call(api.create_client, session, payload)
            # The tier is set by scoring the questionnaire server side, in a
            # second call, rather than being sent as a string on the first.
            # That is what puts a dated, versioned assessment behind the tier
            # every downstream limit is derived from.
            await _call(api.submit_risk_assessment, session, created["id"], answers)
            refreshed = await _call(api.get_client, session, created["id"])
        except api.ApiError as exc:
            set_error(exc.message)
            return
        on_created(refreshed)

    if error:
        solara.Error(error)
    solara.Button("Create client", on_click=submit, color="primary", disabled=submit.pending)
    if submit.pending or load_schema.pending:
        solara.ProgressLinear()


# --- Result panels -----------------------------------------------------------


@solara.component
def PortfolioDiagnosticsPanel(diagnostics):
    if not diagnostics:
        solara.Info("No diagnostics available.")
        return
    with solara.Columns([1, 1]):
        with solara.Card("Portfolio health"):
            solara.Markdown(f"**Sharpe ratio:** {diagnostics.get('sharpe_ratio')}")
            solara.Markdown(f"**Annual return:** {diagnostics.get('annual_return')}")
            solara.Markdown(f"**Annual volatility:** {diagnostics.get('annual_volatility')}")
            solara.Markdown(
                f"**Diversification score:** {diagnostics.get('diversification_score')}/100"
            )
        with solara.Card("Concentration"):
            concentration = diagnostics.get("concentration") or {}
            if concentration:
                df = pd.DataFrame(
                    [{"Symbol": k, "Fraction": v} for k, v in concentration.items()]
                )
                solara.FigurePlotly(
                    px.pie(df, values="Fraction", names="Symbol", title="Portfolio concentration")
                )

    flaws = diagnostics.get("flaws") or []
    with solara.Card("Diagnosed flaws"):
        if flaws:
            for flaw in flaws:
                solara.Markdown(f"- {flaw}")
        else:
            solara.Markdown("No flaws detected.")


@solara.component
def MarketRegimePanel(regime):
    if not regime:
        solara.Info("No market regime data available.")
        return
    title = (
        f"Market regime: {regime.get('regime_label')} "
        f"(confidence {regime.get('confidence')})"
    )
    with solara.Card(title):
        solara.Markdown(regime.get("narrative", ""))


@solara.component
def RecommendationsPanel(suitability):
    if not suitability:
        solara.Info("No suitability data available.")
        return
    with solara.Card("Recommended candidates"):
        recs = suitability.get("adjusted_recommendations") or []
        if recs:
            total = sum(r.get("allocation_amount", 0.0) for r in recs)
            solara.Markdown(f"**Total recommended allocation: ${total:,.0f}**")
            df = pd.DataFrame(
                [
                    {
                        "Ticker": r.get("ticker"),
                        "Allocation ($)": r.get("allocation_amount", 0.0),
                        "Allocation (%)": r.get("allocation_pct", 0.0) * 100,
                        "Addresses flaw": r.get("addresses_flaw"),
                        "Rationale": r.get("regime_fit_rationale"),
                        "Confidence": r.get("confidence"),
                    }
                    for r in recs
                ]
            )
            solara.DataFrame(df)
            solara.FigurePlotly(
                px.pie(df, values="Allocation ($)", names="Ticker",
                       title="Recommended allocation")
            )
        else:
            solara.Markdown("No candidates were approved this cycle.")

    with solara.Card("Suitability / compliance notes"):
        solara.Markdown(f"**Approved:** {suitability.get('approved')}")
        for violation in suitability.get("violations") or []:
            solara.Markdown(f"- {violation}")


@solara.component
def TaxPanel(tax_assessment, tax_blocked=None):
    if not tax_assessment and not tax_blocked:
        return
    tax_assessment = tax_assessment or {}
    with solara.Card("Tax notes"):
        if tax_blocked:
            # Show the enforcement, not just the finding. A flag the client
            # can see next to a recommendation they were still given is worse
            # than no flag at all.
            solara.Warning(
                "Blocked from this cycle's recommendations by the wash-sale rule: "
                + ", ".join(tax_blocked)
            )
        elif tax_assessment.get("wash_sale_flags"):
            solara.Markdown(
                "**Wash-sale flagged tickers:** "
                + ", ".join(tax_assessment["wash_sale_flags"])
            )
        for note in tax_assessment.get("tax_efficiency_notes") or []:
            solara.Markdown(f"- {note}")


@solara.component
def ResultsView(result, session, on_approve):
    if not result:
        return

    if result.get("status") == "pending_approval":
        interrupt = result.get("interrupt") or {}
        solara.Warning(f"This run is paused for human review: {interrupt.get('reason')}")
        MarketRegimePanel(interrupt.get("market_regime"))
        if session.may_approve:
            with solara.Row():
                solara.Button("Approve", on_click=lambda: on_approve(True), color="primary")
                solara.Button("Reject", on_click=lambda: on_approve(False), color="error")
        else:
            # Offering a button the server will refuse teaches the operator
            # that the system is broken rather than that the control is
            # working. Name the role that can clear it instead.
            solara.Info(
                f"Signed in as {session.role}. A compliance user must clear this run -- "
                "the advisor who requested it cannot approve their own work."
            )
        return

    solara.Success("Analysis complete.")

    if result.get("llm_enabled") is False:
        solara.Warning(
            "No GEMINI_API_KEY is configured, so the market regime call, the stock "
            "ranking and the written report below came from the system's "
            "deterministic fallbacks rather than the LLM. The quantitative analysis "
            "and every guardrail are unaffected -- those never use an LLM."
        )

    if result.get("approval_state") == "rejected":
        solara.Error(
            "This run was rejected at human review. The report is retained for the "
            "record; its recommendations were not issued."
        )

    PortfolioDiagnosticsPanel(result.get("portfolio_diagnostics"))
    MarketRegimePanel(result.get("market_regime"))
    RecommendationsPanel(result.get("suitability_result"))
    TaxPanel(result.get("tax_assessment"), result.get("tax_blocked_recommendations"))

    with solara.Card("Full report"):
        solara.Markdown(result.get("final_report") or "No report generated.")


# --- Page --------------------------------------------------------------------


@solara.component
def Page():
    session: Optional[api.Session]
    session, set_session = solara.use_state(None)
    clients: List[Dict[str, Any]]
    clients, set_clients = solara.use_state([])
    selected_client_id, set_selected_client_id = solara.use_state(None)
    showing_new_form, set_showing_new_form = solara.use_state(False)
    result, set_result = solara.use_state(None)
    run_id, set_run_id = solara.use_state(None)
    progress, set_progress = solara.use_state(None)
    error, set_error = solara.use_state(None)

    @solara.lab.use_task(dependencies=[session])
    async def refresh_clients():
        if session is None:
            return
        try:
            set_clients(await _call(api.list_clients, session))
        except api.ApiError as exc:
            set_error(exc.message)

    @solara.lab.use_task(dependencies=None)
    async def run_analysis():
        if session is None or not selected_client_id:
            return
        set_error(None)
        set_result(None)
        set_run_id(None)
        try:
            job = await _call(api.trigger_run, session, selected_client_id)
        except api.ApiError as exc:
            set_error(exc.message)
            return

        # The run is queued, not done. Polling here rather than waiting on the
        # response is what makes a run that outlasts an HTTP timeout -- which
        # is most of them -- survivable, and it lets the operator see which
        # agent is running.
        job_id = job["job_id"]
        waited = 0.0
        while waited < POLL_CEILING_SECONDS:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            waited += POLL_INTERVAL_SECONDS
            try:
                state = await _call(api.get_job, session, job_id)
            except api.ApiError as exc:
                set_error(exc.message)
                return

            set_progress(
                {
                    "pct": state.get("progress_pct") or 0.0,
                    "step": state.get("current_step") or state.get("status"),
                }
            )
            if state.get("status") not in api.TERMINAL_JOB_STATES:
                continue

            set_progress(None)
            if state.get("status") != "succeeded":
                set_error(
                    state.get("error")
                    or f"The run {state.get('status')} without producing a report."
                )
                return

            payload = state.get("result") or {}
            set_run_id(payload.get("run_id"))
            if payload.get("status") == "pending_approval":
                set_result(payload)
                return
            try:
                report = await _call(api.get_report, session, payload["report_id"])
            except api.ApiError as exc:
                set_error(exc.message)
                return
            set_result(api.view_model(report))
            return

        set_error(
            "The run is taking longer than this page will wait. It is still running "
            "on the worker -- reopen this client shortly to see the result."
        )
        set_progress(None)

    @solara.lab.use_task(dependencies=None)
    async def decide(approved: bool):
        if session is None or not run_id:
            return
        set_error(None)
        try:
            outcome = await _call(api.approve_run, session, run_id, approved)
            report_id = (outcome or {}).get("report_id")
            if report_id:
                set_result(api.view_model(await _call(api.get_report, session, report_id)))
            else:
                set_result(outcome)
        except api.ApiError as exc:
            set_error(exc.message)

    if session is None:
        with solara.Card("AI Wealth Manager"):
            LoginForm(set_session)
        return

    busy = run_analysis.pending or decide.pending

    with solara.Sidebar():
        solara.Markdown("# AI Wealth Manager")
        solara.Markdown(
            f"Signed in as **{session.user.get('email', 'unknown')}** ({session.role})"
        )

        label_to_id = {f"{c['name']} (#{c['id']})": c["id"] for c in clients}
        id_to_label = {v: k for k, v in label_to_id.items()}
        if clients:
            solara.Select(
                label="Client",
                value=id_to_label.get(selected_client_id),
                values=list(label_to_id.keys()),
                on_value=lambda label: set_selected_client_id(label_to_id.get(label)),
            )
        else:
            solara.Info("No clients yet.")

        if session.role in ("advisor", "compliance", "admin"):
            solara.Button(
                "New client",
                on_click=lambda: set_showing_new_form(not showing_new_form),
            )

        if selected_client_id and not showing_new_form and session.may_run:
            solara.Button(
                "Run analysis", on_click=run_analysis, color="primary", disabled=busy
            )

        if progress:
            solara.Markdown(f"_{progress['step']}_")
            solara.ProgressLinear(value=progress["pct"])
        elif busy:
            solara.ProgressLinear()

        solara.Button("Sign out", on_click=lambda: set_session(None), text=True)

    with solara.Card("Dashboard"):
        if error:
            solara.Error(error)

        if showing_new_form:
            def created(client):
                set_clients(clients + [client])
                set_selected_client_id(client["id"])
                set_showing_new_form(False)

            NewClientForm(session, created)
        elif not selected_client_id:
            solara.Info("Select or create a client to get started.")
        elif not result:
            client = next((c for c in clients if c["id"] == selected_client_id), None)
            if client:
                solara.Markdown(f"### {client['name']}")
                solara.Markdown(
                    f"Age {client.get('age')} | Risk tolerance: {client['risk_tolerance']} | "
                    f"Horizon: {client['time_horizon_years']} years | "
                    f"Net worth: ${client['net_worth']:,.0f}"
                )
                accounts = client.get("accounts") or []
                if accounts:
                    solara.DataFrame(pd.DataFrame(accounts))
            solara.Info("Click 'Run analysis' in the sidebar to generate a portfolio review.")
        else:
            ResultsView(result, session, lambda approved: decide(approved))


@solara.component
def Layout(children):
    return solara.AppLayout(children=children)
