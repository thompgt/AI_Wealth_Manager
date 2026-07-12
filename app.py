import solara
import httpx
import pandas as pd
import plotly.express as px

from config import settings

API_BASE = "http://localhost:8000/api/v1"
HEADERS = {"X-API-Key": settings.API_AUTH_KEY}

RISK_QUESTIONS = [
    {
        "text": "If your portfolio dropped 20% in a month, you would:",
        "options": [
            ("Sell everything to stop further loss", 1),
            ("Sell some to reduce risk", 2),
            ("Hold and wait it out", 3),
            ("Buy more at the lower price", 4),
        ],
    },
    {
        "text": "How stable is your income over the next few years?",
        "options": [
            ("Very unstable / uncertain", 1),
            ("Somewhat stable", 2),
            ("Stable", 3),
            ("Very stable (e.g. tenured, government)", 4),
        ],
    },
    {
        "text": "How much investment loss can you tolerate in a single year without changing your plan?",
        "options": [
            ("None -- I need my principal protected", 1),
            ("Up to 10%", 2),
            ("Up to 25%", 3),
            ("More than 25%, for higher long-run growth", 4),
        ],
    },
]


def score_to_tier(total: int) -> str:
    # 3 questions, each scored 1-4 -> total in [3, 12]
    if total <= 6:
        return "Conservative"
    if total <= 9:
        return "Moderate"
    return "Aggressive"


def answers_to_tier(answers):
    if not all(a is not None for a in answers):
        return None
    return score_to_tier(sum(answers))


@solara.component
def RiskQuestionnaire(answers, set_answers):
    solara.Markdown("**Risk Tolerance Questionnaire**")
    for i, q in enumerate(RISK_QUESTIONS):
        label_to_score = {label: score for label, score in q["options"]}
        score_to_label = {score: label for label, score in q["options"]}
        current_label = score_to_label.get(answers[i])

        def make_setter(idx, label_map):
            def setter(label):
                new_answers = list(answers)
                new_answers[idx] = label_map.get(label)
                set_answers(new_answers)
            return setter

        solara.Select(
            label=q["text"],
            value=current_label,
            values=[label for label, _ in q["options"]],
            on_value=make_setter(i, label_to_score),
        )

    tier = answers_to_tier(answers)
    if tier:
        solara.Info(f"Computed risk tolerance: **{tier}**")


@solara.component
def NewClientForm(on_created):
    name, set_name = solara.use_state("")
    email, set_email = solara.use_state("")
    age, set_age = solara.use_state(35)
    horizon, set_horizon = solara.use_state(10)
    goals_text, set_goals_text = solara.use_state("retirement")
    net_worth, set_net_worth = solara.use_state(100000.0)
    portfolio_mode, set_portfolio_mode = solara.use_state("New investment (let the agents build it)")
    holdings_rows, set_holdings_rows = solara.use_state([{"symbol": "CASH", "quantity": 100000.0, "cost_basis": 1.0}])
    answers, set_answers = solara.use_state([None, None, None])
    error, set_error = solara.use_state(None)

    solara.InputText("Name", value=name, on_value=set_name)
    solara.InputText("Email", value=email, on_value=set_email)
    solara.InputInt("Age", value=age, on_value=set_age)
    solara.InputInt("Time horizon (years)", value=horizon, on_value=set_horizon)
    solara.InputText("Goals (comma-separated)", value=goals_text, on_value=set_goals_text)
    solara.InputFloat("Total net worth ($)", value=net_worth, on_value=set_net_worth)

    RiskQuestionnaire(answers, set_answers)
    tier = answers_to_tier(answers)

    solara.Select(
        label="Portfolio setup",
        value=portfolio_mode,
        values=["New investment (let the agents build it)", "I have an existing portfolio"],
        on_value=set_portfolio_mode,
    )

    if portfolio_mode == "New investment (let the agents build it)":
        solara.Info(
            f"Your full ${net_worth:,.0f} will start as uninvested cash. Running analysis will "
            "have the agents diagnose that as 100% cash drag and pick specific stocks/ETFs and "
            "dollar amounts to deploy it into."
        )
    else:
        solara.Markdown("**Current Holdings**")
        for idx, row in enumerate(holdings_rows):
            with solara.Row():
                def make_row_setter(i, field):
                    def setter(value):
                        new_rows = [dict(r) for r in holdings_rows]
                        new_rows[i][field] = value
                        set_holdings_rows(new_rows)
                    return setter

                solara.InputText("Symbol", value=row["symbol"], on_value=make_row_setter(idx, "symbol"))
                solara.InputFloat("Quantity ($ for CASH)", value=row["quantity"], on_value=make_row_setter(idx, "quantity"))
                solara.InputFloat("Cost basis", value=row["cost_basis"], on_value=make_row_setter(idx, "cost_basis"))

        def add_row():
            set_holdings_rows(holdings_rows + [{"symbol": "", "quantity": 0.0, "cost_basis": 0.0}])

        solara.Button("+ Add holding", on_click=add_row)
        solara.Markdown(
            "Any uninvested cash you list above (symbol `CASH`) is what the agents will have "
            "available to deploy into new recommendations."
        )

    @solara.lab.use_task(dependencies=None)
    async def submit_task():
        if not tier:
            set_error("Please answer all risk questionnaire questions.")
            return
        set_error(None)
        try:
            if portfolio_mode == "New investment (let the agents build it)":
                holdings_payload = [{"symbol": "CASH", "quantity": net_worth, "cost_basis": 1.0}]
            else:
                holdings_payload = [r for r in holdings_rows if r["symbol"]]

            payload = {
                "name": name,
                "email": email or None,
                "age": age,
                "risk_tolerance": tier,
                "time_horizon_years": horizon,
                "goals": [g.strip() for g in goals_text.split(",") if g.strip()],
                "net_worth": net_worth,
                "holdings": holdings_payload,
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{API_BASE}/clients", json=payload, headers=HEADERS, timeout=30.0)
                if response.status_code == 200:
                    on_created(response.json())
                else:
                    set_error(f"Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            set_error(f"Failed to connect to backend: {str(e)}")

    if error:
        solara.Error(error)
    solara.Button("Create Client", on_click=submit_task, color="primary", disabled=submit_task.pending)
    if submit_task.pending:
        solara.ProgressLinear()


@solara.component
def PortfolioDiagnosticsPanel(diagnostics):
    if not diagnostics:
        solara.Info("No diagnostics available.")
        return
    with solara.Columns([1, 1]):
        with solara.Card("Portfolio Health"):
            solara.Markdown(f"**Sharpe ratio:** {diagnostics.get('sharpe_ratio')}")
            solara.Markdown(f"**Annual return:** {diagnostics.get('annual_return')}")
            solara.Markdown(f"**Annual volatility:** {diagnostics.get('annual_volatility')}")
            solara.Markdown(f"**Diversification score:** {diagnostics.get('diversification_score')}/100")
        with solara.Card("Concentration"):
            concentration = diagnostics.get("concentration") or {}
            if concentration:
                df = pd.DataFrame([{"Symbol": k, "Fraction": v} for k, v in concentration.items()])
                fig = px.pie(df, values="Fraction", names="Symbol", title="Portfolio Concentration")
                solara.FigurePlotly(fig)

    flaws = diagnostics.get("flaws") or []
    with solara.Card("Diagnosed Flaws"):
        if flaws:
            for f in flaws:
                solara.Markdown(f"- {f}")
        else:
            solara.Markdown("No flaws detected.")


@solara.component
def MarketRegimePanel(regime):
    if not regime:
        solara.Info("No market regime data available.")
        return
    with solara.Card(f"Market Regime: {regime.get('regime_label')} (confidence {regime.get('confidence')})"):
        solara.Markdown(regime.get("narrative", ""))


@solara.component
def RecommendationsPanel(suitability):
    if not suitability:
        solara.Info("No suitability/recommendation data available.")
        return
    with solara.Card("Recommended Candidates"):
        recs = suitability.get("adjusted_recommendations") or []
        if recs:
            total_allocated = sum(r.get("allocation_amount", 0.0) for r in recs)
            solara.Markdown(f"**Total recommended allocation: ${total_allocated:,.0f}**")
            df = pd.DataFrame(
                [
                    {
                        "Ticker": r.get("ticker"),
                        "Allocation ($)": r.get("allocation_amount", 0.0),
                        "Allocation (%)": r.get("allocation_pct", 0.0) * 100,
                        "Addresses Flaw": r.get("addresses_flaw"),
                        "Rationale": r.get("regime_fit_rationale"),
                        "Confidence": r.get("confidence"),
                    }
                    for r in recs
                ]
            )
            solara.DataFrame(df)
            fig = px.pie(df, values="Allocation ($)", names="Ticker", title="Recommended Allocation")
            solara.FigurePlotly(fig)
        else:
            solara.Markdown("No candidates were approved this cycle.")

    with solara.Card("Suitability / Compliance Notes"):
        solara.Markdown(f"**Approved:** {suitability.get('approved')}")
        violations = suitability.get("violations") or []
        if violations:
            for v in violations:
                solara.Markdown(f"- {v}")
        else:
            solara.Markdown("No violations recorded.")


@solara.component
def TaxPanel(tax_assessment):
    if not tax_assessment:
        return
    with solara.Card("Tax Notes"):
        flags = tax_assessment.get("wash_sale_flags") or []
        notes = tax_assessment.get("tax_efficiency_notes") or []
        if flags:
            solara.Markdown("**Wash-sale flagged tickers:** " + ", ".join(flags))
        for n in notes:
            solara.Markdown(f"- {n}")


@solara.component
def ResultsView(run_result, on_approve):
    if not run_result:
        return

    status = run_result.get("status")

    if status == "pending_approval":
        interrupt = run_result.get("interrupt") or {}
        solara.Warning(f"This run requires human approval: {interrupt.get('reason')}")
        MarketRegimePanel(interrupt.get("market_regime"))
        with solara.Row():
            solara.Button("Approve", on_click=lambda: on_approve(True), color="primary")
            solara.Button("Reject", on_click=lambda: on_approve(False), color="error")
        return

    solara.Success("Analysis complete.")
    PortfolioDiagnosticsPanel(run_result.get("portfolio_diagnostics"))
    MarketRegimePanel(run_result.get("market_regime"))
    RecommendationsPanel(run_result.get("suitability_result"))
    TaxPanel(run_result.get("tax_assessment"))

    with solara.Card("Full Report"):
        solara.Markdown(run_result.get("final_report") or "No report generated.")


@solara.component
def Page():
    clients, set_clients = solara.use_state([])
    selected_client_id, set_selected_client_id = solara.use_state(None)
    showing_new_form, set_showing_new_form = solara.use_state(False)
    run_result, set_run_result = solara.use_state(None)
    run_id, set_run_id = solara.use_state(None)
    error, set_error = solara.use_state(None)

    async def refresh_clients():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{API_BASE}/clients", headers=HEADERS, timeout=15.0)
                if response.status_code == 200:
                    set_clients(response.json())
        except Exception as e:
            set_error(f"Failed to connect to backend: {str(e)}")

    solara.lab.use_task(refresh_clients, dependencies=[])

    @solara.lab.use_task(dependencies=None)
    async def trigger_run_task():
        if not selected_client_id:
            return
        set_error(None)
        set_run_result(None)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{API_BASE}/clients/{selected_client_id}/run", headers=HEADERS, timeout=120.0
                )
                if response.status_code == 200:
                    data = response.json()
                    set_run_result(data)
                    set_run_id(data.get("run_id"))
                else:
                    set_error(f"Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            set_error(f"Failed to connect to backend: {str(e)}")

    @solara.lab.use_task(dependencies=None)
    async def approve_run_task(approved: bool):
        if not run_id:
            return
        set_error(None)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{API_BASE}/runs/{run_id}/approve",
                    json={"approved": approved, "decided_by": "solara-ui"},
                    headers=HEADERS,
                    timeout=120.0,
                )
                if response.status_code == 200:
                    set_run_result(response.json())
                else:
                    set_error(f"Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            set_error(f"Failed to connect to backend: {str(e)}")

    loading = trigger_run_task.pending or approve_run_task.pending

    def on_client_created(client):
        set_clients(clients + [client])
        set_selected_client_id(client["id"])
        set_showing_new_form(False)

    with solara.Sidebar():
        solara.Markdown("# AI Wealth Manager")

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
            solara.Info("No clients yet -- create one below.")

        solara.Button(
            "Toggle New Client Form",
            on_click=lambda: set_showing_new_form(not showing_new_form),
        )

        if selected_client_id and not showing_new_form:
            solara.Button(
                "Run Analysis",
                on_click=trigger_run_task,
                color="primary",
                disabled=loading,
            )
        if loading:
            solara.ProgressLinear()

    with solara.Card("AI Wealth Manager Dashboard"):
        if error:
            solara.Error(error)

        if showing_new_form:
            NewClientForm(on_client_created)
        elif not selected_client_id:
            solara.Info("Select or create a client to get started.")
        elif not run_result:
            client = next((c for c in clients if c["id"] == selected_client_id), None)
            if client:
                solara.Markdown(f"### {client['name']}")
                solara.Markdown(
                    f"Age {client['age']} | Risk tolerance: {client['risk_tolerance']} | "
                    f"Horizon: {client['time_horizon_years']} years | Net worth: ${client['net_worth']:,.0f}"
                )
                if client["holdings"]:
                    df = pd.DataFrame(client["holdings"])
                    solara.DataFrame(df)
            solara.Info("Click 'Run Analysis' in the sidebar to generate a portfolio review.")
        else:
            ResultsView(run_result, lambda approved: approve_run_task(approved))


@solara.component
def Layout(children):
    return solara.AppLayout(children=children)
