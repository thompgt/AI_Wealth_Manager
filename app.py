import solara
import httpx
import pandas as pd
import plotly.express as px

# Configuration
API_URL = "http://localhost:8000/api/v1/rebalance"

@solara.component
def Page():
    user_id, set_user_id = solara.use_state(1)
    loading, set_loading = solara.use_state(False)
    result, set_result = solara.use_state(None)
    error, set_error = solara.use_state(None)

    async def trigger_rebalance():
        set_loading(True)
        set_error(None)
        set_result(None)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(API_URL, json={"user_id": int(user_id)}, timeout=60.0)
                if response.status_code == 200:
                    set_result(response.json())
                else:
                    set_error(f"Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            set_error(f"Failed to connect to backend: {str(e)}")
        finally:
            set_loading(False)

    with solara.Sidebar():
        solara.Markdown("# 🏦 Wealth Manager Control")
        solara.InputInt("Enter User ID", value=user_id, on_value=set_user_id)
        solara.Button("🚀 Trigger AI Rebalance", on_click=trigger_rebalance, color="primary", disabled=loading)
        if loading:
            solara.ProgressLinear()

    with solara.Card("AI Wealth Manager Dashboard"):
        if error:
            solara.Error(error)
        
        if not result and not error:
            solara.Info("Enter a User ID and click the button to start the autonomous rebalancing workflow.")
            solara.Markdown("""
            ### What happens when you click?
            1. **Macro Analysis**: Gemini AI checks market trends.
            2. **Portfolio Strategy**: Quant algorithms draft new trades.
            3. **Tax Check**: Transaction logs are scanned for wash-sale rules.
            4. **Compliance Audit**: Strict rules are checked before any money moves.
            """)

        if result:
            solara.Success("Rebalance Completed Successfully!")
            
            with solara.Columns([1, 1]):
                with solara.Card("Market Summary"):
                    solara.Markdown(f"**Condition:** {result['market_condition']}")
                    solara.Markdown(f"**Trades Executed:** {len(result['trades_executed'])}")
                
                with solara.Card("Trades Executed"):
                    if result['trades_executed']:
                        df_trades = pd.DataFrame(result['trades_executed'])
                        solara.DataFrame(df_trades)
                    else:
                        solara.Markdown("No trades necessary for this cycle.")

            with solara.Card("New Portfolio Allocation"):
                portfolio = result['final_portfolio']
                df_portfolio = pd.DataFrame([{"Asset": k, "Value": v} for k, v in portfolio.items()])
                fig = px.pie(df_portfolio, values='Value', names='Asset', title="Current Portfolio Breakdown")
                solara.FigurePlotly(fig)
                solara.DataFrame(df_portfolio)

@solara.component
def Layout(children):
    return solara.AppLayout(children=children)
