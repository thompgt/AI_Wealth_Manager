from state import AgentState

def quant_builder_node(state: AgentState) -> AgentState:
    print(">>> [Quant Builder] Drafting portfolio adjustments...")
    condition = state.get("market_condition", "Bull")
    violations = state.get("wash_sale_violations", [])
    
    current_cash = state["current_portfolio"].get("CASH", 0.0)
    proposed_trades = []
    
    # We simulate logic to deploy cash.
    if current_cash > 0:
        if condition == "Bull":
            target_asset = "AAPL"
            alt_asset = "MSFT"
        elif condition == "Bear":
            target_asset = "GOLD"
            alt_asset = "BND"
        else:
            target_asset = "VIXY"
            alt_asset = "CASH"
            
        asset_to_buy = target_asset
        if asset_to_buy in violations:
            print(f"    [Quant Builder] Avoiding {asset_to_buy} due to wash-sale. Switching to {alt_asset}.")
            asset_to_buy = alt_asset
            
        if asset_to_buy != "CASH":
            trade_amount = min(current_cash, 50000.0)
            proposed_trades.append({
                "asset": asset_to_buy,
                "action": "BUY",
                "amount": trade_amount
            })
            print(f"    [Quant Builder] Proposed BUY {trade_amount} of {asset_to_buy}")
    
    # To test the wash-sale, let's explicitly propose a TSLA buy on the first pass if no violations.
    # In the seed data, TSLA was sold at a loss 15 days ago.
    if "TSLA" not in violations and current_cash >= 10000:
        proposed_trades.append({
            "asset": "TSLA",
            "action": "BUY",
            "amount": 10000.0
        })
        print(f"    [Quant Builder] Proposed BUY 10000.0 of TSLA")
        
    state["proposed_trades"] = proposed_trades
    state["needs_rebuild"] = False
    return state
