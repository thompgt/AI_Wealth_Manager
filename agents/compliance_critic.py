from state import AgentState

def compliance_critic_node(state: AgentState) -> AgentState:
    print(">>> [Compliance Critic] Evaluating deterministic rules...")
    proposed_trades = state.get("proposed_trades", [])
    current_portfolio = dict(state["current_portfolio"])
    net_worth = state["net_worth"]
    
    errors = []
    
    for trade in proposed_trades:
        asset = trade["asset"]
        amount = trade["amount"]
        
        # Rule: No penny stocks
        if asset == "DOGE":
            errors.append(f"Cannot buy penny stock: {asset}")
            
        if trade["action"] == "BUY":
            current_portfolio["CASH"] = current_portfolio.get("CASH", 0) - amount
            current_portfolio[asset] = current_portfolio.get(asset, 0) + amount
        elif trade["action"] == "SELL":
            current_portfolio["CASH"] = current_portfolio.get("CASH", 0) + amount
            current_portfolio[asset] = current_portfolio.get(asset, 0) - amount
            
    # Rule: No single asset (excluding CASH) can be over 30% of net worth
    for asset, amount in current_portfolio.items():
        if asset != "CASH" and amount > (0.30 * net_worth):
            errors.append(f"Asset {asset} allocation ({amount}) exceeds 30% of net worth ({net_worth}).")
            
    if errors:
        print(f"    [Compliance Critic] Blocked! Errors: {errors}")
        state["compliance_passed"] = False
        state["compliance_errors"] = errors
        raise ValueError(f"Compliance Violations: {errors}")
    
    print("    [Compliance Critic] Approved! Trades committed to final portfolio.")
    state["compliance_passed"] = True
    state["compliance_errors"] = []
    state["final_portfolio"] = current_portfolio
    
    return state
