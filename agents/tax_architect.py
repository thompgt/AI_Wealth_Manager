from datetime import datetime, timedelta, timezone
from state import AgentState
from db import SessionLocal, TransactionLog

def tax_architect_node(state: AgentState) -> AgentState:
    print(">>> [Tax Architect] Inspecting proposed trades for wash-sale violations...")
    user_id = state["user_id"]
    proposed_trades = state.get("proposed_trades", [])
    violations = state.get("wash_sale_violations", [])
    
    db = SessionLocal()
    thirty_days_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)
    
    recent_sells = db.query(TransactionLog).filter(
        TransactionLog.user_id == user_id,
        TransactionLog.transaction_type == "SELL",
        TransactionLog.timestamp >= thirty_days_ago
    ).all()
    
    sell_assets = {log.asset_symbol for log in recent_sells}
    db.close()
    
    needs_rebuild = False
    new_violations = list(violations)
    
    for trade in proposed_trades:
        if trade["action"] == "BUY" and trade["asset"] in sell_assets:
            if trade["asset"] not in new_violations:
                print(f"    [Tax Architect] FLAG: Wash-sale violation detected for {trade['asset']}!")
                new_violations.append(trade["asset"])
                needs_rebuild = True
                
    if needs_rebuild:
        print("    [Tax Architect] Sending back to Quant Builder for rebuild.")
    else:
        print("    [Tax Architect] All trades cleared wash-sale checks.")
        
    state["wash_sale_violations"] = new_violations
    state["needs_rebuild"] = needs_rebuild
    return state
