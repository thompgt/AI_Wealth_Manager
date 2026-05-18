from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    user_id: int
    net_worth: float
    current_portfolio: Dict[str, float]
    
    # Set by Macro Sentinel
    market_condition: Optional[str] # "Bull", "Bear", "Volatile"
    
    # Set by Quant Builder
    proposed_trades: List[Dict[str, Any]] # [{"asset": "AAPL", "action": "BUY", "amount": 10000}]
    
    # Set by Tax Architect
    wash_sale_violations: List[str] # List of assets that violated the rule
    
    # Set by Compliance Critic
    compliance_passed: bool
    compliance_errors: List[str]
    
    # Final output
    final_portfolio: Dict[str, float]
    
    # Control flow flags
    needs_rebuild: bool
