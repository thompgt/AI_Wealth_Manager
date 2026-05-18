import pytest
from fastapi.testclient import TestClient
from server import app
from db import SessionLocal, UserProfile
from agents.compliance_critic import compliance_critic_node

client = TestClient(app)

def test_valid_scenario_and_wash_sale():
    # Because user 1 has a TSLA wash-sale violation seeded in the DB,
    # the workflow will catch it and propose the alternative asset.
    response = client.post("/api/v1/rebalance", json={"user_id": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    trades = data["trades_executed"]
    assert len(trades) > 0
    # TSLA should have been caught and avoided
    executed_assets = [t["asset"] for t in trades]
    assert "TSLA" not in executed_assets

def test_compliance_violation():
    # We can test the node directly for compliance failure
    state = {
        "user_id": 1,
        "net_worth": 500000.0,
        "current_portfolio": {"CASH": 400000.0},
        "market_condition": "Bull",
        "proposed_trades": [{"asset": "AAPL", "action": "BUY", "amount": 200000.0}], # > 30% of 500k (150k)
        "wash_sale_violations": [],
        "compliance_passed": False,
        "compliance_errors": [],
        "final_portfolio": {},
        "needs_rebuild": False
    }
    
    with pytest.raises(ValueError) as excinfo:
        compliance_critic_node(state)
        
    assert "Compliance Violations" in str(excinfo.value)
    assert "exceeds 30% of net worth" in str(excinfo.value)

def test_penny_stock_violation():
    state = {
        "user_id": 1,
        "net_worth": 500000.0,
        "current_portfolio": {"CASH": 50000.0},
        "market_condition": "Bull",
        "proposed_trades": [{"asset": "DOGE", "action": "BUY", "amount": 10000.0}], 
        "wash_sale_violations": [],
        "compliance_passed": False,
        "compliance_errors": [],
        "final_portfolio": {},
        "needs_rebuild": False
    }
    
    with pytest.raises(ValueError) as excinfo:
        compliance_critic_node(state)
        
    assert "Compliance Violations" in str(excinfo.value)
    assert "Cannot buy penny stock" in str(excinfo.value)
