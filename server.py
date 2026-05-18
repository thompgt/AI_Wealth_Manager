from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from workflow import app_workflow
from db import SessionLocal, UserProfile, TransactionLog
from datetime import datetime

app = FastAPI(title="AI Wealth Manager Engine")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class RebalanceRequest(BaseModel):
    user_id: int

@app.post("/api/v1/rebalance")
def trigger_rebalance(req: RebalanceRequest, db: Session = Depends(get_db)):
    user = db.query(UserProfile).filter(UserProfile.id == req.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    initial_state = {
        "user_id": user.id,
        "net_worth": user.net_worth,
        "current_portfolio": dict(user.portfolio),
        "market_condition": None,
        "proposed_trades": [],
        "wash_sale_violations": [],
        "compliance_passed": False,
        "compliance_errors": [],
        "final_portfolio": {},
        "needs_rebuild": False
    }
    
    try:
        final_state = app_workflow.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    if not final_state.get("compliance_passed"):
        raise HTTPException(status_code=400, detail="Compliance check failed")
        
    # Commit the trades to the db
    # In a real app we'd verify the trade execution here
    for trade in final_state["proposed_trades"]:
        new_log = TransactionLog(
            user_id=user.id,
            asset_symbol=trade["asset"],
            transaction_type=trade["action"],
            amount=trade["amount"],
            price_per_unit=1.0, # Dummy price
            timestamp=datetime.utcnow()
        )
        db.add(new_log)
        
    user.portfolio = final_state["final_portfolio"]
    db.commit()
    
    return {
        "status": "success",
        "market_condition": final_state.get("market_condition"),
        "trades_executed": final_state["proposed_trades"],
        "final_portfolio": final_state["final_portfolio"]
    }
