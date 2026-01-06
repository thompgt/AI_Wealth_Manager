from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.agents.agent_workflow import agent
import uvicorn
import os

# Define the FastAPI app
app = FastAPI(
    title="AI Wealth Manager API",
    description="Backend API for Portfolio Analysis and Report Generation",
    version="0.1.0"
)

# --- Data Models ---

class Holding(BaseModel):
    symbol: str
    quantity: float
    value: float
    asset_class: str # e.g., "Equity", "Crypto", "Bond"

class PortfolioInput(BaseModel):
    user_id: str
    risk_tolerance: str # e.g., "Low", "Moderate", "High"
    holdings: List[Holding]

class ReportResponse(BaseModel):
    user_id: str
    recommendation: str
    status: str

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "AI Wealth Manager API is running"}

@app.post("/generate-report", response_model=ReportResponse)
async def generate_report(portfolio: PortfolioInput):
    """
    Triggers the AI Agent workflow to generate a wealth management report.
    """
    try:
        # Convert Pydantic model to dict for the agent
        portfolio_dict = portfolio.model_dump()
        
        # Run the agent workflow
        # In a real app, this might be a background task (Celery/Arq)
        result = agent.run(portfolio_dict)
        
        return ReportResponse(
            user_id=portfolio.user_id,
            recommendation=result,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
