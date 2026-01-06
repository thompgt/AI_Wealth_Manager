import json
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os

# --- Mock Services ---

def mock_calculate_risk(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock risk calculation engine (e.g., PyPortfolioOpt).
    In reality, this would compute efficient frontier, Sharpe ratios, etc.
    """
    # specific logic based on inputs to make it feel 'real'
    holdings = portfolio.get("holdings", [])
    total_value = sum(h.get("value", 0) for h in holdings)
    
    # Mock result
    return {
        "current_risk_score": 65,  # 0-100
        "suggested_allocation": {
            "stocks": 0.60,
            "bonds": 0.30,
            "crypto": 0.10
        },
        "volatility": "High" if total_value > 10000 else "Moderate",
        "sharpe_ratio": 1.2
    }

#TODO
def mock_fetch_news(portfolio: Dict[str, Any]) -> List[str]:
    """
    Mock NLP News Intelligence Layer.
    In reality, this would query Newspaper3k + Vector DB.
    """
    holdings = [h.get("symbol") for h in portfolio.get("holdings", [])]
    news = []
    
    if "AAPL" in holdings:
        news.append("Apple reports record quarterly earnings, beating expectations.")
    if "TSLA" in holdings:
        news.append("Tesla faces new regulatory scrutiny in Europe regarding FSD.")
    if "BTC" in holdings:
        news.append("Bitcoin surges past key resistance levels amidst global uncertainty.")
        
    if not news:
        news.append("Market remains volatile as inflation data is awaited.")
        
    return news

# --- The Agent Workflow ---

class WealthManagerAgent:
    def __init__(self):
        # Initialize LLM (Ensure GOOGLE_API_KEY is set in env)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY", "mock-key-if-not-set") 
        )
        
        # Define the Prompt
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an elite AI Wealth Manager. 
            
            Analyze the following user data and produce a concise, actionable investment recommendation.
            
            User Portfolio Context:
            {portfolio_context}
            
            Risk Analysis:
            {risk_analysis}
            
            Relevant Market News:
            {news_insights}
            
            Output your response in a professional tone, addressing the user directly. 
            Structure it as:
            1. Portfolio Health Check
            2. Key News Impacts
            3. Recommended Actions
            """
        )
        
        # Build the Chain
        # 1. Calculate Risk (Mock)
        # 2. Fetch News (Mock)
        # 3. Format everything for the prompt
        # 4. Invoke LLM
        self.chain = (
            {
                "portfolio_context": RunnablePassthrough(),
                "risk_analysis": RunnableLambda(mock_calculate_risk),
                "news_insights": RunnableLambda(mock_fetch_news)
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def run(self, user_portfolio: Dict[str, Any]) -> str:
        """
        Executes the wealth manager workflow.
        """
        try:
            # invocing the chain
            result = self.chain.invoke(user_portfolio)
            return result
        except Exception as e:
            return f"Error executing agent workflow: {str(e)}"

# Singleton instance for easy import
agent = WealthManagerAgent()
