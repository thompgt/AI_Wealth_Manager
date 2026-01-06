from typing import Dict, Any, List
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from app.analytics.risk_engine import calculate_portfolio_metrics
from app.services.news_service import get_portfolio_news
from app.core.memory import memory
from app.core.config import settings
import json

# --- Tools ---

@tool
def analyze_portfolio_risk(holdings_json: str) -> str:
    """
    Calculates risk metrics (Sharpe, Volatility, Returns) for a given portfolio.
    Input must be a JSON string of a list of holdings: [{"symbol": "AAPL", "value": 1000}, ...]
    """
    try:
        holdings = json.loads(holdings_json)
        metrics = calculate_portfolio_metrics(holdings)
        return json.dumps(metrics)
    except Exception as e:
        return f"Error calculating risk: {str(e)}"

@tool
def fetch_market_news(tickers_json: str) -> str:
    """
    Searches for latest financial news for a list of tickers.
    Input must be a JSON string of a list of ticker symbols: ["AAPL", "TSLA"]
    """
    try:
        tickers = json.loads(tickers_json)
        news_summary = get_portfolio_news(tickers)
        return news_summary
    except Exception as e:
        return f"Error fetching news: {str(e)}"

# --- Agent Class ---

class WealthManagerAgent:
    def __init__(self):
        # Tools available to the agent
        self.tools = [analyze_portfolio_risk, fetch_market_news]
        
        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.5,
            google_api_key=settings.GOOGLE_API_KEY
        )

        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             """You are an expert AI Wealth Manager. Your goal is to provide personalized, data-driven investment advice.
             
             User Profile:
             {user_profile}
             
             You have access to tools to calculate real-time risk metrics and fetch live news.
             ALWAYS use the 'analyze_portfolio_risk' tool first to get the ground truth on the portfolio's performance.
             Then use 'fetch_market_news' to understand the context for the major holdings.
             
             Synthesize all data into a professional report.
             Structure your response clearly:
             1. **Portfolio Health Check**: Use the metrics (Sharpe, Volatility).
             2. **News Intelligence**: Summarize key drivers found in news.
             3. **Strategic Recommendations**: Actionable advice based on risk profile and market data.
             """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Construct the Agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self, user_id: str, portfolio: Dict[str, Any]) -> str:
        """
        Main entry point.
        """
        # 1. Retrieve User Memory
        user_profile = memory.get_user_profile(user_id)
        profile_str = json.dumps(user_profile, indent=2)
        
        # 2. Prepare Input
        # We pass the portfolio directly in the input string so the model knows what to pass to tools
        holdings = portfolio.get("holdings", [])
        holdings_simple = [{"symbol": h["symbol"], "value": h["value"]} for h in holdings]
        tickers = [h["symbol"] for h in holdings_simple]
        
        input_text = f"""
        Here is my current portfolio:
        {json.dumps(holdings_simple)}
        
        My risk tolerance is: {portfolio.get('risk_tolerance', 'Moderate')}
        
        Please generate a comprehensive wealth report.
        """
        
        # 3. Execute
        try:
            result = self.agent_executor.invoke({
                "input": input_text,
                "user_profile": profile_str
            })
            response = result['output']
            
            # 4. Save Interaction to Memory
            memory.add_interaction(user_id, input_text, response)
            
            return response
        except Exception as e:
            return f"Agent Error: {str(e)}"

# Singleton
agent = WealthManagerAgent()