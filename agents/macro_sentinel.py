import google.generativeai as genai
from config import settings
from state import AgentState

genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

def macro_sentinel_node(state: AgentState) -> AgentState:
    print(">>> [Macro Sentinel] Analyzing market conditions...")
    # In a real app, this would read market streams. We simulate by prompting Gemini.
    prompt = "You are a financial analyst. Based on current global events (assume moderate inflation and steady job growth), classify the market condition into exactly one word: Bull, Bear, or Volatile. Just return the word."
    
    try:
        response = model.generate_content(prompt)
        condition = response.text.strip().capitalize()
        if condition not in ["Bull", "Bear", "Volatile"]:
            condition = "Bull"
    except Exception as e:
        print(f"    [Macro Sentinel] Gemini API error/fallback (Using Bull): {e}")
        condition = "Bull"
        
    print(f"    [Macro Sentinel] Condition determined: {condition}")
    state["market_condition"] = condition
    return state
