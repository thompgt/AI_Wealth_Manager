from duckduckgo_search import DDGS
from typing import List, Dict

def search_financial_news(keywords: List[str], max_results: int = 5) -> List[Dict[str, str]]:
    """
    Searches for financial news using DuckDuckGo.
    """
    results = []
    query = " ".join(keywords) + " financial news"
    
    try:
        with DDGS() as ddgs:
            # DDGS.text() returns a generator
            search_gen = ddgs.text(query, region='wt-wt', safesearch='off', timelimit='d', max_results=max_results)
            for r in search_gen:
                results.append({
                    "title": r.get('title'),
                    "link": r.get('href'),
                    "snippet": r.get('body')
                })
    except Exception as e:
        print(f"Error searching news: {e}")
        return []

    return results

def get_portfolio_news(tickers: List[str]) -> str:
    """
    Aggregates news for a list of tickers into a single string context.
    """
    # Group searches to avoid rate limits or too many queries?
    # For now, just search for the top 3 holdings or the portfolio as a whole if small.
    # Let's search for top 3 holdings.
    
    if not tickers:
        return "No tickers provided for news search."

    context = ""
    for ticker in tickers[:3]: # Limit to top 3 to save time/tokens
        news_items = search_financial_news([ticker], max_results=3)
        if news_items:
            context += f"\n--- News for {ticker} ---\n"
            for item in news_items:
                context += f"- {item['title']}: {item['snippet']} ({item['link']})\n"
    
    if not context:
        return "No specific news found."
        
    return context
