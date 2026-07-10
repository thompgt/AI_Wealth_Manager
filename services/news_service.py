from typing import Dict, List

from ddgs import DDGS


def search_financial_news(keywords: List[str], max_results: int = 5) -> List[Dict[str, str]]:
    """
    Searches for financial news using DuckDuckGo.
    """
    results = []
    query = " ".join(keywords) + " financial news"

    try:
        with DDGS() as ddgs:
            search_gen = ddgs.text(query, region="wt-wt", safesearch="off", timelimit="d", max_results=max_results)
            for r in search_gen:
                results.append(
                    {
                        "title": r.get("title"),
                        "link": r.get("href"),
                        "snippet": r.get("body"),
                    }
                )
    except Exception as e:
        print(f"Error searching news: {e}")
        return []

    return results


def get_portfolio_news(tickers: List[str]) -> str:
    """
    Aggregates news for a list of tickers into a single string context.
    Limited to the top 3 tickers to save time/tokens.
    """
    if not tickers:
        return "No tickers provided for news search."

    context = ""
    for ticker in tickers[:3]:
        news_items = search_financial_news([ticker], max_results=3)
        if news_items:
            context += f"\n--- News for {ticker} ---\n"
            for item in news_items:
                context += f"- {item['title']}: {item['snippet']} ({item['link']})\n"

    if not context:
        return "No specific news found."

    return context
