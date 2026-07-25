from typing import Dict, List

from ddgs import DDGS

from logging_setup import get_logger

logger = get_logger(__name__)

# DuckDuckGo is a best-effort, unauthenticated source: it rate-limits, and it
# has no SLA. News is supplementary context for the regime call (the
# deterministic price signals are the ground truth), so a failure here
# degrades the analysis rather than failing it -- but it must be bounded, or
# a hanging search blocks the whole graph run.
SEARCH_TIMEOUT_SECONDS = 15


def search_financial_news(keywords: List[str], max_results: int = 5) -> List[Dict[str, str]]:
    """
    Searches for financial news using DuckDuckGo. Returns [] on any failure.
    """
    results = []
    query = " ".join(keywords) + " financial news"

    try:
        with DDGS(timeout=SEARCH_TIMEOUT_SECONDS) as ddgs:
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
        logger.warning(
            "News search for %r failed (%s); continuing without news context.", query, e
        )
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
