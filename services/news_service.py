"""News retrieval and deterministic sentiment scoring.

News is supplementary context: the deterministic price signals are the ground
truth for the regime call, and a news failure must degrade the analysis rather
than fail it. But "best effort" was doing a lot of work in the previous
version — in practice the search failed on *every* run (a bogus `wt-wt` region
code produced DNS lookups for `wt.wikipedia.org`), returned `[]`, and nothing
anywhere said the regime agent had been running without news context for its
entire history. Silent permanent degradation is the failure mode this whole
system is supposed to make impossible.

What changed:

* `region="us-en"` — the actual bug. `wt-wt` is not a region this backend
  accepts and it poisoned every query.
* A one-week window rather than one day. A 24-hour window returns nothing for
  most queries on a Monday morning, which reads identically to an outage.
* A bounded TTL cache. Every retry pass re-searched the same terms against a
  source documented as rate-limiting, which is how you get blocked.
* A `degraded` flag on the result, so the caller can record that it ran
  without news instead of quietly pretending it did not matter.
* Lexicon sentiment, computed here and deterministically. It gives the regime
  agent a numeric signal it can use even when the LLM is unavailable, and it
  is reproducible — the same headlines always score the same way.
"""

import re
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from ddgs import DDGS

from db import utcnow
from logging_setup import get_logger
from services.resilience import retry_call
from services.untrusted import fence, sanitize

logger = get_logger(__name__)

SEARCH_TIMEOUT_SECONDS = 15
# A hanging search must not block a graph run; three attempts at 15s each is
# already the outer bound of what a node should spend on optional context.
SEARCH_ATTEMPTS = 2
CACHE_TTL = timedelta(minutes=30)
CACHE_MAX_ENTRIES = 256

_cache: Dict[str, Tuple, ] = {}
_cache_lock = threading.Lock()

# Deliberately small and finance-specific. A general-purpose sentiment model
# scores "Fed holds rates steady" as neutral-positive; in a regime context the
# word that matters is "holds". Weights are ±1 for direction words and ±2 for
# ones that only appear in decisive coverage.
_POSITIVE = {
    "rally": 2, "rallies": 2, "surge": 2, "surges": 2, "soar": 2, "soars": 2,
    "gain": 1, "gains": 1, "rise": 1, "rises": 1, "climb": 1, "climbs": 1,
    "beat": 1, "beats": 1, "upgrade": 2, "upgraded": 2, "outperform": 2,
    "record high": 2, "bullish": 2, "recovery": 1, "rebound": 2, "optimism": 1,
    "strong": 1, "growth": 1, "expansion": 1, "easing": 1, "cut rates": 2,
}
_NEGATIVE = {
    "plunge": 2, "plunges": 2, "crash": 2, "crashes": 2, "slump": 2, "slumps": 2,
    "fall": 1, "falls": 1, "drop": 1, "drops": 1, "decline": 1, "declines": 1,
    "miss": 1, "misses": 1, "downgrade": 2, "downgraded": 2, "underperform": 2,
    "bearish": 2, "recession": 2, "selloff": 2, "sell-off": 2, "correction": 1,
    "inflation": 1, "layoff": 1, "layoffs": 1, "bankruptcy": 2, "default": 2,
    "warning": 1, "warns": 1, "slowdown": 1, "contraction": 2, "volatility": 1,
    "hike rates": 1, "tariff": 1, "tariffs": 1, "crisis": 2,
}


@dataclass
class NewsResult:
    """Headlines plus an explicit statement of whether the fetch worked.

    `degraded` is the important field. Returning an empty list on failure is
    indistinguishable from "there was no news", and that ambiguity is exactly
    how a permanently broken integration survived unnoticed.
    """

    items: List[Dict[str, str]] = field(default_factory=list)
    degraded: bool = False
    reason: Optional[str] = None
    sentiment: float = 0.0  # -1..1
    headline_count: int = 0


def _cache_get(key: str) -> Optional[NewsResult]:
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        cached_at, value = entry
        if utcnow() - cached_at > CACHE_TTL:
            _cache.pop(key, None)
            return None
        return value


def _cache_put(key: str, value: NewsResult) -> None:
    with _cache_lock:
        if len(_cache) >= CACHE_MAX_ENTRIES:
            oldest = min(_cache, key=lambda k: _cache[k][0])
            _cache.pop(oldest, None)
        _cache[key] = (utcnow(), value)


def clear_news_cache() -> None:
    with _cache_lock:
        _cache.clear()


def score_sentiment(texts: List[str]) -> float:
    """Net sentiment in -1..1 across a set of headlines.

    Normalized by total matched weight rather than headline count, so five
    mildly positive headlines do not outweigh one that says "crash". Returns
    0.0 when nothing matched, which means "no signal", not "neutral" -- the
    caller should check `headline_count` to tell them apart.
    """
    positive = negative = 0
    for text in texts:
        lowered = re.sub(r"[^a-z0-9\s-]", " ", (text or "").lower())
        for phrase, weight in _POSITIVE.items():
            if phrase in lowered:
                positive += weight
        for phrase, weight in _NEGATIVE.items():
            if phrase in lowered:
                negative += weight
    total = positive + negative
    if total == 0:
        return 0.0
    return round((positive - negative) / total, 3)


def search_financial_news(keywords: List[str], max_results: int = 5) -> NewsResult:
    """Search for recent financial news. Never raises."""
    query = " ".join(k for k in keywords if k).strip()
    if not query:
        return NewsResult(degraded=True, reason="no search terms supplied")
    query = f"{query} financial news"

    cache_key = f"{query}|{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    def _search() -> List[Dict[str, str]]:
        with DDGS(timeout=SEARCH_TIMEOUT_SECONDS) as ddgs:
            return [
                {
                    "title": r.get("title") or "",
                    "link": r.get("href") or "",
                    "snippet": r.get("body") or "",
                }
                for r in ddgs.text(
                    query,
                    # "wt-wt" was the bug: not a region this backend accepts,
                    # and it produced malformed upstream requests that failed
                    # on every single call.
                    region="us-en",
                    safesearch="off",
                    # One week, not one day. A 24-hour window is empty for
                    # most queries on a Monday morning and looks like an
                    # outage.
                    timelimit="w",
                    max_results=max_results,
                )
            ]

    try:
        items = retry_call(
            _search, attempts=SEARCH_ATTEMPTS, base_delay=1.0, label="news.search"
        )
    except Exception as exc:  # noqa: BLE001 -- news is optional context
        logger.warning("News search for %r failed (%s); continuing without it.", query, exc)
        result = NewsResult(degraded=True, reason=str(exc)[:200])
        _cache_put(cache_key, result)
        return result

    texts = [f"{i['title']} {i['snippet']}" for i in items]
    result = NewsResult(
        items=items,
        degraded=not items,
        reason=None if items else "search returned no headlines",
        sentiment=score_sentiment(texts),
        headline_count=len(items),
    )
    _cache_put(cache_key, result)
    return result


def get_market_news(keywords: Optional[List[str]] = None, max_results: int = 6) -> NewsResult:
    """Macro headlines for the regime agent."""
    return search_financial_news(
        keywords or ["Federal Reserve", "market outlook", "economy"], max_results=max_results
    )


def get_portfolio_news(tickers: List[str], per_ticker: int = 3) -> NewsResult:
    """Aggregate headlines across a portfolio's largest holdings.

    Capped at three symbols: each search is a round trip to a rate-limited
    source, and the marginal value of the fourth name's headlines does not
    justify another few seconds inside a graph node.
    """
    if not tickers:
        return NewsResult(degraded=True, reason="no tickers supplied")

    merged: List[Dict[str, str]] = []
    failures = 0
    for ticker in tickers[:3]:
        result = search_financial_news([ticker], max_results=per_ticker)
        if result.degraded:
            failures += 1
            continue
        for item in result.items:
            item = dict(item)
            item["ticker"] = ticker
            merged.append(item)

    texts = [f"{i['title']} {i['snippet']}" for i in merged]
    return NewsResult(
        items=merged,
        degraded=not merged,
        reason=f"{failures} of {min(len(tickers), 3)} ticker searches failed" if failures else None,
        sentiment=score_sentiment(texts),
        headline_count=len(merged),
    )


def format_for_prompt(result: NewsResult, limit: int = 8) -> str:
    """Render headlines for an LLM prompt, fenced as untrusted data.

    Titles and snippets are written by whoever ranked for the query, so they
    are quoted inside an `<untrusted_data>` element and sanitised so they
    cannot escape it. See `services/untrusted.py` for why a bare bullet list
    was a live channel into the regime call rather than a theoretical one.

    States the degraded case in words rather than sending an empty block: a
    model given no news silently invents context, whereas a model told the
    search failed will say the regime call rests on price signals alone --
    which is the truth.
    """
    if result.degraded or not result.items:
        return (
            "NEWS CONTEXT UNAVAILABLE: the news search failed or returned nothing "
            f"({result.reason or 'unknown reason'}). Base your assessment only on the "
            "quantitative signals provided, and say so."
        )

    lines = []
    for item in result.items[:limit]:
        ticker = sanitize(item.get("ticker") or "", max_chars=12)
        prefix = f"[{ticker}] " if ticker else ""
        title = sanitize(item.get("title"), max_chars=200)
        snippet = sanitize(item.get("snippet"), max_chars=200)
        if not title and not snippet:
            continue
        lines.append(f"- {prefix}{title}: {snippet}" if snippet else f"- {prefix}{title}")

    fenced = fence(
        lines,
        source="a public web search for financial news; the text is written by third parties",
    )
    if not fenced:
        return (
            "NEWS CONTEXT UNAVAILABLE: every headline returned was empty after sanitisation. "
            "Base your assessment only on the quantitative signals provided, and say so."
        )

    # The sentiment score sits outside the fence: it is computed here, from a
    # fixed lexicon, and is the firm's own number rather than the source's.
    return "\n".join(
        [
            f"Aggregate headline sentiment (computed in-house from a fixed lexicon, not "
            f"supplied by the source): {result.sentiment:+.2f} "
            f"(-1 bearish to +1 bullish, from {result.headline_count} headlines)",
            "",
            fenced,
        ]
    )
