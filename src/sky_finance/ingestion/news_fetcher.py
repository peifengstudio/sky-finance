"""
Google RSS news fetcher.

Fetches news articles for a ticker via Google News RSS, using a three-tier
keyword model for more targeted coverage:

  L1 (l1_keywords) — direct company/product terms  → primary EN query
  L2 (l2_topics)   — sector/theme context          → secondary EN query
  L3 (l3_macro)    — macro factors                 → reserved for RAG, not fetched here

Japan tickers also get Japanese-language feeds for both L1 and L2 terms.

Concurrency
-----------
All RSS feeds for a single ticker are fetched concurrently via
httpx.AsyncClient + asyncio.gather. For a US ticker this is typically 2
requests (L1-EN + L2-EN); for a Japan ticker up to 4 (+ L1-JA + L2-JA).
Each per-ticker Celery task calls asyncio.run() to execute the async fetch
inside its own event loop. Celery worker concurrency (--concurrency flag)
is the outer rate-limit that prevents too many tickers hitting Google at once.

Articles are persisted to:
    data/raw/news/{ticker}/{YYYY-MM-DD}.json   (date = UTC fetch date)

Multiple fetches on the same day merge into the existing file, deduplicating
by URL so reruns are safe.
"""

import asyncio
import calendar
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import feedparser
import httpx

logger = logging.getLogger(__name__)

_RSS_EN = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_RSS_JA = "https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"

# Cap keyword lists when building queries to stay within URL length limits
_MAX_L1_TERMS = 4
_MAX_L2_TERMS = 3

_HTTP_TIMEOUT = 30.0
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; sky-finance/1.0; +https://github.com)"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _parse_published_iso(entry: object) -> str | None:
    """
    Return a UTC ISO-8601 string from feedparser's published_parsed field.
    feedparser stores published_parsed as a time.struct_time in UTC.
    Returns None if the field is missing or unparseable.
    """
    parsed = getattr(entry, "published_parsed", None)
    if parsed is None:
        return None
    try:
        ts = calendar.timegm(parsed)  # struct_time (UTC) → Unix timestamp
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except Exception:
        return None


def _parse_entry(entry: object) -> dict[str, Any]:
    source = getattr(entry, "source", None)
    return {
        "title": _strip_html(getattr(entry, "title", "")),
        "url": getattr(entry, "link", ""),
        "published": getattr(entry, "published", ""),
        "published_iso": _parse_published_iso(entry),  # UTC ISO string for DB
        "summary": _strip_html(getattr(entry, "summary", "")),
        "source": source.get("title", "") if isinstance(source, dict) else "",
    }


async def _fetch_feed_async(client: httpx.AsyncClient, url: str) -> list[dict[str, Any]]:
    """Fetch and parse a single RSS feed URL. Returns [] on any HTTP/network error."""
    logger.debug("Fetching RSS: %s", url)
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("RSS HTTP %d for %s", exc.response.status_code, url)
        return []
    except httpx.RequestError as exc:
        logger.warning("RSS request error for %s: %s", url, exc)
        return []

    feed = feedparser.parse(response.text)
    if feed.bozo:
        logger.warning("RSS parse warning for %s: %s", url, feed.bozo_exception)
    return [_parse_entry(e) for e in feed.entries]


async def _fetch_all_feeds(urls: list[str]) -> list[list[dict[str, Any]]]:
    """Fetch all RSS URLs concurrently and return one article list per URL."""
    async with httpx.AsyncClient(
        timeout=_HTTP_TIMEOUT,
        follow_redirects=True,
        headers=_HTTP_HEADERS,
    ) as client:
        return list(await asyncio.gather(*(_fetch_feed_async(client, url) for url in urls)))


def _has_non_ascii(text: str) -> bool:
    """Return True if the string contains non-ASCII characters (e.g. Japanese)."""
    return any(ord(c) > 127 for c in text)


def _raw_path(ticker: str, date_utc: str) -> Path:
    base = Path(__file__).parents[3] / "data" / "raw" / "news" / ticker
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{date_utc}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_news(
    ticker: str,
    market: str,
    l1_keywords: list[str] | None = None,
    l2_topics: list[str] | None = None,
    l3_macro: list[str] | None = None,
) -> dict[str, Any]:
    """
    Fetch news articles for a ticker from Google News RSS.

    Query strategy
    --------------
    Two separate EN queries are built to improve coverage without diluting
    results with overly broad search terms:

      Query 1 (primary):   ticker + L1 keywords
                           → direct company/product news, highest signal
      Query 2 (topic):     ticker + L2 topics   (only if l2_topics provided)
                           → sector/theme context, catches industry-wide moves

    For Japan tickers ('japan'), two additional Japanese-language feeds are
    fetched using the Japanese terms from L1 and L2.

    L3 (macro) is intentionally excluded from RSS fetching — macro terms are
    too generic and produce noise. They are used downstream by the strategy
    engine for RAG context expansion.

    Args:
        ticker:       stock symbol, e.g. "AAPL" or "7203.T".
        market:       "us" or "japan".
        l1_keywords:  direct company/ticker search terms (L1).
        l2_topics:    sector/theme context keywords (L2).
        l3_macro:     macro factors (L3) — stored in payload but NOT fetched.

    Returns:
        dict with keys: ticker, market, fetched_at, keyword_tiers, articles.
    """
    l1 = l1_keywords or []
    l2 = l2_topics or []

    # --- Build URL list (order determines which results appear first) ---
    urls: list[str] = []

    primary_terms = [ticker] + l1[:_MAX_L1_TERMS]
    urls.append(_RSS_EN.format(query=quote_plus(" ".join(primary_terms))))
    logger.debug("L1 query EN (%d terms): %s", len(primary_terms), " ".join(primary_terms))

    if l2:
        topic_terms = [ticker] + l2[:_MAX_L2_TERMS]
        urls.append(_RSS_EN.format(query=quote_plus(" ".join(topic_terms))))
        logger.debug("L2 query EN (%d terms): %s", len(topic_terms), " ".join(topic_terms))

    ja_l2: list[str] = []
    if market == "japan":
        ja_l1 = [k for k in l1 if _has_non_ascii(k)]
        ja_l1_terms = ([ticker] + ja_l1) if ja_l1 else [ticker]
        urls.append(_RSS_JA.format(query=quote_plus(" ".join(ja_l1_terms))))
        logger.debug("L1 query JA: %s", " ".join(ja_l1_terms))

        ja_l2 = [k for k in l2 if _has_non_ascii(k)]
        if ja_l2:
            urls.append(_RSS_JA.format(query=quote_plus(" ".join(ja_l2[:_MAX_L2_TERMS]))))
            logger.debug("L2 query JA: %s", " ".join(ja_l2[:_MAX_L2_TERMS]))

    # --- Fetch all feeds concurrently ---
    per_feed = asyncio.run(_fetch_all_feeds(urls))
    articles = [article for feed_articles in per_feed for article in feed_articles]

    # Deduplicate by URL (preserve order, keep first occurrence)
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for a in articles:
        if a["url"] and a["url"] not in seen:
            seen.add(a["url"])
            unique.append(a)

    logger.info(
        "Fetched %d unique articles for %s (%d feeds, %d EN + %s JA)",
        len(unique),
        ticker,
        len(urls),
        2 if l2 else 1,
        "2" if ja_l2 else ("1" if market == "japan" else "0"),
    )
    return {
        "ticker": ticker,
        "market": market,
        "fetched_at": datetime.now(UTC).isoformat(),
        # Preserve keyword tiers in payload for downstream use (RAG context, debugging)
        "keyword_tiers": {
            "l1": l1,
            "l2": l2,
            "l3": l3_macro or [],
        },
        "articles": unique,
    }


def save_raw_news(payload: dict[str, Any]) -> tuple[Path, int]:
    """
    Persist news payload to disk, merging with any existing file for the same day.

    Deduplicates by URL so multiple hourly fetches on the same day accumulate
    new articles without creating duplicates.

    Args:
        payload: dict returned by fetch_news().

    Returns:
        Tuple of (file Path, total article count after merge).
    """
    ticker = payload["ticker"]
    date_utc = datetime.fromisoformat(payload["fetched_at"]).strftime("%Y-%m-%d")
    path = _raw_path(ticker, date_utc)

    # Merge with existing file if present
    existing_articles: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            existing_articles = json.load(f).get("articles", [])

    existing_urls = {a["url"] for a in existing_articles}
    new_articles = [a for a in payload["articles"] if a["url"] not in existing_urls]

    merged = {**payload, "articles": existing_articles + new_articles}

    with path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    total = len(merged["articles"])
    logger.info(
        "Saved news → %s  (+%d new, %d total)",
        path,
        len(new_articles),
        total,
    )
    return path, total
