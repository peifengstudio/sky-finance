"""
Data cleaner — normalise raw records before LLM processing.

Two cleaning paths:
  - clean_news_article(article: dict) -> dict
  - clean_ohlcv_to_text(record: dict)  -> str   (text for embedding)
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# News cleaning
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_HTML_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_RE.sub("", text or "").strip()


def _normalise_ws(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_news_article(article: dict[str, Any]) -> dict[str, Any]:
    """
    Clean a single news article dict (from news_raw row or fetcher output).

    Operations:
    - Strip HTML tags from title and content
    - Normalise whitespace
    - Truncate content to 2 000 chars (LLM context budget)

    Returns a new dict with cleaned fields.
    """
    title = _normalise_ws(_strip_html(article.get("title", "") or ""))
    content = _normalise_ws(
        _strip_html(article.get("content", "") or article.get("summary", "") or "")
    )

    # Trim to stay within small-model context budget
    if len(content) > 2000:
        content = content[:2000] + "…"

    return {
        **article,
        "title": title,
        "content": content,
    }


# ---------------------------------------------------------------------------
# OHLCV → text representation for embedding
# ---------------------------------------------------------------------------


def ohlcv_to_text(record: dict[str, Any]) -> str:
    """
    Convert a raw_data payload (yfinance) to a human-readable text chunk
    suitable for embedding and semantic search.

    Args:
        record: dict from raw_data.payload (JSONB), containing
                ticker, market, ohlcv list, fundamentals dict.

    Returns:
        A multi-line text string.
    """
    ticker = record.get("ticker", "")
    market = record.get("market", "").upper()
    ohlcv = record.get("ohlcv", [])
    fund = record.get("fundamentals", {})

    # Use the most recent trading day
    latest = ohlcv[-1] if ohlcv else {}
    prev = ohlcv[-2] if len(ohlcv) >= 2 else {}

    # Daily change %
    change_pct = ""
    if latest.get("close") and prev.get("close") and prev["close"] != 0:
        pct = ((latest["close"] - prev["close"]) / prev["close"]) * 100
        change_pct = f"{pct:+.2f}%"

    lines = [
        f"{ticker} ({market}) — {latest.get('date', 'N/A')}",
        f"Open: {latest.get('open')}  Close: {latest.get('close')}  "
        f"High: {latest.get('high')}  Low: {latest.get('low')}  "
        f"Volume: {latest.get('volume')}",
    ]

    if change_pct:
        lines.append(f"Daily change: {change_pct}")

    if fund.get("marketCap"):
        lines.append(f"Market cap: {fund['marketCap']:,.0f} {fund.get('currency', '')}")
    if fund.get("trailingPE"):
        lines.append(f"Trailing P/E: {fund['trailingPE']:.2f}")
    if fund.get("fiftyTwoWeekHigh") and fund.get("fiftyTwoWeekLow"):
        lines.append(f"52-week range: {fund['fiftyTwoWeekLow']} – {fund['fiftyTwoWeekHigh']}")
    if fund.get("sector"):
        lines.append(f"Sector: {fund['sector']}  Industry: {fund.get('industry', '')}")
    if fund.get("shortName"):
        lines.append(f"Company: {fund['shortName']}")

    return "\n".join(lines)
