"""
Unit tests for sky_finance.ingestion.news_fetcher

Covers:
  - fetch_news  (httpx mocked via respx — no live network calls)
  - save_raw_news  (file I/O via tmp_path + monkeypatched _raw_path)
"""

import json
import re
from unittest.mock import MagicMock

import httpx
import respx

from sky_finance.ingestion.news_fetcher import (
    _parse_published_iso,
    fetch_news,
    save_raw_news,
)

# ---------------------------------------------------------------------------
# RSS fixture data
# ---------------------------------------------------------------------------

_RSS_ONE_ARTICLE = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
    <item>
      <title>AAPL Beats Earnings</title>
      <link>https://example.com/news/aapl-1</link>
      <pubDate>Mon, 01 Jan 2024 12:00:00 +0000</pubDate>
      <description>Apple Q4 results exceeded expectations.</description>
    </item>
  </channel>
</rss>
"""

_RSS_EMPTY = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel><title>Google News</title></channel>
</rss>
"""

_GOOGLE_RSS = re.compile(r"https://news\.google\.com/rss/search")


# ---------------------------------------------------------------------------
# fetch_news
# ---------------------------------------------------------------------------


def test_fetch_news_us_returns_articles():
    with respx.mock:
        respx.get(_GOOGLE_RSS).mock(return_value=httpx.Response(200, text=_RSS_ONE_ARTICLE))
        result = fetch_news("AAPL", "us", l1_keywords=["earnings"], l2_topics=["tech"])

    assert result["ticker"] == "AAPL"
    assert result["market"] == "us"
    assert len(result["articles"]) == 1
    assert result["articles"][0]["title"] == "AAPL Beats Earnings"
    assert result["articles"][0]["url"] == "https://example.com/news/aapl-1"


def test_fetch_news_deduplicates_across_feeds():
    # Both L1 and L2 feeds return the same article URL → deduplicated to 1
    with respx.mock:
        respx.get(_GOOGLE_RSS).mock(return_value=httpx.Response(200, text=_RSS_ONE_ARTICLE))
        result = fetch_news("AAPL", "us", l1_keywords=["earnings"], l2_topics=["tech"])

    assert len(result["articles"]) == 1


def test_fetch_news_http_error_on_one_feed_does_not_raise():
    call_count = 0

    def side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        # First feed (L1) returns 404; second (L2) succeeds
        return (
            httpx.Response(404) if call_count == 1 else httpx.Response(200, text=_RSS_ONE_ARTICLE)
        )

    with respx.mock:
        respx.get(_GOOGLE_RSS).mock(side_effect=side_effect)
        result = fetch_news("AAPL", "us", l1_keywords=["earnings"], l2_topics=["tech"])

    assert result["ticker"] == "AAPL"
    # Only the second feed succeeded
    assert len(result["articles"]) == 1


def test_fetch_news_no_l2_builds_one_en_feed():
    with respx.mock:
        route = respx.get(_GOOGLE_RSS).mock(return_value=httpx.Response(200, text=_RSS_EMPTY))
        fetch_news("AAPL", "us", l1_keywords=["iPhone"])

    assert route.call_count == 1


def test_fetch_news_japan_ticker_with_ja_keywords_builds_four_feeds():
    # L1-EN, L2-EN, L1-JA (トヨタ has JA chars), L2-JA (自動車 has JA chars) → 4 feeds
    with respx.mock:
        route = respx.get(_GOOGLE_RSS).mock(return_value=httpx.Response(200, text=_RSS_EMPTY))
        fetch_news(
            "7203.T",
            "japan",
            l1_keywords=["トヨタ", "Toyota"],
            l2_topics=["自動車", "EV market"],
        )

    assert route.call_count == 4


def test_fetch_news_japan_no_ja_keywords_builds_two_feeds():
    # All keywords are ASCII → only L1-EN and L1-JA (ticker only, no JA terms in L2)
    with respx.mock:
        route = respx.get(_GOOGLE_RSS).mock(return_value=httpx.Response(200, text=_RSS_EMPTY))
        fetch_news("7203.T", "japan", l1_keywords=["Toyota"], l2_topics=["EV market"])

    # L1-EN, L2-EN, L1-JA (no JA chars in l1 so just ticker), no L2-JA = 3
    assert route.call_count == 3


def test_fetch_news_l3_macro_not_fetched_but_stored_in_payload():
    with respx.mock:
        route = respx.get(_GOOGLE_RSS).mock(return_value=httpx.Response(200, text=_RSS_EMPTY))
        result = fetch_news("AAPL", "us", l3_macro=["Fed rate cut", "USD strength"])

    # L3 generates no additional feeds — only the L1 feed (no l2_topics given)
    assert route.call_count == 1
    assert result["keyword_tiers"]["l3"] == ["Fed rate cut", "USD strength"]


def test_fetch_news_payload_has_required_keys():
    with respx.mock:
        respx.get(_GOOGLE_RSS).mock(return_value=httpx.Response(200, text=_RSS_EMPTY))
        result = fetch_news("NVDA", "us")

    assert {"ticker", "market", "fetched_at", "keyword_tiers", "articles"} <= result.keys()
    assert isinstance(result["articles"], list)


# ---------------------------------------------------------------------------
# save_raw_news
# ---------------------------------------------------------------------------

_ARTICLE_1 = {
    "url": "https://example.com/news/1",
    "title": "First Article",
    "published": "",
    "published_iso": None,
    "summary": "Summary one.",
    "source": "Reuters",
}
_ARTICLE_2 = {
    "url": "https://example.com/news/2",
    "title": "Second Article",
    "published": "",
    "published_iso": None,
    "summary": "Summary two.",
    "source": "Bloomberg",
}


def _make_payload(articles: list[dict]) -> dict:
    return {
        "ticker": "AAPL",
        "market": "us",
        "fetched_at": "2024-01-15T12:00:00+00:00",
        "keyword_tiers": {"l1": [], "l2": [], "l3": []},
        "articles": articles,
    }


def test_save_raw_news_creates_json_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sky_finance.ingestion.news_fetcher._raw_path",
        lambda ticker, date_utc: tmp_path / f"{date_utc}.json",
    )
    path, total = save_raw_news(_make_payload([_ARTICLE_1]))

    assert path.exists()
    assert total == 1
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ticker"] == "AAPL"
    assert len(data["articles"]) == 1


def test_save_raw_news_merges_new_articles_on_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sky_finance.ingestion.news_fetcher._raw_path",
        lambda ticker, date_utc: tmp_path / f"{date_utc}.json",
    )
    save_raw_news(_make_payload([_ARTICLE_1]))
    _, total = save_raw_news(_make_payload([_ARTICLE_1, _ARTICLE_2]))

    # _ARTICLE_1 is a duplicate (same URL) → merged total is 2, not 3
    assert total == 2


def test_save_raw_news_deduplicates_on_exact_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sky_finance.ingestion.news_fetcher._raw_path",
        lambda ticker, date_utc: tmp_path / f"{date_utc}.json",
    )
    save_raw_news(_make_payload([_ARTICLE_1]))
    _, total = save_raw_news(_make_payload([_ARTICLE_1]))

    assert total == 1


def test_save_raw_news_returns_correct_path(tmp_path, monkeypatch):
    expected = tmp_path / "2024-01-15.json"
    monkeypatch.setattr(
        "sky_finance.ingestion.news_fetcher._raw_path",
        lambda ticker, date_utc: expected,
    )
    path, _ = save_raw_news(_make_payload([_ARTICLE_1]))

    assert path == expected


# ---------------------------------------------------------------------------
# _parse_published_iso — error paths (lines 72, 76-77)
# ---------------------------------------------------------------------------


def test_parse_published_iso_returns_none_when_field_missing():
    entry = MagicMock(spec=[])  # no published_parsed attribute
    result = _parse_published_iso(entry)
    assert result is None


def test_parse_published_iso_returns_none_when_field_is_none():
    entry = MagicMock()
    entry.published_parsed = None
    result = _parse_published_iso(entry)
    assert result is None


def test_parse_published_iso_returns_none_on_timegm_exception():
    entry = MagicMock()
    entry.published_parsed = "not-a-struct-time"  # causes calendar.timegm to raise
    result = _parse_published_iso(entry)
    assert result is None


# ---------------------------------------------------------------------------
# _fetch_feed_async — HTTP error paths (lines 101-103, 107)
# ---------------------------------------------------------------------------


def test_fetch_news_network_error_returns_empty_articles():
    with respx.mock:
        respx.get(_GOOGLE_RSS).mock(side_effect=httpx.ConnectError("connection refused"))
        result = fetch_news("AAPL", "us", l1_keywords=["apple"])
    assert result["articles"] == []


def test_fetch_news_handles_bozo_feed_without_raising():
    # Malformed XML — feedparser sets bozo=True but doesn't crash
    malformed_xml = "<rss><channel><item><title>Test</title></item></channel>"
    with respx.mock:
        respx.get(_GOOGLE_RSS).mock(return_value=httpx.Response(200, text=malformed_xml))
        result = fetch_news("AAPL", "us", l1_keywords=["apple"])
    assert isinstance(result["articles"], list)
