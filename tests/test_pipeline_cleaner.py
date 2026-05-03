"""
Unit tests for sky_finance.pipeline.cleaner

Covers the two public functions:
  - clean_news_article  (HTML strip, whitespace normalisation, truncation)
  - ohlcv_to_text       (OHLCV record → human-readable text for embedding)
"""

from sky_finance.pipeline.cleaner import clean_news_article, ohlcv_to_text

# ---------------------------------------------------------------------------
# clean_news_article
# ---------------------------------------------------------------------------


def test_clean_strips_html_and_normalises_whitespace():
    article = {
        "title": "<b>Apple  Inc.</b>",
        "content": "<p>Strong  quarterly  results.</p>",
    }
    result = clean_news_article(article)
    assert result["title"] == "Apple Inc."
    assert result["content"] == "Strong quarterly results."


def test_clean_truncates_long_content():
    article = {"title": "T", "content": "x" * 3000}
    result = clean_news_article(article)
    # Truncated body = 2 000 chars + "…"
    assert len(result["content"]) == 2001
    assert result["content"].endswith("…")


def test_clean_falls_back_to_summary_field():
    """If 'content' is absent, 'summary' should be used instead."""
    article = {"title": "T", "summary": "Fallback <i>summary</i> text."}
    result = clean_news_article(article)
    assert result["content"] == "Fallback summary text."


# ---------------------------------------------------------------------------
# ohlcv_to_text
# ---------------------------------------------------------------------------


def test_ohlcv_to_text_includes_daily_change():
    record = {
        "ticker": "AAPL",
        "market": "us",
        "ohlcv": [
            {
                "date": "2024-01-02",
                "open": 100,
                "close": 100,
                "high": 105,
                "low": 98,
                "volume": 1_000,
            },
            {
                "date": "2024-01-03",
                "open": 101,
                "close": 110,
                "high": 112,
                "low": 100,
                "volume": 1_200,
            },
        ],
        "fundamentals": {},
    }
    text = ohlcv_to_text(record)
    assert "AAPL" in text
    assert "+10.00%" in text


def test_ohlcv_to_text_renders_fundamentals():
    record = {
        "ticker": "NVDA",
        "market": "us",
        "ohlcv": [
            {
                "date": "2024-01-03",
                "open": 500,
                "close": 510,
                "high": 515,
                "low": 498,
                "volume": 2_000,
            },
        ],
        "fundamentals": {
            "marketCap": 1_200_000_000_000,
            "currency": "USD",
            "trailingPE": 65.4,
            "sector": "Technology",
            "industry": "Semiconductors",
            "shortName": "NVIDIA Corporation",
            "fiftyTwoWeekHigh": 800,
            "fiftyTwoWeekLow": 400,
        },
    }
    text = ohlcv_to_text(record)
    assert "NVDA" in text
    assert "Trailing P/E" in text
    assert "Technology" in text
