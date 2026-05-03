"""
Unit tests for sky_finance.pipeline.llm_summariser

Tests the internal helpers that don't require a live Ollama connection:
  - _validate_sentiment  (normalise / reject invalid sentiment strings)
  - summarise_article    (full call, Ollama mocked out)
"""

from unittest.mock import MagicMock, patch

from sky_finance.pipeline.llm_summariser import _validate_sentiment, summarise_article

# ---------------------------------------------------------------------------
# _validate_sentiment
# ---------------------------------------------------------------------------


def test_validate_sentiment_accepts_all_valid_values():
    assert _validate_sentiment("positive") == "positive"
    assert _validate_sentiment("neutral") == "neutral"
    assert _validate_sentiment("negative") == "negative"


def test_validate_sentiment_is_case_insensitive():
    assert _validate_sentiment("Positive") == "positive"
    assert _validate_sentiment("NEGATIVE") == "negative"


def test_validate_sentiment_falls_back_to_neutral_on_unknown():
    assert _validate_sentiment("bullish") == "neutral"
    assert _validate_sentiment(None) == "neutral"
    assert _validate_sentiment(42) == "neutral"
    assert _validate_sentiment("") == "neutral"


# ---------------------------------------------------------------------------
# summarise_article (Ollama mocked)
# ---------------------------------------------------------------------------


def _make_ollama_response(payload: dict) -> MagicMock:
    """Build a minimal mock that matches the ollama.Client.chat() return shape."""
    import json

    msg = MagicMock()
    msg.content = json.dumps(payload)
    resp = MagicMock()
    resp.message = msg
    return resp


def test_summarise_article_returns_structured_dict():
    fake_payload = {
        "summary": "Apple reported record revenue.",
        "sentiment": "positive",
        "key_facts": ["Revenue up 12%", "iPhone units beat estimates"],
        "topics": ["earnings"],
        "relevance_score": 0.9,
    }
    with patch("sky_finance.pipeline.llm_summariser.ollama.Client") as mock_client_cls:
        mock_client_cls.return_value.chat.return_value = _make_ollama_response(fake_payload)
        result = summarise_article("AAPL", "Apple Q4 Results", "Apple reported record revenue.")

    assert result["sentiment"] == "positive"
    assert result["relevance_score"] == 0.9
    assert isinstance(result["key_facts"], list)


def test_summarise_article_falls_back_on_invalid_json():
    """If the LLM returns non-JSON, the function should return a safe default dict."""
    bad_response = MagicMock()
    bad_response.message.content = "Sorry, I cannot answer that."

    with patch("sky_finance.pipeline.llm_summariser.ollama.Client") as mock_client_cls:
        mock_client_cls.return_value.chat.return_value = bad_response
        result = summarise_article("AAPL", "Title", "Content")

    assert result["sentiment"] == "neutral"
    assert result["summary"] == ""
    assert result["relevance_score"] == 0.0
