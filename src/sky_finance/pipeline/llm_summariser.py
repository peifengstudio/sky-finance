"""
LLM summariser — call local Ollama model to extract structured information
from cleaned news articles.

Model: qwen2.5:3b-instruct  (or qwen3:4b — change LLM_MODEL in settings.toml)
Output: JSON with summary, sentiment, key_facts, topics, relevance_score

Usage:
    from sky_finance.pipeline.llm_summariser import summarise_article
    result = summarise_article(ticker="AAPL", title="...", content="...")
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import ollama

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config  (read once at import time; worker restart picks up changes)
# ---------------------------------------------------------------------------


def _load_llm_config() -> dict[str, Any]:
    import tomllib

    settings_path = Path(__file__).parents[3] / "config" / "settings.toml"
    with settings_path.open("rb") as f:
        cfg = tomllib.load(f)
    result: dict[str, Any] = cfg.get("llm", {})
    return result


_CFG = _load_llm_config()

LLM_MODEL = os.environ.get("OLLAMA_MODEL", _CFG.get("model", "qwen2.5:3b-instruct"))
OLLAMA_HOST = os.environ.get("OLLAMA_BASE_URL", _CFG.get("base_url", "http://localhost:11434"))
MAX_TOKENS = int(_CFG.get("max_tokens", 512))


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a financial analyst assistant. Analyse the news article and extract:

  summary        — 2–3 sentence factual summary
  sentiment      — one of: "positive", "neutral", "negative"
  key_facts      — up to 5 short factual bullet strings
  topics         — relevant tags, e.g. ["earnings","M&A","regulation","macro"]
  relevance_score — 0.0–1.0 how directly this news affects the ticker

Always respond in English regardless of the language of the input article."""

# JSON schema passed to Ollama's format parameter — enforces structure at the
# generation level so the model cannot produce malformed output.
_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "key_facts": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "topics": {"type": "array", "items": {"type": "string"}},
        "relevance_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["summary", "sentiment", "key_facts", "topics", "relevance_score"],
}

_USER_TEMPLATE = """\
Ticker: {ticker}
Title: {title}

Article:
{content}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarise_article(
    ticker: str,
    title: str,
    content: str,
    *,
    model: str = LLM_MODEL,
) -> dict[str, Any]:
    """
    Summarise a single news article using the local Ollama model.

    Args:
        ticker:  stock symbol for context (e.g. "AAPL").
        title:   cleaned article title.
        content: cleaned article body (≤ 2 000 chars recommended).
        model:   Ollama model name override.

    Returns:
        dict with keys: summary, sentiment, key_facts, topics, relevance_score.
        Falls back to a safe default dict if the model returns invalid JSON.
    """
    client = ollama.Client(host=OLLAMA_HOST)
    user_msg = _USER_TEMPLATE.format(ticker=ticker, title=title, content=content)

    logger.debug("LLM summarise [%s] model=%s  content_len=%d", ticker, model, len(content))

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            format=_FORMAT_SCHEMA,
            options={"num_predict": MAX_TOKENS, "temperature": 0.1},
        )
        raw = (response.message.content or "").strip()
        result = json.loads(raw)

        # Validate / normalise required keys
        return {
            "summary": str(result.get("summary", "")),
            "sentiment": _validate_sentiment(result.get("sentiment")),
            "key_facts": list(result.get("key_facts", [])),
            "topics": list(result.get("topics", [])),
            "relevance_score": float(result.get("relevance_score", 0.5)),
        }

    except json.JSONDecodeError as exc:
        logger.warning("LLM returned invalid JSON for %s: %s", ticker, exc)
        return _fallback(ticker)

    except ollama.ResponseError as exc:
        logger.error("Ollama API error for %s: %s", ticker, exc)
        raise

    except Exception as exc:
        logger.error("Unexpected LLM error for %s: %s", ticker, exc)
        raise


def _validate_sentiment(value: object) -> str:
    if isinstance(value, str) and value.lower() in {"positive", "neutral", "negative"}:
        return value.lower()
    return "neutral"


def _fallback(ticker: str) -> dict[str, Any]:
    """Safe default when LLM output cannot be parsed."""
    logger.warning("Using fallback summary for %s", ticker)
    return {
        "summary": "",
        "sentiment": "neutral",
        "key_facts": [],
        "topics": [],
        "relevance_score": 0.0,
    }
