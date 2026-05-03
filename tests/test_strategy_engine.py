"""
Unit tests for sky_finance.strategies.engine

Covers the pure / mockable functions:
  - resolve_tickers  (scope routing logic)
  - build_prompt     (placeholder substitution)
"""

from unittest.mock import MagicMock, patch

import pytest

from sky_finance.strategies.costs import UsageStats, compute_cost
from sky_finance.strategies.engine import (
    _load_model_cfg,
    _rrf_fuse,
    build_prompt,
    resolve_tickers,
    run_with_model,
)

# ---------------------------------------------------------------------------
# resolve_tickers
# ---------------------------------------------------------------------------

_ALL_CONFIGS = [
    {"ticker": "AAPL"},
    {"ticker": "NVDA"},
    {"ticker": "7203.T"},
]


def test_resolve_tickers_global_returns_all():
    with patch("sky_finance.strategies.engine.list_stock_configs", return_value=_ALL_CONFIGS):
        result = resolve_tickers({"scope": "global"})
    assert result == ["AAPL", "NVDA", "7203.T"]


def test_resolve_tickers_single_ticker():
    with patch(
        "sky_finance.strategies.engine.load_stock_config",
        return_value={"ticker": "AAPL", "enabled": True},
    ):
        result = resolve_tickers({"scope": "ticker", "scope_value": "AAPL"})
    assert result == ["AAPL"]


def test_resolve_tickers_group_filters_to_enabled():
    """scope='group' with scope_value as comma-separated tickers keeps only enabled ones."""
    enabled = [{"ticker": "AAPL"}, {"ticker": "NVDA"}]
    with patch("sky_finance.strategies.engine.list_stock_configs", return_value=enabled):
        result = resolve_tickers({"scope": "group", "scope_value": "AAPL, NVDA, MSFT"})
    # MSFT is not in the enabled list, so it should be filtered out
    assert set(result) == {"AAPL", "NVDA"}


def test_resolve_tickers_group_empty_scope_value_returns_empty():
    result = resolve_tickers({"scope": "group", "scope_value": None})
    assert result == []


def test_resolve_tickers_ticker_missing_scope_value_raises():
    with pytest.raises(ValueError, match="scope_value"):
        resolve_tickers({"scope": "ticker"})


def test_resolve_tickers_unknown_scope_raises():
    with pytest.raises(ValueError, match="Unknown scope"):
        resolve_tickers({"scope": "market"})


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_fills_tickers_and_rag_context():
    template = "Analyse {tickers}.\n\nContext:\n{rag_context}"
    result = build_prompt(template, {"AAPL": "chunk A", "NVDA": "chunk B"}, ["AAPL", "NVDA"])
    assert "AAPL" in result
    assert "NVDA" in result
    assert "chunk A" in result
    assert "chunk B" in result
    assert "{tickers}" not in result
    assert "{rag_context}" not in result


def test_build_prompt_single_ticker_placeholder_replaced():
    template = "Deep dive on {ticker}."
    result = build_prompt(template, {"AAPL": "ctx"}, ["AAPL"])
    assert "AAPL" in result
    assert "{ticker}" not in result


def test_build_prompt_includes_company_name_when_provided():
    template = "Report: {tickers}"
    result = build_prompt(
        template,
        {"AAPL": "ctx"},
        ["AAPL"],
        ticker_names={"AAPL": "Apple Inc."},
    )
    assert "Apple Inc." in result


# ---------------------------------------------------------------------------
# run_with_model — provider routing
# ---------------------------------------------------------------------------


def test_run_with_model_ollama():
    cfg = {
        "provider": "ollama",
        "model": "qwen2.5:14b-instruct",
        "base_url": "http://localhost:11434",
        "max_tokens": 2048,
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "message": {"content": "Bullish outlook."},
        "prompt_eval_count": 100,
        "eval_count": 50,
    }

    with patch("sky_finance.strategies.engine._load_model_cfg", return_value=cfg):
        with patch("httpx.post", return_value=mock_resp):
            text, model_id, usage = run_with_model("local", "sys", "user")

    assert text == "Bullish outlook."
    assert model_id == "qwen2.5:14b-instruct"
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.cost_usd == 0.0  # ollama is always free


def test_run_with_model_openai():
    cfg = {"provider": "openai", "model": "gpt-4o-mini", "max_tokens": 4096}

    mock_msg = MagicMock()
    mock_msg.content = "Neutral analysis."
    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 200
    mock_usage.completion_tokens = 80
    mock_usage.prompt_tokens_details = None  # no prompt cache hit

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage

    with patch("sky_finance.strategies.engine._load_model_cfg", return_value=cfg):
        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_resp
            text, model_id, usage = run_with_model("nano", "sys", "user")

    assert text == "Neutral analysis."
    assert model_id == "gpt-4o-mini"
    assert usage.input_tokens == 200
    assert usage.output_tokens == 80


def test_run_with_model_claude():
    cfg = {"provider": "claude", "model": "claude-sonnet-4-6", "max_tokens": 16000}

    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = "Risk signals detected."

    mock_usage = MagicMock()
    mock_usage.input_tokens = 300
    mock_usage.output_tokens = 120
    mock_usage.cache_read_input_tokens = 0
    mock_usage.cache_creation_input_tokens = 0

    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    mock_resp.usage = mock_usage

    with patch("sky_finance.strategies.engine._load_model_cfg", return_value=cfg):
        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = mock_resp
            text, model_id, usage = run_with_model("claude_tier", "sys", "user")

    assert text == "Risk signals detected."
    assert model_id == "claude-sonnet-4-6"
    assert usage.output_tokens == 120


def test_run_with_model_claude_unexpected_block_type_raises():
    cfg = {"provider": "claude", "model": "claude-sonnet-4-6", "max_tokens": 16000}

    mock_block = MagicMock()
    mock_block.type = "tool_use"  # not "text"

    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 5
    mock_usage.cache_read_input_tokens = 0
    mock_usage.cache_creation_input_tokens = 0

    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    mock_resp.usage = mock_usage

    with patch("sky_finance.strategies.engine._load_model_cfg", return_value=cfg):
        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = mock_resp
            with pytest.raises(ValueError, match="Unexpected Claude response block type"):
                run_with_model("claude_tier", "sys", "user")


def test_run_with_model_unknown_provider_raises():
    cfg = {"provider": "grok", "model": "grok-1"}
    with patch("sky_finance.strategies.engine._load_model_cfg", return_value=cfg):
        with pytest.raises(ValueError, match="Unknown provider"):
            run_with_model("bad_tier", "sys", "user")


# ---------------------------------------------------------------------------
# _load_model_cfg — reads config/settings.toml from disk
# ---------------------------------------------------------------------------


def test_load_model_cfg_returns_tier_dict():
    # Uses the real settings.toml in the repo — no mock needed
    result = _load_model_cfg("local")
    assert "provider" in result
    assert "model" in result


def test_load_model_cfg_unknown_tier_raises():
    with pytest.raises(ValueError, match="Unknown model tier"):
        _load_model_cfg("nonexistent_tier_xyz")


# ---------------------------------------------------------------------------
# compute_cost / UsageStats
# ---------------------------------------------------------------------------


def test_compute_cost_known_model_returns_float():
    stats = compute_cost(
        model="claude-sonnet-4-6",
        provider="claude",
        input_tokens=1000,
        output_tokens=200,
    )
    assert isinstance(stats.cost_usd, float)
    assert stats.cost_usd > 0


def test_compute_cost_unknown_model_returns_none():
    stats = compute_cost(
        model="unknown-model-xyz",
        provider="openai",
        input_tokens=100,
        output_tokens=50,
    )
    assert stats.cost_usd is None


# ---------------------------------------------------------------------------
# _rrf_fuse — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


# Helper: build a row tuple (id, title, body, sentiment, score)
def _row(doc_id: int, score: float = 1.0) -> tuple:
    return (doc_id, f"Title {doc_id}", f"Body {doc_id}", "positive", score)


def test_rrf_fuse_returns_both_lists():
    vec_rows = [_row(1, 0.9), _row(2, 0.7)]
    bm25_rows = [_row(3, 0.8), _row(1, 0.6)]
    result = _rrf_fuse(vec_rows, bm25_rows, k=60)
    ids = [r[0] for r in result]
    assert 1 in ids and 2 in ids and 3 in ids


def test_rrf_fuse_doc_in_both_lists_ranks_higher():
    # doc 1 in both lists, doc 2 only in vector, doc 3 only in BM25
    # doc 2 and doc 3 are ranked first in their respective lists — 1/61 each
    # doc 1 is ranked second in both — 1/62 + 1/62 = 0.0323 > 1/61 = 0.0164
    vec_rows = [_row(2, 0.9), _row(1, 0.8)]
    bm25_rows = [_row(3, 0.9), _row(1, 0.8)]
    result = _rrf_fuse(vec_rows, bm25_rows, k=60)
    assert result[0][0] == 1, "doc appearing in both lists should have highest RRF score"


def test_rrf_fuse_respects_top_k():
    vec_rows = [_row(i, 1.0 / i) for i in range(1, 11)]
    bm25_rows = [_row(i, 1.0 / i) for i in range(1, 11)]
    result = _rrf_fuse(vec_rows, bm25_rows, k=60, top_k=3)
    assert len(result) == 3


def test_rrf_fuse_empty_bm25_falls_back_to_vector():
    vec_rows = [_row(1, 0.9), _row(2, 0.7)]
    result = _rrf_fuse(vec_rows, [], k=60, top_k=5)
    assert len(result) == 2
    assert {r[0] for r in result} == {1, 2}


def test_rrf_fuse_empty_vector_falls_back_to_bm25():
    bm25_rows = [_row(10, 0.5), _row(11, 0.3)]
    result = _rrf_fuse([], bm25_rows, k=60, top_k=5)
    assert {r[0] for r in result} == {10, 11}


def test_rrf_fuse_score_is_rrf_formula():
    # Single doc, rank 1 in vector only: score = 1/(60+1) = 1/61 ≈ 0.01639
    # _rrf_fuse rounds to 4 decimal places (0.0164), so allow 1e-4 tolerance
    result = _rrf_fuse([_row(1, 0.9)], [], k=60)
    assert abs(result[0][4] - 1 / 61) < 1e-4


def test_rrf_fuse_both_empty_returns_empty():
    assert _rrf_fuse([], [], k=60) == []


def test_usage_stats_to_dict_contains_all_fields():
    stats = UsageStats(
        model="claude-sonnet-4-6",
        provider="claude",
        input_tokens=500,
        output_tokens=100,
        cached_tokens=200,
        cache_creation_tokens=50,
        cost_usd=0.005,
    )
    d = stats.to_dict()
    assert d["model"] == "claude-sonnet-4-6"
    assert d["cached_tokens"] == 200
    assert d["cost_usd"] == 0.005
