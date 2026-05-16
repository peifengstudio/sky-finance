"""Unit tests for sky_finance.evaluation.judge."""

import json
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from sky_finance.evaluation.judge import (
    _call_anthropic,
    _call_ollama,
    _call_openai,
    _fallback,
    _infer_provider,
    _validate,
    judge,
    score_avg,
)

_VALID_VERDICT = {
    "A": {"faithfulness": 8, "coverage": 7, "actionability": 6},
    "B": {"faithfulness": 5, "coverage": 6, "actionability": 7},
    "winner": "A",
    "reasoning": "Method A was more faithful to source material.",
}


# ---------------------------------------------------------------------------
# _infer_provider
# ---------------------------------------------------------------------------


def test_infer_provider_claude():
    assert _infer_provider("claude-sonnet-4-6") == "anthropic"
    assert _infer_provider("claude-opus-4-7") == "anthropic"


def test_infer_provider_openai_gpt():
    assert _infer_provider("gpt-4o") == "openai"
    assert _infer_provider("gpt-4o-mini") == "openai"


def test_infer_provider_openai_o_series():
    assert _infer_provider("o1") == "openai"
    assert _infer_provider("o3-mini") == "openai"


def test_infer_provider_ollama_fallback():
    assert _infer_provider("qwen2.5:14b") == "ollama"
    assert _infer_provider("llama3.2") == "ollama"
    assert _infer_provider("mistral") == "ollama"


# ---------------------------------------------------------------------------
# score_avg
# ---------------------------------------------------------------------------


def test_score_avg_equal_scores():
    assert score_avg({"faithfulness": 8, "coverage": 8, "actionability": 8}) == 8.0


def test_score_avg_mixed_scores():
    result = score_avg({"faithfulness": 9, "coverage": 6, "actionability": 6})
    assert abs(result - 7.0) < 0.01


def test_score_avg_missing_key_counts_as_zero():
    result = score_avg({"faithfulness": 9, "coverage": 9})
    assert abs(result - 6.0) < 0.01


# ---------------------------------------------------------------------------
# _validate
# ---------------------------------------------------------------------------


def test_validate_passes_on_valid_verdict():
    _validate(_VALID_VERDICT)  # no exception


def test_validate_raises_on_missing_key():
    bad = {k: v for k, v in _VALID_VERDICT.items() if k != "winner"}
    with pytest.raises(ValueError, match="Missing keys"):
        _validate(bad)


def test_validate_raises_on_non_numeric_score():
    bad = {**_VALID_VERDICT, "A": {"faithfulness": "high", "coverage": 7, "actionability": 6}}
    with pytest.raises(ValueError, match="numeric"):
        _validate(bad)


def test_validate_raises_on_bad_winner():
    bad = {**_VALID_VERDICT, "winner": "C"}
    with pytest.raises(ValueError, match="winner"):
        _validate(bad)


# ---------------------------------------------------------------------------
# _fallback
# ---------------------------------------------------------------------------


def test_fallback_returns_tie_with_zero_scores():
    result = _fallback("test error")
    assert result["winner"] == "tie"
    assert result["A"]["faithfulness"] == 0
    assert result["B"]["actionability"] == 0
    assert "test error" in result["reasoning"]


# ---------------------------------------------------------------------------
# _call_anthropic — mocked Anthropic client
# ---------------------------------------------------------------------------


def _make_anthropic_response(tool_input: dict):
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.input = tool_input
    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    mock_resp.usage = MagicMock(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return mock_resp


def test_call_anthropic_returns_tool_use_input():
    mock_resp = _make_anthropic_response(_VALID_VERDICT)
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_resp
    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        result = _call_anthropic("claude-sonnet-4-6", "user message")
    assert result["winner"] == "A"
    assert "faithfulness" in result["A"]


def test_call_anthropic_raises_when_no_tool_use_block():
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    mock_resp.usage = MagicMock(
        input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_resp
    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        with pytest.raises(ValueError, match="No tool_use block"):
            _call_anthropic("claude-sonnet-4-6", "user message")


# ---------------------------------------------------------------------------
# _call_openai — mocked OpenAI client
# ---------------------------------------------------------------------------


def test_call_openai_parses_json_content():
    mock_message = MagicMock()
    mock_message.content = json.dumps(_VALID_VERDICT)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_usage = MagicMock(
        prompt_tokens=100,
        completion_tokens=50,
        prompt_tokens_details=MagicMock(cached_tokens=0),
    )
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_resp
    with patch.dict(sys.modules, {"openai": mock_openai}):
        result = _call_openai("gpt-4o-mini", "user message")

    assert result["winner"] == "A"


# ---------------------------------------------------------------------------
# _call_ollama — mocked via respx
# ---------------------------------------------------------------------------


@respx.mock
def test_call_ollama_parses_response():
    response_body = {
        "message": {"content": json.dumps(_VALID_VERDICT)},
    }
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=response_body)
    )
    result = _call_ollama("qwen2.5:14b", "user message")
    assert result["winner"] == "A"


# ---------------------------------------------------------------------------
# judge() — integration routing tests
# ---------------------------------------------------------------------------


def test_judge_routes_to_anthropic_for_claude():
    mock_resp = _make_anthropic_response(_VALID_VERDICT)
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_resp
    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        result = judge(
            ticker="AAPL",
            company="Apple Inc.",
            query="momentum outlook",
            report_a="Report A text.",
            report_b="Report B text.",
            a_chunks=20,
            b_chunks=20,
            model="claude-sonnet-4-6",
        )
    assert result["winner"] == "A"


def test_judge_routes_to_openai_for_gpt():
    mock_message = MagicMock()
    mock_message.content = json.dumps(_VALID_VERDICT)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = None

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_resp
    with patch.dict(sys.modules, {"openai": mock_openai}):
        result = judge(
            ticker="AAPL",
            company="Apple Inc.",
            query="earnings outlook",
            report_a="A",
            report_b="B",
            a_chunks=20,
            b_chunks=20,
            model="gpt-4o-mini",
        )
    assert result["winner"] == "A"


@respx.mock
def test_judge_routes_to_ollama_for_local_model():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": json.dumps(_VALID_VERDICT)}})
    )
    result = judge(
        ticker="7203.T",
        company="Toyota",
        query="outlook",
        report_a="A",
        report_b="B",
        a_chunks=10,
        b_chunks=10,
        model="qwen2.5:14b",
    )
    assert result["winner"] == "A"


def test_judge_returns_fallback_on_provider_exception():
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value.messages.create.side_effect = RuntimeError("API timeout")
    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        result = judge(
            ticker="AAPL",
            company="Apple",
            query="test",
            report_a="A",
            report_b="B",
            a_chunks=1,
            b_chunks=1,
            model="claude-sonnet-4-6",
        )
    assert result["winner"] == "tie"
    assert result["A"]["faithfulness"] == 0


def test_judge_truncates_long_reports():
    """Verify reports longer than _MAX_REPORT_CHARS are silently truncated (no crash)."""
    mock_resp = _make_anthropic_response(_VALID_VERDICT)
    long_report = "x" * 10_000
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_resp
    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        result = judge(
            ticker="AAPL",
            company="Apple",
            query="q",
            report_a=long_report,
            report_b=long_report,
            a_chunks=20,
            b_chunks=20,
            model="claude-sonnet-4-6",
        )
    assert "winner" in result
