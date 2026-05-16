"""Unit tests for sky_finance.notifications.slack."""

import json
import os
from unittest.mock import patch

import httpx
import pytest
import respx

from sky_finance.notifications.slack import (
    _chunk_text,
    _md_to_mrkdwn,
    post_dev_log,
    post_report,
)

_SLACK_API = "https://slack.com/api/chat.postMessage"
_OK_RESPONSE = httpx.Response(200, json={"ok": True})


# ---------------------------------------------------------------------------
# _md_to_mrkdwn — pure function tests
# ---------------------------------------------------------------------------


def test_md_heading_converted():
    # Heading → *text*, then italic rule re-matches the single asterisks → _text_
    result = _md_to_mrkdwn("# Title")
    assert "Title" in result
    assert "#" not in result


def test_md_heading_strips_hashes_all_levels():
    for level in range(1, 7):
        result = _md_to_mrkdwn(f"{'#' * level} Heading")
        assert "#" not in result
        assert "Heading" in result


def test_md_double_asterisk_converted():
    result = _md_to_mrkdwn("**bold text**")
    assert "bold text" in result
    assert "**" not in result


def test_md_double_underscore_converted():
    result = _md_to_mrkdwn("__bold text__")
    assert "bold text" in result
    assert "__" not in result


def test_md_horizontal_rule_removed():
    result = _md_to_mrkdwn("before\n---\nafter")
    assert "---" not in result
    assert "before" in result
    assert "after" in result


def test_md_triple_blank_lines_collapsed():
    result = _md_to_mrkdwn("a\n\n\n\nb")
    assert "\n\n\n" not in result
    assert "a" in result
    assert "b" in result


def test_md_trailing_whitespace_stripped():
    result = _md_to_mrkdwn("line one   \nline two  ")
    for line in result.splitlines():
        assert line == line.rstrip()


def test_md_plain_text_unchanged():
    assert _md_to_mrkdwn("hello world") == "hello world"


# ---------------------------------------------------------------------------
# _chunk_text — pure function tests
# ---------------------------------------------------------------------------


def test_chunk_text_short_returns_single():
    text = "short text"
    assert _chunk_text(text) == [text]


def test_chunk_text_exactly_at_limit_returns_single():
    text = "x" * 2900
    assert _chunk_text(text) == [text]


def test_chunk_text_splits_at_newline():
    line_a = "a" * 2000
    line_b = "b" * 1000
    text = line_a + "\n" + line_b
    chunks = _chunk_text(text)
    assert len(chunks) == 2
    assert all(len(c) <= 2900 for c in chunks)


def test_chunk_text_hard_cut_when_no_newline():
    text = "x" * 3500
    chunks = _chunk_text(text)
    assert len(chunks) == 2
    assert all(len(c) <= 2900 for c in chunks)


def test_chunk_text_long_text_multiple_chunks():
    text = ("word " * 600).strip()  # ~3000 chars
    chunks = _chunk_text(text, max_len=500)
    assert len(chunks) >= 3
    assert all(len(c) <= 500 for c in chunks)


# ---------------------------------------------------------------------------
# _token / _post — error paths
# ---------------------------------------------------------------------------


def test_post_raises_when_token_missing():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("SLACK_BOT_TOKEN", None)
        from sky_finance.notifications.slack import _token

        with pytest.raises(OSError, match="SLACK_BOT_TOKEN"):
            _token()


@respx.mock
def test_post_raises_on_slack_api_error():
    respx.post(_SLACK_API).mock(
        return_value=httpx.Response(200, json={"ok": False, "error": "channel_not_found"})
    )
    with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"}):
        with pytest.raises(RuntimeError, match="Slack API error"):
            post_dev_log("test message")


# ---------------------------------------------------------------------------
# post_report
# ---------------------------------------------------------------------------


@respx.mock
def test_post_report_block_structure():
    route = respx.post(_SLACK_API).mock(return_value=_OK_RESPONSE)
    with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test", "SLACK_CHANNEL": "#test"}):
        post_report(
            strategy_name="TestStrategy",
            tickers=["AAPL"],
            report="Short report text.",
            model="claude-sonnet-4-6",
            duration_seconds=2.5,
            result_id=42,
        )
    assert route.called
    body = json.loads(route.calls[0].request.content)
    blocks = body["blocks"]
    assert blocks[0]["type"] == "header"
    assert blocks[1]["type"] == "context"
    assert blocks[2]["type"] == "divider"
    assert all(b["type"] == "section" for b in blocks[3:])


@respx.mock
def test_post_report_ticker_names_shown():
    route = respx.post(_SLACK_API).mock(return_value=_OK_RESPONSE)
    with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"}):
        post_report(
            strategy_name="S",
            tickers=["AAPL"],
            ticker_names={"AAPL": "Apple Inc."},
            report="Report.",
            model="gpt-4o",
            duration_seconds=1.0,
            result_id=1,
        )
    body = json.loads(route.calls[0].request.content)
    context_text = str(body["blocks"][1])
    assert "Apple Inc." in context_text


@respx.mock
def test_post_report_none_duration_shows_dash():
    route = respx.post(_SLACK_API).mock(return_value=_OK_RESPONSE)
    with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"}):
        post_report(
            strategy_name="S",
            tickers=[],
            report="x",
            model="m",
            duration_seconds=None,
            result_id=0,
        )
    body = json.loads(route.calls[0].request.content)
    context_text = str(body["blocks"][1])
    assert "—" in context_text


@respx.mock
def test_post_report_long_report_multiple_sections():
    route = respx.post(_SLACK_API).mock(return_value=_OK_RESPONSE)
    long_report = "Analysis line.\n" * 300
    with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"}):
        post_report(
            strategy_name="S",
            tickers=["AAPL"],
            report=long_report,
            model="m",
            duration_seconds=5.0,
            result_id=7,
        )
    body = json.loads(route.calls[0].request.content)
    section_blocks = [b for b in body["blocks"] if b["type"] == "section"]
    assert len(section_blocks) >= 2
    for block in section_blocks:
        assert len(block["text"]["text"]) <= 2900


# ---------------------------------------------------------------------------
# post_dev_log
# ---------------------------------------------------------------------------


@respx.mock
def test_post_dev_log_sends_text_key():
    route = respx.post(_SLACK_API).mock(return_value=_OK_RESPONSE)
    with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test", "SLACK_DEV_CHANNEL": "#dev"}):
        post_dev_log("pipeline completed for AAPL")
    body = json.loads(route.calls[0].request.content)
    assert "text" in body
    assert body["channel"] == "#dev"
    assert "blocks" not in body
