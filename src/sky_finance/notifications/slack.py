"""
Slack client — thin wrapper over the Web API (chat.postMessage).

Two channels:
  SLACK_CHANNEL      — strategy reports (#sky-finance)
  SLACK_DEV_CHANNEL  — task lifecycle logs (#dev-logs)
"""

import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_API = "https://slack.com/api/chat.postMessage"
_BLOCK_MAX = 2900  # Slack section text limit is 3000 chars


def _md_to_mrkdwn(text: str) -> str:
    """Convert basic Markdown to Slack mrkdwn."""
    # ATX headings → bold line
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
    # **bold** / __bold__ → *bold*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = re.sub(r"__(.+?)__", r"*\1*", text)
    # *italic* / _italic_ → _italic_  (only single *)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"_\1_", text)
    # Horizontal rules → blank line (we add a divider block ourselves)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str, max_len: int = _BLOCK_MAX) -> list[str]:
    """Split text into chunks that fit inside a Slack section block."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split = text.rfind("\n", 0, max_len)
        if split == -1:
            split = max_len
        chunks.append(text[:split])
        text = text[split:].lstrip("\n")
    return chunks


def _token() -> str:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        raise OSError("SLACK_BOT_TOKEN is not set")
    return token


def _post(payload: dict[str, Any]) -> None:
    resp = httpx.post(
        _API,
        headers={"Authorization": f"Bearer {_token()}"},
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    if not body.get("ok"):
        raise RuntimeError(f"Slack API error: {body.get('error')} — {body}")


# ---------------------------------------------------------------------------
# Report channel (#sky-finance)
# ---------------------------------------------------------------------------


def post_report(
    *,
    strategy_name: str,
    tickers: list[str],
    ticker_names: dict[str, str] | None = None,
    report: str,
    model: str,
    duration_seconds: float | None,
    result_id: int,
) -> None:
    """Send a completed strategy report to the main channel."""
    channel = os.environ.get("SLACK_CHANNEL", "#sky-finance")

    names = ticker_names or {}

    def label(t: str) -> str:
        n = names.get(t, "")
        return f"{t} ({n})" if n else t

    ticker_str = "  ·  ".join(label(t) for t in tickers) if tickers else "—"
    dur_str = f"{duration_seconds:.1f}s" if duration_seconds is not None else "—"

    body = _md_to_mrkdwn(report)
    chunks = _chunk_text(body)

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Strategy Report: {strategy_name}"},
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"*Tickers:* {ticker_str}"},
                {"type": "mrkdwn", "text": f"*Model:* `{model}`"},
                {"type": "mrkdwn", "text": f"*Duration:* {dur_str}"},
                {"type": "mrkdwn", "text": f"*Result ID:* #{result_id}"},
            ],
        },
        {"type": "divider"},
    ]
    for chunk in chunks:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": chunk}})

    payload = {"channel": channel, "blocks": blocks}
    _post(payload)
    logger.info("Sent report to Slack channel=%s strategy=%r", channel, strategy_name)


# ---------------------------------------------------------------------------
# Dev log channel (#dev-logs)
# ---------------------------------------------------------------------------


def post_dev_log(text: str) -> None:
    """Send a plain-text structured log line to the dev channel."""
    channel = os.environ.get("SLACK_DEV_CHANNEL", "#dev-logs")
    _post({"channel": channel, "text": text})
    logger.debug("Dev log sent to %s: %s", channel, text[:120])
