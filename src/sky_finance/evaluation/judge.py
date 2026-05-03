"""
LLM-as-a-judge: score two RAG-generated reports on three dimensions.

The judge receives Method A and Method B without knowing which retrieval
strategy produced each — this prevents systematic bias toward either approach.
Scoring uses three dimensions that matter for financial research:

  faithfulness   — claims are grounded in the retrieved context (no hallucinations)
  coverage       — multiple market perspectives represented (bull / bear / neutral)
  actionability  — concrete, decision-relevant insights for an investor

All three providers enforce structured output natively so the result is always
a valid, schema-conformant dict — no prompt-level JSON templates needed:

  claude-*          → Anthropic  tool_use  (ANTHROPIC_API_KEY)
  gpt-* / o1 / o3   → OpenAI    response_format json_schema  (OPENAI_API_KEY)
  anything else     → Ollama    format=<schema>  (local, free)

Examples::

    uv run sky-eval --strategy-id 1                                   # claude-sonnet-4-6 (default)
    uv run sky-eval --strategy-id 1 --judge-model gpt-4o-mini         # OpenAI
    uv run sky-eval --strategy-id 1 --judge-model qwen2.5:14b-instruct # Ollama, free
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared output schema — used by all three providers
# ---------------------------------------------------------------------------

_SCORES_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer", "description": "0–10"},
        "coverage": {"type": "integer", "description": "0–10"},
        "actionability": {"type": "integer", "description": "0–10"},
    },
    "required": ["faithfulness", "coverage", "actionability"],
    "additionalProperties": False,
}

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "A": _SCORES_SCHEMA,
        "B": _SCORES_SCHEMA,
        "winner": {
            "type": "string",
            "enum": ["A", "B", "tie"],
            "description": "Which report is better overall",
        },
        "reasoning": {
            "type": "string",
            "description": "2–3 sentences on the decisive difference between A and B",
        },
    },
    "required": ["A", "B", "winner", "reasoning"],
    "additionalProperties": False,
}

# Anthropic tool definition — forces the model to call this and nothing else
_SUBMIT_VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit the evaluation verdict with dimension scores for both reports.",
    "input_schema": _VERDICT_SCHEMA,
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert financial-research evaluator.
Compare two AI-generated equity research reports and score them objectively."""

_USER_TEMPLATE = """\
Evaluate two RAG-powered financial analysis reports for **{ticker}** ({company}).

Research query: "{query}"

---
## Method A  ({a_chunks} context chunks retrieved)

{report_a}

---
## Method B  ({b_chunks} context chunks retrieved)

{report_b}

---
Score each report on three dimensions (integer 0–10):

| Dimension      | Definition |
|----------------|------------|
| faithfulness   | Claims are grounded in the retrieved context; no obvious hallucinations |
| coverage       | Multiple market perspectives represented (bullish, bearish, neutral signals) |
| actionability  | Concrete, decision-relevant insights — not just restating the news |

Choose a winner (A, B, or tie) and provide 2–3 sentences on the decisive difference."""

# Cap report length sent to the judge to stay within context limits
_MAX_REPORT_CHARS = 3500


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def judge(
    ticker: str,
    company: str,
    query: str,
    report_a: str,
    report_b: str,
    a_chunks: int,
    b_chunks: int,
    model: str = "claude-sonnet-4-6",
) -> dict[str, Any]:
    """
    Ask an LLM to compare two reports and return a scored verdict.

    Provider is inferred from the model name:
    - ``claude-*``              → Anthropic (ANTHROPIC_API_KEY)
    - ``gpt-*`` / ``o1`` / ``o3`` → OpenAI  (OPENAI_API_KEY)
    - anything else             → Ollama    (local, no key needed)

    Returns a dict with keys: ``A``, ``B``, ``winner``, ``reasoning``.
    On parse failure a neutral fallback is returned so the eval run continues.
    """
    user_msg = _USER_TEMPLATE.format(
        ticker=ticker,
        company=company or ticker,
        query=query,
        report_a=report_a[:_MAX_REPORT_CHARS],
        report_b=report_b[:_MAX_REPORT_CHARS],
        a_chunks=a_chunks,
        b_chunks=b_chunks,
    )

    provider = _infer_provider(model)
    logger.info("Judge | ticker=%s provider=%s model=%s", ticker, provider, model)

    try:
        if provider == "anthropic":
            result = _call_anthropic(model, user_msg)
        elif provider == "openai":
            result = _call_openai(model, user_msg)
        else:
            result = _call_ollama(model, user_msg)
        _validate(result)
    except Exception as exc:
        logger.warning("Judge failed for %s (%s): %s", ticker, type(exc).__name__, exc)
        result = _fallback(f"Judge error: {exc}")

    logger.info(
        "Judge done | ticker=%s winner=%s A=%s B=%s",
        ticker,
        result.get("winner"),
        result.get("A"),
        result.get("B"),
    )
    return result


# ---------------------------------------------------------------------------
# Provider inference + call helpers
# ---------------------------------------------------------------------------


def _infer_provider(model: str) -> str:
    """Return 'anthropic', 'openai', or 'ollama' based on the model name."""
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    return "ollama"


def _call_anthropic(model: str, user_msg: str) -> dict[str, Any]:
    """Use Anthropic tool_use to force structured output — result is a parsed dict.

    The system prompt is marked ephemeral so repeated judge calls within the
    same eval run (one per ticker) hit the prompt cache — ~90% cost reduction
    on the system-prompt tokens after the first call.
    """
    import anthropic

    client = anthropic.Anthropic(timeout=60)
    response = client.messages.create(  # type: ignore[call-overload]
        model=model,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
        tools=[_SUBMIT_VERDICT_TOOL],
        tool_choice={"type": "tool", "name": "submit_verdict"},
    )
    u = response.usage
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_created = getattr(u, "cache_creation_input_tokens", 0) or 0
    logger.info(
        "Anthropic judge | model=%s | input=%d | output=%d | cache_read=%d | cache_created=%d",
        model,
        u.input_tokens,
        u.output_tokens,
        cache_read,
        cache_created,
    )
    for block in response.content:
        if block.type == "tool_use":
            result: dict[str, Any] = dict(block.input)
            return result
    raise ValueError("No tool_use block in Anthropic response")


def _call_openai(model: str, user_msg: str) -> dict[str, Any]:
    """Use OpenAI response_format json_schema — output is schema-guaranteed JSON.

    OpenAI automatically caches prompts longer than 1024 tokens (50% cost
    reduction).  Cache hit count is logged via prompt_tokens_details.
    """
    from openai import OpenAI

    client = OpenAI(timeout=60)
    response = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_verdict",
                "strict": True,
                "schema": _VERDICT_SCHEMA,
            },
        },
    )
    if u := response.usage:
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        logger.info(
            "OpenAI judge | model=%s | input=%d | output=%d | cached=%d",
            model,
            u.prompt_tokens,
            u.completion_tokens,
            cached,
        )
    parsed: dict[str, Any] = json.loads(response.choices[0].message.content or "{}")
    return parsed


def _call_ollama(model: str, user_msg: str) -> dict[str, Any]:
    """Use Ollama format=<schema> for constrained JSON generation."""
    import httpx

    base_url = "http://localhost:11434"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "options": {"temperature": 0},
        "format": _VERDICT_SCHEMA,
    }
    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{base_url}/api/chat", json=payload)
        resp.raise_for_status()
    parsed: dict[str, Any] = json.loads(resp.json().get("message", {}).get("content", "{}").strip())
    return parsed


# ---------------------------------------------------------------------------
# Result validation helpers
# ---------------------------------------------------------------------------


def _validate(result: dict[str, Any]) -> None:
    """Raise ValueError if the judge response is structurally invalid."""
    required_keys = {"A", "B", "winner", "reasoning"}
    missing = required_keys - result.keys()
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    for label in ("A", "B"):
        scores = result[label]
        for dim in ("faithfulness", "coverage", "actionability"):
            if not isinstance(scores.get(dim), (int, float)):
                raise ValueError(f"scores.{label}.{dim} must be numeric")
    if result["winner"] not in ("A", "B", "tie"):
        raise ValueError(f"winner must be A, B, or tie — got {result['winner']!r}")


def _fallback(reason: str) -> dict[str, Any]:
    return {
        "A": {"faithfulness": 0, "coverage": 0, "actionability": 0},
        "B": {"faithfulness": 0, "coverage": 0, "actionability": 0},
        "winner": "tie",
        "reasoning": f"Judge result is unreliable — {reason}",
    }


def score_avg(scores: dict[str, Any]) -> float:
    """Return the mean of the three scoring dimensions."""
    vals = [float(scores.get(k, 0)) for k in ("faithfulness", "coverage", "actionability")]
    return round(sum(vals) / len(vals), 2)
