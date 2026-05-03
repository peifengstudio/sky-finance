"""
LLM cost estimation for sky-finance strategy runs.

Usage stats are written to strategy_results.metadata["usage"] after every run
so cost trends are visible in the dashboard without querying the API billing page.

Pricing is per 1 million tokens (USD).  Update _PRICING when you change models
or when providers adjust their rates — no other code needs to change.

Key design choice
-----------------
``input_tokens`` in UsageStats always means the **total** prompt-side tokens
(cached + non-cached + cache-creation).  This matches what you'd call "context
size" and is provider-neutral.  The cost formula then subtracts cached /
cache-creation portions and bills each at their own rate.
"""

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Pricing table  (USD per 1 000 000 tokens, estimates where marked)
# ---------------------------------------------------------------------------

# fmt: off
_PRICING: dict[str, dict[str, float]] = {
    # Anthropic — official 2025 published rates
    "claude-sonnet-4-6":         {"in": 3.00,   "out": 15.00,  "cache_read": 0.30,  "cache_write": 3.75},   # noqa: E501
    "claude-opus-4-7":           {"in": 15.00,  "out": 75.00,  "cache_read": 1.50,  "cache_write": 18.75},  # noqa: E501
    "claude-haiku-4-5-20251001": {"in": 0.80,   "out": 4.00,   "cache_read": 0.08,  "cache_write": 1.00},   # noqa: E501
    # OpenAI — 2025 published rates; gpt-5 / gpt-5.4-nano are estimates
    "gpt-4o":                    {"in": 2.50,   "out": 10.00,  "cache_read": 1.25,  "cache_write": 0.0},    # noqa: E501
    "gpt-4o-mini":               {"in": 0.15,   "out": 0.60,   "cache_read": 0.075, "cache_write": 0.0},   # noqa: E501
    "gpt-5":                     {"in": 10.00,  "out": 30.00,  "cache_read": 5.00,  "cache_write": 0.0},    # noqa: E501  # estimate
    "gpt-5.4-nano":              {"in": 0.30,   "out": 1.20,   "cache_read": 0.15,  "cache_write": 0.0},   # noqa: E501  # estimate
    "o3":                        {"in": 10.00,  "out": 40.00,  "cache_read": 2.50,  "cache_write": 0.0},    # noqa: E501
    "o3-mini":                   {"in": 1.10,   "out": 4.40,   "cache_read": 0.55,  "cache_write": 0.0},    # noqa: E501
}
# fmt: on


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class UsageStats:
    model: str
    provider: str  # "openai" | "claude" | "ollama"
    input_tokens: int  # total prompt tokens (cached + new + cache-creation)
    output_tokens: int
    cached_tokens: int = 0  # subset of input served from cache (cheaper rate)
    cache_creation_tokens: int = 0  # tokens written to a new cache entry (Anthropic only)
    cost_usd: float | None = None  # None when model is not in the pricing table

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cost_usd": self.cost_usd,
        }


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


def compute_cost(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> UsageStats:
    """
    Return a UsageStats with estimated cost_usd.

    ``input_tokens`` must be the **total** prompt token count (including cached
    and cache-creation tokens).  The formula bills each segment at its own rate:

        uncached  = input - cached - cache_creation
        cost      = uncached × in_rate
                  + cached × cache_read_rate
                  + cache_creation × cache_write_rate   (Anthropic only)
                  + output × out_rate

    Ollama (local) always returns cost_usd=0.0 — no API fee, though you still
    pay for electricity / GPU amortisation which we don't model here.
    """
    if provider == "ollama":
        cost: float | None = 0.0
    else:
        p = _PRICING.get(model)
        if p is None:
            cost = None
        else:
            uncached = max(0, input_tokens - cached_tokens - cache_creation_tokens)
            cost = round(
                uncached * p["in"] / 1_000_000
                + cached_tokens * p["cache_read"] / 1_000_000
                + cache_creation_tokens * p["cache_write"] / 1_000_000
                + output_tokens * p["out"] / 1_000_000,
                6,
            )

    return UsageStats(
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cost_usd=cost,
    )
