"""
LLM cost estimation for sky-finance strategy runs.

Usage stats are written to strategy_results.metadata["usage"] after every run
so cost trends are visible in the dashboard without querying the API billing page.

Pricing is per 1 million tokens (USD).  Configure it under
``[models.<tier>.pricing]`` in config/settings.toml so changing models or
rates does not require code changes.

Key design choice
-----------------
``input_tokens`` in UsageStats always means the **total** prompt-side tokens
(cached + non-cached + cache-creation).  This matches what you'd call "context
size" and is provider-neutral.  The cost formula then subtracts cached /
cache-creation portions and bills each at their own rate.
"""

from dataclasses import dataclass
from typing import Any

from sky_finance.settings import ModelPricingConfig, get_settings

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
    cost_usd: float | None = None  # None when model pricing is not configured

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


def _pricing_for_model(model: str) -> ModelPricingConfig | None:
    """
    Return pricing configured for a model tier, if present.

    Pricing is stored beside the model ID in config/settings.toml so changing a
    model or its published rates stays configuration-only.
    """
    for tier in get_settings().toml.models.values():
        if tier.model == model:
            return tier.pricing
    return None


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
        p = _pricing_for_model(model)
        if p is None:
            cost = None
        else:
            uncached = max(0, input_tokens - cached_tokens - cache_creation_tokens)
            cost = round(
                uncached * p.input / 1_000_000
                + cached_tokens * p.cache_read / 1_000_000
                + cache_creation_tokens * p.cache_write / 1_000_000
                + output_tokens * p.output / 1_000_000,
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
