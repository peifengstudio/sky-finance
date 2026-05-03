"""
Seed default strategies from config/strategies/*.toml into the database.

Idempotent: uses INSERT … ON CONFLICT (name) DO UPDATE, so running it
multiple times or after editing a TOML file is safe — existing rows are
updated in-place without touching strategy_results history.

Run with:
    uv run python -m sky_finance.strategies.seed
"""

import logging
import tomllib
from pathlib import Path
from typing import Any

from sky_finance.logging_config import setup_logging
from sky_finance.storage.db import get_connection

logger = logging.getLogger(__name__)

_STRATEGIES_DIR = Path(__file__).parents[3] / "config" / "strategies"


def _flatten(data: dict[str, Any]) -> dict[str, Any]:
    rag = data.get("rag", {})
    prompt = data.get("prompt", {})
    return {
        "name": data["name"],
        "description": data.get("description", ""),
        "scope": data.get("scope", "global"),
        "scope_value": data.get("scope_value") or None,
        "rag_query_template": rag.get("query_template", ""),
        "prompt_template": prompt.get("template", ""),
        "model_tier": data.get("model_tier", "local"),
        "schedule": data.get("schedule") or None,
        "enabled": data.get("enabled", True),
        "rag_threshold": rag.get("threshold", 0.55),
        "rag_top_k_positive": rag.get("top_k_positive", 20),
        "rag_top_k_neutral": rag.get("top_k_neutral", 20),
        "rag_top_k_negative": rag.get("top_k_negative", 20),
        "retrieval_mode": rag.get("retrieval_mode", "hybrid"),
    }


def seed_strategies() -> int:
    paths = sorted(_STRATEGIES_DIR.glob("*.toml"))
    if not paths:
        logger.warning("No TOML files found in %s — nothing to seed", _STRATEGIES_DIR)
        return 0

    with get_connection() as conn:
        for path in paths:
            with path.open("rb") as f:
                s = _flatten(tomllib.load(f))
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO strategies
                        (name, description, scope, scope_value,
                         rag_query_template, prompt_template, model_tier,
                         schedule, enabled,
                         rag_threshold, rag_top_k_positive, rag_top_k_neutral,
                         rag_top_k_negative, retrieval_mode)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (name) DO UPDATE SET
                        description        = EXCLUDED.description,
                        scope              = EXCLUDED.scope,
                        scope_value        = EXCLUDED.scope_value,
                        rag_query_template = EXCLUDED.rag_query_template,
                        prompt_template    = EXCLUDED.prompt_template,
                        model_tier         = EXCLUDED.model_tier,
                        schedule           = EXCLUDED.schedule,
                        enabled            = EXCLUDED.enabled,
                        rag_threshold      = EXCLUDED.rag_threshold,
                        rag_top_k_positive = EXCLUDED.rag_top_k_positive,
                        rag_top_k_neutral  = EXCLUDED.rag_top_k_neutral,
                        rag_top_k_negative = EXCLUDED.rag_top_k_negative,
                        retrieval_mode     = EXCLUDED.retrieval_mode,
                        updated_at         = now()
                    """,
                    (
                        s["name"],
                        s["description"],
                        s["scope"],
                        s["scope_value"],
                        s["rag_query_template"],
                        s["prompt_template"],
                        s["model_tier"],
                        s["schedule"],
                        s["enabled"],
                        s["rag_threshold"],
                        s["rag_top_k_positive"],
                        s["rag_top_k_neutral"],
                        s["rag_top_k_negative"],
                        s["retrieval_mode"],
                    ),
                )
            conn.commit()
            logger.info("Seeded strategy %r", s["name"])

    return len(paths)


if __name__ == "__main__":
    setup_logging()
    n = seed_strategies()
    print(f"Seeded {n} strateg{'y' if n == 1 else 'ies'}.")
