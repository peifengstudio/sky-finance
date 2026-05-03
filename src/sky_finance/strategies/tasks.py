"""
Strategy tasks — run RAG queries and model analysis.

Task flow:
  dispatch_strategies
  └─ group( run_strategy_task(strategy_id) × N enabled strategies )
       └─ resolve_tickers → rag_fetch → build_prompt → model → save result → notify
"""

import logging
from typing import Any

from celery import group

from sky_finance.scheduler.celery_app import app

logger = logging.getLogger(__name__)


_SCHEDULER_KEY = "sky_finance:scheduler_enabled"


def _redis() -> Any:
    import os

    import redis

    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))  # type: ignore[no-untyped-call]


def scheduler_enabled() -> bool:
    """Return True only if the web UI has explicitly enabled the beat scheduler."""
    try:
        result: bool = _redis().get(_SCHEDULER_KEY) == b"1"
        return result
    except Exception:
        return False


def set_scheduler_enabled(enabled: bool) -> None:
    try:
        if enabled:
            _redis().set(_SCHEDULER_KEY, "1")
        else:
            _redis().delete(_SCHEDULER_KEY)
    except Exception as exc:
        logger.warning("Failed to set scheduler state in Redis: %s", exc)


def _dev_log(text: str) -> None:
    """Fire-and-forget dev log to Slack. Never raises."""
    try:
        from sky_finance.notifications.slack import post_dev_log

        post_dev_log(text)
    except Exception as exc:
        logger.warning("Dev log to Slack failed: %s", exc)


@app.task(  # type: ignore[untyped-decorator]
    name="sky_finance.strategies.tasks.dispatch_strategies",
    queue="strategies",
    ignore_result=True,
)
def dispatch_strategies() -> None:
    """Fan out one run_strategy_task per enabled strategy in the DB.

    Gated by a Redis flag set via the web UI — does nothing if the
    scheduler has not been explicitly enabled since the last restart.
    """
    if not scheduler_enabled():
        logger.info("Scheduled dispatch skipped — scheduler is disabled (enable via web UI)")
        return

    from sky_finance.storage.db import get_connection
    from sky_finance.strategies.repository import list_strategies

    with get_connection() as conn:
        strategies = list_strategies(conn, enabled_only=True)

    logger.info("Dispatching %d strategies", len(strategies))
    group(run_strategy_task.si(s["id"]) for s in strategies).apply_async()


@app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name="sky_finance.strategies.tasks.run_strategy_task",
    queue="strategies",
    max_retries=2,
    soft_time_limit=480,
    time_limit=540,
)
def run_strategy_task(self: Any, strategy_id: int) -> dict[str, Any]:
    """
    Execute a single strategy by its DB id:
      1. Load strategy definition from DB
      2. resolve_tickers → rag_fetch → build_prompt → model call
      3. Persist result to strategy_results
      4. Post report to Slack (#sky-finance) + lifecycle log to #dev-logs

    Args:
        strategy_id: strategies.id in the database.

    Returns:
        dict with strategy name, tickers, and result id.
    """
    from sky_finance.storage.db import get_connection
    from sky_finance.strategies.engine import run_strategy
    from sky_finance.strategies.repository import (
        get_strategy,
        save_strategy_result,
    )

    try:
        with get_connection() as conn:
            strategy = get_strategy(conn, strategy_id)
            if strategy is None:
                logger.error("Strategy id=%d not found in DB", strategy_id)
                return {"error": f"strategy {strategy_id} not found"}
            if not strategy["enabled"]:
                logger.warning("Strategy id=%d is disabled — skipping", strategy_id)
                return {"skipped": True}

        _dev_log(
            f"⚙️ *Strategy started* | `{strategy['name']}` | "
            f"scope: `{strategy['scope']}"
            f"{'/' + strategy['scope_value'] if strategy.get('scope_value') else ''}`"
            f" | model_tier: `{strategy['model_tier']}` | task: `{self.request.id[:8]}`"
        )

        with get_connection() as conn:
            report, model_id, tickers, ticker_names, ticker_news, started_at, duration, usage = (
                run_strategy(strategy, conn)
            )

            if not tickers:
                _dev_log(
                    f"⚠️ *Strategy skipped (no tickers)* | `{strategy['name']}` "
                    f"| task: `{self.request.id[:8]}`"
                )
                return {"strategy_id": strategy_id, "tickers": [], "result_id": None}

            result_id = save_strategy_result(
                conn,
                strategy_id=strategy["id"],
                strategy_name=strategy["name"],
                tickers=tickers,
                report=report,
                model=model_id,
                started_at=started_at,
                duration_seconds=duration,
                metadata={
                    "ticker_names": ticker_names,
                    "rag_news": ticker_news,
                    "usage": usage.to_dict(),
                },
            )
            conn.commit()

        logger.info(
            "Strategy %r completed — result_id=%d tickers=%s model=%s duration=%.1fs",
            strategy["name"],
            result_id,
            tickers,
            model_id,
            duration,
        )

        _dev_log(
            f"✅ *Strategy done* | `{strategy['name']}` | "
            f"tickers: `{', '.join(tickers)}` | model: `{model_id}` | "
            f"duration: `{duration:.1f}s` | result: `#{result_id}` | "
            f"task: `{self.request.id[:8]}`"
        )

        # Post full report to the main channel
        try:
            from sky_finance.notifications.slack import post_report

            post_report(
                strategy_name=strategy["name"],
                tickers=tickers,
                ticker_names=ticker_names,
                report=report,
                model=model_id,
                duration_seconds=duration,
                result_id=result_id,
            )
        except Exception as exc:
            logger.warning("Failed to post report to Slack: %s", exc)

        return {
            "strategy_id": strategy_id,
            "strategy": strategy["name"],
            "tickers": tickers,
            "result_id": result_id,
        }

    except Exception as exc:
        logger.warning(
            "Strategy id=%d failed: %s — retry %d/%d",
            strategy_id,
            exc,
            self.request.retries,
            self.max_retries,
        )
        _dev_log(
            f"❌ *Strategy failed* | id: `{strategy_id}` | "
            f"error: `{str(exc)[:120]}` | "
            f"retry: `{self.request.retries}/{self.max_retries}` | "
            f"task: `{self.request.id[:8]}`"
        )
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
