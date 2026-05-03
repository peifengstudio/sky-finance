"""
Notification tasks — send Slack messages for strategy digests and alerts.
"""

import logging
from typing import Any

from sky_finance.scheduler.celery_app import app

logger = logging.getLogger(__name__)


@app.task(  # type: ignore[untyped-decorator]
    name="sky_finance.notifications.tasks.send_digest",
    queue="notifications",
    max_retries=3,
    default_retry_delay=30,
    soft_time_limit=60,
    time_limit=90,
)
def send_digest() -> None:
    """
    Compile today's strategy results and send a morning digest to Slack.
    Reads the most recent strategy_results rows (from today) and formats
    a Block Kit message.
    """
    logger.info("Sending morning digest to Slack")
    # TODO: implement sky_finance.notifications.slack and call it here
    raise NotImplementedError("Slack notifier not yet implemented")


@app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name="sky_finance.notifications.tasks.send_alert",
    queue="notifications",
    max_retries=3,
    default_retry_delay=15,
    soft_time_limit=30,
    time_limit=60,
)
def send_alert(self: Any, ticker: str, alert_type: str, payload: dict[str, Any]) -> None:
    """
    Send an immediate Slack alert for a specific event.
    Called ad-hoc by other tasks (e.g., significant price move, pipeline error).

    Args:
        ticker:     stock symbol that triggered the alert.
        alert_type: one of 'price_move' | 'pipeline_error' | 'strategy_signal'.
        payload:    dict with event-specific data to include in the message.
    """
    logger.info("Sending alert [%s] for %s", alert_type, ticker)
    try:
        # TODO: implement sky_finance.notifications.slack and call it here
        raise NotImplementedError("Slack notifier not yet implemented")
    except Exception as exc:
        raise self.retry(exc=exc, countdown=15)
