"""
Celery application entry point.

Three separate processes are required:

  # 1. Worker — executes tasks pulled from Redis queues
  celery -A sky_finance.scheduler.celery_app worker --loglevel=info

  # 2. Beat — pushes scheduled tasks into Redis queues (run exactly one instance)
  celery -A sky_finance.scheduler.celery_app beat --loglevel=info

  # 3. Flower — monitoring UI at http://localhost:5555
  celery -A sky_finance.scheduler.celery_app flower

Or use `uv run honcho start` to start all three.
"""

import os
from typing import Any

from celery import Celery
from celery.schedules import crontab
from celery.signals import celeryd_after_setup, worker_process_init, worker_process_shutdown
from kombu import Exchange, Queue

broker = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

app = Celery("sky_finance", broker=broker, backend=backend)

# ---------------------------------------------------------------------------
# Logging — applied as soon as the worker/beat process is ready
# ---------------------------------------------------------------------------


@celeryd_after_setup.connect  # type: ignore[untyped-decorator]
def _configure_logging(
    sender: Any, instance: Any, loglevel: Any, **kwargs: Any
) -> None:  # noqa: ARG001
    """Wire sky_finance logging into every Celery worker and beat process."""
    import sys

    from sky_finance.logging_config import setup_logging
    from sky_finance.settings import ConfigurationError, validate_settings

    setup_logging(log_level=loglevel if isinstance(loglevel, str) else "INFO")

    try:
        validate_settings()
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


@worker_process_init.connect  # type: ignore[untyped-decorator]
def _open_db_pool(**kwargs: Any) -> None:  # noqa: ARG001
    """Open the connection pool in each worker process after the fork.

    This signal fires once per worker subprocess (not per task), which is the
    correct granularity for pool lifecycle.  Initialising before the fork would
    share file descriptors across processes — a known source of connection errors.
    """
    from sky_finance.storage.db import open_pool

    open_pool()


@worker_process_shutdown.connect  # type: ignore[untyped-decorator]
def _close_db_pool(**kwargs: Any) -> None:  # noqa: ARG001
    """Drain and close the pool when a worker process exits."""
    from sky_finance.storage.db import close_pool

    close_pool()


# ---------------------------------------------------------------------------
# Queue definitions
# Each pipeline stage has its own queue so workers can be scaled independently.
# ---------------------------------------------------------------------------
_default_exchange = Exchange("default", type="direct")

app.conf.task_queues = (
    Queue("ingestion", Exchange("ingestion"), routing_key="ingestion"),
    Queue("pipeline", Exchange("pipeline"), routing_key="pipeline"),
    Queue("storage", Exchange("storage"), routing_key="storage"),
    Queue("strategies", Exchange("strategies"), routing_key="strategies"),
    Queue("notifications", Exchange("notifications"), routing_key="notifications"),
    Queue("default", _default_exchange, routing_key="default"),
)
app.conf.task_default_queue = "default"

# ---------------------------------------------------------------------------
# Task routing — wildcard by module path
# ---------------------------------------------------------------------------
app.conf.task_routes = {
    "sky_finance.ingestion.*": {"queue": "ingestion"},
    "sky_finance.pipeline.*": {"queue": "pipeline"},
    "sky_finance.storage.*": {"queue": "storage"},
    "sky_finance.strategies.*": {"queue": "strategies"},
    "sky_finance.notifications.*": {"queue": "notifications"},
}

# ---------------------------------------------------------------------------
# Core config
# ---------------------------------------------------------------------------
app.config_from_object(
    {
        # Serialisation
        "task_serializer": "json",
        "result_serializer": "json",
        "accept_content": ["json"],
        # Timezone
        "timezone": "UTC",
        "enable_utc": True,
        # Reliability — task is not acknowledged until it completes;
        # if the worker dies mid-task, another worker will retry it.
        "task_acks_late": True,
        "task_reject_on_worker_lost": True,
        # Timeouts (overridable per-task via @app.task decorator)
        "task_soft_time_limit": int(os.environ.get("CELERY_TASK_SOFT_LIMIT", "300")),
        "task_time_limit": int(os.environ.get("CELERY_TASK_HARD_LIMIT", "600")),
        "task_track_started": True,  # expose STARTED state (not just PENDING)
        "result_expires": 60 * 60 * 24,  # keep results in Redis for 24 h
        # Auto-discover tasks
        "imports": [
            "sky_finance.ingestion.tasks",
            "sky_finance.pipeline.tasks",
            "sky_finance.storage.tasks",
            "sky_finance.strategies.tasks",
            "sky_finance.notifications.tasks",
        ],
    }
)

# ---------------------------------------------------------------------------
# Celery Beat schedule  (cron = UTC)
#
# Beat does NOT execute tasks — it only enqueues them at the right time.
# The worker picks them up and runs them.
# ---------------------------------------------------------------------------
app.conf.beat_schedule = {
    # -------------------------------------------------------------------
    # Stock data ingestion
    #
    # US  — 23:00 UTC weekdays
    #   NYSE/NASDAQ closes 16:00 ET.
    #   Winter (EST = UTC-5): close = 21:00 UTC → 23:00 is +2 h
    #   Summer (EDT = UTC-4): close = 20:00 UTC → 23:00 is +3 h
    #   Fixed UTC time avoids any DST handling in code.
    #
    # Japan — 07:30 UTC weekdays
    #   TSE closes 15:30 JST = 06:30 UTC (Japan has no DST).
    #   07:30 UTC is exactly +1 h after close.
    # -------------------------------------------------------------------
    "ingest-us-stocks": {
        "task": "sky_finance.ingestion.tasks.dispatch_ingest_us_stocks",
        "schedule": crontab(minute=0, hour=23, day_of_week="1-5"),
    },
    "ingest-japan-stocks": {
        "task": "sky_finance.ingestion.tasks.dispatch_ingest_japan_stocks",
        "schedule": crontab(minute=30, hour=7, day_of_week="1-5"),
    },
    # News — every hour
    "ingest-news": {
        "task": "sky_finance.ingestion.tasks.dispatch_ingest_news",
        "schedule": crontab(minute=0),
    },
    # Cleaning pipeline — 30 min past every hour
    "run-pipeline": {
        "task": "sky_finance.pipeline.tasks.dispatch_pipeline",
        "schedule": crontab(minute=30),
    },
    # Strategy analysis — 09:00 UTC daily (Mon–Fri)
    # Runs after US ingestion (23:00) + pipeline (23:30) have both completed.
    "run-strategies": {
        "task": "sky_finance.strategies.tasks.dispatch_strategies",
        "schedule": crontab(minute=0, hour=9, day_of_week="1-5"),
    },
    # Slack digest — 09:05 UTC daily (5 min after strategies)
    "send-digest": {
        "task": "sky_finance.notifications.tasks.send_digest",
        "schedule": crontab(minute=5, hour=9, day_of_week="1-5"),
    },
}
