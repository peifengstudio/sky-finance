"""
Database connection pool.

A single ``psycopg_pool.ConnectionPool`` is created once per OS process and
shared across all callers.  This prevents the per-task open/close overhead and
caps the total number of PostgreSQL connections to a configurable maximum.

Usage (identical to the old bare-connection version):

    from sky_finance.storage.db import get_connection

    with get_connection() as conn:
        conn.execute("SELECT 1")
        conn.commit()

Lifecycle
---------
The pool must be opened before the first call to ``get_connection()`` and
closed cleanly on shutdown.  Wire these into the process entry point:

    FastAPI:  open_pool() / close_pool() in the lifespan hook  (dashboard/app.py)
    Celery:   open_pool() in worker_process_init signal         (scheduler/celery_app.py)
              close_pool() in worker_process_shutdown signal

If the pool has not been opened (e.g. in tests or CLI scripts), ``get_connection()``
falls back to a direct ``psycopg.connect()`` so callers keep working without change.

Note on async
-------------
FastAPI routes currently call ``get_connection()`` from ``async def`` handlers,
which blocks the event loop during the DB round-trip.  This is acceptable for a
local-first tool; the pool still prevents connection exhaustion.  A fully async
upgrade (``AsyncConnectionPool`` + ``AsyncConnection``) would require rewriting
all repository functions to use ``await conn.execute(...)`` — tracked as a
follow-up task.
"""

import logging
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_DEFAULT_DSN = "postgresql://skyfinance:skyfinance@localhost:5432/skyfinance"

# One pool per process — initialised by open_pool(), torn down by close_pool().
_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()  # guards _pool assignment only; the pool itself is thread-safe


# ---------------------------------------------------------------------------
# Pool config
# ---------------------------------------------------------------------------


def _pool_cfg() -> dict[str, Any]:
    settings_path = Path(__file__).parents[3] / "config" / "settings.toml"
    try:
        import tomllib

        with settings_path.open("rb") as f:
            result: dict[str, Any] = tomllib.load(f).get("database", {})
            return result
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def open_pool() -> None:
    """
    Create and open the connection pool.

    Safe to call multiple times — subsequent calls are no-ops.
    Raises immediately if the database is unreachable (fail fast at startup).
    """
    global _pool
    with _pool_lock:
        if _pool is not None:
            return
        cfg = _pool_cfg()
        dsn = os.environ.get("DATABASE_URL", _DEFAULT_DSN)
        _pool = ConnectionPool(
            conninfo=dsn,
            min_size=cfg.get("min_connections", 2),
            max_size=cfg.get("max_connections", 10),
            max_waiting=cfg.get("max_waiting", 20),
            timeout=cfg.get("connect_timeout", 5.0),
            open=True,  # connect eagerly — surfaces DB errors at startup, not mid-request
        )
        safe_dsn = dsn.split("@")[-1] if "@" in dsn else dsn
        logger.info(
            "DB pool opened | %s | min=%d max=%d",
            safe_dsn,
            cfg.get("min_connections", 2),
            cfg.get("max_connections", 10),
        )


def close_pool() -> None:
    """Drain in-flight connections and close the pool.  Call on process shutdown."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
            logger.info("DB pool closed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextmanager
def get_connection() -> Generator[psycopg.Connection[tuple[Any, ...]]]:
    """
    Return a database connection as a context manager.

    When the pool is open (production):
        - borrows a connection from the pool
        - on ``with`` exit: commits (or rolls back on exception) and returns the
          connection to the pool — does NOT close it

    When the pool is not open (tests, CLI scripts):
        - opens a bare psycopg.connect() connection
        - on ``with`` exit: commits / rolls back and closes the connection

    Both paths satisfy ``with get_connection() as conn:`` identically.
    ``conn`` is always a ``psycopg.Connection`` instance.
    """
    if _pool is not None:
        with _pool.connection() as conn:
            yield conn
        return

    # Fallback: bare connection for tests / one-off scripts
    dsn = os.environ.get("DATABASE_URL", _DEFAULT_DSN)
    logger.debug("DB pool not initialised — opening bare connection (tests / CLI)")
    with psycopg.connect(dsn) as conn:
        yield conn
