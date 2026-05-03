"""
Web dashboard — FastAPI + Jinja2 + HTMX + Tailwind CSS.

Start with:
    uvicorn sky_finance.dashboard.app:app --reload --port 8000
or via honcho:
    honcho start web
"""

import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse

from sky_finance.logging_config import setup_logging
from sky_finance.settings import ConfigurationError, validate_settings
from sky_finance.storage.db import close_pool, get_connection, open_pool

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    setup_logging()
    try:
        validate_settings()
    except ConfigurationError as exc:
        # Print to stderr before the logger is fully wired — operators need to see this.
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    open_pool()
    logger.info("Dashboard starting — http://localhost:8000")
    yield
    logger.info("Dashboard shutting down")
    close_pool()


app = FastAPI(title="sky-finance", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Routers — imported after app is created to avoid circular refs
# ---------------------------------------------------------------------------

from sky_finance.dashboard.routes.eval import router as eval_router  # noqa: E402
from sky_finance.dashboard.routes.pipeline import router as pipeline_router  # noqa: E402
from sky_finance.dashboard.routes.search import router as search_router  # noqa: E402
from sky_finance.dashboard.routes.stock import router as stock_router  # noqa: E402
from sky_finance.dashboard.routes.strategies import router as strategies_router  # noqa: E402
from sky_finance.dashboard.routes.watchlist import router as watchlist_router  # noqa: E402

app.include_router(watchlist_router)
app.include_router(stock_router)
app.include_router(search_router)
app.include_router(pipeline_router)
app.include_router(strategies_router)
app.include_router(eval_router)


@app.get("/health")
async def health() -> JSONResponse:
    """
    Probe DB, Redis, and Ollama.  Returns 200 when all pass, 503 when any fail.
    """
    from sky_finance.settings import get_settings

    settings = get_settings()
    checks: dict[str, dict[str, Any]] = {}

    # --- PostgreSQL ---
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
        checks["db"] = {"status": "ok"}
    except Exception as exc:
        logger.warning("Health check — DB: %s", exc)
        checks["db"] = {"status": "error", "error": str(exc)}

    # --- Redis ---
    try:
        import redis as _redis

        r = _redis.from_url(settings.env.celery_broker_url, socket_connect_timeout=3)  # type: ignore[no-untyped-call]
        r.ping()
        checks["redis"] = {"status": "ok"}
    except Exception as exc:
        logger.warning("Health check — Redis: %s", exc)
        checks["redis"] = {"status": "error", "error": str(exc)}

    # --- Ollama ---
    try:
        import httpx

        resp = httpx.get(f"{settings.env.ollama_base_url}/api/tags", timeout=3.0)
        resp.raise_for_status()
        checks["ollama"] = {"status": "ok"}
    except Exception as exc:
        logger.warning("Health check — Ollama: %s", exc)
        checks["ollama"] = {"status": "error", "error": str(exc)}

    all_ok = all(c["status"] == "ok" for c in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> RedirectResponse:
    return RedirectResponse(url="https://fav.farm/📈")
