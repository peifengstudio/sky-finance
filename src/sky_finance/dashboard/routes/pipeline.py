"""Pipeline status route — processing counts, recent documents."""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from sky_finance.dashboard import queries
from sky_finance.dashboard._templates import templates
from sky_finance.storage.db import get_connection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page(request: Request) -> HTMLResponse:
    with get_connection() as conn:
        stats = queries.get_pipeline_stats(conn)
        recent_docs = queries.get_recent_documents(conn, limit=20)

    return templates.TemplateResponse(
        request,
        "pipeline.html",
        {
            "request": request,
            "stats": stats,
            "recent_docs": recent_docs,
            "active_page": "pipeline",
        },
    )


@router.get("/pipeline/stats", response_class=HTMLResponse)
async def pipeline_stats_partial(request: Request) -> HTMLResponse:
    """HTMX partial — auto-refreshed every 30 s from the pipeline page."""
    with get_connection() as conn:
        stats = queries.get_pipeline_stats(conn)

    return templates.TemplateResponse(
        request,
        "partials/pipeline_stats.html",
        {"request": request, "stats": stats},
    )
