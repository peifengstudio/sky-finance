"""Semantic search route — embed query → pgvector cosine similarity."""

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from sky_finance.dashboard import queries
from sky_finance.dashboard._templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request) -> HTMLResponse:
    ticker_names = queries.list_ticker_names()
    return templates.TemplateResponse(
        request,
        "search.html",
        {
            "request": request,
            "ticker_names": ticker_names,
            "active_page": "search",
        },
    )


@router.get("/search/results", response_class=HTMLResponse)
async def search_results(
    request: Request,
    q: str = "",
    ticker: str = "",
    source_type: str = "",
    sentiment: str = "",
    threshold: float = 0.70,
) -> HTMLResponse:
    """
    HTMX partial — called on every keypress (debounced 400 ms in the template).
    Embeds the query locally via Ollama then queries pgvector.
    """
    ctx: dict[str, Any] = {"request": request, "results": [], "q": q, "error": None}

    if not q.strip():
        return templates.TemplateResponse(request, "partials/search_results.html", ctx)

    try:
        from sky_finance.storage.db import get_connection
        from sky_finance.storage.embedder import embed_single

        vector = embed_single(q.strip())

        with get_connection() as conn:
            results = queries.semantic_search(
                conn,
                vector,
                ticker=ticker or None,
                source_type=source_type or None,
                sentiment=sentiment or None,
                threshold=threshold,
            )
        ctx["results"] = results

    except Exception as exc:
        logger.warning("Search failed for %r: %s", q, exc)
        ctx["error"] = str(exc)

    return templates.TemplateResponse(request, "partials/search_results.html", ctx)
