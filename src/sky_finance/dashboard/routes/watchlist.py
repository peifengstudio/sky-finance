"""Watchlist route — stock list with market / position filter."""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from sky_finance.dashboard import queries
from sky_finance.dashboard._templates import templates

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def watchlist(
    request: Request,
    market: str = "",
    position: str = "",
) -> HTMLResponse:
    """
    Full watchlist page.

    Query params:
        market:   '' | 'us' | 'japan'
        position: '' | 'held'   (held = shares > 0 only)
    """
    stocks = queries.list_stocks_from_config(market_filter=market or None)
    if position == "held":
        stocks = [s for s in stocks if s["has_position"]]

    ctx = {
        "request": request,
        "stocks": stocks,
        "active_market": market,
        "active_position": position,
        "total": len(stocks),
        "held_count": sum(1 for s in stocks if s["has_position"]),
        "active_page": "watchlist",
    }

    # HTMX partial request — return only the table fragment
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(request, "partials/stock_table.html", ctx)

    return templates.TemplateResponse(request, "watchlist.html", ctx)
