"""Stock detail route — per-ticker config, latest price, news, strategies."""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from sky_finance.dashboard import queries
from sky_finance.dashboard._templates import templates
from sky_finance.storage.db import get_connection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stocks/{ticker}", response_class=HTMLResponse)
async def stock_detail(request: Request, ticker: str) -> HTMLResponse:
    cfg = queries.get_stock_config(ticker)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker!r} not in config")

    with get_connection() as conn:
        latest = queries.get_latest_raw_data(conn, ticker)
        news = queries.get_stock_news(conn, ticker, limit=10)
        strategies = queries.get_strategy_results(conn, ticker, limit=3)

    # Extract OHLCV summary from raw payload for display
    ohlcv_summary = None
    if latest:
        payload = latest["payload"]
        ohlcv = payload.get("ohlcv", [])
        if ohlcv:
            last = ohlcv[-1]
            prev = ohlcv[-2] if len(ohlcv) >= 2 else None
            change_pct = None
            if prev and prev.get("close") and last.get("close"):
                change_pct = ((last["close"] - prev["close"]) / prev["close"]) * 100
            ohlcv_summary = {
                "date": last.get("date"),
                "open": last.get("open"),
                "high": last.get("high"),
                "low": last.get("low"),
                "close": last.get("close"),
                "volume": last.get("volume"),
                "change_pct": round(change_pct, 2) if change_pct is not None else None,
                "fetched_at": latest["fetched_at"],
                "fundamentals": payload.get("fundamentals", {}),
            }

    return templates.TemplateResponse(
        request,
        "stock.html",
        {
            "request": request,
            "cfg": cfg,
            "ohlcv": ohlcv_summary,
            "news": news,
            "strategies": strategies,
            "active_page": "watchlist",
        },
    )


@router.get("/stocks/{ticker}/news", response_class=HTMLResponse)
async def stock_news_more(
    request: Request,
    ticker: str,
    offset: int = 0,
) -> HTMLResponse:
    """HTMX partial — load more news articles (infinite scroll)."""
    with get_connection() as conn:
        news = queries.get_stock_news(conn, ticker, limit=10, offset=offset)

    return templates.TemplateResponse(
        request,
        "partials/news_list.html",
        {"request": request, "news": news, "ticker": ticker, "offset": offset + 10},
    )
