"""
Ingestion tasks — fetch stock data for all enabled tickers.

Schedule (UTC, weekdays only):
  - US stocks   : 23:00 UTC  (≥ 2 h after NYSE/NASDAQ close regardless of DST)
  - Japan stocks: 07:30 UTC  (1 h after TSE close at 06:30 UTC; Japan has no DST)

Task flow:
  Beat
  ├─ dispatch_ingest_us_stocks     →  group( ingest_stock("AAPL",   "us")    × N )
  ├─ dispatch_ingest_japan_stocks  →  group( ingest_stock("7203.T", "japan") × N )
  └─ dispatch_ingest_news          →  group( ingest_news_for_ticker × N )
"""

import logging
from typing import Any

import httpx
from celery import group
from requests.exceptions import RequestException

from sky_finance.config import list_stock_configs, load_stock_config
from sky_finance.scheduler.celery_app import app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_tickers_by_market(market: str) -> list[str]:
    """Return enabled tickers filtered by market ('us' or 'japan')."""
    return [cfg["ticker"] for cfg in list_stock_configs(market=market)]


# ---------------------------------------------------------------------------
# Dispatch tasks  (triggered by Beat, fan out per-ticker work)
# ---------------------------------------------------------------------------


@app.task(  # type: ignore[untyped-decorator]
    name="sky_finance.ingestion.tasks.dispatch_ingest_us_stocks",
    queue="ingestion",
    ignore_result=True,
)
def dispatch_ingest_us_stocks() -> None:
    """
    Triggered at 23:00 UTC on weekdays.
    Fans out one ingest_stock task per enabled US ticker.
    """
    tickers = _load_tickers_by_market("us")
    logger.info("Dispatching US stock ingestion for %d tickers", len(tickers))
    group(ingest_stock.si(ticker, "us") for ticker in tickers).apply_async()


@app.task(  # type: ignore[untyped-decorator]
    name="sky_finance.ingestion.tasks.dispatch_ingest_japan_stocks",
    queue="ingestion",
    ignore_result=True,
)
def dispatch_ingest_japan_stocks() -> None:
    """
    Triggered at 07:30 UTC on weekdays.
    Fans out one ingest_stock task per enabled Japan ticker.
    """
    tickers = _load_tickers_by_market("japan")
    logger.info("Dispatching Japan stock ingestion for %d tickers", len(tickers))
    group(ingest_stock.si(ticker, "japan") for ticker in tickers).apply_async()


@app.task(  # type: ignore[untyped-decorator]
    name="sky_finance.ingestion.tasks.dispatch_ingest_news",
    queue="ingestion",
    ignore_result=True,
)
def dispatch_ingest_news() -> None:
    """Triggered hourly. Fans out news ingestion for all enabled tickers."""
    tickers = [cfg["ticker"] for cfg in list_stock_configs()]
    logger.info("Dispatching news ingestion for %d tickers", len(tickers))
    group(ingest_news_for_ticker.si(ticker) for ticker in tickers).apply_async()


# ---------------------------------------------------------------------------
# Per-ticker tasks
# ---------------------------------------------------------------------------


@app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name="sky_finance.ingestion.tasks.ingest_stock",
    queue="ingestion",
    autoretry_for=(RequestException, OSError),
    retry_backoff=60,  # 60 s → 120 s → 240 s (capped at retry_backoff_max)
    retry_backoff_max=300,
    retry_jitter=True,  # add random ±10 % so workers don't all retry in sync
    max_retries=3,
    soft_time_limit=120,
    time_limit=180,
)
def ingest_stock(self: Any, ticker: str, market: str) -> dict[str, Any]:
    """
    Fetch raw stock data for a single ticker and persist to disk.

    Retries automatically on network / HTTP errors (yfinance uses requests
    internally). Retry schedule with jitter: ~60 s, ~120 s, ~240 s.

    Args:
        ticker: stock symbol, e.g. "AAPL" or "7203.T".
        market: "us" or "japan".

    Returns:
        dict with ticker, market, fetched_at, and path of saved file.
    """
    from sky_finance.ingestion.yfinance_fetcher import (
        fetch_japan_stock,
        fetch_us_stock,
        save_raw,
    )

    logger.info("Ingesting %s stock: %s", market, ticker)

    if market == "us":
        payload = fetch_us_stock(ticker)
    elif market == "japan":
        payload = fetch_japan_stock(ticker)
    else:
        raise ValueError(f"Unknown market: {market!r}")

    path = save_raw(payload)

    from sky_finance.storage.db import get_connection
    from sky_finance.storage.repository import insert_raw_stock

    with get_connection() as conn:
        row_id = insert_raw_stock(conn, ticker, market, payload)
        conn.commit()

    logger.info("Stored raw_data id=%d for %s [%s]", row_id, ticker, market)
    return {"ticker": ticker, "market": market, "path": str(path), "db_id": row_id}


@app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name="sky_finance.ingestion.tasks.ingest_news_for_ticker",
    queue="ingestion",
    autoretry_for=(httpx.HTTPError, httpx.NetworkError, OSError),
    retry_backoff=120,
    retry_backoff_max=600,
    retry_jitter=True,
    max_retries=3,
    soft_time_limit=60,
    time_limit=90,
)
def ingest_news_for_ticker(self: Any, ticker: str) -> dict[str, Any]:
    """
    Fetch Google RSS news articles for a single ticker and persist to disk.

    All RSS feeds for the ticker (L1-EN, L2-EN, L1-JA, L2-JA) are fetched
    concurrently via httpx.AsyncClient. Individual feed failures are logged
    and skipped; task-level retry fires on broader HTTP/network errors.

    Returns:
        dict with ticker, path, and article counts.
    """
    from sky_finance.ingestion.news_fetcher import fetch_news, save_raw_news

    logger.info("Ingesting news: %s", ticker)

    cfg = load_stock_config(ticker) or {}
    market = cfg.get("market", "us")
    ingestion_cfg = cfg.get("ingestion", {})
    l1_keywords = ingestion_cfg.get("l1_keywords", [])
    l2_topics = ingestion_cfg.get("l2_topics", [])
    l3_macro = ingestion_cfg.get("l3_macro", [])

    payload = fetch_news(ticker, market, l1_keywords, l2_topics, l3_macro)

    path, total = save_raw_news(payload)

    from sky_finance.storage.db import get_connection
    from sky_finance.storage.repository import insert_raw_news

    with get_connection() as conn:
        inserted = insert_raw_news(conn, ticker, payload["articles"])
        conn.commit()

    logger.info("Stored %d new news rows for %s (file total: %d)", inserted, ticker, total)
    return {
        "ticker": ticker,
        "path": str(path),
        "total_articles": total,
        "db_inserted": inserted,
    }
