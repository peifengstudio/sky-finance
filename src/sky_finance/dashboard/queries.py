"""
Dashboard DB queries and config helpers.

All DB functions accept an open psycopg3 connection and do NOT commit.
Config helpers read TOML files from config/stocks/.
"""

import logging
from typing import Any

import psycopg

from sky_finance.config import list_stock_configs, load_stock_config

logger = logging.getLogger(__name__)

_CURRENCY_SYMBOL = {"USD": "$", "JPY": "¥", "EUR": "€", "GBP": "£"}


# ---------------------------------------------------------------------------
# Config helpers (TOML — no DB)
# ---------------------------------------------------------------------------


def list_stocks_from_config(market_filter: str | None = None) -> list[dict[str, Any]]:
    """
    Load all enabled stock configs and return a flat list of dicts.
    Adds computed helper fields (currency_symbol, has_position).
    Local overrides in config/stocks/local/ are applied automatically.
    """
    stocks = []
    for cfg in list_stock_configs(market=market_filter):
        price = cfg.get("price", {})
        position = cfg.get("position", {})
        analysis = cfg.get("analysis", {})
        currency = cfg.get("currency", "USD")
        stocks.append(
            {
                "ticker": cfg["ticker"],
                "name": cfg.get("name", ""),
                "market": cfg.get("market", "us"),
                "currency": currency,
                "currency_symbol": _CURRENCY_SYMBOL.get(currency, ""),
                "enabled": cfg.get("enabled", True),
                "buy_price": price.get("buy_price", 0.0),
                "stop_loss": price.get("stop_loss", 0.0),
                "targets": price.get("targets", [0.0, 0.0, 0.0]),
                "alerts": price.get("alerts", [0.0, 0.0, 0.0]),
                "shares": position.get("shares", 0),
                "max_weight": position.get("max_weight", 0.0),
                "notes": position.get("notes", ""),
                "signal_chain": analysis.get("signal_chain", ""),
                "l1_keywords": cfg.get("ingestion", {}).get("l1_keywords", []),
                "l2_topics": cfg.get("ingestion", {}).get("l2_topics", []),
                "l3_macro": cfg.get("ingestion", {}).get("l3_macro", []),
                "strategies": cfg.get("strategies", {}).get("enabled", []),
                "has_position": position.get("shares", 0) > 0,
            }
        )
    return stocks


def get_stock_config(ticker: str) -> dict[str, Any] | None:
    """Load a single stock's config. Returns None if neither shared nor local file exists."""
    cfg = load_stock_config(ticker)
    if cfg is None:
        return None
    currency = cfg.get("currency", "USD")
    price = cfg.get("price", {})
    position = cfg.get("position", {})
    analysis = cfg.get("analysis", {})
    return {
        "ticker": cfg["ticker"],
        "name": cfg.get("name", ""),
        "market": cfg.get("market", "us"),
        "currency": currency,
        "currency_symbol": _CURRENCY_SYMBOL.get(currency, ""),
        "buy_price": price.get("buy_price", 0.0),
        "stop_loss": price.get("stop_loss", 0.0),
        "targets": price.get("targets", [0.0, 0.0, 0.0]),
        "alerts": price.get("alerts", [0.0, 0.0, 0.0]),
        "shares": position.get("shares", 0),
        "max_weight": position.get("max_weight", 0.0),
        "notes": position.get("notes", ""),
        "signal_chain": analysis.get("signal_chain", ""),
        "l1_keywords": cfg.get("ingestion", {}).get("l1_keywords", []),
        "l2_topics": cfg.get("ingestion", {}).get("l2_topics", []),
        "l3_macro": cfg.get("ingestion", {}).get("l3_macro", []),
        "strategies": cfg.get("strategies", {}).get("enabled", []),
    }


def list_ticker_names() -> list[tuple[str, str]]:
    """Return [(ticker, name), ...] for all enabled stocks — used in search dropdown."""
    return [(cfg["ticker"], cfg.get("name", cfg["ticker"])) for cfg in list_stock_configs()]


# ---------------------------------------------------------------------------
# Pipeline stats
# ---------------------------------------------------------------------------


def get_pipeline_stats(conn: psycopg.Connection) -> dict[str, Any]:
    """Counts of processed / pending rows for both raw tables."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*)                                    AS total,
                COUNT(*) FILTER (WHERE processed_at IS NULL)    AS pending,
                COUNT(*) FILTER (WHERE processed_at IS NOT NULL) AS processed,
                MAX(processed_at)                           AS last_processed
            FROM news_raw
        """)
        news = cur.fetchone()

        cur.execute("""
            SELECT
                COUNT(*)                                    AS total,
                COUNT(*) FILTER (WHERE processed_at IS NULL)    AS pending,
                COUNT(*) FILTER (WHERE processed_at IS NOT NULL) AS processed,
                MAX(processed_at)                           AS last_processed
            FROM raw_data
        """)
        prices = cur.fetchone()

        cur.execute("SELECT COUNT(*) FROM documents")
        doc_row = cur.fetchone()
        assert doc_row is not None
        doc_count = doc_row[0]

        cur.execute("SELECT COUNT(*) FROM embeddings")
        emb_row = cur.fetchone()
        assert emb_row is not None
        emb_count = emb_row[0]

    assert news is not None
    assert prices is not None
    return {
        "news": {
            "total": news[0],
            "pending": news[1],
            "processed": news[2],
            "last_processed": news[3],
        },
        "prices": {
            "total": prices[0],
            "pending": prices[1],
            "processed": prices[2],
            "last_processed": prices[3],
        },
        "documents": doc_count,
        "embeddings": emb_count,
    }


# ---------------------------------------------------------------------------
# Recent documents (pipeline output)
# ---------------------------------------------------------------------------


def get_recent_documents(
    conn: psycopg.Connection,
    ticker: str | None = None,
    source_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    filters: list[str] = []
    params: list[Any] = []
    if ticker:
        filters.append("ticker = %s")
        params.append(ticker)
    if source_type:
        filters.append("source_type = %s")
        params.append(source_type)

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params += [limit, offset]

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, ticker, source_type, title, sentiment, created_at
            FROM documents
            {where}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            params,
        )
        rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "ticker": r[1],
            "source_type": r[2],
            "title": r[3],
            "sentiment": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Per-ticker queries
# ---------------------------------------------------------------------------


def get_latest_raw_data(conn: psycopg.Connection, ticker: str) -> dict[str, Any] | None:
    """Most recent yfinance payload for a ticker."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT payload, fetched_at
            FROM raw_data
            WHERE ticker = %s
            ORDER BY fetched_at DESC
            LIMIT 1
            """,
            (ticker,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"payload": row[0], "fetched_at": row[1]}


def get_stock_news(
    conn: psycopg.Connection,
    ticker: str,
    limit: int = 10,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Processed news articles for a ticker (from documents table)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT title, body, sentiment, created_at
            FROM documents
            WHERE ticker = %s AND source_type = 'news'
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            (ticker, limit, offset),
        )
        rows = cur.fetchall()
    return [
        {
            "title": r[0],
            "body": r[1][:300] + "…" if r[1] and len(r[1]) > 300 else r[1],
            "sentiment": r[2],
            "created_at": r[3],
        }
        for r in rows
    ]


def get_strategy_results(
    conn: psycopg.Connection,
    ticker: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Recent strategy_results rows, optionally filtered to a specific ticker."""
    with conn.cursor() as cur:
        if ticker:
            cur.execute(
                """
                SELECT strategy, tickers, report, model, ran_at
                FROM strategy_results
                WHERE %s = ANY(tickers)
                ORDER BY ran_at DESC
                LIMIT %s
                """,
                (ticker, limit),
            )
        else:
            cur.execute(
                """
                SELECT strategy, tickers, report, model, ran_at
                FROM strategy_results
                ORDER BY ran_at DESC
                LIMIT %s
                """,
                (limit,),
            )
        rows = cur.fetchall()

    return [
        {
            "strategy": r[0],
            "tickers": r[1],
            "report": r[2],
            "model": r[3],
            "ran_at": r[4],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Strategies (thin wrappers over strategies.repository)
# ---------------------------------------------------------------------------


def list_strategies(conn: psycopg.Connection, enabled_only: bool = False) -> list[dict[str, Any]]:
    from sky_finance.strategies.repository import list_strategies as _list

    return _list(conn, enabled_only=enabled_only)


def get_strategy(conn: psycopg.Connection, strategy_id: int) -> dict[str, Any] | None:
    from sky_finance.strategies.repository import get_strategy as _get

    return _get(conn, strategy_id)


def create_strategy(conn: psycopg.Connection, data: dict[str, Any]) -> int:
    from sky_finance.strategies.repository import create_strategy as _create

    return _create(conn, data)


def update_strategy(conn: psycopg.Connection, strategy_id: int, data: dict[str, Any]) -> None:
    from sky_finance.strategies.repository import update_strategy as _update

    _update(conn, strategy_id, data)


def delete_strategy(conn: psycopg.Connection, strategy_id: int) -> None:
    from sky_finance.strategies.repository import delete_strategy as _delete

    _delete(conn, strategy_id)


def get_strategy_result(conn: psycopg.Connection, result_id: int) -> dict[str, Any] | None:
    from sky_finance.strategies.repository import get_strategy_result as _get

    return _get(conn, result_id)


def count_strategy_results(
    conn: psycopg.Connection,
    strategy_id: int | None = None,
) -> int:
    from sky_finance.strategies.repository import count_strategy_results as _count

    return _count(conn, strategy_id=strategy_id)


def list_strategy_results(
    conn: psycopg.Connection,
    strategy_id: int | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    from sky_finance.strategies.repository import list_strategy_results as _list

    return _list(conn, strategy_id=strategy_id, limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Semantic search (pgvector)
# ---------------------------------------------------------------------------


def semantic_search(
    conn: psycopg.Connection,
    vector: list[float],
    ticker: str | None = None,
    source_type: str | None = None,
    sentiment: str | None = None,
    limit: int = 10,
    threshold: float = 0.70,
) -> list[dict[str, Any]]:
    """
    Cosine similarity search over the embeddings table.

    Args:
        conn:        open psycopg3 connection.
        vector:      query embedding (must match column dimension).
        ticker:      optional ticker filter.
        source_type: 'news' | 'ohlcv_summary' | None (all).
        limit:       maximum number of results.
        threshold:   minimum similarity score (0–1).

    Returns:
        list of dicts with ticker, source_type, title, body_preview,
        sentiment, created_at, similarity.
    """
    from pgvector.psycopg import register_vector

    register_vector(conn)

    extra_filters = ""
    extra_params: dict[str, Any] = {}
    if ticker:
        extra_filters += " AND e.ticker = %(ticker)s"
        extra_params["ticker"] = ticker
    if source_type:
        extra_filters += " AND e.source_type = %(source_type)s"
        extra_params["source_type"] = source_type
    if sentiment:
        extra_filters += " AND d.sentiment = %(sentiment)s"
        extra_params["sentiment"] = sentiment

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                e.ticker,
                e.source_type,
                d.title,
                LEFT(d.body, 250)            AS body_preview,
                d.sentiment,
                d.created_at,
                1 - (e.embedding <=> %(vec)s::vector) AS similarity
            FROM embeddings e
            JOIN documents d ON d.id = e.document_id
            WHERE 1 - (e.embedding <=> %(vec)s::vector) >= %(threshold)s
            {extra_filters}
            ORDER BY e.embedding <=> %(vec)s::vector
            LIMIT %(limit)s
            """,
            {
                "vec": vector,
                "threshold": threshold,
                "limit": limit,
                **extra_params,
            },
        )
        rows = cur.fetchall()

    return [
        {
            "ticker": r[0],
            "source_type": r[1],
            "title": r[2],
            "body_preview": r[3],
            "sentiment": r[4],
            "created_at": r[5],
            "similarity": round(float(r[6]), 3),
        }
        for r in rows
    ]
