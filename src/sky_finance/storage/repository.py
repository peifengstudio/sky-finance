"""
Repository — all database read/write operations for sky-finance.

All functions accept an open psycopg3 connection and do NOT commit.
The caller (task) owns the transaction.
"""

import logging
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# raw_data  (stock prices + fundamentals)
# ---------------------------------------------------------------------------


def insert_raw_stock(
    conn: psycopg.Connection,
    ticker: str,
    market: str,
    payload: dict[str, Any],
) -> int:
    """
    Insert one yfinance payload into raw_data.

    Args:
        conn:    open psycopg3 connection (caller commits).
        ticker:  stock symbol.
        market:  "us" or "japan".
        payload: dict returned by fetch_us_stock / fetch_japan_stock.

    Returns:
        The new row's id.
    """
    fetched_at = datetime.fromisoformat(payload["fetched_at"])
    source = f"yfinance_{market}"

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO raw_data (ticker, source, fetched_at, payload)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (ticker, source, fetched_at, Jsonb(payload)),
        )
        row_id: int = cur.fetchone()[0]  # type: ignore[index]

    logger.debug("Inserted raw_data id=%d  ticker=%s  source=%s", row_id, ticker, source)
    return row_id


# ---------------------------------------------------------------------------
# news_raw
# ---------------------------------------------------------------------------


def insert_raw_news(
    conn: psycopg.Connection,
    ticker: str,
    articles: list[dict[str, Any]],
) -> int:
    """
    Bulk-insert news articles into news_raw.

    Articles whose URL already exists are silently skipped
    (ON CONFLICT DO NOTHING on the url UNIQUE constraint).

    Args:
        conn:     open psycopg3 connection (caller commits).
        ticker:   stock symbol the articles belong to.
        articles: list of dicts from news_fetcher._parse_entry().
                  Expected keys: title, url, published_iso (optional),
                  summary, source.

    Returns:
        Number of rows actually inserted (duplicates not counted).
    """
    if not articles:
        return 0

    inserted = 0
    with conn.cursor() as cur:
        for article in articles:
            published_at: datetime | None = None
            if iso := article.get("published_iso"):
                try:
                    published_at = datetime.fromisoformat(iso)
                except ValueError:
                    pass

            cur.execute(
                """
                INSERT INTO news_raw
                    (ticker, title, url, published_at, content, source_name)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO NOTHING
                """,
                (
                    ticker,
                    article.get("title", ""),
                    article.get("url", ""),
                    published_at,
                    article.get("summary", ""),
                    article.get("source", ""),
                ),
            )
            inserted += cur.rowcount

    logger.debug("Inserted %d/%d news articles for %s", inserted, len(articles), ticker)
    return inserted


# ---------------------------------------------------------------------------
# Pipeline — fetch unprocessed records
# Uses SELECT … FOR UPDATE SKIP LOCKED so concurrent workers don't collide.
# ---------------------------------------------------------------------------


def fetch_unprocessed_news(
    conn: psycopg.Connection,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Claim a batch of unprocessed news_raw rows.

    Rows are locked for the duration of the caller's transaction; other workers
    skip them automatically (SKIP LOCKED). Mark them processed when done via
    mark_news_processed().

    Why FOR UPDATE SKIP LOCKED instead of a Redis distributed lock?
    A Redis lock requires a separate round-trip to acquire, a TTL to guard
    against crashed workers, and manual release logic.  FOR UPDATE SKIP LOCKED
    gives the same mutual exclusion guarantee using the same Postgres connection
    that already holds the data — no extra dependency, no TTL tuning, and the
    lock is released automatically when the transaction commits or the
    connection drops.  The tradeoff is that it only works when all workers
    share the same database, which is always true here.

    Returns:
        list of dicts with keys: id, ticker, title, content, source_name, published_at.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, ticker, title, content, source_name, published_at
            FROM   news_raw
            WHERE  processed_at IS NULL
            ORDER  BY fetched_at ASC
            LIMIT  %s
            FOR UPDATE SKIP LOCKED
            """,
            (limit,),
        )
        rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "ticker": r[1],
            "title": r[2],
            "content": r[3],
            "source_name": r[4],
            "published_at": r[5],
        }
        for r in rows
    ]


def fetch_unprocessed_raw_stocks(
    conn: psycopg.Connection,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Claim a batch of unprocessed raw_data rows.

    Uses FOR UPDATE SKIP LOCKED for the same reason as fetch_unprocessed_news —
    see that function's docstring for the full rationale.

    Returns:
        list of dicts with keys: id, ticker, source, payload.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, ticker, source, payload
            FROM   raw_data
            WHERE  processed_at IS NULL
            ORDER  BY fetched_at ASC
            LIMIT  %s
            FOR UPDATE SKIP LOCKED
            """,
            (limit,),
        )
        rows = cur.fetchall()

    return [{"id": r[0], "ticker": r[1], "source": r[2], "payload": r[3]} for r in rows]


def mark_news_processed(conn: psycopg.Connection, article_ids: list[int]) -> None:
    """Stamp processed_at = NOW() on the given news_raw rows."""
    if not article_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE news_raw SET processed_at = NOW() WHERE id = ANY(%s)",
            (article_ids,),
        )


def mark_raw_stocks_processed(conn: psycopg.Connection, record_ids: list[int]) -> None:
    """Stamp processed_at = NOW() on the given raw_data rows."""
    if not record_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE raw_data SET processed_at = NOW() WHERE id = ANY(%s)",
            (record_ids,),
        )


# ---------------------------------------------------------------------------
# Pipeline — write cleaned documents
# ---------------------------------------------------------------------------


def insert_document(
    conn: psycopg.Connection,
    *,
    source_type: str,
    source_id: int,
    ticker: str,
    title: str | None,
    body: str,
    sentiment: str | None = None,
    key_facts: list[str] | None = None,
) -> int:
    """
    Insert a cleaned / summarised document.

    Args:
        source_type: 'news' | 'ohlcv_summary'
        source_id:   FK to news_raw.id or raw_data.id
        ticker:      stock symbol
        title:       document title (optional)
        body:        full text of the cleaned document
        sentiment:   'positive' | 'neutral' | 'negative' (optional)
        key_facts:   list of strings (optional, stored as JSONB)

    Returns:
        New document id.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
                (source_type, source_id, ticker, title, body, sentiment, key_facts)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                source_type,
                source_id,
                ticker,
                title,
                body,
                sentiment,
                Jsonb(key_facts or []),
            ),
        )
        doc_id: int = cur.fetchone()[0]  # type: ignore[index]

    logger.debug("Inserted document id=%d  ticker=%s  source_type=%s", doc_id, ticker, source_type)
    return doc_id


# ---------------------------------------------------------------------------
# Pipeline — write embeddings
# ---------------------------------------------------------------------------


def insert_embedding(
    conn: psycopg.Connection,
    *,
    document_id: int,
    ticker: str,
    source_type: str,
    vector: list[float],
) -> int:
    """
    Insert an embedding vector into pgvector.

    Args:
        document_id: FK to documents.id
        ticker:      stock symbol (for filtered ANN search)
        source_type: mirrors documents.source_type
        vector:      dense float vector (must match embeddings column dimension)

    Returns:
        New embedding id.
    """
    from pgvector.psycopg import register_vector

    register_vector(conn)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO embeddings (document_id, ticker, source_type, embedding)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (document_id, ticker, source_type, vector),
        )
        emb_id: int = cur.fetchone()[0]  # type: ignore[index]

    logger.debug("Inserted embedding id=%d  doc_id=%d", emb_id, document_id)
    return emb_id
