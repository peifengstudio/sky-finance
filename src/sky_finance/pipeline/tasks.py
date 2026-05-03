"""
Pipeline tasks — clean raw records, summarise with local LLM, embed and store.

Task flow (runs every 30 min via Beat):

  dispatch_pipeline
  ├─ fetch unprocessed news_raw rows (SELECT … FOR UPDATE SKIP LOCKED)
  │    └─ group( process_news_article.si(id) × N )
  │          each task:
  │            clean text → LLM summarise → insert document → embed → mark processed
  │
  └─ fetch unprocessed raw_data rows
       └─ group( process_stock_record.si(id) × N )
             each task:
               ohlcv_to_text → insert document → embed → mark processed

Workers for this queue should be started with lower concurrency to avoid
overwhelming the local Ollama instance:
    celery worker --queues=pipeline --concurrency=2
"""

import logging
from typing import Any

from celery import group

from sky_finance.scheduler.celery_app import app

logger = logging.getLogger(__name__)

# Batch sizes per dispatch cycle
_NEWS_BATCH = 50
_STOCK_BATCH = 20


# ---------------------------------------------------------------------------
# Dispatch — called by Beat every 30 min
# ---------------------------------------------------------------------------


@app.task(  # type: ignore[untyped-decorator]
    name="sky_finance.pipeline.tasks.dispatch_pipeline",
    queue="pipeline",
    ignore_result=True,
)
def dispatch_pipeline() -> None:
    """
    Claim unprocessed records from both news_raw and raw_data,
    then fan out individual processing tasks.
    """
    from sky_finance.storage.db import get_connection
    from sky_finance.storage.repository import (
        fetch_unprocessed_news,
        fetch_unprocessed_raw_stocks,
    )

    with get_connection() as conn:
        news_rows = fetch_unprocessed_news(conn, limit=_NEWS_BATCH)
        stock_rows = fetch_unprocessed_raw_stocks(conn, limit=_STOCK_BATCH)
        # Commit releases the FOR UPDATE locks — IDs are now "in flight"
        conn.commit()

    news_ids = [r["id"] for r in news_rows]
    stock_ids = [r["id"] for r in stock_rows]

    logger.info(
        "Pipeline dispatch — %d news articles, %d stock records",
        len(news_ids),
        len(stock_ids),
    )

    tasks = []
    if news_ids:
        tasks += [process_news_article.si(article_id) for article_id in news_ids]
    if stock_ids:
        tasks += [process_stock_record.si(record_id) for record_id in stock_ids]

    if tasks:
        group(tasks).apply_async()


# ---------------------------------------------------------------------------
# Per-article pipeline
# ---------------------------------------------------------------------------


@app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name="sky_finance.pipeline.tasks.process_news_article",
    queue="pipeline",
    max_retries=2,
    soft_time_limit=180,  # Ollama can be slow
    time_limit=240,
)
def process_news_article(self: Any, article_id: int) -> dict[str, Any]:
    """
    Full pipeline for a single news article:
      1. Fetch from DB
      2. Clean text
      3. LLM summarise (qwen2.5:3b-instruct)
      4. Insert into documents table
      5. Embed body text (nomic-embed-text)
      6. Insert embedding into pgvector
      7. Mark news_raw row as processed

    Args:
        article_id: news_raw.id

    Returns:
        dict with document_id and embedding_id.
    """
    from sky_finance.pipeline.cleaner import clean_news_article
    from sky_finance.pipeline.llm_summariser import summarise_article
    from sky_finance.storage.db import get_connection
    from sky_finance.storage.embedder import embed_single
    from sky_finance.storage.repository import (
        insert_document,
        insert_embedding,
        mark_news_processed,
    )

    logger.info("Processing news article id=%d", article_id)
    try:
        # --- 1. Fetch raw record ---
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ticker, title, content, source_name FROM news_raw WHERE id = %s",
                    (article_id,),
                )
                row = cur.fetchone()

        if row is None:
            logger.warning("news_raw id=%d not found — skipping", article_id)
            return {"skipped": True}

        ticker, title, content, source_name = row

        # --- 2. Clean ---
        cleaned = clean_news_article({"title": title, "content": content})

        # --- 3. LLM summarise ---
        llm_result = summarise_article(
            ticker=ticker,
            title=cleaned["title"],
            content=cleaned["content"],
        )

        # Build the document body (summary + key facts)
        body_lines = [llm_result["summary"]]
        if llm_result["key_facts"]:
            body_lines += ["", "Key facts:"] + [f"• {f}" for f in llm_result["key_facts"]]
        body = "\n".join(body_lines)

        # --- 4. Insert document ---
        with get_connection() as conn:
            doc_id = insert_document(
                conn,
                source_type="news",
                source_id=article_id,
                ticker=ticker,
                title=cleaned["title"],
                body=body,
                sentiment=llm_result["sentiment"],
                key_facts=llm_result["key_facts"],
            )
            conn.commit()

        # --- 5 & 6. Embed and store ---
        embed_text = f"{ticker}\n{cleaned['title']}\n{body}"
        vector = embed_single(embed_text)

        with get_connection() as conn:
            emb_id = insert_embedding(
                conn,
                document_id=doc_id,
                ticker=ticker,
                source_type="news",
                vector=vector,
            )
            conn.commit()

        # --- 7. Mark processed ---
        with get_connection() as conn:
            mark_news_processed(conn, [article_id])
            conn.commit()

        logger.info(
            "article id=%d → doc id=%d  emb id=%d  sentiment=%s",
            article_id,
            doc_id,
            emb_id,
            llm_result["sentiment"],
        )
        return {"document_id": doc_id, "embedding_id": emb_id}

    except Exception as exc:
        logger.warning("pipeline failed for news id=%d: %s — retrying", article_id, exc)
        raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))


# ---------------------------------------------------------------------------
# Per-stock-record pipeline
# ---------------------------------------------------------------------------


@app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name="sky_finance.pipeline.tasks.process_stock_record",
    queue="pipeline",
    max_retries=2,
    soft_time_limit=60,
    time_limit=90,
)
def process_stock_record(self: Any, record_id: int) -> dict[str, Any]:
    """
    Full pipeline for a single raw_data record (OHLCV + fundamentals):
      1. Fetch payload from DB
      2. Convert to readable text (no LLM needed for price data)
      3. Insert into documents table
      4. Embed the text chunk
      5. Insert embedding into pgvector
      6. Mark raw_data row as processed

    Args:
        record_id: raw_data.id

    Returns:
        dict with document_id and embedding_id.
    """
    from sky_finance.pipeline.cleaner import ohlcv_to_text
    from sky_finance.storage.db import get_connection
    from sky_finance.storage.embedder import embed_single
    from sky_finance.storage.repository import (
        insert_document,
        insert_embedding,
        mark_raw_stocks_processed,
    )

    logger.info("Processing stock record id=%d", record_id)
    try:
        # --- 1. Fetch raw payload ---
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ticker, source, payload FROM raw_data WHERE id = %s",
                    (record_id,),
                )
                row = cur.fetchone()

        if row is None:
            logger.warning("raw_data id=%d not found — skipping", record_id)
            return {"skipped": True}

        ticker, source, payload = row

        # --- 2. Text representation ---
        body = ohlcv_to_text(payload)

        # Latest date from OHLCV for the document title
        ohlcv = payload.get("ohlcv", [])
        date = ohlcv[-1]["date"] if ohlcv else "N/A"
        title = f"{ticker} OHLCV {date}"

        # --- 3. Insert document ---
        with get_connection() as conn:
            doc_id = insert_document(
                conn,
                source_type="ohlcv_summary",
                source_id=record_id,
                ticker=ticker,
                title=title,
                body=body,
            )
            conn.commit()

        # --- 4 & 5. Embed and store ---
        vector = embed_single(body)

        with get_connection() as conn:
            emb_id = insert_embedding(
                conn,
                document_id=doc_id,
                ticker=ticker,
                source_type="ohlcv_summary",
                vector=vector,
            )
            conn.commit()

        # --- 6. Mark processed ---
        with get_connection() as conn:
            mark_raw_stocks_processed(conn, [record_id])
            conn.commit()

        logger.info("record id=%d → doc id=%d  emb id=%d", record_id, doc_id, emb_id)
        return {"document_id": doc_id, "embedding_id": emb_id}

    except Exception as exc:
        logger.warning("pipeline failed for stock id=%d: %s — retrying", record_id, exc)
        raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))
