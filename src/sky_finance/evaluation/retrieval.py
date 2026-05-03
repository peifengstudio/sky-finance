"""
Plain cosine-similarity retrieval — no sentiment bucketing.

Used as the **baseline** in RAG evaluation.  The current production approach
(sentiment-bucketed retrieval in strategies/engine.py) reserves top-k slots
per sentiment class so that minority-sentiment signals always reach the model.
Plain retrieval lets all chunks compete on similarity alone — which tends to
over-represent the dominant sentiment for a given ticker.
"""

import logging
from typing import Any

import psycopg

logger = logging.getLogger(__name__)

_PLAIN_QUERY = """
    SELECT d.title, LEFT(d.body, 400), d.sentiment,
           1 - (e.embedding <=> %(vec)s::vector) AS sim
    FROM embeddings e
    JOIN documents d ON d.id = e.document_id
    WHERE e.ticker = %(ticker)s
      AND 1 - (e.embedding <=> %(vec)s::vector) >= %(threshold)s
    ORDER BY e.embedding <=> %(vec)s::vector
    LIMIT %(top_k)s
"""


def plain_rag_fetch(
    conn: psycopg.Connection,
    query_template: str,
    ticker: str,
    company_name: str = "",
    threshold: float = 0.55,
    top_k: int = 60,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Retrieve ``top_k`` chunks ranked by cosine similarity with no sentiment
    filtering.  The total budget matches the bucketed approach (3 buckets ×
    default 20 each = 60 chunks) so the comparison is fair.

    Returns:
        (context_text, raw_rows) — same shape as rag_fetch() in engine.py.
    """
    from pgvector.psycopg import register_vector

    from sky_finance.storage.embedder import embed_single

    query = query_template.replace("{ticker}", ticker)
    if company_name:
        query = f"{query} {company_name}"
    vector = embed_single(query)

    register_vector(conn)

    with conn.cursor() as cur:
        cur.execute(
            _PLAIN_QUERY,
            {
                "vec": vector,
                "ticker": ticker,
                "threshold": threshold,
                "top_k": top_k,
            },
        )
        raw = cur.fetchall()

    if not raw:
        logger.debug("plain_rag_fetch: no chunks found for ticker=%s", ticker)
        return f"[No relevant documents found for {ticker}]", []

    structured = [
        {"title": title, "body": body, "sentiment": sentiment, "sim": round(float(sim), 3)}
        for title, body, sentiment, sim in raw
    ]

    chunks = []
    for item in structured:
        sentiment_tag = f" [{item['sentiment']}]" if item["sentiment"] else ""
        chunks.append(f"### {item['title']}{sentiment_tag} (sim={item['sim']:.2f})\n{item['body']}")

    logger.debug("plain_rag_fetch: ticker=%s chunks=%d", ticker, len(structured))
    return "\n\n".join(chunks), structured
