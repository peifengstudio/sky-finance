"""
Storage tasks — generate embeddings and upsert into pgvector.
"""

import logging
from typing import Any

from sky_finance.scheduler.celery_app import app

logger = logging.getLogger(__name__)


@app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name="sky_finance.storage.tasks.embed_documents",
    queue="storage",
    max_retries=3,
    default_retry_delay=30,
    soft_time_limit=180,
    time_limit=240,
)
def embed_documents(self: Any, document_ids: list[int]) -> dict[str, Any]:
    """
    Generate embeddings for a batch of documents and upsert into pgvector.
    Calls OpenAI text-embedding-3-small (or local nomic-embed-text via Ollama).

    Args:
        document_ids: list of documents.id values to embed.

    Returns:
        dict with counts of embedded / skipped documents.
    """
    logger.info("Embedding %d documents", len(document_ids))
    try:
        # TODO: implement sky_finance.storage.embedder and call it here
        raise NotImplementedError("Embedder not yet implemented")
    except Exception as exc:
        raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))
