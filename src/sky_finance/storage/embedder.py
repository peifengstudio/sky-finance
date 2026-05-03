"""
Embedding generator — convert text to dense vectors.

Default backend: nomic-embed-text via Ollama (768 dims, local, free).
Alternative:     OpenAI text-embedding-3-small (1536 dims) — set
                 EMBEDDING_BACKEND=openai in .env.

The backend is selected at import time from settings / env vars so all tasks
in a worker process share the same config without re-reading disk on each call.
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_embedding_config() -> dict[str, Any]:
    import tomllib

    settings_path = Path(__file__).parents[3] / "config" / "settings.toml"
    with settings_path.open("rb") as f:
        result: dict[str, Any] = tomllib.load(f).get("embeddings", {})
        return result


_CFG = _load_embedding_config()
BACKEND = os.environ.get("EMBEDDING_BACKEND", _CFG.get("backend", "ollama"))
OLLAMA_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", _CFG.get("model", "nomic-embed-text"))
OLLAMA_HOST = os.environ.get("OLLAMA_BASE_URL", _CFG.get("ollama_host", "http://localhost:11434"))
OPENAI_MODEL = _CFG.get("openai_model", "text-embedding-3-small")
EMBEDDING_DIM = int(_CFG.get("dimensions", 768))
BATCH_SIZE = int(_CFG.get("batch_size", 32))


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via local Ollama (nomic-embed-text)."""
    import ollama

    client = ollama.Client(host=OLLAMA_HOST)
    vectors: list[list[float]] = []
    for text in texts:
        response = client.embeddings(model=OLLAMA_MODEL, prompt=text)
        vectors.append(list(response.embedding))
    return vectors


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via OpenAI API (text-embedding-3-small)."""
    import openai

    client = openai.OpenAI()
    response = client.embeddings.create(model=OPENAI_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embedding vectors for a list of text strings.

    Processes in batches of BATCH_SIZE to avoid memory spikes.

    Args:
        texts: list of strings to embed.

    Returns:
        list of float vectors, same length as input.
    """
    if not texts:
        return []

    logger.info(
        "Embedding %d texts via %s (%s)",
        len(texts),
        BACKEND,
        OLLAMA_MODEL if BACKEND == "ollama" else OPENAI_MODEL,
    )

    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        if BACKEND == "openai":
            vectors.extend(_embed_openai(batch))
        else:
            vectors.extend(_embed_ollama(batch))

    return vectors


def embed_single(text: str) -> list[float]:
    """Convenience wrapper for a single text string."""
    return embed_texts([text])[0]
