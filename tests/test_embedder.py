"""
Unit tests for sky_finance.storage.embedder

Covers backend selection and the public embed_texts / embed_single API
without hitting a live Ollama or OpenAI endpoint.
"""

import sys
from unittest.mock import MagicMock, patch

import sky_finance.storage.embedder as embedder_mod
from sky_finance.storage.embedder import _embed_openai

# ---------------------------------------------------------------------------
# Edge cases that need no I/O
# ---------------------------------------------------------------------------


def test_embed_texts_empty_list_returns_empty():
    result = embedder_mod.embed_texts([])
    assert result == []


# ---------------------------------------------------------------------------
# Backend routing (Ollama default)
# ---------------------------------------------------------------------------


def test_embed_texts_routes_to_ollama_backend():
    fake_vector = [0.1] * 768
    with (
        patch.object(embedder_mod, "BACKEND", "ollama"),
        patch.object(embedder_mod, "_embed_ollama", return_value=[fake_vector]) as mock_ollama,
    ):
        result = embedder_mod.embed_texts(["hello world"])

    mock_ollama.assert_called_once_with(["hello world"])
    assert result == [fake_vector]


def test_embed_texts_routes_to_openai_backend():
    fake_vector = [0.2] * 1536
    with (
        patch.object(embedder_mod, "BACKEND", "openai"),
        patch.object(embedder_mod, "_embed_openai", return_value=[fake_vector]) as mock_openai,
    ):
        result = embedder_mod.embed_texts(["hello world"])

    mock_openai.assert_called_once_with(["hello world"])
    assert result == [fake_vector]


def test_embed_single_unwraps_first_element():
    fake_vector = [0.5] * 768
    with (
        patch.object(embedder_mod, "BACKEND", "ollama"),
        patch.object(embedder_mod, "_embed_ollama", return_value=[fake_vector]),
    ):
        result = embedder_mod.embed_single("test text")

    assert result == fake_vector


# ---------------------------------------------------------------------------
# _embed_openai — direct backend test (lines 60-66)
# ---------------------------------------------------------------------------


def test_embed_openai_calls_client_and_returns_vectors():
    fake_vectors = [[0.1] * 1536, [0.2] * 1536]
    mock_items = [MagicMock(embedding=v) for v in fake_vectors]
    mock_response = MagicMock()
    mock_response.data = mock_items

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.embeddings.create.return_value = mock_response

    with patch.dict(sys.modules, {"openai": mock_openai}):
        result = _embed_openai(["text one", "text two"])

    assert result == fake_vectors
    mock_openai.OpenAI.return_value.embeddings.create.assert_called_once()
