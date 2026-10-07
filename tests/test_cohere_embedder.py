from __future__ import annotations

import json

import httpx
import pytest

from app.embeddings import get_embedder_backend
from app.embeddings.cohere_embedder import CohereEmbedder
from app.embeddings.embedder import EmbeddingError, OllamaEmbedder


def _cohere_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["input_type"] in ("search_document", "search_query")
    vectors = [[float(len(t)), 0.0] for t in body["texts"]]
    return httpx.Response(200, json={"embeddings": {"float": vectors}, "texts": body["texts"]})


async def test_cohere_streams_vectors_matching_input_order():
    transport = httpx.MockTransport(_cohere_handler)
    embedder = CohereEmbedder(api_key="test-key", base_url="http://fake-cohere", transport=transport)

    vectors = await embedder.embed(["hi", "a longer piece of text"])

    assert vectors == [[2.0, 0.0], [22.0, 0.0]]


async def test_cohere_passes_through_input_type():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["input_type"] = json.loads(request.content)["input_type"]
        return httpx.Response(200, json={"embeddings": {"float": [[1.0]]}, "texts": ["q"]})

    transport = httpx.MockTransport(handler)
    embedder = CohereEmbedder(api_key="test-key", base_url="http://fake-cohere", transport=transport)

    await embedder.embed(["q"], input_type="search_query")
    assert seen["input_type"] == "search_query"


async def test_cohere_returns_empty_list_for_no_texts():
    embedder = CohereEmbedder(api_key="test-key")
    assert await embedder.embed([]) == []


async def test_cohere_without_api_key_raises_immediately(monkeypatch):
    # CohereEmbedder(api_key=None) alone isn't enough to prove this: the
    # constructor falls back to settings.cohere_api_key when the explicit
    # arg is falsy, so if the real .env has a key set, that fallback would
    # silently supply one and this test would pass for the wrong reason.
    # Force the setting itself blank so the test is isolated from .env.
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "cohere_api_key", None, raising=False)
    embedder = CohereEmbedder(api_key=None)
    with pytest.raises(EmbeddingError, match="COHERE_API_KEY"):
        await embedder.embed(["hello"])


async def test_cohere_raises_clear_error_when_unreachable():
    # No mock transport, nothing listening: a real connection failure,
    # proving the error path works without needing a live Cohere account.
    embedder = CohereEmbedder(api_key="test-key", base_url="http://127.0.0.1:1", timeout=2.0)
    with pytest.raises(EmbeddingError, match="Could not reach Cohere"):
        await embedder.embed(["hello"])


async def test_cohere_raises_on_bad_status_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    transport = httpx.MockTransport(handler)
    embedder = CohereEmbedder(api_key="bad-key", base_url="http://fake-cohere", transport=transport)
    with pytest.raises(EmbeddingError, match="401"):
        await embedder.embed(["hello"])


async def test_cohere_raises_on_malformed_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"something": "unexpected"})

    transport = httpx.MockTransport(handler)
    embedder = CohereEmbedder(api_key="test-key", base_url="http://fake-cohere", transport=transport)
    with pytest.raises(EmbeddingError, match="Unexpected response"):
        await embedder.embed(["hello"])


def test_factory_picks_ollama_for_default():
    assert isinstance(get_embedder_backend("ollama"), OllamaEmbedder)


def test_factory_picks_cohere():
    assert isinstance(get_embedder_backend("cohere"), CohereEmbedder)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown EMBEDDING_BACKEND"):
        get_embedder_backend("something-else")
