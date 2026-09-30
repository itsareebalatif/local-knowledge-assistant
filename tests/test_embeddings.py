from __future__ import annotations

import json

import httpx
import pytest

from app.embeddings.embedder import EmbeddingError, OllamaEmbedder
from app.embeddings.vector_store import ChromaVectorStore


def _fake_ollama_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    texts = body["input"]
    # Deterministic fake vector per text so assertions are exact.
    vectors = [[float(len(t)), 0.0, 1.0] for t in texts]
    return httpx.Response(200, json={"embeddings": vectors})


async def test_embedder_returns_vectors_in_order():
    transport = httpx.MockTransport(_fake_ollama_handler)
    embedder = OllamaEmbedder(base_url="http://fake-ollama", model="nomic-embed-text", transport=transport)

    vectors = await embedder.embed(["hi", "a longer piece of text"])

    assert vectors == [[2.0, 0.0, 1.0], [22.0, 0.0, 1.0]]


async def test_embedder_returns_empty_list_for_no_texts():
    embedder = OllamaEmbedder(base_url="http://fake-ollama")
    assert await embedder.embed([]) == []


async def test_embedder_raises_clear_error_when_ollama_unreachable():
    # No mock transport, no server listening on this port: a real connection
    # failure, proving the error path works without needing a live Ollama.
    embedder = OllamaEmbedder(base_url="http://127.0.0.1:1", timeout=2.0)
    with pytest.raises(EmbeddingError, match="Is it running"):
        await embedder.embed(["hello"])


async def test_embedder_raises_on_bad_status_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    transport = httpx.MockTransport(handler)
    embedder = OllamaEmbedder(base_url="http://fake-ollama", transport=transport)
    with pytest.raises(EmbeddingError, match="ollama pull"):
        await embedder.embed(["hello"])


def test_chroma_vector_store_add_and_query_round_trip(tmp_path):
    store = ChromaVectorStore(path=str(tmp_path / "chroma"), collection_name="test")
    store.add(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=["doc a", "doc b"],
        metadatas=[{"doc_id": 1}, {"doc_id": 2}],
    )

    hits = store.query([1.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0]["id"] == "a"
    assert hits[0]["document"] == "doc a"
    assert hits[0]["metadata"]["doc_id"] == 1


def test_chroma_vector_store_persists_across_instances(tmp_path):
    path = str(tmp_path / "chroma")
    ChromaVectorStore(path=path, collection_name="test").add(
        ids=["x"], embeddings=[[1.0, 2.0]], documents=["persisted"], metadatas=[{"doc_id": 1}]
    )
    reopened = ChromaVectorStore(path=path, collection_name="test")
    hits = reopened.query([1.0, 2.0], top_k=1)
    assert hits[0]["document"] == "persisted"
