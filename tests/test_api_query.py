from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.dependencies import get_db, get_embedder, get_entity_extractor, get_graph_builder, get_llm, get_vector_store
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.main import app


def _parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.split("\n\n"):
        block = block.strip()
        if block.startswith("data:"):
            events.append(json.loads(block[len("data:") :].strip()))
    return events


def _override_all(db_session, embedder, vector_store, graph_builder, extractor, llm):
    def _get_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_graph_builder] = lambda: graph_builder
    app.dependency_overrides[get_entity_extractor] = lambda: extractor
    app.dependency_overrides[get_llm] = lambda: llm


def test_query_stream_returns_grounded_answer_with_citations(
    db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm
):
    chunk = make_chunk("Ollama runs language models locally on your own machine.")
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id}],
    )
    fake_llm.pieces = ["Ollama ", "runs ", "models ", "locally."]

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    try:
        # No `with`: intentionally not triggering the app's lifespan, since
        # every real resource it would build is already dependency-overridden.
        client = TestClient(app)
        response = client.post("/api/query/stream", json={"user_id": 1, "query": "What does Ollama run models on?"})

        assert response.status_code == 200
        events = _parse_sse(response.text)
        assert any(e["type"] == "token" for e in events)
        final = events[-1]
        assert final["type"] == "done"
        assert "Ollama" in final["answer"]
        assert len(final["citations"]) >= 1
    finally:
        app.dependency_overrides.clear()


def test_query_stream_refuses_without_calling_llm(db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm):
    make_chunk("completely unrelated content about gardening")

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    try:
        client = TestClient(app)
        response = client.post("/api/query/stream", json={"user_id": 1, "query": "quantum computing architecture"})

        events = _parse_sse(response.text)
        assert events == [{"type": "refused", "reason": "no_candidates", "metrics": {"num_candidates": 0}}]
        assert fake_llm.calls == []
    finally:
        app.dependency_overrides.clear()
