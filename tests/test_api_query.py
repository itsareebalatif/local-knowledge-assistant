from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.dependencies import (
    get_db,
    get_embedder,
    get_entity_extractor,
    get_graph_builder,
    get_llm,
    get_reranker,
    get_vector_store,
)
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
    app.dependency_overrides[get_reranker] = lambda: None  # reranking is opt-in; these tests cover RRF's own ordering


def test_query_stream_returns_grounded_answer_with_citations(
    db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm, registered_user, auth_headers
):
    chunk = make_chunk("Ollama runs language models locally on your own machine.", owner=registered_user.user)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": registered_user.user.user_id}],
    )
    fake_llm.pieces = ["Ollama ", "runs ", "models ", "locally."]

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    try:
        # No `with`: intentionally not triggering the app's lifespan, since
        # every real resource it would build is already dependency-overridden.
        client = TestClient(app)
        response = client.post(
            "/api/query/stream", json={"query": "What does Ollama run models on?"}, headers=auth_headers
        )

        assert response.status_code == 200
        events = _parse_sse(response.text)
        assert any(e["type"] == "token" for e in events)
        final = events[-1]
        assert final["type"] == "done"
        assert "Ollama" in final["answer"]
        assert len(final["citations"]) >= 1
    finally:
        app.dependency_overrides.clear()


def test_query_stream_refuses_without_calling_llm(
    db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm, registered_user, auth_headers
):
    make_chunk("completely unrelated content about gardening", owner=registered_user.user)

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/query/stream", json={"query": "quantum computing architecture"}, headers=auth_headers
        )

        events = _parse_sse(response.text)
        assert events == [{"type": "refused", "reason": "no_candidates", "metrics": {"num_candidates": 0}}]
        assert fake_llm.calls == []
    finally:
        app.dependency_overrides.clear()


def test_query_requires_authentication(db_session, fake_embedder, fake_vector_store, fake_llm):
    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    try:
        client = TestClient(app)
        response = client.post("/api/query", json={"query": "anything"})  # no Authorization header
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_query_non_streaming_returns_one_json_response(
    db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm, registered_user, auth_headers
):
    chunk = make_chunk("Ollama runs language models locally on your own machine.", owner=registered_user.user)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": registered_user.user.user_id}],
    )
    fake_llm.pieces = ["Ollama ", "runs ", "models ", "locally."]

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/query", json={"query": "What does Ollama run models on?"}, headers=auth_headers
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "grounded"
        assert "Ollama" in body["answer"]
        assert len(body["citations"]) >= 1
        assert "overall_coverage" in body["grounding"]
    finally:
        app.dependency_overrides.clear()


def test_query_non_streaming_refuses_without_calling_llm(
    db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm, registered_user, auth_headers
):
    make_chunk("completely unrelated content about gardening", owner=registered_user.user)

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/query", json={"query": "quantum computing architecture"}, headers=auth_headers
        )

        body = response.json()
        assert body["status"] == "refused"
        assert body["reason"] == "no_candidates"
        assert body["answer"] == ""
        assert fake_llm.calls == []
    finally:
        app.dependency_overrides.clear()


def test_query_non_streaming_reports_llm_failure_as_error_status(
    db_session, make_chunk, fake_embedder, fake_vector_store, registered_user, auth_headers
):
    from app.llm.base import LLMError

    class FailingLLM:
        model = "fake-failing-llm"

        async def generate_stream(self, system_prompt, user_prompt):
            if True:
                raise LLMError("the model server is unreachable")
            yield ""  # pragma: no cover - unreachable; keeps this an async generator function

    chunk = make_chunk("Ollama runs language models locally on your own machine.", owner=registered_user.user)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": registered_user.user.user_id}],
    )

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), FailingLLM())
    try:
        client = TestClient(app)
        response = client.post(
            "/api/query", json={"query": "What does Ollama run models on?"}, headers=auth_headers
        )

        body = response.json()
        assert body["status"] == "error"
        assert "unreachable" in body["reason"]
    finally:
        app.dependency_overrides.clear()


def test_query_never_surfaces_another_users_chunks(
    db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm, registered_user, auth_headers
):
    from app.models import User

    other = User(email="other@example.com", full_name="Other User", role="USER")
    db_session.add(other)
    db_session.commit()

    other_chunk = make_chunk("Ollama runs language models locally on your own machine.", owner=other)
    fake_vector_store.add(
        ids=[f"chunk-{other_chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[other_chunk.content],
        metadatas=[{"doc_id": other_chunk.doc_id, "chunk_id": other_chunk.chunk_id, "user_id": other.user_id}],
    )

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/query", json={"query": "What does Ollama run models on?"}, headers=auth_headers
        )
        body = response.json()
        assert body["status"] == "refused"  # the only matching chunk belongs to `other`, not registered_user
        assert fake_llm.calls == []
    finally:
        app.dependency_overrides.clear()
