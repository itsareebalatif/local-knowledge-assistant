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
    app.dependency_overrides[get_reranker] = lambda: None


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
        assert events[0]["type"] == "conversation"
        assert events[1:] == [{"type": "refused", "reason": "no_candidates", "metrics": {"num_candidates": 0}}]
        assert fake_llm.calls == []
    finally:
        app.dependency_overrides.clear()


def test_query_requires_authentication(db_session, fake_embedder, fake_vector_store, fake_llm):
    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    try:
        client = TestClient(app)
        response = client.post("/api/query", json={"query": "anything"})
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
            yield ""  # pragma: no cover

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
        assert body["status"] == "refused"
        assert fake_llm.calls == []
    finally:
        app.dependency_overrides.clear()


def test_query_without_a_conversation_id_creates_a_new_conversation(
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
        body = response.json()
        assert body["conversation_id"] is not None

        history = client.get(f"/api/conversations/{body['conversation_id']}", headers=auth_headers).json()
        assert history["title"] == "What does Ollama run models on?"
        assert [m["role"] for m in history["messages"]] == ["user", "assistant"]
        assert history["messages"][0]["content"] == "What does Ollama run models on?"
        assert "Ollama" in history["messages"][1]["content"]
        assert history["messages"][1]["citations"] is not None
    finally:
        app.dependency_overrides.clear()


def test_query_with_an_existing_conversation_id_appends_to_it(
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
        first = client.post("/api/query", json={"query": "What does Ollama run models on?"}, headers=auth_headers).json()
        conversation_id = first["conversation_id"]

        second = client.post(
            "/api/query",
            json={"query": "And what about memory usage?", "conversation_id": conversation_id},
            headers=auth_headers,
        ).json()
        assert second["conversation_id"] == conversation_id

        history = client.get(f"/api/conversations/{conversation_id}", headers=auth_headers).json()
        assert len(history["messages"]) == 4  # two user turns + two assistant turns
    finally:
        app.dependency_overrides.clear()


def test_query_rejects_an_unknown_conversation_id(
    db_session, fake_embedder, fake_vector_store, fake_llm, registered_user, auth_headers
):
    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/query", json={"query": "anything", "conversation_id": 999999}, headers=auth_headers
        )
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_query_persists_a_refusal_with_a_friendly_message(
    db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm, registered_user, auth_headers
):
    make_chunk("completely unrelated content about gardening", owner=registered_user.user)

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/query", json={"query": "quantum computing architecture"}, headers=auth_headers
        )
        conversation_id = response.json()["conversation_id"]

        history = client.get(f"/api/conversations/{conversation_id}", headers=auth_headers).json()
        assert history["messages"][1]["role"] == "assistant"
        assert "couldn't find anything" in history["messages"][1]["content"]
    finally:
        app.dependency_overrides.clear()


def test_query_stream_sends_a_conversation_id_event_first(
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
            "/api/query/stream", json={"query": "What does Ollama run models on?"}, headers=auth_headers
        )
        events = _parse_sse(response.text)
        assert events[0]["type"] == "conversation"
        assert isinstance(events[0]["conversation_id"], int)
    finally:
        app.dependency_overrides.clear()


class _OrderAwareLLM:
    """A stub whose Nth call returns a different canned response — lets a
    test distinguish the query-rewrite call's output from the generation
    call's output, which a single shared FakeLLM instance can't do since it
    returns the same `pieces` for every call regardless of order."""

    model = "order-aware-stub"

    def __init__(self, responses: list[list[str]]):
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    async def generate_stream(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        for piece in self._responses[len(self.calls) - 1]:
            yield piece


def test_query_follow_up_is_rewritten_before_retrieval_runs(
    db_session, make_chunk, fake_embedder, fake_vector_store, registered_user, auth_headers
):
    # The follow-up "what about his siblings?" shares no real vocabulary
    # with this chunk — only a rewrite that resolves "his" to "Hagrid"
    # should be able to retrieve it via BM25.
    chunk = make_chunk("Hagrid had a younger half-brother named Grawp, a giant.", owner=registered_user.user)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": registered_user.user.user_id}],
    )
    llm = _OrderAwareLLM(
        responses=[
            ["Hagrid ", "was ", "the ", "gamekeeper."],  # turn 1 generation (no history yet, no rewrite call)
            ["What ", "are ", "Hagrid's ", "siblings?"],  # turn 2 rewrite call
            ["Hagrid's ", "half-brother ", "is ", "Grawp."],  # turn 2 generation
        ]
    )

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), llm)
    try:
        client = TestClient(app)
        first = client.post("/api/query", json={"query": "who was Hagrid?"}, headers=auth_headers)
        conversation_id = first.json()["conversation_id"]

        second = client.post(
            "/api/query",
            json={"query": "what about his siblings?", "conversation_id": conversation_id},
            headers=auth_headers,
        )
        body = second.json()
        assert body["status"] == "grounded"
        assert "Grawp" in body["answer"]
        assert len(llm.calls) == 3  # turn 1 generation, turn 2 rewrite, turn 2 generation
        rewrite_user_prompt = llm.calls[1][1]
        assert "what about his siblings?" in rewrite_user_prompt
    finally:
        app.dependency_overrides.clear()


def test_query_first_turn_never_calls_rewrite(
    db_session, make_chunk, fake_embedder, fake_vector_store, registered_user, auth_headers
):
    chunk = make_chunk("Ollama runs language models locally on your own machine.", owner=registered_user.user)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": registered_user.user.user_id}],
    )
    llm = _OrderAwareLLM(responses=[["Ollama ", "runs ", "models ", "locally."]])

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), llm)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/query", json={"query": "What does Ollama run models on?"}, headers=auth_headers
        )
        assert response.json()["status"] == "grounded"
        assert len(llm.calls) == 1  # generation only — no history yet, so no rewrite call
    finally:
        app.dependency_overrides.clear()
