from __future__ import annotations

import json

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.config import get_settings
from app.dependencies import (
    get_db,
    get_embedder,
    get_entity_extractor,
    get_graph_builder,
    get_llm,
    get_llm_chain,
    get_rag_graph,
    get_reranker,
    get_vector_store,
)
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.llm.orchestration import build_rag_graph
from app.main import app
from app.services.conversation_service import append_message, create_conversation


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
    fake_llm.pieces = ["Ollama ", "runs ", "models ", "locally ", "[1]."]

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
    fake_llm.pieces = ["Ollama ", "runs ", "models ", "locally ", "[1]."]

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


def test_query_stream_falls_back_to_next_backend_without_surfacing_the_first_failure(
    db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm, registered_user, auth_headers
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
    fake_llm.pieces = ["Ollama ", "runs ", "models ", "locally ", "[1]."]

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    # The primary (first) backend in the chain always fails before producing
    # any output; the second is the normal fake_llm that succeeds.
    app.dependency_overrides[get_llm_chain] = lambda: [FailingLLM(), fake_llm]
    try:
        client = TestClient(app)
        response = client.post(
            "/api/query/stream", json={"query": "What does Ollama run models on?"}, headers=auth_headers
        )

        assert response.status_code == 200
        events = _parse_sse(response.text)
        # The failed first attempt must never reach the client as an "error"
        # event — only the successful fallback's events should be visible.
        assert not any(e["type"] == "error" for e in events)
        final = events[-1]
        assert final["type"] == "done"
        assert "Ollama" in final["answer"]
    finally:
        app.dependency_overrides.clear()


def test_circuit_breaker_skips_a_recently_failed_backend_on_the_next_request(
    db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm, registered_user, auth_headers, monkeypatch
):
    from app.llm.base import LLMError

    # One failure is enough to trip it, and the cooldown is long enough
    # that the second request below can't outlast it by accident.
    monkeypatch.setattr(get_settings(), "circuit_breaker_failure_threshold", 1, raising=False)
    monkeypatch.setattr(get_settings(), "circuit_breaker_cooldown_seconds", 300.0, raising=False)

    class FailingLLM:
        model = "fake-failing-llm"
        provider_name = "FailingProvider"

        def __init__(self):
            self.call_count = 0

        async def generate_stream(self, system_prompt, user_prompt):
            self.call_count += 1
            raise LLMError("the model server is unreachable")
            yield ""  # pragma: no cover

    failing = FailingLLM()
    chunk = make_chunk("Ollama runs language models locally on your own machine.", owner=registered_user.user)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": registered_user.user.user_id}],
    )
    fake_llm.pieces = ["Ollama ", "runs ", "models ", "locally ", "[1]."]

    # A real Store (in-memory is enough here) is what makes this a test of
    # the circuit breaker specifically — it's what persists "this backend
    # just failed" *across* the two separate requests below, each of which
    # creates its own new conversation (no checkpointer needed: the health
    # record lives in the store, not scoped to any one conversation/thread).
    store_backed_graph = build_rag_graph(store=InMemoryStore())

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), failing)
    app.dependency_overrides[get_llm_chain] = lambda: [failing, fake_llm]
    app.dependency_overrides[get_rag_graph] = lambda: store_backed_graph
    try:
        client = TestClient(app)
        first = client.post("/api/query", json={"query": "What does Ollama run models on?"}, headers=auth_headers)
        assert first.json()["status"] == "grounded"
        assert failing.call_count == 1  # tried once, failed, fell back to fake_llm

        second = client.post("/api/query", json={"query": "What does Ollama run models on?"}, headers=auth_headers)
        assert second.json()["status"] == "grounded"
        # A brand-new conversation, same process — the circuit breaker (not
        # conversation history) is what skips the known-bad backend here.
        assert failing.call_count == 1
    finally:
        app.dependency_overrides.clear()


def test_remember_phrase_is_stored_and_applied_to_a_later_conversation(
    db_session, make_chunk, fake_embedder, fake_vector_store, registered_user, auth_headers
):
    chunk = make_chunk("Ollama runs language models locally on your own machine.", owner=registered_user.user)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": registered_user.user.user_id}],
    )

    class RecordingLLM:
        model = "recording-llm"

        def __init__(self):
            self.system_prompts: list[str] = []

        async def generate_stream(self, system_prompt, user_prompt):
            self.system_prompts.append(system_prompt)
            yield "Ollama "
            yield "runs "
            yield "models "
            yield "locally "
            yield "[1]."

    llm = RecordingLLM()
    # A real Store, shared across both requests below — exactly like the
    # circuit breaker test above, but this time proving a fact extracted
    # from one conversation (new conversation_id each call) is available
    # to a completely separate one, which is the whole point of putting it
    # in the Store rather than in that one conversation's own Message rows.
    store_backed_graph = build_rag_graph(store=InMemoryStore())

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), llm)
    app.dependency_overrides[get_rag_graph] = lambda: store_backed_graph
    try:
        client = TestClient(app)
        first = client.post(
            "/api/query",
            json={"query": "Remember that I prefer short, one-sentence answers. What does Ollama run models on?"},
            headers=auth_headers,
        )
        assert first.json()["status"] == "grounded"
        # Applied starting with the very message that defines it.
        assert "prefer short, one-sentence answers" in llm.system_prompts[0]

        second = client.post(
            "/api/query", json={"query": "What does Ollama run models on?"}, headers=auth_headers
        )
        assert second.json()["status"] == "grounded"
        # A brand-new conversation — the fact came from the Store, not from
        # this conversation's own (nonexistent) prior history.
        assert "prefer short, one-sentence answers" in llm.system_prompts[1]
    finally:
        app.dependency_overrides.clear()


async def test_resume_continues_an_interrupted_run_from_its_checkpoint(
    db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm, registered_user, auth_headers
):
    chunk = make_chunk("Ollama runs language models locally on your own machine.", owner=registered_user.user)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": registered_user.user.user_id}],
    )
    fake_llm.pieces = ["Ollama ", "runs ", "models ", "locally ", "[1]."]

    checkpointer = InMemorySaver()
    # Same checkpointer, two compiled graphs: one that stops right before
    # "generate" (standing in for the server crashing at that exact point),
    # one that doesn't — this is what app.dependencies.get_rag_graph would
    # hand a real request in production, since interrupt_before is a
    # test-only construct (see build_rag_graph's docstring note).
    interrupting_graph = build_rag_graph(checkpointer=checkpointer, interrupt_before=["generate"])
    resuming_graph = build_rag_graph(checkpointer=checkpointer)

    conversation = create_conversation(db_session, registered_user.user.user_id, "What does Ollama run models on?")
    append_message(db_session, conversation, "user", "What does Ollama run models on?")
    thread_config = {
        "configurable": {
            "thread_id": str(conversation.conversation_id),
            "db": db_session,
            "embedder": fake_embedder,
            "vector_store": fake_vector_store,
            "graph_builder": CooccurrenceGraphBuilder(),
            "extractor": SpacyEntityExtractor(),
            "llm": fake_llm,
            "llm_chain": [fake_llm],
            "reranker": None,
            # Not graph state: SQLAlchemy Message objects aren't
            # msgpack-serializable, and the checkpointer persists state
            # after every step (see the matching fix in query_routes.py).
            "recent_messages": [],
            "meta_history": [],
        }
    }

    # Simulate the crash: this run stops before "generate" and never
    # touches the LLM at all.
    await interrupting_graph.ainvoke(
        {
            "query": "What does Ollama run models on?",
            "user_id": registered_user.user.user_id,
        },
        config=thread_config,
    )
    assert fake_llm.calls == []

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    app.dependency_overrides[get_rag_graph] = lambda: resuming_graph
    try:
        client = TestClient(app)
        response = client.post(f"/api/query/{conversation.conversation_id}/resume", headers=auth_headers)

        assert response.status_code == 200
        events = _parse_sse(response.text)
        final = events[-1]
        assert final["type"] == "done"
        assert "Ollama" in final["answer"]
        # Only one generation call — retrieval before the "crash" wasn't redone.
        assert len(fake_llm.calls) == 1
    finally:
        app.dependency_overrides.clear()


def test_resume_rejects_a_conversation_with_nothing_pending(
    db_session, fake_embedder, fake_vector_store, fake_llm, registered_user, auth_headers
):
    checkpointer = InMemorySaver()
    graph = build_rag_graph(checkpointer=checkpointer)
    conversation = create_conversation(db_session, registered_user.user.user_id, "hello")

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), fake_llm)
    app.dependency_overrides[get_rag_graph] = lambda: graph
    try:
        client = TestClient(app)
        response = client.post(f"/api/query/{conversation.conversation_id}/resume", headers=auth_headers)
        assert response.status_code == 409
    finally:
        app.dependency_overrides.clear()


class _SequentialLLM:
    """Like _OrderAwareLLM below, but defined here (used before that class's
    definition point in this file) — returns the Nth canned response on the
    Nth call, letting a test see a bad first attempt get silently replaced
    by a better regenerated one."""

    model = "sequential-stub"

    def __init__(self, responses: list[list[str]]):
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    async def generate_stream(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        for piece in self._responses[len(self.calls) - 1]:
            yield piece


def test_query_stream_regenerates_a_fully_ungrounded_first_answer(
    db_session, make_chunk, fake_embedder, fake_vector_store, registered_user, auth_headers
):
    chunk = make_chunk("Ollama runs language models locally on your own machine.", owner=registered_user.user)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": registered_user.user.user_id}],
    )
    # First attempt shares no vocabulary with the chunk at all (fully
    # ungrounded); second attempt actually answers from it.
    llm = _SequentialLLM(
        responses=[
            ["Bananas ", "grow ", "on ", "trees ", "in ", "tropical ", "climates."],
            ["Ollama ", "runs ", "models ", "locally ", "[1]."],
        ]
    )

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), llm)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/query/stream", json={"query": "What does Ollama run models on?"}, headers=auth_headers
        )

        assert response.status_code == 200
        events = _parse_sse(response.text)
        # The discarded first attempt's tokens ("Bananas...") must never
        # reach the client, and there must be no spurious "refused" event
        # for it either — only the regenerated, grounded answer is visible.
        assert not any(e["type"] == "refused" for e in events)
        final = events[-1]
        assert final["type"] == "done"
        assert "Ollama" in final["answer"]
        assert "Bananas" not in response.text
        assert len(llm.calls) == 2
        assert "support" in llm.calls[1][0]  # retry hint was appended to the 2nd call's system prompt
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
    fake_llm.pieces = ["Ollama ", "runs ", "models ", "locally ", "[1]."]

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
    test distinguish the query-router call's output from the generation
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
            ["Hagrid ", "was ", "the ", "gamekeeper."],  # turn 1 generation (no history yet, no router call)
            ["ROUTE: hybrid\nQUERY: What are Hagrid's siblings?"],  # turn 2 router call
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
        assert len(llm.calls) == 3  # turn 1 generation, turn 2 router, turn 2 generation
        router_user_prompt = llm.calls[1][1]
        assert "what about his siblings?" in router_user_prompt
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


def test_meta_question_about_the_conversation_never_touches_retrieval(
    db_session, make_chunk, fake_embedder, fake_vector_store, registered_user, auth_headers
):
    # Nothing is indexed for this document on purpose — a meta question like
    # "what did I ask earlier?" must never reach retrieve_and_verify at all,
    # so it must never be refused for lacking a relevant chunk either.
    make_chunk("Ollama runs language models locally on your own machine.", owner=registered_user.user)

    llm = _OrderAwareLLM(
        responses=[
            ["Ollama ", "runs ", "locally."],  # turn 1 generation (no history yet, no router call)
            ["ROUTE: meta\nQUERY: what did I ask earlier?"],  # turn 2 router call
            ["You ", "asked ", "about ", "Ollama."],  # turn 2 transcript-answering call
        ]
    )

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(), llm)
    try:
        client = TestClient(app)
        first = client.post("/api/query", json={"query": "What does Ollama run models on?"}, headers=auth_headers)
        conversation_id = first.json()["conversation_id"]

        second = client.post(
            "/api/query",
            json={"query": "what did I ask earlier?", "conversation_id": conversation_id},
            headers=auth_headers,
        )
        body = second.json()
        assert body["status"] == "conversation"
        assert body["answer"] == "You asked about Ollama."
        assert body["citations"] == []
        assert body["grounding"] is None
        assert len(llm.calls) == 3  # turn 1 generation, turn 2 router, turn 2 transcript answer — no retrieval call

        # Regression for a real, confirmed-live bug: the current question had
        # already been appended to the conversation before the transcript was
        # read, so it showed up as the transcript's own last line — "what did
        # I ask earlier?" would answer itself instead of the real prior turn.
        transcript_prompt = llm.calls[2][1]
        assert "What does Ollama run models on?" in transcript_prompt
        assert transcript_prompt.count("what did I ask earlier?") == 1
    finally:
        app.dependency_overrides.clear()
