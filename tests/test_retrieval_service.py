from __future__ import annotations

from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.services.retrieval_service import retrieve_and_verify


async def test_retrieve_and_verify_returns_grounded_when_bm25_and_vector_agree(
    db_session, make_chunk, fake_embedder, fake_vector_store, user
):
    chunk = make_chunk("Ollama runs local language models directly on your own machine.")
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": user.user_id}],
    )
    graph_builder = CooccurrenceGraphBuilder()  # empty — this test isn't exercising graph expansion
    extractor = SpacyEntityExtractor()

    outcome = await retrieve_and_verify(
        db_session, fake_embedder, fake_vector_store, graph_builder, extractor,
        "What does Ollama run models on?", user.user_id,
    )

    assert outcome.status == "grounded"
    assert chunk.chunk_id in {c.chunk_id for c in outcome.candidates}
    # Two embed calls, not one: the first is vector_search's query embedding
    # (proves the vector path actually ran); the second is MMR re-embedding
    # the fused candidates' own text to measure diversity between them —
    # a different question from "how similar is this to the query."
    assert len(fake_embedder.calls) == 2
    assert fake_embedder.calls[0] == ["What does Ollama run models on?"]
    assert fake_embedder.calls[1] == [chunk.content]
    found = next(c for c in outcome.candidates if c.chunk_id == chunk.chunk_id)
    assert "bm25" in found.sources and "vector" in found.sources  # both sources found it independently


async def test_retrieve_and_verify_refuses_when_nothing_relevant_exists(
    db_session, make_chunk, fake_embedder, fake_vector_store, user
):
    make_chunk("Completely unrelated content about gardening tips and soil pH.")
    # fake_vector_store deliberately left empty: nothing embedded to find.
    graph_builder = CooccurrenceGraphBuilder()
    extractor = SpacyEntityExtractor()

    outcome = await retrieve_and_verify(
        db_session, fake_embedder, fake_vector_store, graph_builder, extractor,
        "quantum computing architecture", user.user_id,
    )

    assert outcome.status == "refused"
    assert outcome.reason == "no_candidates"
    assert outcome.candidates == []


async def test_retrieve_and_verify_runs_reranker_when_one_is_provided(
    db_session, make_chunk, fake_embedder, fake_vector_store, fake_reranker, user
):
    chunk = make_chunk("Ollama runs local language models directly on your own machine.")
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": user.user_id}],
    )

    outcome = await retrieve_and_verify(
        db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(),
        "What does Ollama run models on?", user.user_id, fake_reranker,
    )

    assert outcome.status == "grounded"
    assert len(fake_reranker.calls) == 1  # proves the rerank step actually ran
    query_arg, num_candidates_in = fake_reranker.calls[0]
    assert query_arg == "What does Ollama run models on?"
    assert num_candidates_in >= 1


async def test_retrieve_and_verify_skips_reranking_when_no_reranker_given(
    db_session, make_chunk, fake_embedder, fake_vector_store, user
):
    chunk = make_chunk("Ollama runs local language models directly on your own machine.")
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": user.user_id}],
    )

    # No sixth argument at all — the default (reranker=None) must still work,
    # exactly as it did before reranking existed. Guards against reranking
    # ever becoming accidentally mandatory.
    outcome = await retrieve_and_verify(
        db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(),
        "What does Ollama run models on?", user.user_id,
    )

    assert outcome.status == "grounded"


async def test_retrieve_and_verify_never_crosses_users(db_session, make_chunk, fake_embedder, fake_vector_store, user):
    from app.models import User

    other = User(email="other@example.com", full_name="Other User", role="USER")
    db_session.add(other)
    db_session.commit()

    other_chunk = make_chunk("Ollama runs local language models directly on your own machine.", owner=other)
    fake_vector_store.add(
        ids=[f"chunk-{other_chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[other_chunk.content],
        metadatas=[{"doc_id": other_chunk.doc_id, "chunk_id": other_chunk.chunk_id, "user_id": other.user_id}],
    )

    outcome = await retrieve_and_verify(
        db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(),
        "What does Ollama run models on?", user.user_id,
    )

    assert outcome.status == "refused"  # the only matching chunk belongs to someone else
