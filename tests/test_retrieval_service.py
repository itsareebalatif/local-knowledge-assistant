from __future__ import annotations

from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.services.retrieval_service import retrieve_and_verify


async def test_retrieve_and_verify_returns_grounded_when_bm25_and_vector_agree(
    db_session, make_chunk, fake_embedder, fake_vector_store
):
    chunk = make_chunk("Ollama runs local language models directly on your own machine.")
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id}],
    )
    graph_builder = CooccurrenceGraphBuilder()  # empty — this test isn't exercising graph expansion
    extractor = SpacyEntityExtractor()

    outcome = await retrieve_and_verify(
        db_session, fake_embedder, fake_vector_store, graph_builder, extractor, "What does Ollama run models on?"
    )

    assert outcome.status == "grounded"
    assert chunk.chunk_id in {c.chunk_id for c in outcome.candidates}
    assert len(fake_embedder.calls) == 1  # proves the vector path actually ran
    found = next(c for c in outcome.candidates if c.chunk_id == chunk.chunk_id)
    assert "bm25" in found.sources and "vector" in found.sources  # both sources found it independently


async def test_retrieve_and_verify_refuses_when_nothing_relevant_exists(
    db_session, make_chunk, fake_embedder, fake_vector_store
):
    make_chunk("Completely unrelated content about gardening tips and soil pH.")
    # fake_vector_store deliberately left empty: nothing embedded to find.
    graph_builder = CooccurrenceGraphBuilder()
    extractor = SpacyEntityExtractor()

    outcome = await retrieve_and_verify(
        db_session, fake_embedder, fake_vector_store, graph_builder, extractor, "quantum computing architecture"
    )

    assert outcome.status == "refused"
    assert outcome.reason == "no_candidates"
    assert outcome.candidates == []
