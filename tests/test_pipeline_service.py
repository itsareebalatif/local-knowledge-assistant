from __future__ import annotations

from sqlalchemy import select

from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.models import Chunk, ChunkEntityJunction
from app.services.pipeline_service import ingest_index_and_graph


async def test_full_pipeline_embeds_and_builds_graph(db_session, user, fake_embedder, fake_vector_store):
    graph_builder = CooccurrenceGraphBuilder()
    extractor = SpacyEntityExtractor()
    content = b"# Karachi Office\n\nOllama runs models for the Karachi office team.\n"

    outcome = await ingest_index_and_graph(
        db_session, user.user_id, content, "notes.md", fake_embedder, fake_vector_store, graph_builder, extractor
    )

    assert outcome.status == "ingested"
    assert len(fake_embedder.calls) == 1  # one batch call, not one per chunk

    chunks = db_session.execute(select(Chunk).where(Chunk.doc_id == outcome.document_id)).scalars().all()
    assert all(c.embedding_id is not None for c in chunks)
    assert all(c.embedding_id in fake_vector_store.records for c in chunks)

    junctions = db_session.execute(select(ChunkEntityJunction)).scalars().all()
    assert len(junctions) > 0  # entities were actually linked to chunks
    assert graph_builder.graph.number_of_nodes() > 0


async def test_pipeline_skips_embed_and_graph_for_duplicate_file(
    db_session, user, fake_embedder, fake_vector_store
):
    graph_builder = CooccurrenceGraphBuilder()
    extractor = SpacyEntityExtractor()
    content = b"# Doc\n\nOllama runs in Karachi.\n"

    first = await ingest_index_and_graph(
        db_session, user.user_id, content, "notes.md", fake_embedder, fake_vector_store, graph_builder, extractor
    )
    assert first.status == "ingested"
    calls_after_first = len(fake_embedder.calls)
    nodes_after_first = graph_builder.graph.number_of_nodes()

    second = await ingest_index_and_graph(
        db_session, user.user_id, content, "notes.md", fake_embedder, fake_vector_store, graph_builder, extractor
    )

    assert second.status == "duplicate_file"
    assert len(fake_embedder.calls) == calls_after_first  # no new embedding calls
    assert graph_builder.graph.number_of_nodes() == nodes_after_first  # no new graph work
