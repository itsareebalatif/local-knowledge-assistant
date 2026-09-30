from __future__ import annotations

import hashlib

from sqlalchemy import select

from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.models import Chunk, ChunkEntityJunction, Document, EntityNode, RelationEdge
from app.services.graph_service import build_graph_for_chunk


def test_entity_extractor_maps_labels_to_erd_types():
    extractor = SpacyEntityExtractor()
    entities = extractor.extract("Ollama was built to run models like qwen locally in Karachi.")
    types_found = {t for _, t in entities}
    # Exact NER output can vary a bit by spaCy version; what matters is the
    # label mapping and noise filtering, not this model's exact guesses.
    assert types_found <= {"Person", "Organization", "Location", "Concept"}
    assert any(name == "Karachi" for name, _ in entities)


def test_entity_extractor_dedupes_repeated_mentions():
    extractor = SpacyEntityExtractor()
    entities = extractor.extract("Karachi is a big city. I love Karachi.")
    karachi_hits = [e for e in entities if e[0] == "Karachi"]
    assert len(karachi_hits) == 1


def test_entity_extractor_handles_empty_text():
    assert SpacyEntityExtractor().extract("") == []
    assert SpacyEntityExtractor().extract("   ") == []


def test_graph_builder_adds_nodes_and_weighted_edges():
    builder = CooccurrenceGraphBuilder()
    builder.add_chunk_entities(1, [("Ollama", "Organization"), ("Karachi", "Location")])
    builder.add_chunk_entities(2, [("Ollama", "Organization"), ("Karachi", "Location")])

    assert builder.graph.number_of_nodes() == 2
    neighbors = builder.neighbors("Ollama", "Organization")
    assert neighbors == ["Location:karachi"]
    edge_weight = builder.graph["Organization:ollama"]["Location:karachi"]["weight"]
    assert edge_weight == 2  # co-occurred in two different chunks


def test_graph_builder_save_and_load_round_trip(tmp_path):
    builder = CooccurrenceGraphBuilder()
    builder.add_chunk_entities(1, [("Ollama", "Organization"), ("Karachi", "Location")])
    path = tmp_path / "graph.json"
    builder.save(path)

    reloaded = CooccurrenceGraphBuilder.load(path)
    assert reloaded.graph.number_of_nodes() == 2
    assert reloaded.neighbors("Ollama", "Organization") == ["Location:karachi"]


def test_graph_builder_load_missing_file_returns_empty_graph(tmp_path):
    builder = CooccurrenceGraphBuilder.load(tmp_path / "does_not_exist.json")
    assert builder.graph.number_of_nodes() == 0


_doc_counter = 0


def _make_chunk(db_session, user, content: str) -> Chunk:
    global _doc_counter
    _doc_counter += 1
    doc = Document(
        user_id=user.user_id,
        file_path=f"x{_doc_counter}.md",
        file_name=f"x{_doc_counter}.md",
        file_type="MD",
        hash_checksum=hashlib.sha256(f"doc{_doc_counter}".encode()).hexdigest(),
    )
    db_session.add(doc)
    db_session.flush()
    chunk = Chunk(
        doc_id=doc.doc_id,
        chunk_index=0,
        chunk_hash=hashlib.sha256(f"chunk{_doc_counter}".encode()).hexdigest(),
        content=content,
        token_count=10,
    )
    db_session.add(chunk)
    db_session.commit()
    return chunk


def test_build_graph_for_chunk_persists_entities_and_edges(db_session, user):
    chunk = _make_chunk(db_session, user, "Ollama runs models for teams in Karachi.")
    graph_builder = CooccurrenceGraphBuilder()
    extractor = SpacyEntityExtractor()

    entities = build_graph_for_chunk(db_session, graph_builder, extractor, chunk)
    assert len(entities) >= 1

    stored_entities = db_session.execute(select(EntityNode)).scalars().all()
    assert len(stored_entities) == len(entities)

    junctions = db_session.execute(
        select(ChunkEntityJunction).where(ChunkEntityJunction.chunk_id == chunk.chunk_id)
    ).scalars().all()
    assert len(junctions) == len(entities)

    if len(entities) >= 2:
        edges = db_session.execute(select(RelationEdge)).scalars().all()
        assert len(edges) >= 1
        assert edges[0].weight == 1.0


def test_build_graph_for_chunk_increments_weight_on_repeat_cooccurrence(db_session, user):
    graph_builder = CooccurrenceGraphBuilder()
    extractor = SpacyEntityExtractor()
    text = "Ollama and Karachi appear together in this sentence."

    chunk_a = _make_chunk(db_session, user, text)
    build_graph_for_chunk(db_session, graph_builder, extractor, chunk_a)

    chunk_b = _make_chunk(db_session, user, text)
    entities_b = build_graph_for_chunk(db_session, graph_builder, extractor, chunk_b)

    if len(entities_b) >= 2:
        edges = db_session.execute(select(RelationEdge)).scalars().all()
        assert len(edges) == 1  # same pair, not a duplicate edge
        assert edges[0].weight == 2.0


def test_build_graph_for_chunk_with_no_entities_is_a_no_op(db_session, user):
    chunk = _make_chunk(db_session, user, "1 2 3, and, or, but.")
    graph_builder = CooccurrenceGraphBuilder()
    entities = build_graph_for_chunk(db_session, graph_builder, SpacyEntityExtractor(), chunk)
    assert entities == []
