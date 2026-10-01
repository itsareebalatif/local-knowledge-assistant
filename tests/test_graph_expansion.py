from __future__ import annotations

from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.graph.graph_expansion import expand_via_graph


class _FakeExtractor:
    """Deterministic stand-in for SpacyEntityExtractor — this test is about
    graph traversal logic, not NER accuracy (that's covered in test_graph.py),
    so it shouldn't be sensitive to a real model's exact classifications."""

    def __init__(self, entities: list[tuple[str, str]]):
        self._entities = entities

    def extract(self, text: str) -> list[tuple[str, str]]:
        return self._entities


def test_expands_to_neighbor_entity_chunks(db_session, make_chunk):
    chunk_a = make_chunk("chunk mentioning both Ollama and qwen together")
    chunk_b = make_chunk("chunk mentioning only qwen")

    graph_builder = CooccurrenceGraphBuilder()
    graph_builder.add_chunk_entities(chunk_a.chunk_id, [("Ollama", "Organization"), ("qwen", "Concept")])
    graph_builder.add_chunk_entities(chunk_b.chunk_id, [("qwen", "Concept")])

    extractor = _FakeExtractor([("Ollama", "Organization")])
    hits = expand_via_graph(db_session, graph_builder, extractor, "tell me about Ollama", top_k=5)

    hit_ids = {h.chunk_id for h in hits}
    # Both chunks come back: they're both linked to "qwen", Ollama's 1-hop neighbor.
    assert hit_ids == {chunk_a.chunk_id, chunk_b.chunk_id}
    assert all(h.sources == ["graph"] for h in hits)


def test_query_entity_not_in_graph_returns_empty(db_session, make_chunk):
    make_chunk("some chunk")
    graph_builder = CooccurrenceGraphBuilder()  # empty graph
    extractor = _FakeExtractor([("Nonexistent", "Organization")])
    assert expand_via_graph(db_session, graph_builder, extractor, "query", top_k=5) == []


def test_query_entity_with_no_neighbors_returns_empty(db_session, make_chunk):
    chunk = make_chunk("chunk mentioning only Ollama, nothing else notable")
    graph_builder = CooccurrenceGraphBuilder()
    graph_builder.add_chunk_entities(chunk.chunk_id, [("Ollama", "Organization")])  # single entity, no edges

    extractor = _FakeExtractor([("Ollama", "Organization")])
    assert expand_via_graph(db_session, graph_builder, extractor, "query", top_k=5) == []


def test_no_entities_in_query_returns_empty(db_session, make_chunk):
    graph_builder = CooccurrenceGraphBuilder()
    extractor = _FakeExtractor([])  # nothing extracted from the query
    assert expand_via_graph(db_session, graph_builder, extractor, "query", top_k=5) == []


def test_ranks_by_edge_weight_and_respects_top_k(db_session, make_chunk):
    strong_neighbor_chunk = make_chunk("chunk about qwen, seen with Ollama many times")
    weak_neighbor_chunk = make_chunk("chunk about llama, seen with Ollama once")

    graph_builder = CooccurrenceGraphBuilder()
    # 3 co-occurrences of Ollama+qwen vs 1 of Ollama+llama.
    for _ in range(3):
        graph_builder.add_chunk_entities(strong_neighbor_chunk.chunk_id, [("Ollama", "Organization"), ("qwen", "Concept")])
    graph_builder.add_chunk_entities(weak_neighbor_chunk.chunk_id, [("Ollama", "Organization"), ("llama", "Concept")])

    extractor = _FakeExtractor([("Ollama", "Organization")])
    hits = expand_via_graph(db_session, graph_builder, extractor, "query", top_k=1)

    assert len(hits) == 1
    assert hits[0].chunk_id == strong_neighbor_chunk.chunk_id
