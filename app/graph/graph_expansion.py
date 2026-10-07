"""1-hop NetworkX graph expansion for retrieval (FR-4.2).

Extracts entities from the query itself, finds each one's 1-hop neighbors in
the co-occurrence graph, and pulls the chunks linked to those *neighbors* —
surfacing chunks that are conceptually related to the query even when they
share no vocabulary with it and aren't embedding-similar either (e.g. a
query about "Ollama" pulling in a chunk that only mentions "qwen2.5:3b",
because the graph has learned the two co-occur often).

The co-occurrence graph itself is shared/global across users (it's entity
structure, not document content), so the chunk_ids it names must be filtered
down to ones the requesting user actually owns — and that filter has to run
BEFORE the top_k cut, not after: truncating first could crowd out a user's
own results with higher-weight chunks that happen to belong to someone else.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.models import Chunk, Document
from app.search.types import CandidateChunk


def expand_via_graph(
    db: Session,
    graph_builder: CooccurrenceGraphBuilder,
    extractor: SpacyEntityExtractor,
    query: str,
    top_k: int,
    user_id: int,
) -> list[CandidateChunk]:
    query_entities = extractor.extract(query)
    if not query_entities:
        return []

    graph = graph_builder.graph
    # chunk_id -> strongest co-occurrence weight linking it to any query entity.
    chunk_weight: dict[int, float] = {}

    for name, entity_type in query_entities:
        key = CooccurrenceGraphBuilder.node_key(name, entity_type)
        if key not in graph:
            continue
        for neighbor_key in graph.neighbors(key):
            weight = graph[key][neighbor_key].get("weight", 1)
            for chunk_id in graph.nodes[neighbor_key].get("chunk_ids", []):
                chunk_weight[chunk_id] = max(chunk_weight.get(chunk_id, 0), weight)

    if not chunk_weight:
        return []

    rows = db.execute(
        select(Chunk)
        .join(Document, Chunk.doc_id == Document.doc_id)
        .where(Chunk.chunk_id.in_(chunk_weight.keys()), Document.user_id == user_id)
    ).scalars().all()
    rows_by_id = {row.chunk_id: row for row in rows}

    ranked_chunk_ids = sorted(rows_by_id, key=lambda cid: chunk_weight[cid], reverse=True)[:top_k]

    return [
        CandidateChunk(
            chunk_id=chunk_id,
            doc_id=rows_by_id[chunk_id].doc_id,
            content=rows_by_id[chunk_id].content,
            score=chunk_weight[chunk_id],
            sources=["graph"],
        )
        for chunk_id in ranked_chunk_ids
    ]
