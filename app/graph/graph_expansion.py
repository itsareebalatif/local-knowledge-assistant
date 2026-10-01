
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.models import Chunk
from app.search.types import CandidateChunk


def expand_via_graph(
    db: Session,
    graph_builder: CooccurrenceGraphBuilder,
    extractor: SpacyEntityExtractor,
    query: str,
    top_k: int,
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

    ranked_chunk_ids = sorted(chunk_weight, key=lambda cid: chunk_weight[cid], reverse=True)[:top_k]

    rows = db.execute(select(Chunk).where(Chunk.chunk_id.in_(ranked_chunk_ids))).scalars().all()
    rows_by_id = {row.chunk_id: row for row in rows}

    candidates = []
    for chunk_id in ranked_chunk_ids:
        chunk = rows_by_id.get(chunk_id)
        if chunk is None:
            continue  # graph and DB can drift if a chunk was deleted after the graph was built
        candidates.append(
            CandidateChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                score=chunk_weight[chunk_id],
                sources=["graph"],
            )
        )
    return candidates
