"""Builds the knowledge graph for a newly ingested chunk (FR-2.1, FR-2.2).

spaCy extracts entities from the chunk's text (no LLM); every entity gets an
EntityNode row (deduped by name+type), a chunk_entity_junction row links it
to this chunk, and every pair of entities that co-occurred in the chunk gets
a RelationEdge (or has its weight bumped if that edge already exists). The
same is mirrored into the in-memory NetworkX graph so both stay in sync —
SQL for durable, queryable storage; NetworkX for fast in-process traversal.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.models import Chunk, ChunkEntityJunction, EntityNode, RelationEdge

logger = logging.getLogger(__name__)


def _get_or_create_entity(db: Session, name: str, entity_type: str) -> EntityNode:
    entity = db.execute(
        select(EntityNode).where(EntityNode.name == name, EntityNode.type == entity_type)
    ).scalar_one_or_none()
    if entity is None:
        entity = EntityNode(name=name, type=entity_type)
        db.add(entity)
        db.flush()  # need entity.entity_id for the junction/edge rows below
    return entity


def _link_chunk_to_entity(db: Session, chunk_id: int, entity_id: int) -> None:
    exists = db.execute(
        select(ChunkEntityJunction).where(
            ChunkEntityJunction.chunk_id == chunk_id, ChunkEntityJunction.entity_id == entity_id
        )
    ).scalar_one_or_none()
    if exists is None:
        db.add(ChunkEntityJunction(chunk_id=chunk_id, entity_id=entity_id))


def _upsert_relation(db: Session, entity_id_a: int, entity_id_b: int) -> None:
    # Canonical (min, max) order: co-occurrence is undirected, so A-B and B-A
    # must be the same edge, not counted twice.
    source_id, target_id = sorted((entity_id_a, entity_id_b))
    edge = db.execute(
        select(RelationEdge).where(
            RelationEdge.source_entity_id == source_id,
            RelationEdge.target_entity_id == target_id,
            RelationEdge.relation_type == "CO_OCCURS_WITH",
        )
    ).scalar_one_or_none()
    if edge is None:
        db.add(
            RelationEdge(
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation_type="CO_OCCURS_WITH",
                weight=1.0,
                confidence_score=1.0,
            )
        )
    else:
        edge.weight += 1.0


def build_graph_for_chunk(
    db: Session,
    graph_builder: CooccurrenceGraphBuilder,
    extractor: SpacyEntityExtractor,
    chunk: Chunk,
) -> list[EntityNode]:
    """Extract entities from one chunk, link them to it, and connect every
    pair that co-occurred — in both the SQL tables and the in-memory graph."""
    found = extractor.extract(chunk.content)
    if not found:
        return []

    entities = [_get_or_create_entity(db, name, entity_type) for name, entity_type in found]

    for entity in entities:
        _link_chunk_to_entity(db, chunk.chunk_id, entity.entity_id)

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            _upsert_relation(db, entities[i].entity_id, entities[j].entity_id)

    db.commit()

    graph_builder.add_chunk_entities(chunk.chunk_id, found)
    logger.info("Chunk %s: linked %d entities", chunk.chunk_id, len(entities))
    return entities
