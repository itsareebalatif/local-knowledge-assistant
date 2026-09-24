"""EntityNode model — a graph entity extracted from chunks via spaCy NER (non-LLM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.chunk_entity_junction import ChunkEntityJunction
    from app.models.relation_edge import RelationEdge


class EntityNode(Base):
    __tablename__ = "entity_nodes"

    entity_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)  # Person, Organization, Location, Concept, ...

    chunk_links: Mapped[list["ChunkEntityJunction"]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )
    outgoing_edges: Mapped[list["RelationEdge"]] = relationship(
        foreign_keys="RelationEdge.source_entity_id", back_populates="source_entity", cascade="all, delete-orphan"
    )
    incoming_edges: Mapped[list["RelationEdge"]] = relationship(
        foreign_keys="RelationEdge.target_entity_id", back_populates="target_entity", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Same entity name+type should resolve to one node (coreference merge, SRS 4.5).
        UniqueConstraint("name", "type", name="uq_entity_name_type"),
    )
