"""RelationEdge model — a directed edge between two entity nodes in the co-occurrence graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.entity_node import EntityNode


class RelationEdge(Base):
    __tablename__ = "relation_edges"

    edge_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entity_nodes.entity_id", ondelete="CASCADE"), index=True, nullable=False
    )
    target_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entity_nodes.entity_id", ondelete="CASCADE"), index=True, nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False)  # USES, DEPENDS_ON, AUTHORED_BY, ...
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    source_entity: Mapped["EntityNode"] = relationship(
        foreign_keys=[source_entity_id], back_populates="outgoing_edges"
    )
    target_entity: Mapped["EntityNode"] = relationship(
        foreign_keys=[target_entity_id], back_populates="incoming_edges"
    )

    __table_args__ = (
        UniqueConstraint(
            "source_entity_id", "target_entity_id", "relation_type", name="uq_edge_source_target_relation"
        ),
    )
