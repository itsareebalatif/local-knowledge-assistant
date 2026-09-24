"""ChunkEntityJunction model — composite junction table mapping chunks <-> entities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.chunk import Chunk
    from app.models.entity_node import EntityNode


class ChunkEntityJunction(Base):
    __tablename__ = "chunk_entity_junction"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"), primary_key=True, index=True
    )
    entity_id: Mapped[int] = mapped_column(
        ForeignKey("entity_nodes.entity_id", ondelete="CASCADE"), primary_key=True, index=True
    )

    chunk: Mapped["Chunk"] = relationship(back_populates="entity_links")
    entity: Mapped["EntityNode"] = relationship(back_populates="chunk_links")
