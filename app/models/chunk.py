"""Chunk model — a text segment produced by chunking a document."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.chunk_entity_junction import ChunkEntityJunction
    from app.models.document import Document


class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True, nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # Post-chunking SHA-256 fingerprint — FR-1.4 skips re-embedding on a match.
    chunk_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Foreign identifier pointing at the vector-store record (Chroma/FAISS id).
    embedding_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")
    entity_links: Mapped[list["ChunkEntityJunction"]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("doc_id", "chunk_index", name="uq_chunk_doc_index"),)
