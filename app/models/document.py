"""Document model — an ingested source file owned by a user."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models._utils import utcnow

if TYPE_CHECKING:
    from app.models.chunk import Chunk
    from app.models.metadata import DocumentMetadata
    from app.models.user import User


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True, nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    # Pre-chunking SHA-256 fingerprint — FR-1.4 checks this before parsing.
    hash_checksum: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    owner: Mapped["User"] = relationship(back_populates="documents")
    metadata_entries: Mapped[list["DocumentMetadata"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        # Same user re-uploading the identical file is a duplicate; different
        # users may legitimately own copies of the same file.
        UniqueConstraint("user_id", "hash_checksum", name="uq_document_user_hash"),
    )
