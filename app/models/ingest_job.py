
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models._utils import utcnow

if TYPE_CHECKING:
    from app.models.user import User

# "processing" while the background task runs; terminal states mirror
# IngestOutcome.status ("ingested" | "duplicate_file" | "unsupported" |
# "parse_failed"), plus "error" for an unexpected exception in the task.
JOB_STATUS_PROCESSING = "processing"


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    job_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=JOB_STATUS_PROCESSING)
    document_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    new_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reused_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    warnings: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    owner: Mapped[User] = relationship()
