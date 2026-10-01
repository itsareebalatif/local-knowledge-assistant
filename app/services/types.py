
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.search.types import CandidateChunk

IngestStatus = Literal["ingested", "duplicate_file", "unsupported", "parse_failed"]
RetrievalStatus = Literal["grounded", "refused"]


@dataclass
class IngestOutcome:
    filename: str
    status: IngestStatus
    file_hash: str = ""
    document_id: int | None = None
    total_chunks: int = 0
    new_chunks: int = 0
    reused_chunks: int = 0
    pending_embedding_chunk_ids: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class RetrievalOutcome:
    query: str
    status: RetrievalStatus
    candidates: list[CandidateChunk] = field(default_factory=list)
    reason: str = ""
    metrics: dict = field(default_factory=dict)
