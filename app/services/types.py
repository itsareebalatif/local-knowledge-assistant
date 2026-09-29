
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

IngestStatus = Literal["ingested", "duplicate_file", "unsupported", "parse_failed"]


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
