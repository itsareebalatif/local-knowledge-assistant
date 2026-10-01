
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateChunk:
    chunk_id: int
    doc_id: int
    content: str
    score: float
    sources: list[str] = field(default_factory=list)
