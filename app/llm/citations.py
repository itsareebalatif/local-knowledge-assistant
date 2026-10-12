from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document
from app.search.types import CandidateChunk

_SNIPPET_LENGTH = 200


@dataclass
class Citation:
    marker: int
    chunk_id: int
    doc_id: int
    file_name: str
    source_url: str
    snippet: str


def build_citations(db: Session, candidates: list[CandidateChunk]) -> list[Citation]:
    # Marker numbering must match app/llm/prompts.py's build_user_prompt exactly.
    doc_ids = {c.doc_id for c in candidates}
    docs = db.execute(select(Document).where(Document.doc_id.in_(doc_ids))).scalars().all()
    docs_by_id = {d.doc_id: d for d in docs}

    citations = []
    for i, candidate in enumerate(candidates):
        doc = docs_by_id.get(candidate.doc_id)
        file_name = doc.file_name if doc is not None else f"document-{candidate.doc_id}"
        snippet = candidate.content[:_SNIPPET_LENGTH]
        if len(candidate.content) > _SNIPPET_LENGTH:
            snippet += "…"
        citations.append(
            Citation(
                marker=i + 1,
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                file_name=file_name,
                source_url=f"/api/documents/{candidate.doc_id}",
                snippet=snippet,
            )
        )
    return citations
