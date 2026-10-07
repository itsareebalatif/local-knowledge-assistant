"""Document listing — lets a logged-in user see what they've actually
uploaded: which files, how many chunks each produced, and a running total.
Nothing here was previously exposed; before this, the only way to find out
was to query the database directly.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.schemas import DocumentListResponse, DocumentOut
from app.dependencies import get_current_user, get_db
from app.models import Chunk, Document, User

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"], summary="List your uploaded documents")
def list_documents(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> DocumentListResponse:
    rows = db.execute(
        select(Document, func.count(Chunk.chunk_id))
        .outerjoin(Chunk, Chunk.doc_id == Document.doc_id)
        .where(Document.user_id == current_user.user_id)
        .group_by(Document.doc_id)
        .order_by(Document.created_at.desc())
    ).all()

    documents = [
        DocumentOut(
            doc_id=doc.doc_id,
            file_name=doc.file_name,
            file_type=doc.file_type,
            chunk_count=chunk_count,
            created_at=doc.created_at,
        )
        for doc, chunk_count in rows
    ]

    return DocumentListResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=sum(d.chunk_count for d in documents),
    )
