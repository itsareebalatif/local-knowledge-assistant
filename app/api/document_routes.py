from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.schemas import DocumentListResponse, DocumentOut
from app.config import get_settings
from app.dependencies import get_current_user, get_db, get_graph_builder, get_vector_store
from app.models import Chunk, Document, User
from app.services.document_service import delete_document

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


@router.delete(
    "/documents/{doc_id}",
    tags=["documents"],
    summary="Delete a document — removes its chunks, vectors, and graph references too",
)
def delete_document_route(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    vector_store=Depends(get_vector_store),
    graph_builder=Depends(get_graph_builder),
) -> dict:
    deleted = delete_document(db, vector_store, graph_builder, current_user.user_id, doc_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    graph_builder.save(get_settings().graph_store_path)
    return {"status": "deleted", "doc_id": doc_id}
