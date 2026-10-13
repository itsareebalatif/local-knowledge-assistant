from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.schemas import ChunkOut
from app.dependencies import get_current_user, get_db
from app.models import Chunk, Document, User

router = APIRouter()


@router.get("/chunks/{chunk_id}", response_model=ChunkOut, tags=["citations"], summary="View a cited chunk's source text")
def get_chunk(chunk_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ChunkOut:
    chunk = db.get(Chunk, chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    document = db.get(Document, chunk.doc_id)
    if document is None or document.user_id != current_user.user_id:
        # Same 404 as "doesn't exist" — confirming a chunk belongs to someone else would leak info.
        raise HTTPException(status_code=404, detail="Chunk not found")

    return ChunkOut(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, file_name=document.file_name, content=chunk.content)
