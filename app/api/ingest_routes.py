"""File ingestion endpoint (FR-1.1, Day-3 API surface).

Wraps app.services.pipeline_service.ingest_index_and_graph — the exact same
ingest -> embed -> graph pipeline `kengine ingest` uses — behind a multipart
file upload. The in-memory graph is persisted to disk right after a
successful ingest, so a later GET /graph (or a server restart) reflects
what this call just added.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.api.schemas import IngestResponse
from app.config import get_settings
from app.dependencies import get_current_user, get_db, get_embedder, get_entity_extractor, get_graph_builder, get_vector_store
from app.models import User
from app.services.pipeline_service import ingest_index_and_graph

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, tags=["ingest"], summary="Ingest a file into the knowledge base")
async def ingest_file(
    file: UploadFile = File(..., description="PDF, Markdown, HTML, DOCX, or TXT file"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    embedder=Depends(get_embedder),
    vector_store=Depends(get_vector_store),
    graph_builder=Depends(get_graph_builder),
    extractor=Depends(get_entity_extractor),
) -> IngestResponse:
    file_bytes = await file.read()
    outcome = await ingest_index_and_graph(
        db, current_user.user_id, file_bytes, file.filename, embedder, vector_store, graph_builder, extractor
    )
    if outcome.status == "ingested":
        graph_builder.save(get_settings().graph_store_path)

    return IngestResponse(
        status=outcome.status,
        document_id=outcome.document_id,
        total_chunks=outcome.total_chunks,
        new_chunks=outcome.new_chunks,
        reused_chunks=outcome.reused_chunks,
        warnings=outcome.warnings,
        error=outcome.error,
    )
