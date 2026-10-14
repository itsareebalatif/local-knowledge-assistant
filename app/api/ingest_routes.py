from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.schemas import IngestJobAccepted, IngestJobOut
from app.config import get_settings
from app.dependencies import (
    get_current_user,
    get_db,
    get_embedder,
    get_entity_extractor,
    get_graph_builder,
    get_session_factory,
    get_vector_store,
)
from app.models import IngestJob, User
from app.services.pipeline_service import run_ingest_job

router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["ingest"],
    summary="Queue a file for ingestion into the knowledge base",
)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF, Markdown, HTML, DOCX, or TXT file"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    embedder=Depends(get_embedder),
    vector_store=Depends(get_vector_store),
    graph_builder=Depends(get_graph_builder),
    extractor=Depends(get_entity_extractor),
    session_factory=Depends(get_session_factory),
) -> IngestJobAccepted:
    # The upload's bytes must be read now: UploadFile's underlying stream
    # doesn't survive past this request, so it can't be read later from the
    # background task. Parsing, chunking, embedding, and graph-building are
    # what's deferred — that's the slow part that was blocking the request
    # long enough to trip hosting-platform proxy timeouts on large files.
    file_bytes = await file.read()

    job = IngestJob(user_id=current_user.user_id, filename=file.filename or "upload")
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(
        run_ingest_job,
        job.job_id,
        current_user.user_id,
        file_bytes,
        file.filename,
        embedder,
        vector_store,
        graph_builder,
        extractor,
        get_settings().graph_store_path,
        session_factory,
    )

    return IngestJobAccepted(job_id=job.job_id)


@router.get(
    "/ingest/{job_id}",
    response_model=IngestJobOut,
    tags=["ingest"],
    summary="Check the status of a queued ingestion job",
)
def get_ingest_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IngestJobOut:
    job = db.get(IngestJob, job_id)
    if job is None or job.user_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingest job not found")

    return IngestJobOut(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status,
        document_id=job.document_id,
        total_chunks=job.total_chunks,
        new_chunks=job.new_chunks,
        reused_chunks=job.reused_chunks,
        warnings=job.warnings,
        error=job.error,
    )
