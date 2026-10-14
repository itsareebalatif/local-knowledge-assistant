from __future__ import annotations

import logging
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.embedder import EmbedderBackend
from app.embeddings.vector_store import VectorStore
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.models import Chunk, IngestJob
from app.services.embedding_service import embed_pending_chunks
from app.services.graph_service import build_graph_for_chunk
from app.services.ingest_service import ingest_document
from app.services.types import IngestOutcome

logger = logging.getLogger(__name__)


async def ingest_index_and_graph(
    db: Session,
    user_id: int,
    file_bytes: bytes,
    filename: str,
    embedder: EmbedderBackend,
    vector_store: VectorStore,
    graph_builder: CooccurrenceGraphBuilder,
    extractor: SpacyEntityExtractor,
    file_path: str | None = None,
) -> IngestOutcome:
    outcome = await ingest_document(db, user_id, file_bytes, filename, file_path=file_path)
    if outcome.status != "ingested":
        return outcome

    await embed_pending_chunks(db, embedder, vector_store, outcome.pending_embedding_chunk_ids)

    chunks = db.execute(select(Chunk).where(Chunk.doc_id == outcome.document_id)).scalars().all()
    for chunk in chunks:
        build_graph_for_chunk(db, graph_builder, extractor, chunk)

    return outcome


async def run_ingest_job(
    job_id: int,
    user_id: int,
    file_bytes: bytes,
    filename: str,
    embedder: EmbedderBackend,
    vector_store: VectorStore,
    graph_builder: CooccurrenceGraphBuilder,
    extractor: SpacyEntityExtractor,
    graph_store_path: str,
    session_factory: Callable[[], Session],
) -> None:
    # Runs as a FastAPI BackgroundTask, after the response that created the
    # job row has already been sent — so it must not touch the request's
    # own (by-then-closed) DB session. It opens its own, independent one.
    db = session_factory()
    try:
        job = db.get(IngestJob, job_id)
        if job is None:  # pragma: no cover - defensive, job rows are never deleted
            return
        try:
            outcome = await ingest_index_and_graph(
                db, user_id, file_bytes, filename, embedder, vector_store, graph_builder, extractor
            )
        except Exception as exc:
            logger.exception("Background ingest job %s failed", job_id)
            job.status = "error"
            job.error = str(exc)
            db.commit()
            return

        if outcome.status == "ingested":
            graph_builder.save(graph_store_path)

        job.status = outcome.status
        job.document_id = outcome.document_id
        job.total_chunks = outcome.total_chunks
        job.new_chunks = outcome.new_chunks
        job.reused_chunks = outcome.reused_chunks
        job.warnings = outcome.warnings
        job.error = outcome.error
        db.commit()
    finally:
        db.close()
