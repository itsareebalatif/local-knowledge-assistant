#orchestrator of ingestion pipeline
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.embedder import EmbedderBackend
from app.embeddings.vector_store import VectorStore
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.models import Chunk
from app.services.embedding_service import embed_pending_chunks
from app.services.graph_service import build_graph_for_chunk
from app.services.ingest_service import ingest_document
from app.services.types import IngestOutcome


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
