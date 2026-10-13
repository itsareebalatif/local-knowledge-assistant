from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.vector_store import VectorStore
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.models import Chunk, Document


def delete_document(
    db: Session,
    vector_store: VectorStore,
    graph_builder: CooccurrenceGraphBuilder,
    user_id: int,
    doc_id: int,
) -> bool:
    document = db.execute(
        select(Document).where(Document.doc_id == doc_id, Document.user_id == user_id)
    ).scalar_one_or_none()
    if document is None:
        return False

    chunks = db.execute(select(Chunk).where(Chunk.doc_id == doc_id)).scalars().all()
    chunk_ids = {c.chunk_id for c in chunks}
    candidate_embedding_ids = {c.embedding_id for c in chunks if c.embedding_id is not None}

    db.delete(document)
    db.commit()

    if candidate_embedding_ids:
        # embedding_id may be shared with chunks from other documents (dedup on content hash)
        still_referenced = set(
            db.execute(
                select(Chunk.embedding_id).where(Chunk.embedding_id.in_(candidate_embedding_ids))
            ).scalars()
        )
        orphaned_embedding_ids = candidate_embedding_ids - still_referenced
        vector_store.delete(ids=list(orphaned_embedding_ids))

    if chunk_ids:
        graph_builder.remove_chunks(chunk_ids)

    return True
