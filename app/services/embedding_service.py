
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.embedder import EmbedderBackend
from app.embeddings.vector_store import VectorStore
from app.models import Chunk, Document

logger = logging.getLogger(__name__)


async def embed_pending_chunks(
    db: Session,
    embedder: EmbedderBackend,
    vector_store: VectorStore,
    chunk_ids: list[int],
) -> int:

    if not chunk_ids:
        return 0

    rows = db.execute(
        select(Chunk, Document.user_id).join(Document, Chunk.doc_id == Document.doc_id).where(Chunk.chunk_id.in_(chunk_ids))
    ).all()
    rows = [(chunk, owner_id) for chunk, owner_id in rows if chunk.embedding_id is None]
    if not rows:
        return 0

    chunks = [chunk for chunk, _ in rows]
    texts = [c.content for c in chunks]
    vectors = await embedder.embed(texts, input_type="search_document")

    vector_ids = [f"chunk-{c.chunk_id}" for c in chunks]
    # user_id is read straight from the DB (via the chunk's owning document)
    # rather than trusted from a caller argument — this is the metadata that
    # app.search.vector_search's where={"user_id": ...} filter relies on to
    # keep one user's query from ever matching another user's vectors.
    metadatas = [{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": owner_id} for chunk, owner_id in rows]
    vector_store.add(ids=vector_ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    for chunk, vector_id in zip(chunks, vector_ids):
        chunk.embedding_id = vector_id
    db.commit()

    logger.info("Embedded %d chunks", len(chunks))
    return len(chunks)
