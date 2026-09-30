
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.embedder import OllamaEmbedder
from app.embeddings.vector_store import VectorStore
from app.models import Chunk

logger = logging.getLogger(__name__)


async def embed_pending_chunks(
    db: Session,
    embedder: OllamaEmbedder,
    vector_store: VectorStore,
    chunk_ids: list[int],
) -> int:
    
    if not chunk_ids:
        return 0

    chunks = db.execute(select(Chunk).where(Chunk.chunk_id.in_(chunk_ids))).scalars().all()
    chunks = [c for c in chunks if c.embedding_id is None]
    if not chunks:
        return 0

    texts = [c.content for c in chunks]
    vectors = await embedder.embed(texts)

    vector_ids = [f"chunk-{c.chunk_id}" for c in chunks]
    metadatas = [{"doc_id": c.doc_id, "chunk_id": c.chunk_id} for c in chunks]
    vector_store.add(ids=vector_ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    for chunk, vector_id in zip(chunks, vector_ids):
        chunk.embedding_id = vector_id
    db.commit()

    logger.info("Embedded %d chunks", len(chunks))
    return len(chunks)
