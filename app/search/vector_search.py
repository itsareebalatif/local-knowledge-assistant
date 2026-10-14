from __future__ import annotations

import asyncio

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.embedder import EmbedderBackend
from app.embeddings.vector_store import VectorStore
from app.models import Chunk
from app.search.types import CandidateChunk


async def vector_search(
    db: Session, embedder: EmbedderBackend, vector_store: VectorStore, query: str, top_k: int, user_id: int
) -> list[CandidateChunk]:
    vectors = await embedder.embed([query], input_type="search_query")
    if not vectors:
        return []

    hits = vector_store.query(vectors[0], top_k=top_k, where={"user_id": user_id})
    if not hits:
        return []

    # The vector store is a separate system from the SQL database, kept in
    # sync by application code (ingestion writes to both, deletion is meant
    # to clean up both) rather than a single transaction — so a bug, crash,
    # or manual intervention on either side can leave a vector in Chroma
    # whose chunk/document no longer exists in SQL. Confirmed live: exactly
    # this happened, and the stale vector kept being retrieved and answered
    # from indefinitely, with no error, because nothing downstream ever
    # checked SQL again. chunk.doc_id has ON DELETE CASCADE from
    # documents, so a chunk_id still present in `chunks` guarantees its
    # document still exists too — one existence check covers both.
    candidate_chunk_ids = {hit["metadata"]["chunk_id"] for hit in hits}

    def _query_live_chunk_ids() -> set[int]:
        # Run on a worker thread, not the event loop: this function runs
        # concurrently with BM25/graph (via asyncio.gather in
        # retrieval_service.py), which already offload their own db access
        # the same way — a single SQLAlchemy Session isn't safe to touch
        # from more than one place at once.
        return set(db.execute(select(Chunk.chunk_id).where(Chunk.chunk_id.in_(candidate_chunk_ids))).scalars())

    live_chunk_ids = await asyncio.to_thread(_query_live_chunk_ids)

    return [
        CandidateChunk(
            chunk_id=hit["metadata"]["chunk_id"],
            doc_id=hit["metadata"]["doc_id"],
            content=hit["document"],
            score=hit["distance"],
            sources=["vector"],
        )
        for hit in hits
        if hit["metadata"]["chunk_id"] in live_chunk_ids
    ]
