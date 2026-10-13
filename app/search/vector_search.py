from __future__ import annotations

from app.embeddings.embedder import EmbedderBackend
from app.embeddings.vector_store import VectorStore
from app.search.types import CandidateChunk


async def vector_search(
    embedder: EmbedderBackend, vector_store: VectorStore, query: str, top_k: int, user_id: int
) -> list[CandidateChunk]:
    vectors = await embedder.embed([query], input_type="search_query")
    if not vectors:
        return []

    hits = vector_store.query(vectors[0], top_k=top_k, where={"user_id": user_id})
    return [
        CandidateChunk(
            chunk_id=hit["metadata"]["chunk_id"],
            doc_id=hit["metadata"]["doc_id"],
            content=hit["document"],
            score=hit["distance"],
            sources=["vector"],
        )
        for hit in hits
    ]
