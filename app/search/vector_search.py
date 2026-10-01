
from __future__ import annotations

from app.embeddings.embedder import OllamaEmbedder
from app.embeddings.vector_store import VectorStore
from app.search.types import CandidateChunk


async def vector_search(embedder: OllamaEmbedder, vector_store: VectorStore, query: str, top_k: int) -> list[CandidateChunk]:
    vectors = await embedder.embed([query])
    if not vectors:
        return []

    hits = vector_store.query(vectors[0], top_k=top_k)
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
