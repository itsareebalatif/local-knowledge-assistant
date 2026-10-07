"""Dense vector search over the Chroma collection (FR-3.3).

Reuses the metadata already stored at embedding time (embedding_service.py
writes doc_id/chunk_id/user_id into Chroma's metadatas, and the chunk text
itself into documents) — no extra DB round-trip needed to build a
CandidateChunk.

`user_id` is required, not optional: it's passed straight to Chroma's
native metadata `where` filter, so a user's query can only ever match
vectors belonging to their own documents. This is one of the two places
(the other is bm25_search.py's SQL filter and graph_expansion.py's chunk
lookup) that actually enforces per-user data isolation — see the "Known
limitations" note this closed in the README.
"""

from __future__ import annotations

from app.embeddings.embedder import EmbedderBackend
from app.embeddings.vector_store import VectorStore
from app.search.types import CandidateChunk


async def vector_search(
    embedder: EmbedderBackend, vector_store: VectorStore, query: str, top_k: int, user_id: int
) -> list[CandidateChunk]:
    # "search_query" (not "search_document") — Cohere's API distinguishes
    # the two for better retrieval quality; Ollama ignores the parameter.
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
