from __future__ import annotations

from app.search.vector_search import vector_search


async def test_maps_vector_store_hits_to_candidate_chunks(fake_embedder, fake_vector_store):
    fake_vector_store.add(
        ids=["chunk-1", "chunk-2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["first chunk text", "second chunk text"],
        metadatas=[{"doc_id": 10, "chunk_id": 1}, {"doc_id": 10, "chunk_id": 2}],
    )

    hits = await vector_search(fake_embedder, fake_vector_store, "some query", top_k=5)

    assert len(hits) == 2
    assert hits[0].chunk_id == 1
    assert hits[0].doc_id == 10
    assert hits[0].content == "first chunk text"
    assert hits[0].sources == ["vector"]
    assert len(fake_embedder.calls) == 1
    assert fake_embedder.calls[0] == ["some query"]


async def test_respects_top_k(fake_embedder, fake_vector_store):
    fake_vector_store.add(
        ids=["a", "b", "c"],
        embeddings=[[1.0], [2.0], [3.0]],
        documents=["a", "b", "c"],
        metadatas=[{"doc_id": 1, "chunk_id": 1}, {"doc_id": 1, "chunk_id": 2}, {"doc_id": 1, "chunk_id": 3}],
    )
    hits = await vector_search(fake_embedder, fake_vector_store, "q", top_k=2)
    assert len(hits) == 2


async def test_empty_store_returns_empty_list(fake_embedder, fake_vector_store):
    assert await vector_search(fake_embedder, fake_vector_store, "anything", top_k=5) == []
