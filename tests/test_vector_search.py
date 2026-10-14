from __future__ import annotations

from app.search.vector_search import vector_search

USER_ID = 1


async def test_maps_vector_store_hits_to_candidate_chunks(db_session, user, make_chunk, fake_embedder, fake_vector_store):
    chunk1 = make_chunk("first chunk text")
    chunk2 = make_chunk("second chunk text")
    fake_vector_store.add(
        ids=[f"chunk-{chunk1.chunk_id}", f"chunk-{chunk2.chunk_id}"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=[chunk1.content, chunk2.content],
        metadatas=[
            {"doc_id": chunk1.doc_id, "chunk_id": chunk1.chunk_id, "user_id": user.user_id},
            {"doc_id": chunk2.doc_id, "chunk_id": chunk2.chunk_id, "user_id": user.user_id},
        ],
    )

    hits = await vector_search(db_session, fake_embedder, fake_vector_store, "some query", top_k=5, user_id=user.user_id)

    assert len(hits) == 2
    assert hits[0].chunk_id == chunk1.chunk_id
    assert hits[0].doc_id == chunk1.doc_id
    assert hits[0].content == "first chunk text"
    assert hits[0].sources == ["vector"]
    assert len(fake_embedder.calls) == 1
    assert fake_embedder.calls[0] == ["some query"]


async def test_respects_top_k(db_session, user, make_chunk, fake_embedder, fake_vector_store):
    chunks = [make_chunk(c) for c in ("a", "b", "c")]
    fake_vector_store.add(
        ids=[f"chunk-{c.chunk_id}" for c in chunks],
        embeddings=[[1.0], [2.0], [3.0]],
        documents=[c.content for c in chunks],
        metadatas=[{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "user_id": user.user_id} for c in chunks],
    )
    hits = await vector_search(db_session, fake_embedder, fake_vector_store, "q", top_k=2, user_id=user.user_id)
    assert len(hits) == 2


async def test_empty_store_returns_empty_list(db_session, fake_embedder, fake_vector_store):
    assert await vector_search(db_session, fake_embedder, fake_vector_store, "anything", top_k=5, user_id=USER_ID) == []


async def test_does_not_return_another_users_vectors(db_session, user, make_chunk, fake_embedder, fake_vector_store):
    chunk = make_chunk("someone else's private chunk")
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        # A real chunk, but stored under a different user_id than the query — the point of
        # this test is the user filter, not the row's existence, so chunk_id must be real.
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": 999}],
    )
    hits = await vector_search(db_session, fake_embedder, fake_vector_store, "anything", top_k=5, user_id=user.user_id)
    assert hits == []


async def test_excludes_a_vector_whose_chunk_no_longer_exists_in_sql(db_session, user, fake_embedder, fake_vector_store):
    # Regression for a real, confirmed-live bug: a document deletion that
    # cleaned up SQL but left its vectors behind in Chroma (for any reason —
    # a bug, a crash mid-delete, a manual edit) kept being retrieved and
    # answered from indefinitely, with citations pointing at a document that
    # no longer existed. No chunk with this id was ever created via
    # make_chunk, so `chunks` has no matching row — simulating exactly that
    # orphaned-vector state.
    fake_vector_store.add(
        ids=["chunk-99999"],
        embeddings=[[1.0, 0.0]],
        documents=["stale content from a deleted document"],
        metadatas=[{"doc_id": 99999, "chunk_id": 99999, "user_id": user.user_id}],
    )
    hits = await vector_search(db_session, fake_embedder, fake_vector_store, "anything", top_k=5, user_id=user.user_id)
    assert hits == []


async def test_a_live_chunk_is_not_excluded_by_the_orphan_check(db_session, user, make_chunk, fake_embedder, fake_vector_store):
    # The orphan filter must not accidentally exclude real, live chunks —
    # paired with the test above so a bug that excludes everything (instead
    # of just orphans) would also be caught.
    chunk = make_chunk("a perfectly real, still-existing chunk")
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": user.user_id}],
    )
    hits = await vector_search(db_session, fake_embedder, fake_vector_store, "anything", top_k=5, user_id=user.user_id)
    assert len(hits) == 1
    assert hits[0].chunk_id == chunk.chunk_id
