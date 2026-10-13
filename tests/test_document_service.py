from __future__ import annotations

import hashlib

from sqlalchemy import text

from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.models import Chunk, Document
from app.services.document_service import delete_document


def _make_document(db_session, owner, n: int) -> Document:
    doc = Document(
        user_id=owner.user_id,
        file_path=f"doc{n}.md",
        file_name=f"doc{n}.md",
        file_type="MD",
        hash_checksum=hashlib.sha256(f"doc-service-{n}".encode()).hexdigest(),
    )
    db_session.add(doc)
    db_session.flush()
    return doc


def _make_chunk(db_session, doc: Document, index: int, content: str, embedding_id: str | None) -> Chunk:
    chunk = Chunk(
        doc_id=doc.doc_id,
        chunk_index=index,
        chunk_hash=hashlib.sha256(content.encode()).hexdigest(),
        content=content,
        token_count=len(content.split()),
        embedding_id=embedding_id,
    )
    db_session.add(chunk)
    db_session.flush()
    return chunk


def test_delete_document_removes_the_document_and_its_chunks_from_sql(db_session, user, fake_vector_store):
    doc = _make_document(db_session, user, 1)
    chunk = _make_chunk(db_session, doc, 0, "some content", embedding_id="chunk-1")
    db_session.commit()
    chunk_id, doc_id = chunk.chunk_id, doc.doc_id

    deleted = delete_document(db_session, fake_vector_store, CooccurrenceGraphBuilder(), user.user_id, doc_id)

    assert deleted is True
    assert db_session.get(Document, doc_id) is None
    assert db_session.get(Chunk, chunk_id) is None


def test_delete_document_removes_its_fts_entry(db_session, user, fake_vector_store):
    doc = _make_document(db_session, user, 2)
    chunk = _make_chunk(db_session, doc, 0, "a uniquely searchable phrase", embedding_id=None)
    db_session.commit()
    chunk_id = chunk.chunk_id

    delete_document(db_session, fake_vector_store, CooccurrenceGraphBuilder(), user.user_id, doc.doc_id)

    row = db_session.execute(text("SELECT rowid FROM chunks_fts WHERE rowid = :cid"), {"cid": chunk_id}).first()
    assert row is None


def test_delete_document_removes_its_vector_from_the_vector_store(db_session, user, fake_vector_store):
    doc = _make_document(db_session, user, 3)
    _make_chunk(db_session, doc, 0, "unique content", embedding_id="chunk-unique")
    db_session.commit()
    fake_vector_store.add(ids=["chunk-unique"], embeddings=[[1.0, 0.0]], documents=["unique content"], metadatas=[{}])

    delete_document(db_session, fake_vector_store, CooccurrenceGraphBuilder(), user.user_id, doc.doc_id)

    assert "chunk-unique" not in fake_vector_store.records


def test_delete_document_keeps_a_shared_vector_alive_when_another_chunk_still_uses_it(db_session, user, fake_vector_store):
    doc_a = _make_document(db_session, user, 4)
    doc_b = _make_document(db_session, user, 5)
    _make_chunk(db_session, doc_a, 0, "duplicated content", embedding_id="chunk-shared")
    _make_chunk(db_session, doc_b, 0, "duplicated content", embedding_id="chunk-shared")
    db_session.commit()
    fake_vector_store.add(
        ids=["chunk-shared"], embeddings=[[1.0, 0.0]], documents=["duplicated content"], metadatas=[{}]
    )

    delete_document(db_session, fake_vector_store, CooccurrenceGraphBuilder(), user.user_id, doc_a.doc_id)

    assert "chunk-shared" in fake_vector_store.records


def test_delete_document_removes_chunk_ids_from_the_graph(db_session, user, fake_vector_store):
    doc = _make_document(db_session, user, 6)
    chunk = _make_chunk(db_session, doc, 0, "Ollama content", embedding_id=None)
    db_session.commit()
    graph_builder = CooccurrenceGraphBuilder()
    graph_builder.add_chunk_entities(chunk.chunk_id, [("Ollama", "Product")])
    assert graph_builder.neighbors("Ollama", "Product") == []
    assert CooccurrenceGraphBuilder.node_key("Ollama", "Product") in graph_builder.graph

    delete_document(db_session, fake_vector_store, graph_builder, user.user_id, doc.doc_id)

    assert CooccurrenceGraphBuilder.node_key("Ollama", "Product") not in graph_builder.graph


def test_delete_document_keeps_a_graph_node_alive_when_another_chunk_still_references_it(db_session, user, fake_vector_store):
    doc_a = _make_document(db_session, user, 7)
    doc_b = _make_document(db_session, user, 8)
    chunk_a = _make_chunk(db_session, doc_a, 0, "mentions Ollama", embedding_id=None)
    chunk_b = _make_chunk(db_session, doc_b, 0, "also mentions Ollama", embedding_id=None)
    db_session.commit()
    graph_builder = CooccurrenceGraphBuilder()
    graph_builder.add_chunk_entities(chunk_a.chunk_id, [("Ollama", "Product")])
    graph_builder.add_chunk_entities(chunk_b.chunk_id, [("Ollama", "Product")])

    delete_document(db_session, fake_vector_store, graph_builder, user.user_id, doc_a.doc_id)

    key = CooccurrenceGraphBuilder.node_key("Ollama", "Product")
    assert key in graph_builder.graph
    assert graph_builder.graph.nodes[key]["chunk_ids"] == [chunk_b.chunk_id]


def test_delete_document_returns_false_for_an_unknown_doc_id(db_session, user, fake_vector_store):
    deleted = delete_document(db_session, fake_vector_store, CooccurrenceGraphBuilder(), user.user_id, 999999)
    assert deleted is False


def test_delete_document_returns_false_for_another_users_document(db_session, user, fake_vector_store):
    from app.models import User

    other = User(email="otherowner@example.com", full_name="Other", role="USER")
    db_session.add(other)
    db_session.commit()
    doc = _make_document(db_session, other, 9)
    db_session.commit()

    deleted = delete_document(db_session, fake_vector_store, CooccurrenceGraphBuilder(), user.user_id, doc.doc_id)

    assert deleted is False
    assert db_session.get(Document, doc.doc_id) is not None
