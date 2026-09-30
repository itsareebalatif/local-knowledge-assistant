from __future__ import annotations

import hashlib

from sqlalchemy import text

from app.models import Chunk, Document


def _make_document(db_session, user, suffix: str) -> Document:
    doc = Document(
        user_id=user.user_id,
        file_path=f"{suffix}.md",
        file_name=f"{suffix}.md",
        file_type="MD",
        hash_checksum=hashlib.sha256(suffix.encode()).hexdigest(),
    )
    db_session.add(doc)
    db_session.flush()
    return doc


def _make_chunk(db_session, doc: Document, content: str, index: int = 0) -> Chunk:
    chunk = Chunk(
        doc_id=doc.doc_id,
        chunk_index=index,
        chunk_hash=hashlib.sha256(f"{doc.doc_id}:{index}:{content}".encode()).hexdigest(),
        content=content,
        token_count=len(content.split()),
    )
    db_session.add(chunk)
    db_session.commit()
    return chunk


def test_fts_table_and_triggers_exist(db_session):
    rows = db_session.execute(
        text("SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')")
    ).scalars().all()
    assert "chunks_fts" in rows
    assert "chunks_fts_after_insert" in rows
    assert "chunks_fts_after_update" in rows
    assert "chunks_fts_after_delete" in rows


def test_insert_trigger_indexes_new_chunk(db_session, user):
    doc = _make_document(db_session, user, "a")
    chunk = _make_chunk(db_session, doc, "Ollama runs language models entirely on the local machine.")

    hits = db_session.execute(
        text("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'Ollama'")
    ).scalars().all()
    assert hits == [chunk.chunk_id]


def test_match_ranks_relevant_chunk_above_unrelated_one(db_session, user):
    doc = _make_document(db_session, user, "b")
    relevant = _make_chunk(db_session, doc, "Ollama is a tool for running local language models.", index=0)
    _make_chunk(db_session, doc, "The weather today is sunny with a light breeze.", index=1)

    # bm25() returns lower (more negative) = more relevant, so ascending order.
    rows = db_session.execute(
        text("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'Ollama' ORDER BY bm25(chunks_fts) ASC")
    ).scalars().all()
    assert rows == [relevant.chunk_id]


def test_update_trigger_reindexes_changed_content(db_session, user):
    doc = _make_document(db_session, user, "c")
    chunk = _make_chunk(db_session, doc, "original wording about pineapples")

    chunk.content = "completely different wording about rockets"
    db_session.commit()

    assert db_session.execute(
        text("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'pineapples'")
    ).scalars().all() == []
    assert db_session.execute(
        text("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'rockets'")
    ).scalars().all() == [chunk.chunk_id]


def test_delete_trigger_removes_chunk_from_index(db_session, user):
    doc = _make_document(db_session, user, "d")
    chunk = _make_chunk(db_session, doc, "a chunk that will be deleted shortly")

    db_session.delete(chunk)
    db_session.commit()

    assert db_session.execute(
        text("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'deleted'")
    ).scalars().all() == []
