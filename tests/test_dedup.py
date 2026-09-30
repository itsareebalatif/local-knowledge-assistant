from __future__ import annotations

from sqlalchemy import select

from app.hashing import sha256_bytes, sha256_text
from app.models import Chunk, Document, User, UserAuth
from app.services.ingest_service import ingest_document


async def test_new_file_is_ingested_with_chunks_pending_embedding(db_session, user, sample_markdown_bytes):
    outcome = await ingest_document(db_session, user.user_id, sample_markdown_bytes, "readme.md")

    assert outcome.status == "ingested"
    assert outcome.document_id is not None
    assert outcome.total_chunks > 0
    assert outcome.new_chunks == outcome.total_chunks
    assert outcome.reused_chunks == 0
    assert len(outcome.pending_embedding_chunk_ids) == outcome.total_chunks
    assert outcome.file_hash == sha256_bytes(sample_markdown_bytes)

    doc = db_session.get(Document, outcome.document_id)
    assert doc.hash_checksum == outcome.file_hash
    chunks = db_session.execute(select(Chunk).where(Chunk.doc_id == doc.doc_id)).scalars().all()
    assert len(chunks) == outcome.total_chunks
    assert all(c.embedding_id is None for c in chunks)


async def test_duplicate_file_halts_before_parsing(db_session, user, sample_markdown_bytes, monkeypatch):
    first = await ingest_document(db_session, user.user_id, sample_markdown_bytes, "readme.md")
    assert first.status == "ingested"

    call_count = 0
    import app.services.ingest_service as svc

    real_ingest_bytes = svc.ingest_bytes

    async def _spy(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await real_ingest_bytes(*args, **kwargs)

    monkeypatch.setattr(svc, "ingest_bytes", _spy)

    second = await ingest_document(db_session, user.user_id, sample_markdown_bytes, "readme.md")

    assert second.status == "duplicate_file"
    assert second.document_id == first.document_id
    assert call_count == 0, "parser/chunker must never run for a duplicate file"

    docs = db_session.execute(select(Document).where(Document.user_id == user.user_id)).scalars().all()
    assert len(docs) == 1, "no second Document row should be created for a duplicate"


async def test_same_file_different_users_both_ingest(db_session, user, sample_markdown_bytes):
    other = User(email="user@gmail.com", full_name="Other User", role="USER")
    db_session.add(other)
    db_session.flush()
    db_session.add(UserAuth(user_id=other.user_id, password_hash="hashed"))
    db_session.commit()

    first = await ingest_document(db_session, user.user_id, sample_markdown_bytes, "readme.md")
    second = await ingest_document(db_session, other.user_id, sample_markdown_bytes, "readme.md")

    assert first.status == "ingested"
    assert second.status == "ingested"
    assert first.document_id != second.document_id


async def test_chunk_level_dedup_reuses_existing_embedding(db_session, user):
    shared_paragraph = "This exact paragraph will appear in two different documents for the duplication test."
    doc_a_bytes = f"# Doc A\n\n{shared_paragraph}\n\n## Unique To A\n\nOnly in document A.".encode()
    doc_b_bytes = f"# Doc B\n\n{shared_paragraph}\n\n## Unique To B\n\nOnly in document B.".encode()

    first = await ingest_document(db_session, user.user_id, doc_a_bytes, "a.md")
    assert first.status == "ingested"

    shared_hash = sha256_text(shared_paragraph)
    shared_chunk = db_session.execute(select(Chunk).where(Chunk.chunk_hash == shared_hash)).scalars().first()
    assert shared_chunk is not None
    shared_chunk.embedding_id = "vec-shared-123"
    db_session.commit()

    second = await ingest_document(db_session, user.user_id, doc_b_bytes, "b.md")
    assert second.status == "ingested"
    assert second.reused_chunks == 1
    assert second.new_chunks == second.total_chunks - 1

    doc_b_chunks = db_session.execute(select(Chunk).where(Chunk.doc_id == second.document_id)).scalars().all()
    reused = next(c for c in doc_b_chunks if c.chunk_hash == shared_hash)
    assert reused.embedding_id == "vec-shared-123"
    assert reused.chunk_id not in second.pending_embedding_chunk_ids

    other_chunk = next(c for c in doc_b_chunks if c.chunk_hash != shared_hash)
    assert other_chunk.embedding_id is None
    assert other_chunk.chunk_id in second.pending_embedding_chunk_ids


async def test_parse_failure_leaves_no_document_row(db_session, user):
    outcome = await ingest_document(db_session, user.user_id, b"not a real pdf", "broken.pdf")

    assert outcome.status == "parse_failed"
    assert outcome.error

    docs = db_session.execute(select(Document).where(Document.user_id == user.user_id)).scalars().all()
    assert docs == [], "a failed parse must not leave a dangling Document row blocking future retries"


async def test_unsupported_file_type_rejected_before_any_db_write(db_session, user):
    outcome = await ingest_document(db_session, user.user_id, b"whatever", "archive.zip")

    assert outcome.status == "unsupported"
    docs = db_session.execute(select(Document).where(Document.user_id == user.user_id)).scalars().all()
    assert docs == []
