from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.hashing import sha256_bytes, sha256_text
from app.ingestion.exceptions import UnsupportedFileTypeError
from app.ingestion.parsers import file_type_for_filename
from app.ingestion.pipeline import DEFAULT_MAX_TOKENS, ingest_bytes
from app.models import Chunk, Document
from app.services.types import IngestOutcome

logger = logging.getLogger(__name__)


async def ingest_document(
    db: Session,
    user_id: int,
    file_bytes: bytes,
    filename: str,
    file_path: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> IngestOutcome:

    file_hash = sha256_bytes(file_bytes)

    try:
        file_type = file_type_for_filename(filename)
    except UnsupportedFileTypeError as exc:
        logger.warning("Rejecting unsupported file %s before any DB write: %s", filename, exc)
        return IngestOutcome(filename=filename, status="unsupported", file_hash=file_hash, error=str(exc))


    existing_doc = db.execute(
        select(Document).where(Document.user_id == user_id, Document.hash_checksum == file_hash)
    ).scalar_one_or_none()
    if existing_doc is not None:
        logger.info(
            "Duplicate file for user_id=%s (doc_id=%s, hash=%s…%s) — halting before parse/chunk",
            user_id,
            existing_doc.doc_id,
            file_hash[:8],
            file_hash[-4:],
        )
        return IngestOutcome(
            filename=filename,
            status="duplicate_file",
            file_hash=file_hash,
            document_id=existing_doc.doc_id,
        )

    document = Document(
        user_id=user_id,
        file_path=file_path or filename,
        file_name=filename,
        file_type=file_type,
        hash_checksum=file_hash,
    )
    db.add(document)
    db.flush() 


    result = await ingest_bytes(file_bytes, filename, max_tokens=max_tokens)
    if not result.ok:
        db.rollback() 
        logger.warning("Parsing failed for %s: %s", filename, result.error)
        return IngestOutcome(filename=filename, status="parse_failed", file_hash=file_hash, error=result.error)




    new_chunks = 0
    reused_chunks = 0
    pending_embedding_chunk_ids: list[int] = []

    for draft in result.chunks:
        chunk_hash = sha256_text(draft.content)
        existing_chunk = db.execute(
            select(Chunk).where(Chunk.chunk_hash == chunk_hash, Chunk.embedding_id.is_not(None))
        ).scalars().first()

        chunk_row = Chunk(
            doc_id=document.doc_id,
            chunk_index=draft.chunk_index,
            chunk_hash=chunk_hash,
            content=draft.content,
            token_count=draft.token_count,
        )
        if existing_chunk is not None:
            chunk_row.embedding_id = existing_chunk.embedding_id
            reused_chunks += 1
        else:
            new_chunks += 1
        db.add(chunk_row)
        db.flush() 
        if chunk_row.embedding_id is None:
            pending_embedding_chunk_ids.append(chunk_row.chunk_id)

    db.commit()

    logger.info(
        "Ingested %s: doc_id=%s, %d chunks (%d new, %d reused embeddings)",
        filename,
        document.doc_id,
        len(result.chunks),
        new_chunks,
        reused_chunks,
    )
    return IngestOutcome(
        filename=filename,
        status="ingested",
        file_hash=file_hash,
        document_id=document.doc_id,
        total_chunks=len(result.chunks),
        new_chunks=new_chunks,
        reused_chunks=reused_chunks,
        pending_embedding_chunk_ids=pending_embedding_chunk_ids,
        warnings=result.warnings,
    )
