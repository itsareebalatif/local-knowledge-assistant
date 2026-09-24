"""Ingestion pipeline: bytes in, IngestResult (parsed + chunked) out.

This is deliberately the only place that (a) picks a parser by extension and
(b) catches parsing exceptions. Every other module in app/ingestion is a pure
function/class over data — this is where NFR-7 ("damaged or unsupported
files must be safely skipped and logged without stopping the rest of the
import process") is actually enforced: a bad file becomes a failed
`IngestResult`, never a raised exception, so a caller looping over many files
can keep going. Chunk-level SHA-256 hashing / DB persistence is a separate
step (not part of this task) and consumes this function's output.
"""

from __future__ import annotations

import asyncio
import logging

from app.ingestion.chunker import HeaderAwareChunker
from app.ingestion.exceptions import IngestionError, UnsupportedFileTypeError
from app.ingestion.parsers import file_type_for_filename, get_parser
from app.ingestion.types import IngestResult

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 512


async def ingest_bytes(file_bytes: bytes, filename: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> IngestResult:
    """Parse + chunk one file's bytes. Never raises — failures come back as
    `IngestResult(ok=False, error=...)`."""
    try:
        file_type = file_type_for_filename(filename)
    except UnsupportedFileTypeError as exc:
        logger.warning("Skipping unsupported file %s: %s", filename, exc)
        return IngestResult(filename=filename, file_type="UNKNOWN", ok=False, error=str(exc))

    try:
        parser = get_parser(file_type)
        parsed = await parser.parse(file_bytes, filename)
        chunks = HeaderAwareChunker(max_tokens=max_tokens).chunk(parsed.blocks)
    except IngestionError as exc:
        logger.warning("Failed to ingest %s (%s): %s", filename, file_type, exc)
        return IngestResult(filename=filename, file_type=file_type, ok=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - a single bad file must never kill the batch
        logger.exception("Unexpected error ingesting %s (%s)", filename, file_type)
        return IngestResult(filename=filename, file_type=file_type, ok=False, error=f"Unexpected error: {exc}")

    return IngestResult(
        filename=filename,
        file_type=file_type,
        ok=True,
        chunks=chunks,
        meta=parsed.meta,
        warnings=parsed.warnings,
    )


async def ingest_many(files: list[tuple[bytes, str]], max_tokens: int = DEFAULT_MAX_TOKENS) -> list[IngestResult]:
    """Ingest several files concurrently. One failure never cancels the rest."""
    return await asyncio.gather(*(ingest_bytes(data, name, max_tokens) for data, name in files))
