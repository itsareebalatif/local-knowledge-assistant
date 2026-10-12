from __future__ import annotations

import asyncio
import logging

from app.config import get_settings
from app.ingestion.chunker import HeaderAwareChunker
from app.ingestion.exceptions import IngestionError, UnsupportedFileTypeError
from app.ingestion.parsers import file_type_for_filename, get_parser
from app.ingestion.types import IngestResult

logger = logging.getLogger(__name__)

# bound once at import time from CHUNK_TOKEN_SIZE
DEFAULT_MAX_TOKENS = get_settings().chunk_token_size


async def ingest_bytes(file_bytes: bytes, filename: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> IngestResult:
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
    return await asyncio.gather(*(ingest_bytes(data, name, max_tokens) for data, name in files))
