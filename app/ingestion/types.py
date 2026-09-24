"""Shared data types passed between parsers, the cleaner, and the chunker.

Keeping these as plain dataclasses (rather than the ORM models) is deliberate:
the ingestion package has no idea a database exists. It turns raw file bytes
into `ChunkDraft`s; the caller decides how to persist them (SHA-256 hashing /
dedup and the actual `Chunk` row insert are a separate pipeline step).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BlockType = Literal["heading", "paragraph", "table", "code"]


@dataclass
class Block:
    """One structural unit of a parsed document, in document order.

    `level` is only meaningful for headings: 1-6, mirroring H1-H6 / Markdown
    `#`..`######` / DOCX "Heading N" styles / font-size-inferred PDF headings.
    """

    type: BlockType
    text: str
    level: int = 0


@dataclass
class ParsedDocument:
    """Output of a format parser: a flat, ordered list of blocks + metadata."""

    blocks: list[Block] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ChunkDraft:
    """Output of the chunker: not yet a DB row (no chunk_hash / doc_id yet)."""

    chunk_index: int
    content: str
    token_count: int
    heading_path: list[str] = field(default_factory=list)


@dataclass
class IngestResult:
    """Top-level pipeline result for one file — success or a logged failure.

    Modeled directly on NFR-7 ("damaged or unsupported files must be safely
    skipped and logged without stopping the rest of the import process"): the
    pipeline never raises for a single bad file, it returns this instead.
    """

    filename: str
    file_type: str
    ok: bool
    chunks: list[ChunkDraft] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
