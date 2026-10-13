
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BlockType = Literal["heading", "paragraph", "table", "code"]


@dataclass
class Block:

    type: BlockType
    text: str
    level: int = 0


@dataclass
class ParsedDocument:
    blocks: list[Block] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ChunkDraft:
    chunk_index: int
    content: str
    token_count: int
    heading_path: list[str] = field(default_factory=list)


@dataclass
class IngestResult:
    filename: str
    file_type: str
    ok: bool
    chunks: list[ChunkDraft] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
