"""Plain-text parser. No heading structure — every blank-line-delimited
paragraph is a body block at level 0."""

from __future__ import annotations

from app.ingestion.cleaning import normalize_text
from app.ingestion.exceptions import ParsingError
from app.ingestion.parsers.base import BaseParser
from app.ingestion.types import Block, ParsedDocument


class TxtParser(BaseParser):
    file_type = "TXT"

    def parse_sync(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as exc:  # pragma: no cover - latin-1 practically never fails
                raise ParsingError(f"Could not decode {filename} as text: {exc}") from exc

        text = normalize_text(text)
        blocks = [Block(type="paragraph", text=para) for para in text.split("\n\n") if para.strip()]
        return ParsedDocument(blocks=blocks, meta={"source_format": "txt"})
