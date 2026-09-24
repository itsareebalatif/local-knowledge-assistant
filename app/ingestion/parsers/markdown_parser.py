"""Markdown parser.

Parses raw Markdown structurally (ATX headings, fenced code blocks, pipe
tables, blank-line-delimited paragraphs) via regex/line-scanning rather than
rendering to HTML first — this keeps the heading hierarchy and block
boundaries exact instead of guessing them back out of rendered markup, and
avoids pulling in a full markdown-it/mistune dependency for a fairly
mechanical grammar subset.

Setext-style headings (`Title\\n=====`) are not handled; ATX (`#`..`######`)
covers the overwhelming majority of real notes/docs.
"""

from __future__ import annotations

import re

from app.ingestion.cleaning import normalize_text
from app.ingestion.parsers.base import BaseParser
from app.ingestion.types import Block, ParsedDocument

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_FENCE = re.compile(r"^(`{3,}|~{3,})")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


class MarkdownParser(BaseParser):
    file_type = "MD"

    def parse_sync(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        text = file_bytes.decode("utf-8", errors="replace")
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        blocks: list[Block] = []
        paragraph_buf: list[str] = []
        table_buf: list[str] = []
        in_fence = False
        fence_buf: list[str] = []

        def flush_paragraph() -> None:
            if paragraph_buf:
                joined = normalize_text(" ".join(paragraph_buf))
                if joined:
                    blocks.append(Block(type="paragraph", text=joined))
                paragraph_buf.clear()

        def flush_table() -> None:
            if table_buf:
                blocks.append(Block(type="table", text=normalize_text("\n".join(table_buf))))
                table_buf.clear()

        for raw_line in lines:
            line = raw_line.rstrip()

            if in_fence:
                fence_buf.append(raw_line)
                if _FENCE.match(line.strip()):
                    in_fence = False
                    blocks.append(Block(type="code", text=normalize_text("\n".join(fence_buf))))
                    fence_buf = []
                continue

            if _FENCE.match(line.strip()):
                flush_paragraph()
                flush_table()
                in_fence = True
                fence_buf = [raw_line]
                continue

            heading_match = _ATX_HEADING.match(line.strip())
            if heading_match:
                flush_paragraph()
                flush_table()
                level = len(heading_match.group(1))
                blocks.append(Block(type="heading", text=heading_match.group(2).strip(), level=level))
                continue

            if _TABLE_ROW.match(line):
                flush_paragraph()
                table_buf.append(line.strip())
                continue
            flush_table()

            if not line.strip():
                flush_paragraph()
                continue

            paragraph_buf.append(line.strip())

        flush_paragraph()
        flush_table()
        if in_fence and fence_buf:
            # Unterminated fence — still capture what we have rather than dropping it.
            blocks.append(Block(type="code", text=normalize_text("\n".join(fence_buf))))

        return ParsedDocument(blocks=blocks, meta={"source_format": "markdown"})
