
from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass

import pdfplumber
from pypdf import PdfReader

from app.ingestion.cleaning import normalize_text, strip_repeated_page_furniture
from app.ingestion.exceptions import ParsingError
from app.ingestion.parsers.base import BaseParser
from app.ingestion.types import Block, ParsedDocument

_LINE_Y_TOLERANCE = 2.0
_PARAGRAPH_GAP_FACTOR = 1.6
_HEADER_BAND_HEIGHT_RATIO = 0.08
_HEADER_BAND_TOLERANCE = 3.0
_HEADER_MIN_PAGE_RATIO = 0.3


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    top: float
    bottom: float


class PdfParser(BaseParser):
    file_type = "PDF"

    def parse_sync(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        try:
            return self._parse_with_pdfplumber(file_bytes)
        except ParsingError:
            raise
        except Exception as plumber_error:
            try:
                return self._parse_with_pypdf(file_bytes)
            except Exception as pypdf_error:
                raise ParsingError(
                    f"Could not parse PDF {filename} (pdfplumber: {plumber_error}; pypdf: {pypdf_error})"
                ) from pypdf_error

    # -- pdfplumber path: layout-aware, with heading detection -----------------

    def _parse_with_pdfplumber(self, file_bytes: bytes) -> ParsedDocument:
        warnings: list[str] = []
        page_lines: list[list[_Line]] = []
        page_heights: list[float] = []
        page_tables: list[list[list[str]]] = []

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                raise ParsingError("PDF has no pages")
            for page in pdf.pages:
                page_lines.append(self._extract_lines(page))
                page_heights.append(page.height)
                try:
                    page_tables.append(page.extract_tables())
                except Exception:  # a broken table extractor shouldn't sink the whole page
                    page_tables.append([])
                    warnings.append(f"Table extraction failed on page {page.page_number}")

        page_lines = self._strip_positional_running_headers(page_lines, page_heights)

        all_lines = [line for lines in page_lines for line in lines]
        if not all_lines:
            return ParsedDocument(blocks=[], meta={"source_format": "pdf", "page_count": str(len(page_lines))},
                                   warnings=["No extractable text (scanned/image-only PDF?)"])

        body_size = self._dominant_size(all_lines)

        raw_page_texts = ["\n".join(line.text for line in lines) for lines in page_lines]
        cleaned_page_texts = strip_repeated_page_furniture(raw_page_texts)
        kept_lines_per_page = [
            {line.strip() for line in cleaned.split("\n") if line.strip()} for cleaned in cleaned_page_texts
        ]

        blocks: list[Block] = []
        for lines, kept, tables in zip(page_lines, kept_lines_per_page, page_tables):
            # `kept` went through normalize_text() inside strip_repeated_page_furniture
            # (NFKC — e.g. expands "…" into multiple characters), so a raw
            # ln.text never matches it exactly. Confirmed live: this silently
            # dropped any line containing such a character from the whole
            # document. Normalize this side the same way before comparing.
            blocks.extend(
                self._lines_to_blocks(
                    [ln for ln in lines if normalize_text(ln.text.strip()) in kept], body_size
                )
            )
            for table in tables:
                text = self._table_to_text(table)
                if text:
                    blocks.append(Block(type="table", text=text))

        return ParsedDocument(
            blocks=blocks,
            meta={"source_format": "pdf", "page_count": str(len(page_lines))},
            warnings=warnings,
        )

    @classmethod
    def _strip_positional_running_headers(
        cls, page_lines: list[list[_Line]], page_heights: list[float]
    ) -> list[list[_Line]]:
        # A running header's text changes page to page (chapter title, page
        # number), so text-frequency matching (strip_repeated_page_furniture)
        # can't reliably catch it — but its vertical position is identical
        # on every page it appears on. Confirmed live: a real running header
        # sat at exactly the same `top` (to 0.1pt) on every affected page,
        # distinct from front-matter and chapter-opener pages.
        candidates = []
        for i, lines in enumerate(page_lines):
            if not lines or i >= len(page_heights) or page_heights[i] <= 0:
                continue
            first = lines[0]
            if first.top / page_heights[i] <= _HEADER_BAND_HEIGHT_RATIO and len(first.text.split()) <= 8:
                candidates.append((i, first.top))

        min_pages = max(3, int(len(page_lines) * _HEADER_MIN_PAGE_RATIO))
        if len(candidates) < min_pages:
            return page_lines

        tops = sorted(top for _, top in candidates)
        median_top = tops[len(tops) // 2]
        header_page_indices = {i for i, top in candidates if abs(top - median_top) <= _HEADER_BAND_TOLERANCE}
        if len(header_page_indices) < min_pages:
            return page_lines

        return [lines[1:] if i in header_page_indices and lines else lines for i, lines in enumerate(page_lines)]

    def _extract_lines(self, page) -> list[_Line]:
        chars = page.chars
        if not chars:
            return []
        rows: list[list[dict]] = []
        for ch in sorted(chars, key=lambda c: (round(c["top"] / _LINE_Y_TOLERANCE), c["x0"])):
            row_key = round(ch["top"] / _LINE_Y_TOLERANCE)
            if rows and rows[-1] and round(rows[-1][-1]["top"] / _LINE_Y_TOLERANCE) == row_key:
                rows[-1].append(ch)
            else:
                rows.append([ch])

        lines: list[_Line] = []
        for row in rows:
            row_sorted = sorted(row, key=lambda c: c["x0"])
            text = self._row_text(row_sorted).strip()
            if not text:
                continue
            sizes = [c["size"] for c in row_sorted]
            fontnames = [c.get("fontname", "") for c in row_sorted]
            lines.append(
                _Line(
                    text=text,
                    size=sum(sizes) / len(sizes),
                    bold=any("bold" in f.lower() for f in fontnames),
                    top=min(c["top"] for c in row_sorted),
                    bottom=max(c["bottom"] for c in row_sorted),
                )
            )
        return lines

    @staticmethod
    def _row_text(row_sorted: list[dict]) -> str:
        # Many PDFs (ebook conversions especially) position words with a
        # horizontal gap instead of an explicit space glyph, so chars.text
        # alone loses every word boundary. A gap over ~10% of the
        # character's own size reliably means "next word" rather than
        # kerning — confirmed live against a real file: intra-word gaps
        # were ~0.001-0.003x size, word-boundary gaps ~0.26x size.
        parts: list[str] = []
        prev: dict | None = None
        for ch in row_sorted:
            if prev is not None:
                gap = ch["x0"] - prev["x1"]
                if gap > prev["size"] * 0.1 and not parts[-1].endswith(" ") and ch["text"] != " ":
                    parts.append(" ")
            parts.append(ch["text"])
            prev = ch
        return "".join(parts)

    def _dominant_size(self, lines: list[_Line]) -> float:
        # Round to the nearest 0.5pt so near-identical body sizes bucket together.
        rounded = Counter(round(line.size * 2) / 2 for line in lines)
        return rounded.most_common(1)[0][0]

    def _classify_heading_level(self, line: _Line, body_size: float) -> int:
        if body_size <= 0:
            return 0
        ratio = line.size / body_size
        word_count = len(line.text.split())
        if ratio >= 1.4:
            return 1
        if ratio >= 1.25:
            return 2
        if ratio >= 1.15:
            return 3
        if line.bold and ratio >= 1.02 and word_count <= 14 and not line.text.rstrip().endswith((".", ",", ";")):
            return 4
        return 0

    def _lines_to_blocks(self, lines: list[_Line], body_size: float) -> list[Block]:
        blocks: list[Block] = []
        para_lines: list[str] = []
        prev_bottom: float | None = None
        prev_line_height = 0.0

        def flush_paragraph() -> None:
            if para_lines:
                text = normalize_text(" ".join(para_lines))
                if text:
                    blocks.append(Block(type="paragraph", text=text))
                para_lines.clear()

        for line in lines:
            level = self._classify_heading_level(line, body_size)
            if level:
                flush_paragraph()
                blocks.append(Block(type="heading", text=normalize_text(line.text), level=level))
                prev_bottom = line.bottom
                prev_line_height = line.bottom - line.top
                continue

            gap = (line.top - prev_bottom) if prev_bottom is not None else 0
            if prev_bottom is not None and prev_line_height > 0 and gap > prev_line_height * _PARAGRAPH_GAP_FACTOR:
                flush_paragraph()
            para_lines.append(line.text)
            prev_bottom = line.bottom
            prev_line_height = line.bottom - line.top

        flush_paragraph()
        return blocks

    def _table_to_text(self, table: list[list[str | None]]) -> str:
        rows = [" | ".join((cell or "").strip() for cell in row) for row in table]
        return normalize_text("\n".join(r for r in rows if r.strip()))

    # -- pypdf fallback: flat text only, no heading detection -----------------

    def _parse_with_pypdf(self, file_bytes: bytes) -> ParsedDocument:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # try an empty password before giving up
            except Exception as exc:
                raise ParsingError(f"PDF is encrypted: {exc}") from exc

        page_texts = [page.extract_text() or "" for page in reader.pages]
        cleaned_pages = strip_repeated_page_furniture(page_texts)
        blocks = [
            Block(type="paragraph", text=para)
            for page_text in cleaned_pages
            for para in page_text.split("\n\n")
            if para.strip()
        ]
        return ParsedDocument(
            blocks=blocks,
            meta={"source_format": "pdf", "page_count": str(len(reader.pages))},
            warnings=["Parsed with pypdf fallback: heading detection unavailable"],
        )
