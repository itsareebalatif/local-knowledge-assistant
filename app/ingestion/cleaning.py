"""Text cleanup / layout stripping shared by every parser (FR-1.2).

Each parser is responsible for calling these on the raw strings it extracts
before wrapping them into `Block`s, so a Chunk's `content` never carries page
furniture, control characters, or inconsistent whitespace into the vector
store.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*(?:/\s*\d{1,4})?\s*$", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Unicode-normalize, drop control characters, collapse runs of whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_BLANK_LINE.sub("\n\n", text)
    return text.strip()


def is_page_number_line(line: str) -> bool:
    """True for lines that are just a page number ('Page 3', '12', '3/40')."""
    return bool(_PAGE_NUMBER_LINE.match(line.strip()))


def strip_repeated_page_furniture(pages: list[str], min_repeat_ratio: float = 0.6) -> list[str]:
    """Drop lines that recur near-identically across most pages of a document.

    Running headers/footers ("Confidential — Acme Corp", page numbers,
    document titles repeated on every page) add noise and duplicate tokens to
    every chunk if left in. A line that appears on >= `min_repeat_ratio` of
    pages is treated as furniture and removed from all of them. Requires at
    least 3 pages to avoid false positives on short documents.
    """
    if len(pages) < 3:
        return [normalize_text(p) for p in pages]

    line_counts: Counter[str] = Counter()
    per_page_lines: list[list[str]] = []
    for page in pages:
        lines = [line.strip() for line in page.split("\n") if line.strip()]
        per_page_lines.append(lines)
        line_counts.update(set(lines))

    threshold = max(2, int(len(pages) * min_repeat_ratio))
    furniture = {line for line, count in line_counts.items() if count >= threshold or is_page_number_line(line)}

    cleaned_pages = []
    for lines in per_page_lines:
        kept = [line for line in lines if line not in furniture]
        cleaned_pages.append(normalize_text("\n".join(kept)))
    return cleaned_pages
