from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*(?:/\s*\d{1,4})?\s*$", re.IGNORECASE)
_LEADING_OR_TRAILING_NUMBER = re.compile(r"^\d{1,4}\s+|\s+\d{1,4}$")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_BLANK_LINE.sub("\n\n", text)
    return text.strip()


def is_page_number_line(line: str) -> bool:
    return bool(_PAGE_NUMBER_LINE.match(line.strip()))


def _furniture_key(line: str) -> str:
    # A running header's page number changes every page (e.g. "Chapter
    # Title 21" vs "Chapter Title 23"), so exact-line matching never counts
    # it as repeated — strip one leading/trailing number before comparing.
    return _LEADING_OR_TRAILING_NUMBER.sub("", line).strip()


def strip_repeated_page_furniture(pages: list[str], min_repeat_ratio: float = 0.6) -> list[str]:
    if len(pages) < 3:
        return [normalize_text(p) for p in pages]

    key_counts: Counter[str] = Counter()
    per_page_lines: list[list[str]] = []
    for page in pages:
        lines = [line.strip() for line in page.split("\n") if line.strip()]
        per_page_lines.append(lines)
        key_counts.update({_furniture_key(line) for line in lines})

    threshold = max(2, int(len(pages) * min_repeat_ratio))
    furniture_keys = {key for key, count in key_counts.items() if count >= threshold}

    cleaned_pages = []
    for lines in per_page_lines:
        kept = [line for line in lines if _furniture_key(line) not in furniture_keys and not is_page_number_line(line)]
        cleaned_pages.append(normalize_text("\n".join(kept)))
    return cleaned_pages
