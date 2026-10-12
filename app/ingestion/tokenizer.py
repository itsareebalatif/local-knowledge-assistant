from __future__ import annotations

import re

# Regex-based approximation, not tiktoken, to keep ingestion fully offline (no CDN fetch of BPE ranks)
_TOKEN_PATTERN = re.compile(
    r"""
    [A-Za-z]+'[A-Za-z]+   # contractions: don't, it's
    | [A-Za-z]+            # words
    | \d+(?:[.,]\d+)*       # numbers, incl. decimals/thousands
    | [^\sA-Za-z0-9]        # any single punctuation / symbol character
    """,
    re.VERBOSE,
)


def count_tokens(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(_TOKEN_PATTERN.findall(text))
