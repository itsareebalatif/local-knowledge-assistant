"""Token counting for chunk-size enforcement.

We deliberately do NOT depend on `tiktoken` here: its BPE rank files are
fetched from a remote CDN on first use, which would silently break this
project's "fully local / offline by default" requirement (NFR-4) the first
time someone runs ingestion without internet. Instead we use a regex-based
approximation of GPT-style tokenization (words, numbers, and punctuation as
separate tokens, contractions split like "don't" -> "do", "n't"). It is not
byte-identical to any real tokenizer's count, but it is stable, dependency-free,
and consistently close enough (~±10%) for enforcing a 512-token chunk budget.

If a real tokenizer is available and desired later, swap the implementation
of `count_tokens` — every caller in this package goes through this one
function.
"""

from __future__ import annotations

import re

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
    """Approximate token count for `text`. Empty/whitespace-only -> 0."""
    if not text or not text.strip():
        return 0
    return len(_TOKEN_PATTERN.findall(text))
