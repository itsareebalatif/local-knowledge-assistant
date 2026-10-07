"""BM25 keyword search over chunks_fts (FR-3.3).

Only responsible for returning a correctly ranked list — the raw bm25()
score is kept on CandidateChunk.score purely for logging, since RRF fusion
(see rrf_fusion.py) only needs this list's rank order, not the score itself.

`user_id` is required and filtered in SQL (joined through documents), not
applied afterward — the same per-user isolation principle as
vector_search.py's Chroma `where` filter.
"""

from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.search.types import CandidateChunk

# Word tokens only: FTS5 MATCH syntax treats quotes, colons, hyphens, and
# keywords like AND/OR/NOT specially. Quoting each token individually and
# joining with OR sidesteps all of that — every token is searched as a
# literal phrase, and a raw query like `qwen2.5:3b OR "hack"` can't be
# misread as FTS5 query syntax.
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _build_match_query(query: str) -> str | None:
    tokens = _TOKEN_PATTERN.findall(query)
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)


def bm25_search(db: Session, query: str, top_k: int, user_id: int) -> list[CandidateChunk]:
    match_query = _build_match_query(query)
    if match_query is None:
        return []

    rows = db.execute(
        text(
            """
            SELECT c.chunk_id, c.doc_id, c.content, bm25(chunks_fts) AS raw_score
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.rowid
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE chunks_fts MATCH :match_query
              AND d.user_id = :user_id
            ORDER BY raw_score ASC
            LIMIT :top_k
            """
        ),
        {"match_query": match_query, "top_k": top_k, "user_id": user_id},
    ).all()

    return [
        CandidateChunk(chunk_id=row.chunk_id, doc_id=row.doc_id, content=row.content, score=row.raw_score, sources=["bm25"])
        for row in rows
    ]
