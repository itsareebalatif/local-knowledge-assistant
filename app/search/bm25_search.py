from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.search.types import CandidateChunk

# Quoting each token individually as a literal avoids FTS5 MATCH syntax
# misreading raw query text (e.g. colons, AND/OR/NOT).
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

_CAMEL_BOUNDARY_SIMPLE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_BOUNDARY_ACRONYM = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


def _build_match_query(query: str) -> str | None:
    query = _CAMEL_BOUNDARY_SIMPLE.sub(" ", query)
    query = _CAMEL_BOUNDARY_ACRONYM.sub(" ", query)
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
