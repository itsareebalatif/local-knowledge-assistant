from __future__ import annotations

from app.llm.citations import build_citations
from app.search.types import CandidateChunk


def test_build_citations_numbers_in_order_and_resolves_file_names(db_session, make_chunk):
    chunk_a = make_chunk("first content")
    chunk_b = make_chunk("second content " * 60)  # long enough to force truncation

    candidates = [
        CandidateChunk(chunk_id=chunk_a.chunk_id, doc_id=chunk_a.doc_id, content=chunk_a.content, score=1.0),
        CandidateChunk(chunk_id=chunk_b.chunk_id, doc_id=chunk_b.doc_id, content=chunk_b.content, score=0.9),
    ]

    citations = build_citations(db_session, candidates)

    assert [c.marker for c in citations] == [1, 2]
    assert citations[0].chunk_id == chunk_a.chunk_id
    assert citations[0].snippet == "first content"
    assert citations[0].file_name.endswith(".md")
    assert citations[0].source_url == f"/api/documents/{chunk_a.doc_id}"

    assert citations[1].snippet.endswith("…")
    assert len(citations[1].snippet) == 201  # 200 chars + ellipsis


def test_build_citations_handles_missing_document_gracefully(db_session):
    # A candidate referencing a doc_id that doesn't exist in the DB — should
    # never crash, just fall back to a placeholder name.
    candidate = CandidateChunk(chunk_id=999, doc_id=999, content="orphan content", score=1.0)
    citations = build_citations(db_session, [candidate])
    assert citations[0].file_name == "document-999"


def test_build_citations_empty_candidates_returns_empty_list(db_session):
    assert build_citations(db_session, []) == []
