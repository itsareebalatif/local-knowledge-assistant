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

    citations = build_citations(db_session, candidates, "Some fact [1]. Another fact [2].")

    assert [c.marker for c in citations] == [1, 2]
    assert citations[0].chunk_id == chunk_a.chunk_id
    assert citations[0].snippet == "first content"
    assert citations[0].file_name.endswith(".md")
    assert citations[0].source_url == f"/api/documents/{chunk_a.doc_id}"

    assert citations[1].snippet.endswith("…")
    assert len(citations[1].snippet) == 201  # 200 chars + ellipsis


def test_build_citations_handles_missing_document_gracefully(db_session):
    candidate = CandidateChunk(chunk_id=999, doc_id=999, content="orphan content", score=1.0)
    citations = build_citations(db_session, [candidate], "A claim [1].")
    assert citations[0].file_name == "document-999"


def test_build_citations_empty_candidates_returns_empty_list(db_session):
    assert build_citations(db_session, [], "anything") == []


def test_build_citations_excludes_candidates_the_answer_never_cited(db_session, make_chunk):
    # Regression for a real, confirmed-live bug: citations used to be built
    # from every retrieved candidate unconditionally, with no connection to
    # what the answer text actually said. A refusal like "I don't know" has
    # no [n] markers in it at all, but still showed "Sources (5)" built from
    # whatever retrieval happened to find. Only markers the answer actually
    # references should produce a citation — including the case of zero.
    chunk_a = make_chunk("first content")
    chunk_b = make_chunk("second content")
    candidates = [
        CandidateChunk(chunk_id=chunk_a.chunk_id, doc_id=chunk_a.doc_id, content=chunk_a.content, score=1.0),
        CandidateChunk(chunk_id=chunk_b.chunk_id, doc_id=chunk_b.doc_id, content=chunk_b.content, score=0.9),
    ]

    no_citations = build_citations(db_session, candidates, "I don't know. The passages don't mention that.")
    assert no_citations == []

    partial_citations = build_citations(db_session, candidates, "Only the first fact is relevant [1].")
    assert [c.marker for c in partial_citations] == [1]
    assert partial_citations[0].chunk_id == chunk_a.chunk_id


def test_build_citations_recognizes_full_width_bracket_markers(db_session, make_chunk):
    # Regression for a real, confirmed-live bug: an OpenRouter model cited
    # correctly but wrote the marker as 【 1 】 (full-width brackets)
    # instead of the prompted ASCII [1] — the plain-ASCII pattern never
    # matched it, so a genuinely correct, well-cited answer showed zero
    # sources at all.
    chunk = make_chunk("Sylvia sells herbal cures without Rory's knowledge.")
    candidate = CandidateChunk(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, content=chunk.content, score=1.0)

    citations = build_citations(db_session, [candidate], "Sylvia runs a secret side business 【1】.")

    assert [c.marker for c in citations] == [1]
    assert citations[0].chunk_id == chunk.chunk_id
