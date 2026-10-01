from __future__ import annotations

from app.search.bm25_search import bm25_search


def test_ranks_relevant_chunk_above_unrelated_one(db_session, make_chunk):
    relevant = make_chunk("Ollama is a tool for running local language models on your own machine.")
    make_chunk("The weather today is sunny with a light breeze.")

    hits = bm25_search(db_session, "Ollama local models", top_k=5)

    assert len(hits) == 1
    assert hits[0].chunk_id == relevant.chunk_id
    assert hits[0].sources == ["bm25"]


def test_returns_up_to_top_k_ordered_by_relevance(db_session, make_chunk):
    best = make_chunk("Ollama Ollama Ollama runs language models locally.")
    make_chunk("Ollama is mentioned once here.")
    make_chunk("This chunk does not mention the keyword at all.")

    hits = bm25_search(db_session, "Ollama", top_k=1)

    assert len(hits) == 1
    assert hits[0].chunk_id == best.chunk_id


def test_query_with_punctuation_does_not_crash_fts5(db_session, make_chunk):
    make_chunk("Ollama runs qwen2.5:3b entirely offline.")
    # A raw query containing FTS5-special characters (quotes, colon, "OR")
    # must not raise — this is exactly what the token-quoting is for.
    hits = bm25_search(db_session, 'qwen2.5:3b OR "malicious" NOT chunks', top_k=5)
    assert isinstance(hits, list)  # no exception is the actual assertion here


def test_empty_or_meaningless_query_returns_no_hits(db_session, make_chunk):
    make_chunk("Some content that exists in the database.")
    assert bm25_search(db_session, "", top_k=5) == []
    assert bm25_search(db_session, "   ", top_k=5) == []
    assert bm25_search(db_session, "!!!???", top_k=5) == []


def test_no_matching_content_returns_empty_list(db_session, make_chunk):
    make_chunk("Completely unrelated content about gardening.")
    assert bm25_search(db_session, "quantum computing", top_k=5) == []
