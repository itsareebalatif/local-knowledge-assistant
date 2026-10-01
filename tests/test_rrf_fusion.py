from __future__ import annotations

from app.search.rrf_fusion import reciprocal_rank_fusion
from app.search.types import CandidateChunk


def _c(chunk_id: int, content: str = "text") -> CandidateChunk:
    return CandidateChunk(chunk_id=chunk_id, doc_id=1, content=content, score=0.0)


def test_chunk_ranked_first_in_all_lists_scores_highest():
    fused = reciprocal_rank_fusion(
        {
            "bm25": [_c(1), _c(2)],
            "vector": [_c(1), _c(3)],
            "graph": [_c(1), _c(4)],
        }
    )
    assert fused[0].chunk_id == 1
    assert fused[0].sources == ["bm25", "graph", "vector"]


def test_agreement_across_sources_beats_a_single_top_rank():
    # chunk 2 is #1 in exactly one list; chunk 3 is #2 in all three lists.
    # RRF should favor consistent agreement over a single first-place finish.
    fused = reciprocal_rank_fusion(
        {
            "bm25": [_c(2), _c(3)],
            "vector": [_c(4), _c(3)],
            "graph": [_c(5), _c(3)],
        }
    )
    assert fused[0].chunk_id == 3


def test_scores_are_normalized_between_zero_and_one():
    fused = reciprocal_rank_fusion({"bm25": [_c(1), _c(2), _c(3)]})
    assert all(0.0 <= c.score <= 1.0 for c in fused)
    assert fused[0].score == 1.0  # sole list, ranked #1 -> the theoretical max


def test_respects_top_n():
    fused = reciprocal_rank_fusion({"bm25": [_c(i) for i in range(20)]}, top_n=5)
    assert len(fused) == 5


def test_no_candidates_anywhere_returns_empty_list():
    assert reciprocal_rank_fusion({"bm25": [], "vector": [], "graph": []}) == []


def test_preserves_content_and_doc_id_from_first_occurrence():
    fused = reciprocal_rank_fusion({"bm25": [_c(1, content="hello world")]})
    assert fused[0].content == "hello world"
    assert fused[0].doc_id == 1
