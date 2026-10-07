from __future__ import annotations

import pytest

from app.search.mmr import _cosine_similarity, _rank_relevance, mmr_select
from app.search.types import CandidateChunk


def test_rank_relevance_runs_from_one_down_to_zero():
    assert _rank_relevance(4) == [1.0, pytest.approx(2 / 3), pytest.approx(1 / 3), 0.0]


def test_rank_relevance_of_a_single_candidate_is_perfect():
    assert _rank_relevance(1) == [1.0]


def test_cosine_similarity_of_identical_vectors_is_one():
    assert _cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_negative_one():
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_handles_a_zero_vector_without_dividing_by_zero():
    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_mmr_select_returns_empty_list_for_no_candidates():
    assert mmr_select([], [], k=5) == []


def test_mmr_select_with_lambda_one_behaves_like_plain_top_k():
    # lambda=1.0 zeroes out the diversity term entirely — pure rank-based
    # relevance, identical to just keeping the candidates' existing order
    # (they're assumed already relevance-sorted on the way in).
    candidates = [
        CandidateChunk(chunk_id=2, doc_id=1, content="high", score=0.9, sources=[]),
        CandidateChunk(chunk_id=3, doc_id=1, content="mid", score=0.5, sources=[]),
        CandidateChunk(chunk_id=1, doc_id=1, content="low", score=0.2, sources=[]),
    ]
    # Embeddings are irrelevant when lambda=1.0 — made identical on purpose
    # to prove the diversity term truly contributes nothing at this setting.
    embeddings = [[1.0, 0.0]] * 3

    selected = mmr_select(candidates, embeddings, k=3, lambda_param=1.0)

    assert [c.chunk_id for c in selected] == [2, 3, 1]  # unchanged input order


def test_mmr_select_prefers_a_diverse_candidate_over_a_near_duplicate_of_the_top_pick():
    # Candidate 2 is a near-duplicate of candidate 1 (same embedding
    # direction) and scores higher than candidate 3, but candidate 3 is
    # completely different (orthogonal embedding) — with a heavy diversity
    # weighting, picking candidate 3 second should beat re-picking
    # something that looks almost exactly like what's already selected.
    candidates = [
        CandidateChunk(chunk_id=1, doc_id=1, content="A", score=0.9, sources=[]),
        CandidateChunk(chunk_id=2, doc_id=1, content="A-near-duplicate", score=0.85, sources=[]),
        CandidateChunk(chunk_id=3, doc_id=1, content="completely different", score=0.6, sources=[]),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.99, 0.01],  # nearly identical direction to candidate 1
        [0.0, 1.0],  # orthogonal to both
    ]

    selected = mmr_select(candidates, embeddings, k=2, lambda_param=0.2)  # diversity-weighted

    assert [c.chunk_id for c in selected] == [1, 3]


def test_mmr_select_respects_k_even_with_more_candidates_available():
    candidates = [CandidateChunk(chunk_id=i, doc_id=1, content=f"chunk {i}", score=1.0 / i, sources=[]) for i in range(1, 6)]
    embeddings = [[float(i), 0.0] for i in range(1, 6)]

    selected = mmr_select(candidates, embeddings, k=2, lambda_param=0.5)

    assert len(selected) == 2


def test_mmr_select_is_not_fooled_by_a_sharply_collapsed_score_distribution():
    # Regression test for a real failure: a cross-encoder reranker gave one
    # candidate a strong score and let everything else collapse toward
    # 0.000 — including a genuinely on-topic runner-up (score 0.009). Using
    # those raw scores as MMR's relevance term let near-zero noise among
    # the "losers" outrank the runner-up just for being more different.
    # Rank-based relevance must keep the runner-up competitive regardless
    # of how skewed the underlying scores are.
    candidates = [
        CandidateChunk(chunk_id=31, doc_id=1, content="top match", score=0.438, sources=[]),
        CandidateChunk(chunk_id=3, doc_id=1, content="genuine runner-up", score=0.009, sources=[]),
        CandidateChunk(chunk_id=16, doc_id=1, content="filler", score=0.003, sources=[]),
        CandidateChunk(chunk_id=7, doc_id=1, content="filler", score=0.002, sources=[]),
        CandidateChunk(chunk_id=18, doc_id=1, content="filler", score=0.001, sources=[]),
        CandidateChunk(chunk_id=4, doc_id=1, content="noise", score=0.000, sources=[]),
        CandidateChunk(chunk_id=35, doc_id=1, content="noise", score=0.000, sources=[]),
        CandidateChunk(chunk_id=17, doc_id=1, content="noise", score=0.000, sources=[]),
    ]
    # All candidates maximally distinct from each other (orthogonal unit
    # vectors) — isolates the test to the relevance term alone, since every
    # diversity penalty between any pair is identically 0.0.
    embeddings = [[1.0 if i == j else 0.0 for j in range(8)] for i in range(8)]

    selected = mmr_select(candidates, embeddings, k=5, lambda_param=0.5)

    assert candidates[1].chunk_id in [c.chunk_id for c in selected]  # the genuine runner-up survives


def test_mmr_select_never_picks_the_same_candidate_twice():
    candidates = [CandidateChunk(chunk_id=i, doc_id=1, content=f"chunk {i}", score=0.5, sources=[]) for i in range(1, 4)]
    embeddings = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]  # all identical — worst case for accidental re-selection

    selected = mmr_select(candidates, embeddings, k=3, lambda_param=0.5)

    assert len({c.chunk_id for c in selected}) == 3
