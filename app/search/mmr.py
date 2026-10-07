"""Maximal Marginal Relevance (MMR) — the final candidate-selection step,
replacing a plain "take the top-k by score" cut.

RRF fusion (and the optional cross-encoder reranker) both produce a single
relevance ranking, with no notion of whether two high-scoring candidates
are actually near-duplicates of each other. A plain top-k off that ranking
can hand the LLM several chunks that all say roughly the same thing,
spending the context budget on redundant information instead of genuinely
different facts. MMR greedily builds the final set one candidate at a
time, each pick balancing two things: how relevant it is against how
different it is from what's already been selected (the highest cosine
similarity to any already-picked candidate's embedding — the higher that
is, the more this candidate just repeats something already covered).

Relevance is derived from each candidate's *rank position* in the incoming
list (assumed already sorted relevance-descending — true of both RRF
fusion's and the reranker's output), not from its raw `.score`. Raw scores
from different upstream stages live on wildly different scales: RRF's
fused score is a modest, fairly even spread, while a cross-encoder's
sigmoid output is often sharply peaked at just the single best match and
collapses everything else toward zero. Mixing that near-zero noise
directly into a 0-1 diversity term lets it dominate the choice among the
"losers," arbitrarily dropping a perfectly good runner-up in favor of
something merely more different (confirmed by a live run: a reranker
score of 0.009 for a clearly on-topic chunk got outranked by several
0.000-score chunks purely on diversity). Rank position only relies on
order, which every upstream stage already got right regardless of its
score's scale or shape.

`mmr_lambda` controls the trade-off: 1.0 behaves like plain top-k (pure
relevance-by-rank, no diversity term at all — reduces to the candidates'
original order); 0.0 ignores relevance entirely and just maximizes spread
between picks.
"""

from __future__ import annotations

import math

from app.search.types import CandidateChunk


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _rank_relevance(n: int) -> list[float]:
    """1.0 for the first candidate, 0.0 for the last, evenly spaced in
    between — a relevance signal that depends only on position, not on
    whatever scale or shape the upstream scorer happened to produce."""
    if n == 1:
        return [1.0]
    return [(n - 1 - rank) / (n - 1) for rank in range(n)]


def mmr_select(
    candidates: list[CandidateChunk],
    embeddings: list[list[float]],
    k: int,
    lambda_param: float = 0.5,
) -> list[CandidateChunk]:
    """`candidates` must already be sorted relevance-descending (both RRF
    fusion and the reranker return their results this way) — that order is
    what the relevance term is derived from. `embeddings[i]` must be the
    embedding of `candidates[i].content`, same order, same length."""
    if not candidates:
        return []

    relevance = _rank_relevance(len(candidates))
    remaining = list(range(len(candidates)))
    selected: list[int] = []

    while remaining and len(selected) < k:
        best_index = remaining[0]
        best_mmr_score = float("-inf")
        for i in remaining:
            diversity_penalty = max((_cosine_similarity(embeddings[i], embeddings[j]) for j in selected), default=0.0)
            mmr_score = lambda_param * relevance[i] - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_index = i
        selected.append(best_index)
        remaining.remove(best_index)

    return [candidates[i] for i in selected]
