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
    # Rank-based, not raw score: upstream scores (RRF vs. cross-encoder) live on
    # incomparable scales, so position-only avoids a near-zero score dominating diversity.
    if n == 1:
        return [1.0]
    return [(n - 1 - rank) / (n - 1) for rank in range(n)]


def mmr_select(
    candidates: list[CandidateChunk],
    embeddings: list[list[float]],
    k: int,
    # 0.5 balances relevance vs. diversity; 1.0 = plain top-k, 0.0 = max diversity only.
    lambda_param: float = 0.5,
) -> list[CandidateChunk]:
    # candidates must already be sorted relevance-descending; embeddings[i] matches candidates[i].
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
