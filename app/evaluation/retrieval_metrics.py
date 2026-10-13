
from __future__ import annotations


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_ids)
    return hits / len(top_k)


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def context_precision_at_k(relevance_flags: list[bool]) -> float:
    # RAGAS-style: rank-weighted, so relevant items ranked earlier score higher
    total_relevant = sum(relevance_flags)
    if total_relevant == 0:
        return 0.0

    weighted_sum = 0.0
    relevant_so_far = 0
    for rank, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            relevant_so_far += 1
            weighted_sum += relevant_so_far / rank

    return weighted_sum / total_relevant


def context_recall(claim_attributable_flags: list[bool]) -> float:
    # RAGAS-style: fraction of ground-truth claims attributable to retrieved context
    if not claim_attributable_flags:
        return 1.0
    return sum(claim_attributable_flags) / len(claim_attributable_flags)
