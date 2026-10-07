
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
    """RAGAS-style Context Precision@K: rank-weighted precision over a
    retrieved list's per-item relevance verdicts (already decided
    elsewhere — usually an LLM judge, since a free-text ground truth has
    no fixed chunk_ids a mechanical check could compare against).

    A relevant item at rank 1 contributes Precision@1 = 1.0; the same item
    buried at rank 10 contributes only Precision@10 = 0.1 (if nothing else
    above it is relevant) — so putting the relevant context first scores
    higher than putting it last, which plain Precision@k can't distinguish
    (it only cares how many of the top k are relevant, not where)."""
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
    """RAGAS-style Context Recall: of the ground truth answer's individual
    claims (split into sentences), what fraction can actually be
    attributed to the retrieved context? Unlike Context Precision (which
    judges each *retrieved chunk* for relevance to the answer), this judges
    each *ground-truth claim* for whether the retrieved context supports
    it — the opposite direction, and the only way to catch a shortlist
    that looks relevant on every chunk but still misses something the
    correct answer actually depends on.

    No rank weighting: a ground-truth claim doesn't have a "position" in
    the retrieved list the way a chunk does, so this is a plain proportion,
    not a weighted sum like context_precision_at_k.

    A ground truth with no claims at all (empty list) is vacuously fully
    recalled — same convention as recall_at_k's empty-relevant-set case."""
    if not claim_attributable_flags:
        return 1.0
    return sum(claim_attributable_flags) / len(claim_attributable_flags)
