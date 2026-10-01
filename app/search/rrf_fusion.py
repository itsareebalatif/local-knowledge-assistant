
from __future__ import annotations

from app.search.types import CandidateChunk


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[CandidateChunk]], k: int = 60, top_n: int = 10
) -> list[CandidateChunk]:
    raw_scores: dict[int, float] = {}
    sources_by_chunk: dict[int, set[str]] = {}
    first_seen: dict[int, CandidateChunk] = {}

    for source_name, candidates in ranked_lists.items():
        for rank, candidate in enumerate(candidates, start=1):
            raw_scores[candidate.chunk_id] = raw_scores.get(candidate.chunk_id, 0.0) + 1.0 / (k + rank)
            sources_by_chunk.setdefault(candidate.chunk_id, set()).add(source_name)
            first_seen.setdefault(candidate.chunk_id, candidate)

    if not raw_scores:
        return []

    max_possible = len(ranked_lists) / (k + 1)

    fused = [
        CandidateChunk(
            chunk_id=base.chunk_id,
            doc_id=base.doc_id,
            content=base.content,
            score=(raw_scores[chunk_id] / max_possible) if max_possible > 0 else 0.0,
            sources=sorted(sources_by_chunk[chunk_id]),
        )
        for chunk_id, base in first_seen.items()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused[:top_n]
