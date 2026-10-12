from __future__ import annotations

from dataclasses import dataclass, field

from app.search.types import CandidateChunk


@dataclass
class GroundingVerdict:
    sufficient: bool
    reason: str
    passing_candidates: list[CandidateChunk] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def evaluate_context_sufficiency(query: str, candidates: list[CandidateChunk]) -> GroundingVerdict:
    # Used to also gate on keyword coverage + semantic similarity of the raw
    # question against the candidates. Dropped: it judged the *question's*
    # wording, not whether retrieval actually found the right content, so
    # typos, paraphrases, and plural/tense mismatches ("borrow" vs
    # "borrowed") got refused even when the correct chunk was already
    # sitting at the top of `candidates`. Whether the *answer* is actually
    # supported is now checked once, after generation, by
    # app.llm.grounding_verifier against the real answer text instead of
    # the question.
    if not candidates:
        return GroundingVerdict(sufficient=False, reason="no_candidates", metrics={"num_candidates": 0})
    return GroundingVerdict(
        sufficient=True,
        reason="ok",
        passing_candidates=candidates,
        metrics={"num_candidates": len(candidates), "best_score": max(c.score for c in candidates)},
    )
