
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings
from app.search.keyword_utils import significant_terms, term_coverage
from app.search.types import CandidateChunk


@dataclass
class GroundingVerdict:
    sufficient: bool
    reason: str
    passing_candidates: list[CandidateChunk] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def evaluate_context_sufficiency(
    query: str,
    candidates: list[CandidateChunk],
    min_score: float | None = None,
    min_keyword_coverage: float | None = None,
    min_candidates: int | None = None,
) -> GroundingVerdict:
    settings = get_settings()
    min_score = settings.min_retrieval_score if min_score is None else min_score
    min_keyword_coverage = settings.min_keyword_coverage if min_keyword_coverage is None else min_keyword_coverage
    min_candidates = settings.min_grounding_candidates if min_candidates is None else min_candidates

    if not candidates:
        return GroundingVerdict(sufficient=False, reason="no_candidates", metrics={"num_candidates": 0})

    passing = [c for c in candidates if c.score >= min_score]
    if len(passing) < min_candidates:
        return GroundingVerdict(
            sufficient=False,
            reason="below_score_threshold",
            metrics={
                "num_passing": len(passing),
                "num_required": min_candidates,
                "best_score": max(c.score for c in candidates),
                "min_score": min_score,
            },
        )

    terms = significant_terms(query)
    coverage = term_coverage(terms, " ".join(c.content for c in passing))
    if coverage < min_keyword_coverage:
        return GroundingVerdict(
            sufficient=False,
            reason="low_keyword_coverage",
            metrics={
                "keyword_coverage": coverage,
                "min_keyword_coverage": min_keyword_coverage,
                "query_terms": terms,
            },
        )

    return GroundingVerdict(
        sufficient=True,
        reason="ok",
        passing_candidates=passing,
        metrics={
            "num_passing": len(passing),
            "best_score": max(c.score for c in passing),
            "keyword_coverage": coverage,
        },
    )
