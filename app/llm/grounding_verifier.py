from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import get_settings
from app.search.keyword_utils import significant_terms, term_coverage
from app.search.types import CandidateChunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


_NO_ANSWER_PATTERN = re.compile(
    r"\b(i\s+don'?t\s+know|i\s+do\s+not\s+know|not\s+enough\s+information|cannot\s+determine|can'?t\s+determine)\b",
    re.IGNORECASE,
)
_NO_ANSWER_MAX_WORDS = 20


@dataclass
class SentenceGrounding:
    sentence: str
    coverage: float
    grounded: bool


@dataclass
class GroundingCheckResult:
    overall_coverage: float | None
    sentences: list[SentenceGrounding] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)


def verify_grounding(
    answer: str,
    candidates: list[CandidateChunk],
    min_sentence_coverage: float | None = None,
) -> GroundingCheckResult:
    settings = get_settings()
    min_sentence_coverage = (
        settings.min_sentence_grounding_coverage if min_sentence_coverage is None else min_sentence_coverage
    )

    context_text = " ".join(c.content for c in candidates)
    raw_sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer.strip()) if s.strip()]

    if not raw_sentences:
        return GroundingCheckResult(overall_coverage=1.0)

    if len(answer.split()) <= _NO_ANSWER_MAX_WORDS and _NO_ANSWER_PATTERN.search(answer):
        return GroundingCheckResult(overall_coverage=None)

    sentence_results = []
    for sentence in raw_sentences:
        terms = significant_terms(sentence)
        coverage = term_coverage(terms, context_text)
        sentence_results.append(
            SentenceGrounding(sentence=sentence, coverage=coverage, grounded=coverage >= min_sentence_coverage)
        )

    overall = sum(r.coverage for r in sentence_results) / len(sentence_results)
    unsupported = [r.sentence for r in sentence_results if not r.grounded]

    return GroundingCheckResult(overall_coverage=overall, sentences=sentence_results, unsupported_sentences=unsupported)
