"""Post-generation Grounding Verifier (FR-4.5 Pipeline 2).

The pre-retrieval gate (app/search/grounding_gate.py) already confirmed the
*retrieved context* was worth answering from — this checks the opposite
direction: did the *model's actual answer* stay inside that context, or did
it wander off and state something the passages never said? An LLM can still
hallucinate even with good context in front of it.

Deliberately non-LLM, same philosophy as spaCy entity extraction: a second
model call to grade the first model's output would be slow, non-deterministic,
and itself capable of being wrong. Instead, each sentence of the answer is
checked for keyword overlap with the retrieved context — cheap, fast,
explainable, and this is a warning/annotation layer, not a second hard gate:
the answer still reaches the user, flagged, rather than being silently
discarded after the user already waited for it to generate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import get_settings
from app.search.keyword_utils import significant_terms, term_coverage
from app.search.types import CandidateChunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class SentenceGrounding:
    sentence: str
    coverage: float
    grounded: bool


@dataclass
class GroundingCheckResult:
    overall_coverage: float
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
