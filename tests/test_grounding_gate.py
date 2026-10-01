from __future__ import annotations

from app.search.grounding_gate import evaluate_context_sufficiency
from app.search.types import CandidateChunk


def _c(chunk_id: int, content: str, score: float) -> CandidateChunk:
    return CandidateChunk(chunk_id=chunk_id, doc_id=1, content=content, score=score)


def test_no_candidates_refuses():
    verdict = evaluate_context_sufficiency("what is Ollama?", [])
    assert verdict.sufficient is False
    assert verdict.reason == "no_candidates"


def test_candidates_all_below_score_threshold_refuses():
    candidates = [_c(1, "Ollama runs local models", 0.1), _c(2, "Ollama runs local models", 0.05)]
    verdict = evaluate_context_sufficiency("what is Ollama?", candidates, min_score=0.5, min_candidates=1)
    assert verdict.sufficient is False
    assert verdict.reason == "below_score_threshold"
    assert verdict.metrics["best_score"] == 0.1


def test_good_score_but_no_shared_vocabulary_refuses_on_keyword_coverage():
    # High score, but the content shares no real words with the query.
    candidates = [_c(1, "The weather today is unrelated content entirely.", 0.9)]
    verdict = evaluate_context_sufficiency(
        "what programming language does Ollama support?", candidates, min_score=0.5, min_keyword_coverage=0.3
    )
    assert verdict.sufficient is False
    assert verdict.reason == "low_keyword_coverage"


def test_sufficient_context_passes_and_returns_passing_candidates():
    candidates = [
        _c(1, "Ollama supports running language models locally on your machine.", 0.9),
        _c(2, "unrelated low scoring chunk", 0.05),
    ]
    verdict = evaluate_context_sufficiency(
        "what programming language does Ollama support?",
        candidates,
        min_score=0.5,
        min_keyword_coverage=0.3,
        min_candidates=1,
    )
    assert verdict.sufficient is True
    assert verdict.reason == "ok"
    assert [c.chunk_id for c in verdict.passing_candidates] == [1]


def test_query_with_only_stopwords_does_not_penalize_coverage():
    # "what is it" has no significant terms once stopwords are stripped —
    # coverage can't meaningfully be computed, so it shouldn't cause a refusal.
    candidates = [_c(1, "some perfectly fine content", 0.9)]
    verdict = evaluate_context_sufficiency("what is it", candidates, min_score=0.5, min_candidates=1)
    assert verdict.sufficient is True
    assert verdict.metrics["keyword_coverage"] == 1.0


def test_min_candidates_requires_enough_passing_not_just_one():
    candidates = [_c(1, "Ollama runs models locally", 0.9)]
    verdict = evaluate_context_sufficiency(
        "Ollama models", candidates, min_score=0.5, min_candidates=2, min_keyword_coverage=0.1
    )
    assert verdict.sufficient is False
    assert verdict.reason == "below_score_threshold"
    assert verdict.metrics["num_passing"] == 1
    assert verdict.metrics["num_required"] == 2
