from __future__ import annotations

from app.search.grounding_gate import evaluate_context_sufficiency
from app.search.types import CandidateChunk


def _c(chunk_id: int, content: str, score: float) -> CandidateChunk:
    return CandidateChunk(chunk_id=chunk_id, doc_id=1, content=content, score=score)


def test_no_candidates_refuses():
    verdict = evaluate_context_sufficiency("what is Ollama?", [])
    assert verdict.sufficient is False
    assert verdict.reason == "no_candidates"


def test_any_candidates_at_all_pass_regardless_of_wording_overlap():
    # No keyword/semantic check anymore — whether the answer is actually
    # supported is judged once, after generation, against the real answer
    # text (app.llm.grounding_verifier), not here against the raw question.
    candidates = [_c(1, "The weather today is unrelated content entirely.", 0.9)]
    verdict = evaluate_context_sufficiency("what programming language does Ollama support?", candidates)
    assert verdict.sufficient is True
    assert verdict.reason == "ok"
    assert verdict.passing_candidates == candidates


def test_metrics_report_candidate_count_and_best_score():
    candidates = [_c(1, "low score", 0.05), _c(2, "high score", 0.9)]
    verdict = evaluate_context_sufficiency("anything", candidates)
    assert verdict.metrics == {"num_candidates": 2, "best_score": 0.9}
