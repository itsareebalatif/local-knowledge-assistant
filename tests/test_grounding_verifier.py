from __future__ import annotations

from app.llm.grounding_verifier import verify_grounding
from app.search.types import CandidateChunk


def _c(content: str) -> CandidateChunk:
    return CandidateChunk(chunk_id=1, doc_id=1, content=content, score=1.0)


def test_grounded_sentence_has_high_coverage_and_is_not_flagged():
    candidates = [_c("Ollama runs language models locally on your own machine.")]
    result = verify_grounding("Ollama runs language models locally.", candidates)
    assert result.overall_coverage > 0.5
    assert result.unsupported_sentences == []


def test_hallucinated_sentence_is_flagged_as_unsupported():
    candidates = [_c("Ollama runs language models locally on your own machine.")]
    answer = "Ollama runs language models locally. The moon is made of green cheese and unicorns exist."
    result = verify_grounding(answer, candidates, min_sentence_coverage=0.3)

    assert len(result.unsupported_sentences) == 1
    assert "unicorns" in result.unsupported_sentences[0].lower()


def test_empty_answer_returns_trivial_full_coverage():
    result = verify_grounding("", [_c("something")])
    assert result.overall_coverage == 1.0
    assert result.sentences == []


def test_overall_coverage_averages_across_sentences():
    candidates = [_c("Ollama runs models locally on your machine using quantization.")]
    answer = "Ollama runs models locally. Bananas grow on tropical palm trees in warm climates."
    result = verify_grounding(answer, candidates, min_sentence_coverage=0.5)
    assert 0.0 < result.overall_coverage < 1.0
    assert len(result.sentences) == 2
