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


def test_i_dont_know_reports_no_coverage_instead_of_a_trivial_100_percent():
    # Regression: "know"/"don" happen to appear in almost any real passage of
    # prose, which previously made a plain refusal show up as "100% grounded"
    # — confirmed live against a real Harry Potter chunk.
    candidates = [_c("Mr. Dursley knew his neighbours well and didn't want any trouble.")]
    result = verify_grounding("I don't know.", candidates)
    assert result.overall_coverage is None
    assert result.unsupported_sentences == []


def test_do_not_know_variant_also_reports_no_coverage():
    candidates = [_c("Some unrelated passage text.")]
    result = verify_grounding("I do not know based on the provided context.", candidates)
    assert result.overall_coverage is None


def test_a_longer_answer_that_merely_mentions_not_knowing_is_still_verified():
    # Only SHORT, refusal-only answers are treated as "no claims made" — a
    # longer answer with real content shouldn't have its scoring suppressed
    # just because one clause expresses uncertainty.
    candidates = [_c("Ollama runs language models locally on your own machine.")]
    answer = (
        "Ollama runs language models locally on your own machine [1]. "
        "I don't know the exact version number though, since the context doesn't mention it, "
        "but the core local-execution behavior is clearly documented above."
    )
    result = verify_grounding(answer, candidates, min_sentence_coverage=0.3)
    assert result.overall_coverage is not None
