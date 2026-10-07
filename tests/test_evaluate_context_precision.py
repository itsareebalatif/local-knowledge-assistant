from __future__ import annotations

import json

import pytest

from app.evaluation.evaluate_context_precision import (
    GoldenCase,
    _collect_with_retry,
    _parse_claim_verdicts,
    _parse_verdicts,
    evaluate_context_precision,
    load_golden_cases,
)
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.llm.base import LLMError


def test_parse_verdicts_reads_relevant_and_not_relevant_lines():
    judge_output = "Chunk 1: RELEVANT\nChunk 2: NOT_RELEVANT\nChunk 3: relevant\n"
    assert _parse_verdicts(judge_output, num_chunks=3) == [True, False, True]


def test_parse_verdicts_defaults_missing_or_malformed_lines_to_not_relevant():
    # Chunk 2 never shows up in the output at all — must not default to
    # relevant just because the judge forgot it or produced garbage.
    judge_output = "Chunk 1: RELEVANT\nI'm not sure about the rest.\n"
    assert _parse_verdicts(judge_output, num_chunks=2) == [True, False]


def test_parse_verdicts_ignores_out_of_range_chunk_numbers():
    judge_output = "Chunk 1: RELEVANT\nChunk 99: RELEVANT\n"
    assert _parse_verdicts(judge_output, num_chunks=1) == [True]


def test_parse_claim_verdicts_reads_attributable_and_not_attributable_lines():
    judge_output = "Claim 1: ATTRIBUTABLE\nClaim 2: NOT_ATTRIBUTABLE\nClaim 3: attributable\n"
    assert _parse_claim_verdicts(judge_output) == [True, False, True]


def test_parse_claim_verdicts_derives_count_from_the_highest_claim_number_seen():
    # The judge decides how many atomic claims exist — nothing upstream
    # tells it a fixed count, unlike chunk verdicts (we supplied those).
    judge_output = "Claim 1: ATTRIBUTABLE\nClaim 2: NOT_ATTRIBUTABLE\nClaim 3: ATTRIBUTABLE\nClaim 4: ATTRIBUTABLE\n"
    assert _parse_claim_verdicts(judge_output) == [True, False, True, True]


def test_parse_claim_verdicts_fills_a_skipped_number_with_not_attributable():
    judge_output = "Claim 1: ATTRIBUTABLE\nClaim 3: ATTRIBUTABLE\n"  # claim 2 missing entirely
    assert _parse_claim_verdicts(judge_output) == [True, False, True]


def test_parse_claim_verdicts_returns_empty_list_when_nothing_parses():
    assert _parse_claim_verdicts("I refuse to answer in that format.") == []


def test_parse_claim_and_chunk_verdicts_dont_cross_contaminate():
    # Both sections share the same "<n>: LABEL" shape — proves the chunk
    # parser doesn't accidentally pick up claim lines or vice versa.
    judge_output = "Chunk 1: RELEVANT\nClaim 1: NOT_ATTRIBUTABLE\n"
    assert _parse_verdicts(judge_output, num_chunks=1) == [True]
    assert _parse_claim_verdicts(judge_output) == [False]


async def test_evaluate_context_precision_scores_a_well_retrieved_case(
    db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm, user
):
    relevant = make_chunk("Ollama runs local language models directly on your own machine.")
    unrelated = make_chunk("Completely unrelated content about gardening and soil pH.")
    fake_vector_store.add(
        ids=[f"chunk-{relevant.chunk_id}", f"chunk-{unrelated.chunk_id}"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=[relevant.content, unrelated.content],
        metadatas=[
            {"doc_id": relevant.doc_id, "chunk_id": relevant.chunk_id, "user_id": user.user_id},
            {"doc_id": unrelated.doc_id, "chunk_id": unrelated.chunk_id, "user_id": user.user_id},
        ],
    )
    fake_llm.pieces = [
        "Chunk 1: RELEVANT\n",
        "Chunk 2: NOT_RELEVANT\n",
        "Claim 1: ATTRIBUTABLE\n",
    ]

    cases = [GoldenCase(id=1, question="What does Ollama run models on?", ground_truth="Your own machine, locally.")]

    report = await evaluate_context_precision(
        db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(),
        fake_llm, cases, user.user_id, k=5,
    )

    assert report.k == 5
    assert len(report.results) == 1
    result = report.results[0]
    assert result.case_id == 1
    assert result.num_retrieved >= 1
    assert len(fake_llm.calls) == 1  # one judge call per case, covering both metrics, not per chunk/claim
    assert report.mean_context_precision == result.context_precision
    assert report.mean_context_recall == result.context_recall
    assert result.context_recall == 1.0  # the single claim was marked ATTRIBUTABLE
    assert result.claim_flags == [True]


async def test_evaluate_context_precision_scores_zero_when_nothing_is_retrieved(
    db_session, fake_embedder, fake_vector_store, fake_llm, user
):
    # Nothing ingested at all — the pipeline has no candidates to offer.
    cases = [GoldenCase(id=1, question="completely unmatched question", ground_truth="anything")]

    report = await evaluate_context_precision(
        db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(),
        fake_llm, cases, user.user_id, k=5,
    )

    result = report.results[0]
    assert result.context_precision == 0.0
    assert result.context_recall == 0.0  # no judge call happened, so this must not fall back to the vacuous 1.0
    assert result.claim_flags == []  # claim count is only known once the judge actually runs
    assert result.num_retrieved == 0
    assert len(fake_llm.calls) == 0  # no retrieved chunks means no point asking the judge


class _FlakyLLM:
    """Raises LLMError the first N calls, then succeeds — simulates Groq's
    429 rate-limit response recovering once the retry wait has passed."""

    def __init__(self, failures_before_success: int, error_message: str = "rate limited"):
        self.failures_before_success = failures_before_success
        self.error_message = error_message
        self.calls = 0

    async def generate_stream(self, system_prompt: str, user_prompt: str):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise LLMError(self.error_message)
        yield "Chunk 1: RELEVANT\n"


def _fake_sleep_recording_into(sleeps: list[float]):
    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return _sleep


async def test_collect_with_retry_recovers_after_transient_failures(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("asyncio.sleep", _fake_sleep_recording_into(sleeps))
    llm = _FlakyLLM(failures_before_success=2, error_message="Groq returned 429: try again in 4.92s")

    result = await _collect_with_retry(llm, "system", "user")

    assert result == "Chunk 1: RELEVANT\n"
    assert llm.calls == 3
    # Parsed the server's own suggested wait (4.92s) rather than falling
    # back to the fixed default, with a small buffer added on top.
    assert sleeps == [pytest.approx(5.42), pytest.approx(5.42)]


async def test_collect_with_retry_falls_back_to_a_default_wait_when_no_hint_is_present(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("asyncio.sleep", _fake_sleep_recording_into(sleeps))
    llm = _FlakyLLM(failures_before_success=1, error_message="Could not reach Groq.")

    await _collect_with_retry(llm, "system", "user")

    assert sleeps == [5.0]


async def test_collect_with_retry_gives_up_after_the_max_attempts(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _fake_sleep_recording_into([]))
    llm = _FlakyLLM(failures_before_success=999, error_message="permanently broken")

    with pytest.raises(LLMError):
        await _collect_with_retry(llm, "system", "user")


def test_load_golden_cases_strips_citation_markers_from_ground_truth(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            [{"id": 1, "question": "What is X?", "ground_truth": "X is a thing[cite: 10]. It does Y[cite: 11, 12]."}]
        )
    )

    cases = load_golden_cases(path)

    assert len(cases) == 1
    assert cases[0].id == 1
    assert cases[0].question == "What is X?"
    assert "[cite" not in cases[0].ground_truth
    assert cases[0].ground_truth == "X is a thing. It does Y."
