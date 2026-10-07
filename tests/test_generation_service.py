from __future__ import annotations

from app.llm.base import LLMError
from app.search.types import CandidateChunk
from app.services.generation_service import generate_answer_stream
from app.services.types import RetrievalOutcome


async def test_refused_outcome_never_touches_llm(db_session, fake_llm):
    outcome = RetrievalOutcome(query="q", status="refused", reason="no_candidates", metrics={"num_candidates": 0})

    events = [e async for e in generate_answer_stream(db_session, fake_llm, outcome)]

    assert events == [{"type": "refused", "reason": "no_candidates", "metrics": {"num_candidates": 0}}]
    assert fake_llm.calls == []


async def test_grounded_outcome_streams_tokens_then_final_event(db_session, make_chunk, fake_llm):
    chunk = make_chunk("Ollama runs language models locally on your machine.")
    candidate = CandidateChunk(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, content=chunk.content, score=1.0)
    outcome = RetrievalOutcome(query="what does Ollama do?", status="grounded", candidates=[candidate])
    fake_llm.pieces = ["Ollama ", "runs ", "language ", "models ", "locally."]

    events = [e async for e in generate_answer_stream(db_session, fake_llm, outcome)]

    token_events = [e for e in events if e["type"] == "token"]
    assert [e["text"] for e in token_events] == fake_llm.pieces

    final = events[-1]
    assert final["type"] == "done"
    assert final["answer"] == "Ollama runs language models locally."
    assert len(final["citations"]) == 1
    assert final["citations"][0]["marker"] == 1
    assert "overall_coverage" in final["grounding"]

    assert len(fake_llm.calls) == 1
    system_prompt, user_prompt = fake_llm.calls[0]
    assert "what does Ollama do?" in user_prompt
    assert "[1]" in user_prompt
    assert "don't know" in system_prompt.lower()


async def test_llm_error_yields_error_event_not_exception(db_session, make_chunk):
    class FailingLLM:
        model = "fake-failing-llm"

        async def generate_stream(self, system_prompt, user_prompt):
            if True:
                raise LLMError("boom")
            yield ""  # pragma: no cover - unreachable; keeps this an async generator function

    chunk = make_chunk("some content")
    candidate = CandidateChunk(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, content=chunk.content, score=1.0)
    outcome = RetrievalOutcome(query="q", status="grounded", candidates=[candidate])

    events = [e async for e in generate_answer_stream(db_session, FailingLLM(), outcome)]

    assert events == [{"type": "error", "message": "boom"}]
