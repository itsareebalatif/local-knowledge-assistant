from __future__ import annotations

from app.llm.base import LLMError
from app.llm.query_rewriter import rewrite_query
from app.models import Message


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


async def test_rewrite_query_passes_through_unchanged_with_no_history(fake_llm):
    result = await rewrite_query(fake_llm, "what about his siblings?", [])
    assert result == "what about his siblings?"
    assert fake_llm.calls == []  # no history to resolve against — no LLM call at all


async def test_rewrite_query_calls_the_llm_with_history_and_returns_its_output(fake_llm):
    fake_llm.pieces = ["What ", "are ", "Hagrid's ", "siblings?"]
    history = [_msg("user", "who was Hagrid?"), _msg("assistant", "Hagrid was the Hogwarts gamekeeper.")]

    result = await rewrite_query(fake_llm, "what about his siblings?", history)

    assert result == "What are Hagrid's siblings?"
    assert len(fake_llm.calls) == 1
    system_prompt, user_prompt = fake_llm.calls[0]
    assert "who was Hagrid?" in user_prompt
    assert "Hagrid was the Hogwarts gamekeeper." in user_prompt
    assert "what about his siblings?" in user_prompt


async def test_rewrite_query_falls_back_to_original_on_llm_error():
    class FailingLLM:
        model = "failing"

        async def generate_stream(self, system_prompt, user_prompt):
            raise LLMError("unreachable")
            yield ""  # pragma: no cover

    history = [_msg("user", "who was Hagrid?"), _msg("assistant", "Hagrid was the Hogwarts gamekeeper.")]
    result = await rewrite_query(FailingLLM(), "what about his siblings?", history)
    assert result == "what about his siblings?"


async def test_rewrite_query_falls_back_to_original_on_empty_output(fake_llm):
    fake_llm.pieces = ["   "]
    history = [_msg("user", "who was Hagrid?"), _msg("assistant", "Hagrid was the Hogwarts gamekeeper.")]
    result = await rewrite_query(fake_llm, "what about his siblings?", history)
    assert result == "what about his siblings?"


async def test_rewrite_query_returns_a_standalone_question_unchanged(fake_llm):
    # When the model (correctly) echoes a standalone question back verbatim,
    # that's exactly the pass-through behavior we want — nothing special
    # beyond "trust what the LLM returned" needs to happen here.
    fake_llm.pieces = ["What ", "is ", "Ollama?"]
    history = [_msg("user", "who was Hagrid?"), _msg("assistant", "Hagrid was the Hogwarts gamekeeper.")]
    result = await rewrite_query(fake_llm, "What is Ollama?", history)
    assert result == "What is Ollama?"
