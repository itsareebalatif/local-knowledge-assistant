from __future__ import annotations

from app.llm.base import LLMError
from app.models import Message
from app.services.conversation_qa_service import _MAX_HISTORY_MESSAGES, answer_from_history_stream, build_system_prompt


def _history(*messages: tuple[str, str]) -> list[Message]:
    return [Message(role=role, content=content) for role, content in messages]


def test_build_system_prompt_accepts_an_explicit_persona_override():
    prompt = build_system_prompt(assistant_name="Acme Support Bot")
    assert "You are Acme Support Bot" in prompt


def test_build_system_prompt_shares_the_persona_setting_with_the_document_prompt():
    from app.config import get_settings
    from app.llm.prompts import build_system_prompt as build_document_prompt

    name = get_settings().assistant_name
    assert f"You are {name}." in build_system_prompt()
    assert f"You are {name}," in build_document_prompt()


async def test_answers_directly_from_the_transcript_with_no_retrieval(fake_llm):
    fake_llm.pieces = ["You ", "asked ", "about ", "Hagrid."]
    history = _history(("user", "who was Hagrid?"), ("assistant", "Hagrid was the Hogwarts gamekeeper."))

    events = [e async for e in answer_from_history_stream(fake_llm, history, "what did I ask earlier?")]

    assert events[-1] == {
        "type": "done",
        "answer": "You asked about Hagrid.",
        "citations": [],
        "grounding": None,
    }
    assert len(fake_llm.calls) == 1
    system_prompt, user_prompt = fake_llm.calls[0]
    assert "who was Hagrid?" in user_prompt
    assert "uploaded document" in system_prompt
    assert "Never claim something came from an uploaded document" in system_prompt


async def test_the_current_question_is_never_included_as_part_of_the_history(fake_llm):
    # Regression for a real, confirmed-live bug: the caller used to pass the
    # live `conversation.messages` relationship *after* the current question
    # had already been appended to it, so the transcript's own last line was
    # the current question — "what did I ask earlier?" answered itself. The
    # fix makes the caller responsible for passing a pre-append snapshot;
    # this proves the function trusts exactly what it's given and doesn't
    # independently re-fetch anything that could include the current turn.
    fake_llm.pieces = ["ok"]
    history = _history(("user", "who was Hagrid?"), ("assistant", "Hagrid was the Hogwarts gamekeeper."))

    [e async for e in answer_from_history_stream(fake_llm, history, "what did I ask earlier?")]

    _, user_prompt = fake_llm.calls[0]
    assert user_prompt.count("what did I ask earlier?") == 1  # only in "Current question:", not in the transcript
    assert "who was Hagrid?" in user_prompt


async def test_caps_the_transcript_sent_to_the_llm(fake_llm):
    fake_llm.pieces = ["ok"]
    messages = [("user", f"question {i}") if i % 2 == 0 else ("assistant", f"answer {i}") for i in range(40)]
    history = _history(*messages)

    events = [e async for e in answer_from_history_stream(fake_llm, history, "what did I ask first?")]
    assert events  # sanity: it still produced output

    _, user_prompt = fake_llm.calls[-1]
    assert "question 0" not in user_prompt  # the oldest messages were dropped
    assert f"question {40 - _MAX_HISTORY_MESSAGES}" in user_prompt  # the window's oldest surviving message


async def test_llm_failure_yields_an_error_event_not_a_crash():
    class FailingLLM:
        model = "failing"

        async def generate_stream(self, system_prompt, user_prompt):
            raise LLMError("the model server is unreachable")
            yield ""  # pragma: no cover

    history = _history(("user", "hello"), ("assistant", "hi there"))
    events = [e async for e in answer_from_history_stream(FailingLLM(), history, "what did I say?")]

    assert events == [{"type": "error", "message": "the model server is unreachable"}]
