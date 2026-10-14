from __future__ import annotations

from app.llm.base import LLMError
from app.llm.query_router import ROUTE_DOMAIN, ROUTE_HYBRID, ROUTE_META, classify_and_rewrite
from app.models import Message


def _msg(role: str, content: str, citations: list[dict] | None = None) -> Message:
    return Message(role=role, content=content, citations=citations)


async def test_no_history_always_resolves_to_domain_with_no_llm_call(fake_llm):
    decision = await classify_and_rewrite(fake_llm, "what is Ollama?", [])
    assert decision.route == ROUTE_DOMAIN
    assert decision.search_query == "what is Ollama?"
    assert fake_llm.calls == []


async def test_hybrid_route_returns_the_rewritten_standalone_query(fake_llm):
    fake_llm.pieces = ["ROUTE: hybrid\nQUERY: What are Hagrid's siblings?"]
    history = [_msg("user", "who was Hagrid?"), _msg("assistant", "Hagrid was the Hogwarts gamekeeper.")]

    decision = await classify_and_rewrite(fake_llm, "what about his siblings?", history)

    assert decision.route == ROUTE_HYBRID
    assert decision.search_query == "What are Hagrid's siblings?"
    assert len(fake_llm.calls) == 1
    _system_prompt, user_prompt = fake_llm.calls[0]
    assert "who was Hagrid?" in user_prompt
    assert "what about his siblings?" in user_prompt


async def test_meta_route_is_recognized_for_a_question_about_the_conversation(fake_llm):
    fake_llm.pieces = ["ROUTE: meta\nQUERY: what did I ask earlier?"]
    history = [_msg("user", "who was Hagrid?"), _msg("assistant", "Hagrid was the Hogwarts gamekeeper.")]

    decision = await classify_and_rewrite(fake_llm, "what did I ask earlier?", history)

    assert decision.route == ROUTE_META


async def test_domain_route_keeps_the_query_unchanged(fake_llm):
    fake_llm.pieces = ["ROUTE: domain\nQUERY: What is Ollama?"]
    history = [_msg("user", "who was Hagrid?"), _msg("assistant", "Hagrid was the Hogwarts gamekeeper.")]

    decision = await classify_and_rewrite(fake_llm, "What is Ollama?", history)

    assert decision.route == ROUTE_DOMAIN
    assert decision.search_query == "What is Ollama?"


async def test_falls_back_to_domain_unchanged_on_llm_error():
    class FailingLLM:
        model = "failing"

        async def generate_stream(self, system_prompt, user_prompt):
            raise LLMError("unreachable")
            yield ""  # pragma: no cover

    history = [_msg("user", "who was Hagrid?"), _msg("assistant", "Hagrid was the Hogwarts gamekeeper.")]
    decision = await classify_and_rewrite(FailingLLM(), "what about his siblings?", history)
    assert decision.route == ROUTE_DOMAIN
    assert decision.search_query == "what about his siblings?"


async def test_falls_back_to_domain_unchanged_on_unparseable_output(fake_llm):
    fake_llm.pieces = ["I'm not sure what you mean."]
    history = [_msg("user", "who was Hagrid?"), _msg("assistant", "Hagrid was the Hogwarts gamekeeper.")]

    decision = await classify_and_rewrite(fake_llm, "what about his siblings?", history)

    assert decision.route == ROUTE_DOMAIN
    assert decision.search_query == "what about his siblings?"


async def test_a_route_value_wrapped_in_angle_brackets_still_parses(fake_llm):
    # Regression for a real, confirmed-live bug: the model sometimes echoes
    # the prompt's own "<meta|domain|hybrid>" placeholder syntax literally
    # ("ROUTE: <meta>" instead of "ROUTE: meta"), and an exact-match lookup
    # missed it, silently sending a genuine meta question into document
    # retrieval where it got refused as "not enough information."
    fake_llm.pieces = ["ROUTE: <meta>\nQUERY: <what did I ask earlier?>"]
    history = [_msg("user", "who was Hagrid?"), _msg("assistant", "Hagrid was the Hogwarts gamekeeper.")]

    decision = await classify_and_rewrite(fake_llm, "what did I ask earlier?", history)

    assert decision.route == ROUTE_META
    assert decision.search_query == "what did I ask earlier?"


async def test_an_invalid_route_label_falls_back_to_domain_but_keeps_a_valid_rewritten_query(fake_llm):
    fake_llm.pieces = ["ROUTE: something-weird\nQUERY: What are Hagrid's siblings?"]
    history = [_msg("user", "who was Hagrid?"), _msg("assistant", "Hagrid was the Hogwarts gamekeeper.")]

    decision = await classify_and_rewrite(fake_llm, "what about his siblings?", history)

    assert decision.route == ROUTE_DOMAIN
    assert decision.search_query == "What are Hagrid's siblings?"


async def test_the_last_assistant_turns_cited_snippets_are_included_for_pronoun_resolution(fake_llm):
    # Regression for a real, confirmed-live bug: resolving "it" against only
    # the recorded answer text ("Hagrid brought the baby on the motorcycle")
    # is ambiguous between "the baby" and "the motorcycle" — the answer never
    # restates which one was literally borrowed. The real cited source text
    # ("Borrowed it, Professor Dumbledore, sir... Young Sirius Black lent it
    # me") isn't ambiguous at all. It must reach the rewrite prompt.
    fake_llm.pieces = ["ROUTE: hybrid\nQUERY: Where did Hagrid get the motorbike?"]
    history = [
        _msg("user", "Who brought the baby on the flying motorcycle?"),
        _msg(
            "assistant",
            "Hagrid brought the baby on the flying motorcycle.",
            citations=[
                {
                    "marker": 1,
                    "chunk_id": 42,
                    "doc_id": 7,
                    "file_name": "book.pdf",
                    "source_url": "/api/documents/7",
                    "snippet": "Borrowed it, Professor Dumbledore, sir... Young Sirius Black lent it me.",
                }
            ],
        ),
    ]

    await classify_and_rewrite(fake_llm, "Where did he get it?", history)

    assert len(fake_llm.calls) == 1
    _system_prompt, user_prompt = fake_llm.calls[0]
    assert "Young Sirius Black lent it me" in user_prompt
    assert "not just the paraphrased answer above" in user_prompt


async def test_no_source_passages_section_when_the_last_answer_has_no_citations(fake_llm):
    # A meta answer (or a refusal) has no citations at all — the prompt must
    # not break or add an empty/misleading "source passages" section.
    fake_llm.pieces = ["ROUTE: domain\nQUERY: What is Ollama?"]
    history = [_msg("user", "what did I ask earlier?"), _msg("assistant", "You asked about Hagrid.")]

    await classify_and_rewrite(fake_llm, "What is Ollama?", history)

    _system_prompt, user_prompt = fake_llm.calls[0]
    assert "Source passages" not in user_prompt


async def test_only_the_most_recent_assistant_turns_citations_are_used(fake_llm):
    # Older turns' sources would be a less relevant distraction — the prompt
    # already frames this as "the most recent exchange" elsewhere, so the
    # citations included must match that, not every turn in the window.
    fake_llm.pieces = ["ROUTE: hybrid\nQUERY: anything"]
    history = [
        _msg("user", "who was Hagrid?"),
        _msg("assistant", "Hagrid was the gamekeeper.", citations=[{"marker": 1, "snippet": "OLD_TURN_SNIPPET"}]),
        _msg("user", "what about Dumbledore?"),
        _msg("assistant", "Dumbledore was the headmaster.", citations=[{"marker": 1, "snippet": "NEW_TURN_SNIPPET"}]),
    ]

    await classify_and_rewrite(fake_llm, "what about his office?", history)

    _system_prompt, user_prompt = fake_llm.calls[0]
    assert "NEW_TURN_SNIPPET" in user_prompt
    assert "OLD_TURN_SNIPPET" not in user_prompt
