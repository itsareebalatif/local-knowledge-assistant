from __future__ import annotations

from app.llm.base import LLMBackend, LLMError
from app.models import Message

REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user's follow-up question into a standalone question, using the "
    "most recent exchange in the conversation to resolve pronouns or implicit "
    "references (e.g. 'he', 'that', 'what about her'). If the follow-up is already "
    "a standalone question that doesn't depend on the conversation, return it "
    "completely unchanged — do not paraphrase or improve a question that's already "
    "fine. Never answer the question, never add information not already present in "
    "the conversation, and respond with ONLY the rewritten question and nothing else."
)


def _build_rewrite_prompt(query: str, recent_messages: list[Message]) -> str:
    history_lines = [f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in recent_messages]
    history_text = "\n".join(history_lines)
    return f"Recent conversation:\n{history_text}\n\nFollow-up question: {query}\n\nStandalone question:"


async def rewrite_query(llm: LLMBackend, query: str, recent_messages: list[Message]) -> str:
    """Resolves a follow-up's pronouns/implicit references against the most
    recent turn before retrieval runs — used only for retrieval and the
    generation prompt's "Question:" line, never stored in place of what the
    user actually typed. A first turn (no history) always passes through
    unchanged, with no LLM call at all. Fails open: any rewrite failure
    (LLM error, empty output) falls back to the original query rather than
    blocking the real request over an enhancement step."""
    if not recent_messages:
        return query

    prompt = _build_rewrite_prompt(query, recent_messages)
    pieces: list[str] = []
    try:
        async for piece in llm.generate_stream(REWRITE_SYSTEM_PROMPT, prompt):
            pieces.append(piece)
    except LLMError:
        return query

    rewritten = "".join(pieces).strip()
    return rewritten or query
