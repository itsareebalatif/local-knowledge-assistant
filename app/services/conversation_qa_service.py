from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from app.config import get_settings
from app.llm.base import LLMBackend, LLMError
from app.models import Message

logger = logging.getLogger(__name__)

# Caps the transcript sent to the LLM. A long-running conversation's full
# history would otherwise grow the prompt unboundedly — the same token-budget
# risk that empties a reasoning model's visible output when it burns its
# whole cap on a huge input (confirmed live with Groq's gpt-oss-20b earlier
# in this project). 20 messages is 10 user/assistant turns, generous for
# "what did I ask earlier" while staying bounded.
_MAX_HISTORY_MESSAGES = 20


def build_system_prompt(assistant_name: str | None = None) -> str:
    """Shares the `assistant_name` persona setting with app.llm.prompts so
    the assistant presents as one consistent identity whether it's answering
    from documents or from the conversation itself."""
    settings = get_settings()
    assistant_name = settings.assistant_name if assistant_name is None else assistant_name

    return (
        f"ROLE\n"
        f"You are {assistant_name}. In this particular conversation turn, your job has "
        "switched: instead of answering a question about an uploaded document, you are "
        "answering a question about the conversation itself — things like 'what did I "
        "ask earlier?', 'summarize what we've discussed so far', or 'which of my "
        "questions didn't work?'. The user is asking you to look back at your own shared "
        "history together, not to look something up in a file.\n\n"
        "WHY THIS MATTERS\n"
        "This system normally answers strictly from uploaded documents and refuses to "
        "answer anything the documents don't cover — that's what keeps it trustworthy. "
        "But a question about the conversation itself was never a document question to "
        "begin with, so sending it through document search would either find nothing "
        "relevant and wrongly refuse, or find an unrelated passage and risk basing an "
        "answer about this chat on content from a book or report that has nothing to do "
        "with it. You are the dedicated path for this case: the one place in the system "
        "that is allowed to treat the conversation transcript itself as the source of "
        "truth, instead of the documents.\n\n"
        "TASK\n"
        "Use ONLY the transcript below to answer. The transcript is the complete record "
        "you have access to of what was said in this conversation (it may be truncated "
        "to the most recent messages if the conversation has been very long).\n\n"
        "RULES\n"
        "- The transcript ends BEFORE the current question — the current question is "
        "never itself part of the history being asked about, so don't mistake the "
        "question you're currently answering for something asked earlier.\n"
        "- If the transcript doesn't contain enough information to answer — for example "
        "the user asks about a turn that happened before the visible window of history — "
        "say so plainly rather than guessing or inventing something that sounds "
        "plausible.\n"
        "- Never claim something came from an uploaded document, and never pull in "
        "outside knowledge about the document's subject matter; this response is purely "
        "about what was actually said in this chat, nothing else.\n"
        "- Keep your tone conversational and direct, as if you're genuinely recalling "
        "the conversation with the user rather than reciting a transcript back at them."
    )


def _build_prompt(history: list[Message], query: str) -> str:
    # `history` must already exclude the current question — the caller is
    # responsible for taking this snapshot before the current question is
    # appended to the conversation. Confirmed live: passing the live
    # `conversation.messages` relationship after that append put the current
    # question in as the transcript's own last line, so "what did I ask
    # earlier?" answered itself instead of a genuinely prior question.
    trimmed = history[-_MAX_HISTORY_MESSAGES:]
    lines = [f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in trimmed]
    transcript = "\n".join(lines) if lines else "(no earlier messages in this conversation)"
    return f"Conversation transcript:\n{transcript}\n\nCurrent question: {query}\n\nAnswer:"


async def answer_from_history_stream(llm: LLMBackend, history: list[Message], query: str) -> AsyncIterator[dict]:
    prompt = _build_prompt(history, query)
    answer_parts: list[str] = []
    try:
        async for piece in llm.generate_stream(build_system_prompt(), prompt):
            answer_parts.append(piece)
            yield {"type": "token", "text": piece}
    except LLMError as exc:
        logger.warning("Conversation-memory answer failed for query %r: %s", query, exc)
        yield {"type": "error", "message": str(exc)}
        return

    yield {
        "type": "done",
        "answer": "".join(answer_parts),
        "citations": [],
        "grounding": None,
    }
