from __future__ import annotations

from dataclasses import dataclass

from app.llm.base import LLMBackend, LLMError
from app.models import Message

ROUTE_META = "meta"
ROUTE_DOMAIN = "domain"
ROUTE_HYBRID = "hybrid"
_VALID_ROUTES = {ROUTE_META, ROUTE_DOMAIN, ROUTE_HYBRID}

# Kept deliberately tighter than the two answer-facing prompts below (see
# app/llm/prompts.py and app/services/conversation_qa_service.py). This one
# runs on every single follow-up question, not just once per full answer, and
# a long system prompt makes a reasoning model spend more hidden "thinking"
# tokens before it writes anything — confirmed live earlier in this project
# that burning the whole token budget on reasoning leaves the visible output
# empty. A router prompt that's too elaborate would make that failure more
# likely on the very step meant to run cheaply and reliably every time.
ROUTER_SYSTEM_PROMPT = (
    "ROLE\n"
    "You are a routing classifier that sits in front of a document "
    "question-answering system. You are not the one who answers the user's question — "
    "your only job is to decide, before anything else happens, where their latest "
    "question should go.\n\n"
    "WHY THIS MATTERS\n"
    "This system can answer two very different kinds of questions: questions about the "
    "user's uploaded documents, and questions about the conversation the user is having "
    "with the assistant right now (e.g. 'what did I ask you earlier?'). If a question "
    "about the conversation gets sent to document search by mistake, it will either find "
    "nothing and wrongly tell the user their documents don't cover it, or worse, find an "
    "unrelated passage and answer from the wrong source entirely. Your classification is "
    "what keeps each question going to the right place.\n\n"
    "TASK\n"
    "Classify the user's latest question against the recent conversation into exactly "
    "one of three routes below, and provide a query to use next.\n\n"
    "ROUTES\n"
    "- meta — the question is about this conversation itself, not about any uploaded "
    'document (e.g. "what did I ask earlier", "summarize what we discussed", "which '
    'question broke").\n'
    "- domain — the question stands on its own and doesn't depend on anything said "
    "earlier in the conversation (a question can be about the documents and still be "
    "domain, as long as it doesn't need the conversation to make sense).\n"
    "- hybrid — the question is a follow-up that depends on the recent conversation to "
    'make sense (pronouns, "what about...", implicit references) but is still asking '
    "about the uploaded documents, not about the conversation itself.\n\n"
    "OUTPUT FORMAT\n"
    "Respond with exactly two lines and nothing else — no extra words, no brackets, no "
    "punctuation around the value. Write the bare word, like this real example:\n"
    "ROUTE: meta\n"
    "QUERY: what did I ask earlier?\n"
    "The ROUTE value must be exactly one of these three bare words: meta, domain, or "
    "hybrid.\n\n"
    "RULES\n"
    "- For the QUERY line: if route is hybrid, rewrite the question into a standalone "
    "question using the conversation to resolve pronouns or implicit references, so it "
    "can be searched against the documents on its own, without needing the earlier "
    "turns. Otherwise repeat the original question unchanged.\n"
    "- Never answer the question itself, and never add information not already present "
    "in the conversation — your output is a routing decision, not an answer."
)


@dataclass
class RouteDecision:
    route: str
    search_query: str


def _build_router_prompt(query: str, recent_messages: list[Message]) -> str:
    history_lines = [f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in recent_messages]
    history_text = "\n".join(history_lines)
    prompt = f"Recent conversation:\n{history_text}\n\nLatest question: {query}"

    # Resolving a pronoun against the conversation's own recorded answer can
    # fail when that answer's phrasing doesn't happen to restate the exact
    # noun a later "it"/"them" needs — confirmed live: an answer said "Hagrid
    # brought the baby on the motorcycle" without repeating "the motorbike"
    # as the specific thing that was borrowed, so a later "where did he get
    # it?" had two candidate nouns (the baby, the motorcycle) to resolve
    # against and picked the wrong one. The real cited source text doesn't
    # have that ambiguity — grounding resolution in it instead of just the
    # paraphrase closes the gap. Only the most recent assistant turn's
    # citations are used, matching "most recent exchange" elsewhere in this
    # prompt — older turns' sources would be a less relevant distraction.
    last_assistant_citations = next(
        (m.citations for m in reversed(recent_messages) if m.role == "assistant" and m.citations), None
    )
    if last_assistant_citations:
        snippets = [c["snippet"] for c in last_assistant_citations if c.get("snippet")]
        if snippets:
            sources_text = "\n".join(f"- {s}" for s in snippets)
            prompt += (
                "\n\nSource passages the most recent answer was actually grounded in "
                "(use these — not just the paraphrased answer above — to resolve any "
                f"pronoun or implicit reference in the latest question):\n{sources_text}"
            )
    return prompt


_WRAPPING_CHARS = "<>\"'*. \t"


def _parse_decision(raw: str, fallback_query: str) -> RouteDecision:
    # Confirmed live: the model sometimes echoes the system prompt's own
    # "<meta|domain|hybrid>" placeholder syntax literally, returning
    # "ROUTE: <meta>" instead of "ROUTE: meta". An exact-match lookup
    # against _VALID_ROUTES then misses it and silently falls back to
    # domain — sending a genuine meta question into document retrieval,
    # where it gets refused for "not enough information in the passages"
    # even though the classification itself was actually correct. Stripping
    # the wrapping punctuation before matching makes this robust to that
    # without needing the model to format it perfectly.
    route = ROUTE_DOMAIN
    search_query = fallback_query
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("route:"):
            candidate = line.split(":", 1)[1].strip(_WRAPPING_CHARS).lower()
            if candidate in _VALID_ROUTES:
                route = candidate
        elif line.lower().startswith("query:"):
            candidate = line.split(":", 1)[1].strip().strip(_WRAPPING_CHARS)
            if candidate:
                search_query = candidate
    return RouteDecision(route=route, search_query=search_query)


async def classify_and_rewrite(llm: LLMBackend, query: str, recent_messages: list[Message]) -> RouteDecision:
    """Decides, in one LLM call, whether a question is about the conversation
    itself (meta), stands alone (domain), or is a document follow-up that
    needs history to resolve (hybrid) — and for hybrid, rewrites it into a
    standalone question in the same call. One call instead of two separate
    classify-then-rewrite steps, same fail-open shape as the rewriter it
    replaces: a first turn (no history) always resolves to domain with the
    query unchanged, with no LLM call at all, and any error or unparseable
    output falls back to the exact same thing — domain, query unchanged —
    which is the pipeline's original, pre-routing behavior."""
    if not recent_messages:
        return RouteDecision(route=ROUTE_DOMAIN, search_query=query)

    prompt = _build_router_prompt(query, recent_messages)
    pieces: list[str] = []
    try:
        async for piece in llm.generate_stream(ROUTER_SYSTEM_PROMPT, prompt):
            pieces.append(piece)
    except LLMError:
        return RouteDecision(route=ROUTE_DOMAIN, search_query=query)

    return _parse_decision("".join(pieces).strip(), query)
