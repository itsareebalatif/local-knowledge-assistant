"""Pipeline 2: Prompt Synthesis, Grounded Generation & Citation (FR-4.5).

Consumes the RetrievalOutcome Pipeline 1 (retrieval_service.py) already
produced. If Pipeline 1 already refused, this yields a single refusal event
and never touches the LLM — that separation (Pipeline 1 can veto Pipeline 2
outright) is the entire point of the two-pipeline split, not something this
function re-checks itself.

Otherwise: builds the fact-restricted prompt, streams the answer piece by
piece (so an API layer can forward it over SSE token-by-token), and once the
full answer is assembled, runs the post-generation grounding verifier and
builds citations before yielding the final event.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import asdict

from sqlalchemy.orm import Session

from app.llm.base import LLMBackend, LLMError
from app.llm.citations import build_citations
from app.llm.grounding_verifier import verify_grounding
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.services.types import RetrievalOutcome

logger = logging.getLogger(__name__)


async def generate_answer_stream(
    db: Session, llm: LLMBackend, retrieval_outcome: RetrievalOutcome
) -> AsyncIterator[dict]:
    if retrieval_outcome.status != "grounded":
        yield {"type": "refused", "reason": retrieval_outcome.reason, "metrics": retrieval_outcome.metrics}
        return

    candidates = retrieval_outcome.candidates
    user_prompt, _marker_by_chunk_id = build_user_prompt(retrieval_outcome.query, candidates)

    answer_parts: list[str] = []
    try:
        async for piece in llm.generate_stream(SYSTEM_PROMPT, user_prompt):
            answer_parts.append(piece)
            yield {"type": "token", "text": piece}
    except LLMError as exc:
        logger.warning("LLM generation failed for query %r: %s", retrieval_outcome.query, exc)
        yield {"type": "error", "message": str(exc)}
        return

    full_answer = "".join(answer_parts)
    verification = verify_grounding(full_answer, candidates)
    citations = build_citations(db, candidates)

    if verification.unsupported_sentences:
        logger.info(
            "Answer to %r has %d unsupported sentence(s) out of %d (coverage=%.2f)",
            retrieval_outcome.query,
            len(verification.unsupported_sentences),
            len(verification.sentences),
            verification.overall_coverage,
        )

    yield {
        "type": "done",
        "answer": full_answer,
        "citations": [asdict(c) for c in citations],
        "grounding": {
            "overall_coverage": verification.overall_coverage,
            "unsupported_sentences": verification.unsupported_sentences,
        },
    }
