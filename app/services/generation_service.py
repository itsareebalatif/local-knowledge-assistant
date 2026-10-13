from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import asdict

from sqlalchemy.orm import Session

from app.llm.base import LLMBackend, LLMError
from app.llm.citations import build_citations
from app.llm.grounding_verifier import verify_grounding
from app.llm.prompts import build_system_prompt, build_user_prompt
from app.observability import observe, update
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
    with observe("llm-generate", as_type="generation", model=llm.model, input=user_prompt) as gen_obs:
        try:
            async for piece in llm.generate_stream(build_system_prompt(), user_prompt):
                answer_parts.append(piece)
                yield {"type": "token", "text": piece}
        except LLMError as exc:
            logger.warning("LLM generation failed for query %r: %s", retrieval_outcome.query, exc)
            update(gen_obs, level="ERROR", status_message=str(exc))
            yield {"type": "error", "message": str(exc)}
            return

        full_answer = "".join(answer_parts)
        # None for Ollama; update() drops None rather than sending a misleading zero
        usage = getattr(llm, "last_usage", None)
        update(gen_obs, output=full_answer, usage_details=usage)

    with observe(
        "grounding-verifier",
        as_type="evaluator",
        input={"answer": full_answer, "candidate_count": len(candidates)},
    ) as obs:
        verification = verify_grounding(full_answer, candidates)
        update(
            obs,
            output={
                "overall_coverage": verification.overall_coverage,
                "unsupported_sentences": verification.unsupported_sentences,
            },
        )

    citations = build_citations(db, candidates)

    if verification.unsupported_sentences:
        logger.info(
            "Answer to %r has %d unsupported sentence(s) out of %d (coverage=%.2f)",
            retrieval_outcome.query,
            len(verification.unsupported_sentences),
            len(verification.sentences),
            verification.overall_coverage,
        )

    grounding = {
        "overall_coverage": verification.overall_coverage,
        "unsupported_sentences": verification.unsupported_sentences,
    }

    # Every sentence failed grounding — the model answered from its own
    # knowledge rather than the retrieved chunks. `overall_coverage is None`
    # is the deliberate "I don't know" case (see grounding_verifier) and is
    # already a safe non-answer, not a hallucination to catch here.
    if verification.sentences and len(verification.unsupported_sentences) == len(verification.sentences):
        yield {"type": "refused", "reason": "ungrounded_answer", "metrics": grounding}
        return

    yield {
        "type": "done",
        "answer": full_answer,
        "citations": [asdict(c) for c in citations],
        "grounding": grounding,
    }
