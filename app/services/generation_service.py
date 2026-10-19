from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import asdict

from sqlalchemy.orm import Session

from app.llm.base import LLMBackend, LLMError
from app.llm.citations import build_citations
from app.llm.grounding_verifier import verify_grounding
from app.llm.prompts import append_user_facts, build_system_prompt, build_user_prompt
from app.observability import observe, update
from app.services.types import RetrievalOutcome

logger = logging.getLogger(__name__)


async def generate_answer_stream(
    db: Session,
    llm: LLMBackend,
    retrieval_outcome: RetrievalOutcome,
    retry_hint: str | None = None,
    user_facts: list[str] | None = None,
) -> AsyncIterator[dict]:
    if retrieval_outcome.status != "grounded":
        yield {"type": "refused", "reason": retrieval_outcome.reason, "metrics": retrieval_outcome.metrics}
        return

    candidates = retrieval_outcome.candidates
    user_prompt, _marker_by_chunk_id = build_user_prompt(retrieval_outcome.query, candidates)
    # Set only on a self-correction retry (app.llm.orchestration), after a
    # prior attempt came back empty, fully ungrounded, or only partially
    # supported — appended rather than templated into build_system_prompt so
    # the first, normal attempt's prompt (the overwhelming majority of
    # calls) is completely unaffected.
    system_prompt = build_system_prompt()
    if retry_hint:
        system_prompt = f"{system_prompt}\n\n{retry_hint}"
    system_prompt = append_user_facts(system_prompt, user_facts)

    answer_parts: list[str] = []
    with observe("llm-generate", as_type="generation", model=llm.model, input=user_prompt) as gen_obs:
        try:
            async for piece in llm.generate_stream(system_prompt, user_prompt):
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

    # Real, confirmed-live bug: a model (observed with a reasoning model
    # over OpenRouter) can stream zero real content and still finish
    # "successfully" — no LLMError, no detect_reasoning_exhaustion trip,
    # just an empty answer. verify_grounding("", ...) treats "no sentences
    # to check" as vacuously 100% grounded (see its own test,
    # test_empty_answer_returns_trivial_full_coverage) — correct for that
    # function in isolation, but it let a blank answer reach the user
    # labeled "Grounding coverage: 100%", which reads as a real, confident,
    # fully-supported answer that happens to have no visible text. Caught
    # here, before grounding verification, not inside it.
    if not full_answer.strip():
        logger.warning("LLM returned an empty answer for query %r", retrieval_outcome.query)
        yield {"type": "refused", "reason": "empty_answer", "metrics": {}}
        return

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

    citations = build_citations(db, candidates, full_answer)

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
