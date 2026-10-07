"""Formal validation of the two hallucination-prevention guardrails.

This file exists separately from the scattered unit tests in
test_grounding_gate.py / test_grounding_verifier.py / test_retrieval_service.py
/ test_generation_service.py on purpose: those prove each piece works in
isolation; this proves the two-guardrail STORY end-to-end, as adversarial
scenarios, in one place a reviewer can read top to bottom to answer "does
this system actually prevent hallucination":

  Guardrail 1 (pre-generation, Pipeline 1): app.search.grounding_gate — can
  the system find anything good enough to answer from? If not, refuse
  before the LLM is ever touched.

  Guardrail 2 (post-generation, Pipeline 2): app.llm.grounding_verifier —
  even with good context, did the model's actual answer stay inside it? If
  not, flag which sentences aren't supported.

Neither guardrail silently lets a bad answer through unflagged — that's
what every test below actually checks.
"""

from __future__ import annotations

from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.services.generation_service import generate_answer_stream
from app.services.retrieval_service import retrieve_and_verify


class HallucinatingLLM:
    """A model that ignores its context entirely and states unrelated
    'facts' instead — the thing Guardrail 2 exists to catch."""

    def __init__(self, claim: str):
        self.claim = claim
        self.calls: list[tuple[str, str]] = []
        self.model = "fake-hallucinating-llm"

    async def generate_stream(self, system_prompt: str, user_prompt: str):
        self.calls.append((system_prompt, user_prompt))
        yield self.claim


class FaithfulLLM:
    """A model that only ever repeats what's actually in the prompt's
    context — the control case proving the verifier doesn't cry wolf on a
    good answer."""

    def __init__(self, answer: str):
        self.answer = answer
        self.calls: list[tuple[str, str]] = []
        self.model = "fake-faithful-llm"

    async def generate_stream(self, system_prompt: str, user_prompt: str):
        self.calls.append((system_prompt, user_prompt))
        yield self.answer


# ---------------------------------------------------------------------
# Guardrail 1: refuse BEFORE generation when there's nothing to ground on
# ---------------------------------------------------------------------


async def test_guardrail_1_refuses_a_query_with_no_relevant_content_at_all(
    db_session, make_chunk, fake_embedder, fake_vector_store, user
):
    make_chunk("Notes about gardening, soil pH, and composting techniques.")
    # fake_vector_store deliberately left empty — nothing embeddable found.
    llm = HallucinatingLLM("The Eiffel Tower was built in 1850 by aliens.")

    outcome = await retrieve_and_verify(
        db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(),
        "What is the capital of a fictional country that doesn't exist?", user.user_id,
    )
    assert outcome.status == "refused"

    events = [e async for e in generate_answer_stream(db_session, llm, outcome)]
    assert events == [{"type": "refused", "reason": outcome.reason, "metrics": outcome.metrics}]
    assert llm.calls == [], "Guardrail 1 failed: the LLM was invoked despite no grounded context"


async def test_guardrail_1_refuses_when_retrieved_content_shares_no_vocabulary_with_query(
    db_session, make_chunk, fake_embedder, fake_vector_store, user
):
    # Score passes (it's the only/best match returned) but the actual words
    # in the query never appear in what was retrieved — this is the
    # embedding-false-positive case keyword coverage exists to catch.
    chunk = make_chunk("The quarterly financial report shows revenue growth of twelve percent.")
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": user.user_id}],
    )
    llm = HallucinatingLLM("Ollama supports Python, JavaScript, and Rust bindings.")

    outcome = await retrieve_and_verify(
        db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(),
        "What programming languages does Ollama support?", user.user_id,
    )

    assert outcome.status == "refused"
    assert outcome.reason == "low_keyword_coverage"

    events = [e async for e in generate_answer_stream(db_session, llm, outcome)]
    assert events[0]["type"] == "refused"
    assert llm.calls == [], "Guardrail 1 failed: the LLM was invoked on irrelevant context"


# ---------------------------------------------------------------------
# Guardrail 2: flag a hallucinated sentence AFTER generation, even with good context
# ---------------------------------------------------------------------


async def test_guardrail_2_flags_an_answer_that_ignores_good_context(
    db_session, make_chunk, fake_embedder, fake_vector_store, user
):
    chunk = make_chunk("Ollama runs language models locally on your own machine using llama.cpp.")
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": user.user_id}],
    )
    # Context is good — retrieval WILL ground this query. The model just
    # chooses to answer with something the context never said.
    llm = HallucinatingLLM("The moon is made of green cheese and unicorns live there.")

    outcome = await retrieve_and_verify(
        db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(),
        "What does Ollama run models on?", user.user_id,
    )
    assert outcome.status == "grounded", "test setup error: this case should pass Guardrail 1"

    events = [e async for e in generate_answer_stream(db_session, llm, outcome)]
    assert llm.calls, "the LLM should have been invoked — Guardrail 1 passed"

    final = events[-1]
    assert final["type"] == "done"
    assert final["grounding"]["unsupported_sentences"], (
        "Guardrail 2 failed: a hallucinated answer was delivered with no grounding flag at all"
    )
    assert final["grounding"]["overall_coverage"] < 0.3


async def test_guardrail_2_does_not_flag_an_answer_that_stays_in_context(
    db_session, make_chunk, fake_embedder, fake_vector_store, user
):
    chunk = make_chunk("Ollama runs language models locally on your own machine using llama.cpp.")
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": user.user_id}],
    )
    llm = FaithfulLLM("Ollama runs language models locally on your own machine.")

    outcome = await retrieve_and_verify(
        db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(),
        "What does Ollama run models on?", user.user_id,
    )
    assert outcome.status == "grounded"

    events = [e async for e in generate_answer_stream(db_session, llm, outcome)]
    final = events[-1]

    assert final["grounding"]["unsupported_sentences"] == [], (
        "Guardrail 2 false-positived on a faithful, well-grounded answer"
    )
    assert final["grounding"]["overall_coverage"] > 0.7


async def test_guardrail_2_flags_only_the_specific_hallucinated_sentence_not_the_whole_answer(
    db_session, make_chunk, fake_embedder, fake_vector_store, user
):
    # Realistic case: a partially-grounded answer — one true sentence, one
    # fabricated one. The verifier should isolate exactly the bad sentence,
    # not blanket-reject (or blanket-accept) the whole response.
    chunk = make_chunk("Ollama runs language models locally on your own machine.")
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": user.user_id}],
    )
    mixed_answer = (
        "Ollama runs language models locally on your own machine. "
        "It was invented in 1823 by a committee of sentient dolphins."
    )
    llm = FaithfulLLM(mixed_answer)

    outcome = await retrieve_and_verify(
        db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(),
        "What does Ollama run models on?", user.user_id,
    )
    events = [e async for e in generate_answer_stream(db_session, llm, outcome)]
    final = events[-1]

    unsupported = final["grounding"]["unsupported_sentences"]
    assert len(unsupported) == 1
    assert "dolphins" in unsupported[0].lower()
    assert "runs language models locally" not in unsupported[0].lower()
