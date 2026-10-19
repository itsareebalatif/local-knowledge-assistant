from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy, StreamWriter

from app.config import get_settings
from app.llm.base import LLMBackend
from app.llm.query_router import ROUTE_DOMAIN, ROUTE_META, classify_and_rewrite
from app.observability import observe, update
from app.search.types import CandidateChunk
from app.services.conversation_qa_service import answer_from_history_stream
from app.services.generation_service import generate_answer_stream
from app.services.retrieval_service import (
    fuse_and_verify,
    run_bm25_search,
    run_graph_search,
    run_vector_search,
    source_top_k,
)
from app.services.types import RetrievalOutcome


_HEALTH_NAMESPACE = ("llm_backend_health",)


def _backend_key(llm: LLMBackend) -> str:
    return getattr(llm, "provider_name", None) or llm.model


async def _is_cooling_down(store: BaseStore | None, llm: LLMBackend) -> bool:
    if store is None:
        return False
    settings = get_settings()
    record = await store.aget(_HEALTH_NAMESPACE, _backend_key(llm))
    if record is None:
        return False
    failures = record.value.get("consecutive_failures", 0)
    last_failure_at = record.value.get("last_failure_at")
    if failures < settings.circuit_breaker_failure_threshold or last_failure_at is None:
        return False
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_failure_at)).total_seconds()
    return elapsed < settings.circuit_breaker_cooldown_seconds


async def _record_backend_outcome(store: BaseStore | None, llm: LLMBackend, succeeded: bool) -> None:
    if store is None:
        return
    key = _backend_key(llm)
    if succeeded:
        await store.aput(_HEALTH_NAMESPACE, key, {"consecutive_failures": 0, "last_failure_at": None})
        return
    record = await store.aget(_HEALTH_NAMESPACE, key)
    failures = (record.value.get("consecutive_failures", 0) if record else 0) + 1
    await store.aput(
        _HEALTH_NAMESPACE,
        key,
        {"consecutive_failures": failures, "last_failure_at": datetime.now(timezone.utc).isoformat()},
    )



_USER_MEMORY_NAMESPACE = ("user_memory",)
_MAX_REMEMBERED_FACTS = 20

_REMEMBER_PATTERN = re.compile(r"\bremember(?:\s+that)?\s+(.+)", re.IGNORECASE)


async def _recall_and_update_memory(store: BaseStore | None, user_id: int, query: str) -> list[str]:
    if store is None:
        return []
    key = str(user_id)
    record = await store.aget(_USER_MEMORY_NAMESPACE, key)
    facts = list(record.value.get("facts", [])) if record else []

    match = _REMEMBER_PATTERN.search(query)
    if not match:
        return facts
    fact = match.group(1).strip().rstrip(".!?")
    if not fact or fact in facts:
        return facts

    updated = (facts + [fact])[-_MAX_REMEMBERED_FACTS:]
    await store.aput(_USER_MEMORY_NAMESPACE, key, {"facts": updated})
    return updated




class RagState(TypedDict, total=False):
    query: str
    user_id: int
    route: str
    search_query: str
    user_facts: list[str]
    bm25_hits: list[CandidateChunk]
    vector_hits: list[CandidateChunk]
    graph_hits: list[CandidateChunk]
    retrieval_outcome: RetrievalOutcome
    llm_index: int
    regen_count: int
    attempt_accepted: bool
    status: str
    reason: str
    answer: str
    citations: list[dict]
    grounding: dict | None


async def _stream_and_collect(event_source, buffer: list[dict]) -> dict[str, Any]:
    status = "error"
    reason = "no_response_generated"
    answer = ""
    citations: list[dict] = []
    grounding: dict | None = None
    async for event in event_source:
        buffer.append(event)
        if event["type"] == "done":
            status = "done"
            answer = event["answer"]
            citations = event["citations"]
            grounding = event["grounding"]
        elif event["type"] == "error":
            status = "error"
            reason = event["message"]
        elif event["type"] == "refused":
            status = "refused"
            reason = event["reason"]
    return {"status": status, "reason": reason, "answer": answer, "citations": citations, "grounding": grounding}


_LOW_GROUNDING_RETRY_HINT = (
    "A previous attempt at this answer either included claims the passages don't "
    "support, came back empty, or wasn't grounded at all. This time, only state what "
    "you can point to a specific numbered passage for — if you can't support a claim "
    "that way, drop it instead of guessing."
)


def _needs_regeneration(collected: dict[str, Any]) -> bool:
    if collected["status"] == "refused":
        return collected.get("reason") in ("ungrounded_answer", "empty_answer")
    if collected["status"] == "done":
        coverage = (collected.get("grounding") or {}).get("overall_coverage")
        return coverage is not None and coverage < get_settings().min_regeneration_coverage
    return False


async def _route_node(state: RagState, config: RunnableConfig, store: BaseStore) -> dict:
    cfg = config["configurable"]
    # Always runs first on every turn — the one place that's guaranteed to
    # reset this turn's loop counters before "meta"/"generate" read them
    # (see the module-level note on why a stale counter from a prior turn
    # on this same checkpointed thread would otherwise leak forward).
    reset = {"llm_index": 0, "regen_count": 0, "attempt_accepted": False}
    user_facts = await _recall_and_update_memory(store, state["user_id"], state["query"])
    if not get_settings().smart_routing_enabled:
        return {**reset, "route": ROUTE_DOMAIN, "search_query": state["query"], "user_facts": user_facts}
    with observe("query-router", input={"query": state["query"]}) as obs:
        decision = await classify_and_rewrite(cfg["llm"], state["query"], cfg["recent_messages"])
        update(obs, output={"route": decision.route, "search_query": decision.search_query})
    return {**reset, "route": decision.route, "search_query": decision.search_query, "user_facts": user_facts}


def _route_branch(state: RagState) -> list[str] | str:
    # Fans out to all three search nodes in parallel instead of a single
    # "retrieve" node running its own internal asyncio.gather — each one is
    # now a distinct graph step with its own RetryPolicy and its own span,
    # all converging on "fuse" once every one of them has returned.
    if state["route"] == ROUTE_META:
        return "meta"
    return ["bm25_search", "vector_search", "graph_search"]


async def _meta_node(state: RagState, config: RunnableConfig, writer: StreamWriter, store: BaseStore) -> dict:
    cfg = config["configurable"]
    llm_chain = cfg["llm_chain"]
    llm_index = state.get("llm_index", 0)
    is_last_backend = llm_index >= len(llm_chain) - 1
    llm = llm_chain[llm_index]

    if not is_last_backend and await _is_cooling_down(store, llm):
        # Known-bad right now (tripped the circuit breaker recently) — skip
        # it without even trying, instead of making this request pay its
        # full timeout to rediscover what the last few requests already did.
        return {"llm_index": llm_index + 1, "attempt_accepted": False}

    buffered: list[dict] = []
    with observe("meta-attempt", input={"llm_index": llm_index}) as obs:
        collected = await _stream_and_collect(
            answer_from_history_stream(llm, cfg["meta_history"], state["query"], state.get("user_facts")), buffered
        )
        update(obs, output={"status": collected["status"]})

    produced_output = any(event["type"] in ("token", "done") for event in buffered)
    await _record_backend_outcome(store, llm, succeeded=produced_output or collected["status"] != "error")
    if collected["status"] == "error" and not produced_output and not is_last_backend:
        # Discarded outright — the client never sees this attempt.
        return {"llm_index": llm_index + 1, "attempt_accepted": False}

    for event in buffered:
        writer(event)
    if collected["status"] == "done":
        collected["status"] = "conversation"
    return {**collected, "attempt_accepted": True}


def _meta_branch(state: RagState) -> str:
    return END if state.get("attempt_accepted") else "meta"


async def _bm25_node(state: RagState, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    bm25_top_k, _, _, _, _ = source_top_k(state["search_query"])
    hits = await run_bm25_search(cfg["db"], state["search_query"], bm25_top_k, state["user_id"])
    return {"bm25_hits": hits}


async def _vector_node(state: RagState, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    _, vector_top_k, _, _, _ = source_top_k(state["search_query"])
    hits = await run_vector_search(
        cfg["db"], cfg["embedder"], cfg["vector_store"], state["search_query"], vector_top_k, state["user_id"]
    )
    return {"vector_hits": hits}


async def _graph_node(state: RagState, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    _, _, graph_top_k, _, _ = source_top_k(state["search_query"])
    hits = await run_graph_search(
        cfg["db"], cfg["graph_builder"], cfg["extractor"], state["search_query"], graph_top_k, state["user_id"]
    )
    return {"graph_hits": hits}


async def _fuse_node(state: RagState, config: RunnableConfig) -> dict:
    cfg = config["configurable"]
    outcome = await fuse_and_verify(
        state["search_query"],
        state["bm25_hits"],
        state["vector_hits"],
        state["graph_hits"],
        cfg["embedder"],
        cfg.get("reranker"),
    )
    return {"retrieval_outcome": outcome}


def _fuse_branch(state: RagState) -> str:
    return "generate" if state["retrieval_outcome"].status == "grounded" else "refused"


async def _refused_node(state: RagState, writer: StreamWriter) -> dict:
    outcome = state["retrieval_outcome"]
    writer({"type": "refused", "reason": outcome.reason, "metrics": outcome.metrics})
    return {"status": "refused", "reason": outcome.reason, "answer": "", "citations": [], "grounding": None}


async def _generate_node(
    state: RagState, config: RunnableConfig, writer: StreamWriter, store: BaseStore
) -> dict:
    cfg = config["configurable"]
    llm_chain = cfg["llm_chain"]
    llm_index = state.get("llm_index", 0)
    regen_count = state.get("regen_count", 0)
    is_last_backend = llm_index >= len(llm_chain) - 1
    retry_hint = _LOW_GROUNDING_RETRY_HINT if regen_count > 0 else None
    llm = llm_chain[llm_index]

    if not is_last_backend and await _is_cooling_down(store, llm):
        return {"llm_index": llm_index + 1, "attempt_accepted": False}

    buffered: list[dict] = []
    attempt_input = {"llm_index": llm_index, "regen_count": regen_count, "retry_hint_used": bool(retry_hint)}
    with observe("generate-attempt", input=attempt_input) as obs:
        collected = await _stream_and_collect(
            generate_answer_stream(
                cfg["db"], llm, state["retrieval_outcome"], retry_hint=retry_hint, user_facts=state.get("user_facts")
            ),
            buffered,
        )
        coverage = (collected.get("grounding") or {}).get("overall_coverage")
        update(obs, output={"status": collected["status"], "coverage": coverage})

    produced_output = any(event["type"] in ("token", "done") for event in buffered)
    # Only a network/timeout-level failure counts against the circuit
    # breaker — a weak-but-real answer (caught by _needs_regeneration below)
    # means the backend itself is working fine.
    await _record_backend_outcome(store, llm, succeeded=produced_output or collected["status"] != "error")
    if collected["status"] == "error" and not produced_output and not is_last_backend:
        # This backend is down — try the next one in the chain. Discarded
        # outright; the client never sees this attempt.
        return {"llm_index": llm_index + 1, "attempt_accepted": False}

    is_last_regen = regen_count >= get_settings().max_regeneration_attempts
    if _needs_regeneration(collected) and not is_last_regen:
        # The backend answered, but badly — restart the fallback chain from
        # the primary for the retry, this time with a stricter prompt.
        return {"llm_index": 0, "regen_count": regen_count + 1, "attempt_accepted": False}

    # This is the attempt being kept — now, and only now, the client sees it.
    for event in buffered:
        writer(event)
    if collected["status"] == "done":
        collected["status"] = "grounded"
    return {**collected, "attempt_accepted": True}


def _generate_branch(state: RagState) -> str:
    return END if state.get("attempt_accepted") else "generate"


def build_rag_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
):
    graph = StateGraph(RagState)
    graph.add_node("route", _route_node)
    graph.add_node("meta", _meta_node)
   
    graph.add_node("bm25_search", _bm25_node, retry_policy=RetryPolicy(max_attempts=3))
    graph.add_node("vector_search", _vector_node, retry_policy=RetryPolicy(max_attempts=3))
    graph.add_node("graph_search", _graph_node, retry_policy=RetryPolicy(max_attempts=3))
    graph.add_node("fuse", _fuse_node)
    graph.add_node("refused", _refused_node)
    graph.add_node("generate", _generate_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges("route", _route_branch, ["meta", "bm25_search", "vector_search", "graph_search"])
    # All three searches must land before fuse runs — LangGraph's superstep
    # barrier makes "fuse" wait for every incoming edge, not just the first.
    graph.add_edge("bm25_search", "fuse")
    graph.add_edge("vector_search", "fuse")
    graph.add_edge("graph_search", "fuse")
    graph.add_conditional_edges("fuse", _fuse_branch, {"generate": "generate", "refused": "refused"})
    # Fallback (switch backend) and self-correction (retry with a stricter
    # prompt) are both real cycles here, not a Python loop inside one node
    # — "meta"/"generate" edge back to themselves while attempt_accepted is
    # False, each re-entry its own checkpointed step, until one attempt is
    # accepted and the edge goes to END instead.
    graph.add_conditional_edges("meta", _meta_branch, ["meta", END])
    graph.add_conditional_edges("generate", _generate_branch, ["generate", END])
    graph.add_edge("refused", END)
    # interrupt_before is only ever passed by tests to simulate a crash —
    # production never configures it. A real crash kills the request
    # handler before it could reach any "is this done" check anyway, so
    # there's nothing to defend here that an actual outage would hit; see
    # the "resume" endpoint (app/api/query_routes.py) for the other half of
    # this — continuing a thread whose last checkpoint has a pending node.
    return graph.compile(checkpointer=checkpointer, store=store, interrupt_before=interrupt_before)


_fallback_graph = None


def get_fallback_rag_graph():
    """A checkpointer-less, store-less compiled graph, for any caller that
    never ran app.main's lifespan (tests, scripts, etc.) and so has no
    persistent checkpointer/store to hand it — see app.dependencies.
    get_rag_graph, which prefers the real graph built at startup and only
    falls back to this one when that isn't available. Same graph either
    way — without a store, the circuit breaker is simply always "not
    cooling down" (see _is_cooling_down's `store is None` check), so this
    degrades to plain backend fallback with no history across calls."""
    global _fallback_graph
    if _fallback_graph is None:
        _fallback_graph = build_rag_graph()
    return _fallback_graph
