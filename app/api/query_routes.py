from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from langfuse import propagate_attributes
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from app.api.schemas import QueryRequest, QueryResponse
from app.dependencies import (
    get_current_user,
    get_db,
    get_embedder,
    get_entity_extractor,
    get_graph_builder,
    get_llm,
    get_llm_chain,
    get_rag_graph,
    get_reranker,
    get_vector_store,
)
from app.models import Conversation, Message, User
from app.observability import observe, update
from app.services.conversation_qa_service import _MAX_HISTORY_MESSAGES
from app.services.conversation_service import append_message, create_conversation, get_conversation

router = APIRouter()

_ROUTER_HISTORY_TURNS = 3  # enough for "what did I ask earlier" to reach back past the last reply


def _recent_messages(conversation: Conversation) -> list[Message]:
    # Must be read before the current question is appended — see the
    # module docstring note in conversation_qa_service on why a snapshot
    # taken after that append put the current question into its own history.
    return conversation.messages[-(_ROUTER_HISTORY_TURNS * 2) :]


def _meta_history(conversation: Conversation) -> list[Message]:
    return conversation.messages[-_MAX_HISTORY_MESSAGES:]


def _sse_format(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _stream_graph_events(graph, inputs, config: dict, final_values: dict):
    # `final_values` is populated as a side effect (mirrors streamlit_app.py's
    # own `final_event` pattern) so the caller can read the terminal node's
    # merged state once this generator is fully drained, without this
    # generator itself needing to "return" anything mid-stream. "custom"
    # carries each token/done/error/refused event live; "values" carries
    # the full state after every node — only the last one matters here.
    async for mode, chunk in graph.astream(inputs, config=config, stream_mode=["custom", "values"]):
        if mode == "custom":
            yield _sse_format(chunk)
        else:
            final_values.clear()
            final_values.update(chunk)


def _assistant_text_for(status_: str, reason: str, answer: str) -> str:
    if status_ in ("grounded", "conversation"):
        return answer
    if status_ == "refused":
        return _refusal_message(reason)
    return f"Error: {reason}"


def _refusal_message(reason: str) -> str:
    if reason == "no_candidates":
        return "I couldn't find anything in your documents related to that. Try rephrasing, or upload a document that covers this topic."
    if reason == "low_keyword_coverage":
        return "I don't have enough information in your documents to answer that confidently. Try rephrasing using wording closer to what's actually in the document, or ask about something more specific."
    if reason == "empty_answer":
        return "The model didn't generate a response for that question. Try asking again, or rephrase the question."
    return "I don't have enough grounded information in your documents to answer that."


def _resolve_conversation(db: Session, user: User, conversation_id: int | None, first_question: str):
    if conversation_id is None:
        return create_conversation(db, user.user_id, first_question)
    conversation = get_conversation(db, user.user_id, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


@router.post("/query", response_model=QueryResponse, tags=["query"], summary="Ask a question, get one JSON response")
async def query(
    payload: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    embedder=Depends(get_embedder),
    vector_store=Depends(get_vector_store),
    graph_builder=Depends(get_graph_builder),
    extractor=Depends(get_entity_extractor),
    llm=Depends(get_llm),
    llm_chain=Depends(get_llm_chain),
    reranker=Depends(get_reranker),
    graph=Depends(get_rag_graph),
) -> QueryResponse:
    conversation = _resolve_conversation(db, current_user, payload.conversation_id, payload.query)
    recent_messages = _recent_messages(conversation)
    meta_history = _meta_history(conversation)
    append_message(db, conversation, "user", payload.query)

    with (
        propagate_attributes(user_id=str(current_user.user_id), tags=["api-query", "langgraph"]),
        observe("rag-query-langgraph", input={"query": payload.query}) as root,
    ):
        config = {
            # thread_id scopes checkpointed state to this conversation, so
            # graph.aget_state_history(config) replays every past turn's
            # routing/retrieval/generation steps for it, not just this one.
            # recent_messages/meta_history live here, not in graph state —
            # they're SQLAlchemy Message objects, not msgpack-serializable,
            # and the checkpointer tries to persist every state field after
            # each step. Putting them in state crashed the stream (with a
            # real, confirmed-live "TypeError: Type is not msgpack
            # serializable: Message") right after the answer had already
            # been delivered, on every turn past the first in a
            # conversation (the first turn's history is empty, so nothing
            # unserializable was in it yet — which is why it looked
            # intermittent). `configurable` is per-call and never
            # checkpointed, same as db/embedder/llm below.
            "configurable": {
                "thread_id": str(conversation.conversation_id),
                "db": db,
                "embedder": embedder,
                "vector_store": vector_store,
                "graph_builder": graph_builder,
                "extractor": extractor,
                "llm": llm,
                "llm_chain": llm_chain,
                "reranker": reranker,
                "recent_messages": recent_messages,
                "meta_history": meta_history,
            }
        }
        result = await graph.ainvoke(
            {
                "query": payload.query,
                "user_id": current_user.user_id,
            },
            config=config,
        )
        status_ = result.get("status", "error")
        reason = result.get("reason", "no_response_generated")
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        grounding = result.get("grounding")

        update(root, output={"status": status_, "answer": answer})

    if status_ in ("grounded", "conversation"):
        assistant_text = answer
    elif status_ == "refused":
        assistant_text = _refusal_message(reason)
    else:
        assistant_text = f"Error: {reason}"
    append_message(db, conversation, "assistant", assistant_text, citations=citations or None, grounding=grounding)

    return QueryResponse(
        status=status_,
        reason=reason,
        answer=answer,
        citations=citations,
        grounding=grounding,
        conversation_id=conversation.conversation_id,
    )


@router.post("/query/stream", tags=["query"], summary="Ask a question, stream the answer as SSE")
async def query_stream(
    payload: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    embedder=Depends(get_embedder),
    vector_store=Depends(get_vector_store),
    graph_builder=Depends(get_graph_builder),
    extractor=Depends(get_entity_extractor),
    llm=Depends(get_llm),
    llm_chain=Depends(get_llm_chain),
    reranker=Depends(get_reranker),
    graph=Depends(get_rag_graph),
) -> StreamingResponse:
    conversation = _resolve_conversation(db, current_user, payload.conversation_id, payload.query)
    recent_messages = _recent_messages(conversation)
    meta_history = _meta_history(conversation)
    append_message(db, conversation, "user", payload.query)

    async def event_stream():
        yield _sse_format({"type": "conversation", "conversation_id": conversation.conversation_id})

        with (
            propagate_attributes(user_id=str(current_user.user_id), tags=["api-query-stream", "langgraph"]),
            observe("rag-query-langgraph", input={"query": payload.query}) as root,
        ):
            config = {
                # See the matching comment on the non-streaming `query()`
                # handler above: recent_messages/meta_history belong in
                # `configurable`, never in graph state — they're
                # non-serializable SQLAlchemy objects, and state gets
                # checkpointed after every step.
                "configurable": {
                    "thread_id": str(conversation.conversation_id),
                    "db": db,
                    "embedder": embedder,
                    "vector_store": vector_store,
                    "graph_builder": graph_builder,
                    "extractor": extractor,
                    "llm": llm,
                    "llm_chain": llm_chain,
                    "reranker": reranker,
                    "recent_messages": recent_messages,
                    "meta_history": meta_history,
                }
            }
            final_values: dict = {}
            inputs = {
                "query": payload.query,
                "user_id": current_user.user_id,
            }
            async for chunk in _stream_graph_events(graph, inputs, config, final_values):
                yield chunk

            final_status = final_values.get("status", "error")
            final_reason = final_values.get("reason", "no_response_generated")
            final_answer = final_values.get("answer", "")
            final_citations = final_values.get("citations", [])
            final_grounding = final_values.get("grounding")
            update(root, output={"status": final_status, "answer": final_answer})

        assistant_text = _assistant_text_for(final_status, final_reason, final_answer)
        append_message(
            db, conversation, "assistant", assistant_text, citations=final_citations or None, grounding=final_grounding
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post(
    "/query/{conversation_id}/resume",
    tags=["query"],
    summary="Resume an interrupted query (e.g. after a server restart) from its last completed step",
)
async def resume_query(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    embedder=Depends(get_embedder),
    vector_store=Depends(get_vector_store),
    graph_builder=Depends(get_graph_builder),
    extractor=Depends(get_entity_extractor),
    llm=Depends(get_llm),
    llm_chain=Depends(get_llm_chain),
    reranker=Depends(get_reranker),
    graph=Depends(get_rag_graph),
) -> StreamingResponse:
    conversation = get_conversation(db, current_user.user_id, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    config = {
        "configurable": {
            "thread_id": str(conversation_id),
            "db": db,
            "embedder": embedder,
            "vector_store": vector_store,
            "graph_builder": graph_builder,
            "extractor": extractor,
            "llm": llm,
            "llm_chain": llm_chain,
            "reranker": reranker,
            # Needed in case the pending node being resumed into is "route"
            # or "meta" — both read these from `configurable`, never from
            # state (see the comment on the other two endpoints above).
            "recent_messages": _recent_messages(conversation),
            "meta_history": _meta_history(conversation),
        }
    }

    try:
        snapshot = await graph.aget_state(config)
    except ValueError as exc:
        # No checkpointer on this graph (e.g. the store/checkpointer-less
        # fallback graph some callers get) — resume is inherently a
        # checkpointer feature, nothing to recover without one.
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="This server has no checkpointer configured."
        ) from exc

    if not snapshot.next:
        # The last turn already ran to completion (or this conversation
        # never started one) — there's no pending node to continue from.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Nothing to resume for this conversation — its last turn already finished.",
        )

    async def event_stream():
        with (
            propagate_attributes(user_id=str(current_user.user_id), tags=["api-query-resume", "langgraph"]),
            observe("rag-query-langgraph-resume", input={"conversation_id": conversation_id}) as root,
        ):
            final_values: dict = {}
            # None (not a fresh input dict) tells LangGraph to continue this
            # thread from its last checkpoint instead of starting a new run
            # — whatever retrieval/routing already finished before the
            # interruption isn't redone; execution just picks up at the
            # pending node (snapshot.next, checked above).
            async for chunk in _stream_graph_events(graph, None, config, final_values):
                yield chunk

            final_status = final_values.get("status", "error")
            final_reason = final_values.get("reason", "no_response_generated")
            final_answer = final_values.get("answer", "")
            final_citations = final_values.get("citations", [])
            final_grounding = final_values.get("grounding")
            update(root, output={"status": final_status, "answer": final_answer})

        assistant_text = _assistant_text_for(final_status, final_reason, final_answer)
        append_message(
            db, conversation, "assistant", assistant_text, citations=final_citations or None, grounding=final_grounding
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
