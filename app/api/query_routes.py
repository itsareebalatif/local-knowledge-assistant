from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from langfuse import propagate_attributes
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from app.api.schemas import QueryRequest, QueryResponse
from app.config import get_settings
from app.dependencies import (
    get_current_user,
    get_db,
    get_embedder,
    get_entity_extractor,
    get_graph_builder,
    get_llm,
    get_reranker,
    get_vector_store,
)
from app.llm.query_router import ROUTE_META, classify_and_rewrite
from app.models import Conversation, Message, User
from app.observability import observe, update
from app.services.conversation_qa_service import _MAX_HISTORY_MESSAGES, answer_from_history_stream
from app.services.conversation_service import append_message, create_conversation, get_conversation
from app.services.generation_service import generate_answer_stream
from app.services.retrieval_service import retrieve_and_verify

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


def _refusal_message(reason: str) -> str:
    if reason == "no_candidates":
        return "I couldn't find anything in your documents related to that. Try rephrasing, or upload a document that covers this topic."
    if reason == "low_keyword_coverage":
        return "I don't have enough information in your documents to answer that confidently. Try rephrasing using wording closer to what's actually in the document, or ask about something more specific."
    return "I don't have enough grounded information in your documents to answer that."


def _resolve_conversation(db: Session, user: User, conversation_id: int | None, first_question: str):
    if conversation_id is None:
        return create_conversation(db, user.user_id, first_question)
    conversation = get_conversation(db, user.user_id, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


async def _event_source(
    db: Session,
    embedder,
    vector_store,
    graph_builder,
    extractor,
    llm,
    reranker,
    meta_history: list[Message],
    user_id: int,
    route: str,
    search_query: str,
    original_query: str,
) -> AsyncIterator[dict]:
    # Meta questions ("what did I ask earlier") are about this chat, not the
    # documents — answered straight from the stored transcript, with no
    # retrieval, no grounding gate, and no chance of a document-retrieval
    # refusal on a question that was never about the documents to begin with.
    # `meta_history` is a snapshot taken before the current question was
    # appended to the conversation, so it never includes the current
    # question as if it were a prior one.
    if route == ROUTE_META:
        async for event in answer_from_history_stream(llm, meta_history, original_query):
            yield event
        return

    retrieval_outcome = await retrieve_and_verify(
        db, embedder, vector_store, graph_builder, extractor, search_query, user_id, reranker
    )
    async for event in generate_answer_stream(db, llm, retrieval_outcome):
        yield event


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
    reranker=Depends(get_reranker),
) -> QueryResponse:
    conversation = _resolve_conversation(db, current_user, payload.conversation_id, payload.query)
    recent_messages = _recent_messages(conversation)
    meta_history = _meta_history(conversation)
    append_message(db, conversation, "user", payload.query)

    with propagate_attributes(user_id=str(current_user.user_id), tags=["api-query"]):
        route = "domain"
        search_query = payload.query
        if get_settings().smart_routing_enabled:
            with observe("query-router", input={"query": payload.query}) as router_obs:
                decision = await classify_and_rewrite(llm, payload.query, recent_messages)
                route, search_query = decision.route, decision.search_query
                update(router_obs, output={"route": route, "search_query": search_query})

        with observe(
            "rag-query", input={"query": payload.query, "search_query": search_query, "route": route}
        ) as root:
            status_ = "error"
            reason = "no_response_generated"
            answer = ""
            citations: list[dict] = []
            grounding: dict | None = None

            async for event in _event_source(
                db, embedder, vector_store, graph_builder, extractor, llm, reranker,
                meta_history, current_user.user_id, route, search_query, payload.query,
            ):
                if event["type"] == "done":
                    status_ = "conversation" if route == ROUTE_META else "grounded"
                    answer = event["answer"]
                    citations = event["citations"]
                    grounding = event["grounding"]
                elif event["type"] == "error":
                    status_ = "error"
                    reason = event["message"]
                elif event["type"] == "refused":
                    status_ = "refused"
                    reason = event["reason"]

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
    reranker=Depends(get_reranker),
) -> StreamingResponse:
    conversation = _resolve_conversation(db, current_user, payload.conversation_id, payload.query)
    recent_messages = _recent_messages(conversation)
    meta_history = _meta_history(conversation)
    append_message(db, conversation, "user", payload.query)

    async def event_stream():
        yield _sse_format({"type": "conversation", "conversation_id": conversation.conversation_id})

        with propagate_attributes(user_id=str(current_user.user_id), tags=["api-query-stream"]):
            route = "domain"
            search_query = payload.query
            if get_settings().smart_routing_enabled:
                with observe("query-router", input={"query": payload.query}) as router_obs:
                    decision = await classify_and_rewrite(llm, payload.query, recent_messages)
                    route, search_query = decision.route, decision.search_query
                    update(router_obs, output={"route": route, "search_query": search_query})

            with observe(
                "rag-query", input={"query": payload.query, "search_query": search_query, "route": route}
            ) as root:
                final_status = "error"
                final_reason = "no_response_generated"
                final_answer = ""
                final_citations: list[dict] = []
                final_grounding: dict | None = None

                async for event in _event_source(
                    db, embedder, vector_store, graph_builder, extractor, llm, reranker,
                    meta_history, current_user.user_id, route, search_query, payload.query,
                ):
                    if event["type"] == "done":
                        final_status = "conversation" if route == ROUTE_META else "grounded"
                        final_answer = event["answer"]
                        final_citations = event["citations"]
                        final_grounding = event["grounding"]
                    elif event["type"] == "error":
                        final_status = "error"
                        final_reason = event["message"]
                    elif event["type"] == "refused":
                        final_status = "refused"
                        final_reason = event["reason"]
                    yield _sse_format(event)
                update(root, output={"status": final_status, "answer": final_answer})

        if final_status in ("grounded", "conversation"):
            assistant_text = final_answer
        elif final_status == "refused":
            assistant_text = _refusal_message(final_reason)
        else:
            assistant_text = f"Error: {final_reason}"
        append_message(
            db, conversation, "assistant", assistant_text, citations=final_citations or None, grounding=final_grounding
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
