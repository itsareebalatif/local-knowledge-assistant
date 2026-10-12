from __future__ import annotations

import json

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
from app.llm.query_rewriter import rewrite_query
from app.models import Conversation, User
from app.observability import observe, update
from app.services.conversation_service import append_message, create_conversation, get_conversation
from app.services.generation_service import generate_answer_stream
from app.services.retrieval_service import retrieve_and_verify

router = APIRouter()

_REWRITE_HISTORY_TURNS = 1  # the single most recent user+assistant pair


def _recent_messages(conversation: Conversation) -> list:
    return conversation.messages[-(_REWRITE_HISTORY_TURNS * 2) :]


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
    append_message(db, conversation, "user", payload.query)

    with propagate_attributes(user_id=str(current_user.user_id), tags=["api-query"]):
        search_query = payload.query
        if get_settings().query_rewrite_enabled:
            with observe("query-rewrite", input={"query": payload.query}) as rewrite_obs:
                search_query = await rewrite_query(llm, payload.query, recent_messages)
                update(rewrite_obs, output={"rewritten_query": search_query})

        with observe("rag-query", input={"query": payload.query, "search_query": search_query}) as root:
            retrieval_outcome = await retrieve_and_verify(
                db, embedder, vector_store, graph_builder, extractor, search_query, current_user.user_id, reranker
            )

            status_ = retrieval_outcome.status
            reason = retrieval_outcome.reason
            answer = ""
            citations: list[dict] = []
            grounding: dict | None = None

            async for event in generate_answer_stream(db, llm, retrieval_outcome):
                if event["type"] == "done":
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

    if status_ == "grounded":
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
    append_message(db, conversation, "user", payload.query)

    async def event_stream():
        yield _sse_format({"type": "conversation", "conversation_id": conversation.conversation_id})

        with propagate_attributes(user_id=str(current_user.user_id), tags=["api-query-stream"]):
            search_query = payload.query
            if get_settings().query_rewrite_enabled:
                with observe("query-rewrite", input={"query": payload.query}) as rewrite_obs:
                    search_query = await rewrite_query(llm, payload.query, recent_messages)
                    update(rewrite_obs, output={"rewritten_query": search_query})

            with observe("rag-query", input={"query": payload.query, "search_query": search_query}) as root:
                retrieval_outcome = await retrieve_and_verify(
                    db, embedder, vector_store, graph_builder, extractor, search_query, current_user.user_id, reranker
                )
                final_status = retrieval_outcome.status
                final_reason = retrieval_outcome.reason
                final_answer = ""
                final_citations: list[dict] = []
                final_grounding: dict | None = None

                async for event in generate_answer_stream(db, llm, retrieval_outcome):
                    if event["type"] == "done":
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

        if final_status == "grounded":
            assistant_text = final_answer
        elif final_status == "refused":
            assistant_text = _refusal_message(final_reason)
        else:
            assistant_text = f"Error: {final_reason}"
        append_message(
            db, conversation, "assistant", assistant_text, citations=final_citations or None, grounding=final_grounding
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
