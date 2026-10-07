"""Query endpoints (FR-4.6).

Two flavors of the same pipeline (Pipeline 1 retrieval+gate -> Pipeline 2
generation+verification+citations):

  POST /query         waits for the full answer, returns one JSON response —
                       simple for scripts, the CLI's non-streaming callers,
                       or any client that doesn't need token-by-token output.
  POST /query/stream   Server-Sent Events, tokens as they're generated —
                       what a chat UI actually wants.

Both refuse to call the LLM at all when Pipeline 1's grounding gate already
said no — this is where that refusal actually surfaces to a client. Both
also only ever search the authenticated caller's own documents: `user_id`
comes from `get_current_user`, never from the request body.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
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
    get_reranker,
    get_vector_store,
)
from app.models import User
from app.observability import observe, update
from app.services.generation_service import generate_answer_stream
from app.services.retrieval_service import retrieve_and_verify

router = APIRouter()


def _sse_format(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


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
    with propagate_attributes(user_id=str(current_user.user_id), tags=["api-query"]):
        with observe("rag-query", input={"query": payload.query}) as root:
            retrieval_outcome = await retrieve_and_verify(
                db, embedder, vector_store, graph_builder, extractor, payload.query, current_user.user_id, reranker
            )

            status = retrieval_outcome.status
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
                    status = "error"
                    reason = event["message"]

            update(root, output={"status": status, "answer": answer})

    return QueryResponse(status=status, reason=reason, answer=answer, citations=citations, grounding=grounding)


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
    async def event_stream():
        with propagate_attributes(user_id=str(current_user.user_id), tags=["api-query-stream"]):
            with observe("rag-query", input={"query": payload.query}) as root:
                retrieval_outcome = await retrieve_and_verify(
                    db, embedder, vector_store, graph_builder, extractor, payload.query, current_user.user_id, reranker
                )
                final_answer = ""
                async for event in generate_answer_stream(db, llm, retrieval_outcome):
                    if event["type"] == "done":
                        final_answer = event["answer"]
                    yield _sse_format(event)
                update(root, output={"status": retrieval_outcome.status, "answer": final_answer})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
