"""Streaming query endpoint (FR-4.6).

Pipeline 1 (retrieval + grounding gate) feeds directly into Pipeline 2
(generation + post-generation verification + citations); the whole thing is
streamed to the client as Server-Sent Events so tokens appear as they're
generated instead of after the full answer is ready.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from app.api.schemas import QueryRequest
from app.dependencies import get_db, get_embedder, get_entity_extractor, get_graph_builder, get_llm, get_vector_store
from app.services.generation_service import generate_answer_stream
from app.services.retrieval_service import retrieve_and_verify

router = APIRouter()


def _sse_format(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@router.post("/query/stream")
async def query_stream(
    payload: QueryRequest,
    db: Session = Depends(get_db),
    embedder=Depends(get_embedder),
    vector_store=Depends(get_vector_store),
    graph_builder=Depends(get_graph_builder),
    extractor=Depends(get_entity_extractor),
    llm=Depends(get_llm),
) -> StreamingResponse:
    async def event_stream():
        retrieval_outcome = await retrieve_and_verify(
            db, embedder, vector_store, graph_builder, extractor, payload.query
        )
        async for event in generate_answer_stream(db, llm, retrieval_outcome):
            yield _sse_format(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
