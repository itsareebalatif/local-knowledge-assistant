from __future__ import annotations

import asyncio
import logging

from sqlalchemy.orm import Session

from app.config import get_settings
from app.embeddings.embedder import EmbedderBackend
from app.embeddings.vector_store import VectorStore
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.graph.graph_expansion import expand_via_graph
from app.observability import observe, update
from app.search.bm25_search import bm25_search
from app.search.grounding_gate import evaluate_context_sufficiency
from app.search.mmr import mmr_select
from app.search.reranker import Reranker
from app.search.rrf_fusion import reciprocal_rank_fusion
from app.search.types import CandidateChunk
from app.search.vector_search import vector_search
from app.services.types import RetrievalOutcome

logger = logging.getLogger(__name__)


def _chunk_ids(candidates: list[CandidateChunk]) -> list[int]:
    return [c.chunk_id for c in candidates]


async def _traced_bm25(db: Session, query: str, top_k: int, user_id: int) -> list[CandidateChunk]:
    with observe("bm25-search", as_type="retriever", input={"query": query}) as obs:
        hits = await asyncio.to_thread(bm25_search, db, query, top_k, user_id)
        update(obs, output={"num_hits": len(hits), "chunk_ids": _chunk_ids(hits)})
        return hits


async def _traced_vector(
    db: Session, embedder: EmbedderBackend, vector_store: VectorStore, query: str, top_k: int, user_id: int
) -> list[CandidateChunk]:
    with observe("vector-search", as_type="retriever", input={"query": query}) as obs:
        hits = await vector_search(db, embedder, vector_store, query, top_k, user_id)
        update(obs, output={"num_hits": len(hits), "chunk_ids": _chunk_ids(hits)})
        return hits


async def _traced_graph(
    db: Session,
    graph_builder: CooccurrenceGraphBuilder,
    extractor: SpacyEntityExtractor,
    query: str,
    top_k: int,
    user_id: int,
) -> list[CandidateChunk]:
    with observe("graph-expansion", as_type="retriever", input={"query": query}) as obs:
        hits = await asyncio.to_thread(expand_via_graph, db, graph_builder, extractor, query, top_k, user_id)
        update(obs, output={"num_hits": len(hits), "chunk_ids": _chunk_ids(hits)})
        return hits


async def retrieve_and_verify(
    db: Session,
    embedder: EmbedderBackend,
    vector_store: VectorStore,
    graph_builder: CooccurrenceGraphBuilder,
    extractor: SpacyEntityExtractor,
    query: str,
    user_id: int,
    reranker: Reranker | None = None,
) -> RetrievalOutcome:
    settings = get_settings()

    # user_id is required on all three sources for per-user data isolation
    bm25_hits, vector_hits, graph_hits = await asyncio.gather(
        _traced_bm25(db, query, settings.bm25_top_k, user_id),
        _traced_vector(db, embedder, vector_store, query, settings.vector_top_k, user_id),
        _traced_graph(db, graph_builder, extractor, query, settings.graph_top_k, user_id),
    )

    with observe(
        "rrf-fusion",
        input={"bm25_count": len(bm25_hits), "vector_count": len(vector_hits), "graph_count": len(graph_hits)},
    ) as obs:
        fused = reciprocal_rank_fusion(
            {"bm25": bm25_hits, "vector": vector_hits, "graph": graph_hits},
            k=settings.rrf_k,
            top_n=settings.rrf_top_n,
        )
        update(obs, output={"fused_count": len(fused), "chunk_ids": _chunk_ids(fused)})

    if reranker is not None and fused:
        # Don't truncate here when MMR will also run, so MMR gets the full pool to pick from
        rerank_pool_size = len(fused) if settings.mmr_enabled else settings.rerank_top_k
        with observe("rerank", input={"candidate_count": len(fused), "model": settings.rerank_model}) as obs:
            fused = await asyncio.to_thread(reranker.rerank, query, fused, rerank_pool_size)
            update(obs, output={"chunk_ids": _chunk_ids(fused), "scores": [round(c.score, 4) for c in fused]})

    if settings.mmr_enabled and fused:
        with observe("mmr-selection", input={"candidate_count": len(fused), "lambda": settings.mmr_lambda}) as obs:
            # Re-embeds candidates (not the query) so MMR can measure candidate-to-candidate similarity
            with observe(
                "embed-candidates", as_type="embedding", model=embedder.model, input={"num_texts": len(fused)}
            ) as embed_obs:
                candidate_embeddings = await embedder.embed([c.content for c in fused], input_type="search_document")
                update(embed_obs, output={"num_embeddings": len(candidate_embeddings)})

            fused = mmr_select(fused, candidate_embeddings, k=settings.mmr_top_k, lambda_param=settings.mmr_lambda)
            update(obs, output={"selected_chunk_ids": _chunk_ids(fused)})

    with observe(
        "grounding-gate", as_type="evaluator", input={"query": query, "candidate_count": len(fused)}
    ) as obs:
        verdict = evaluate_context_sufficiency(query, fused)
        update(obs, output={"sufficient": verdict.sufficient, "reason": verdict.reason, "metrics": verdict.metrics})

    if not verdict.sufficient:
        logger.info("Retrieval refused for query %r: %s (%s)", query, verdict.reason, verdict.metrics)
        return RetrievalOutcome(query=query, status="refused", reason=verdict.reason, metrics=verdict.metrics)

    logger.info("Retrieval grounded for query %r: %s", query, verdict.metrics)
    return RetrievalOutcome(
        query=query, status="grounded", candidates=verdict.passing_candidates, reason="ok", metrics=verdict.metrics
    )
