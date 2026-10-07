"""Pipeline 1: Independent Retrieval Pipeline (FR-4.5).

Runs BM25, vector, and graph-expansion search concurrently (asyncio.gather —
genuine parallelism, not one after another), fuses the three ranked lists
with RRF, optionally reranks and/or diversifies the result (MMR), then runs
the context sufficiency / grounding gate before anything reaches an LLM.
Returns a status-tagged RetrievalOutcome — same convention as
IngestOutcome — never raises for a "no good context found" case, since
that's an expected, not exceptional, outcome.
"""

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
from app.search.bm25_search import bm25_search
from app.search.grounding_gate import evaluate_context_sufficiency
from app.search.mmr import mmr_select
from app.search.reranker import Reranker
from app.search.rrf_fusion import reciprocal_rank_fusion
from app.search.vector_search import vector_search
from app.services.types import RetrievalOutcome

logger = logging.getLogger(__name__)


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

    # user_id is threaded into all three sources — this is Pipeline 1's half
    # of per-user data isolation (the other half is chunk_routes.py's
    # ownership check on citation lookups). Nothing here is optional: a
    # missing user_id would mean a query could surface another user's chunks.
    bm25_hits, vector_hits, graph_hits = await asyncio.gather(
        asyncio.to_thread(bm25_search, db, query, settings.bm25_top_k, user_id),
        vector_search(embedder, vector_store, query, settings.vector_top_k, user_id),
        asyncio.to_thread(expand_via_graph, db, graph_builder, extractor, query, settings.graph_top_k, user_id),
    )

    fused = reciprocal_rank_fusion(
        {"bm25": bm25_hits, "vector": vector_hits, "graph": graph_hits},
        k=settings.rrf_k,
        top_n=settings.rrf_top_n,
    )

    if reranker is not None and fused:
        # When MMR will also run, don't truncate here — leave the full
        # pool intact so MMR has a real set of candidates to pick a
        # diverse subset from, rather than reselecting from a handful
        # reranking already narrowed down to.
        rerank_pool_size = len(fused) if settings.mmr_enabled else settings.rerank_top_k
        # CPU-bound model inference, not I/O — to_thread keeps it from
        # blocking the event loop, same reasoning as bm25_search above.
        fused = await asyncio.to_thread(reranker.rerank, query, fused, rerank_pool_size)

    if settings.mmr_enabled and fused:
        # Re-embeds the candidate texts themselves (not the query) so MMR
        # can measure how similar candidates are to each other — a
        # different question than how similar each one is to the query,
        # which fusion/reranking already answered via candidate.score.
        candidate_embeddings = await embedder.embed([c.content for c in fused], input_type="search_document")
        fused = mmr_select(fused, candidate_embeddings, k=settings.mmr_top_k, lambda_param=settings.mmr_lambda)

    verdict = evaluate_context_sufficiency(query, fused)

    if not verdict.sufficient:
        logger.info("Retrieval refused for query %r: %s (%s)", query, verdict.reason, verdict.metrics)
        return RetrievalOutcome(query=query, status="refused", reason=verdict.reason, metrics=verdict.metrics)

    logger.info("Retrieval grounded for query %r: %s", query, verdict.metrics)
    return RetrievalOutcome(
        query=query, status="grounded", candidates=verdict.passing_candidates, reason="ok", metrics=verdict.metrics
    )
