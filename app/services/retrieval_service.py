"""Pipeline 1: Independent Retrieval Pipeline (FR-4.5).

Runs BM25, vector, and graph-expansion search concurrently (asyncio.gather —
genuine parallelism, not one after another), fuses the three ranked lists
with RRF, then runs the context sufficiency / grounding gate before anything
reaches an LLM. Returns a status-tagged RetrievalOutcome — same convention
as IngestOutcome — never raises for a "no good context found" case, since
that's an expected, not exceptional, outcome.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.orm import Session

from app.config import get_settings
from app.embeddings.embedder import OllamaEmbedder
from app.embeddings.vector_store import VectorStore
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.graph.graph_expansion import expand_via_graph
from app.search.bm25_search import bm25_search
from app.search.grounding_gate import evaluate_context_sufficiency
from app.search.rrf_fusion import reciprocal_rank_fusion
from app.search.vector_search import vector_search
from app.services.types import RetrievalOutcome

logger = logging.getLogger(__name__)


async def retrieve_and_verify(
    db: Session,
    embedder: OllamaEmbedder,
    vector_store: VectorStore,
    graph_builder: CooccurrenceGraphBuilder,
    extractor: SpacyEntityExtractor,
    query: str,
) -> RetrievalOutcome:
    settings = get_settings()

    bm25_hits, vector_hits, graph_hits = await asyncio.gather(
        asyncio.to_thread(bm25_search, db, query, settings.bm25_top_k),
        vector_search(embedder, vector_store, query, settings.vector_top_k),
        asyncio.to_thread(expand_via_graph, db, graph_builder, extractor, query, settings.graph_top_k),
    )

    fused = reciprocal_rank_fusion(
        {"bm25": bm25_hits, "vector": vector_hits, "graph": graph_hits},
        k=settings.rrf_k,
        top_n=settings.rrf_top_n,
    )

    verdict = evaluate_context_sufficiency(query, fused)

    if not verdict.sufficient:
        logger.info("Retrieval refused for query %r: %s (%s)", query, verdict.reason, verdict.metrics)
        return RetrievalOutcome(query=query, status="refused", reason=verdict.reason, metrics=verdict.metrics)

    logger.info("Retrieval grounded for query %r: %s", query, verdict.metrics)
    return RetrievalOutcome(
        query=query, status="grounded", candidates=verdict.passing_candidates, reason="ok", metrics=verdict.metrics
    )
