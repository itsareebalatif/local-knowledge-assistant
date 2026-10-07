"""Retrieval evaluation harness.

Runs a labeled set of (query, relevant_chunk_ids) cases through the real
retrieval pipeline — BM25 + vector + graph expansion, fused by RRF, same as
app.services.retrieval_service — and reports Recall@k, Precision@k and MRR
against the known-correct chunk_ids, plus whether the grounding gate would
have accepted or refused each case.

This needs ground truth: a small set of questions about your own ingested
documents where you already know which chunk_id(s) answer them. There's no
way around hand-labeling that — these metrics measure retrieval quality
against a known answer, they don't invent one. Load cases from a JSON file
of the shape: [{"query": "...", "relevant_chunk_ids": [12, 47]}, ...] via
load_eval_cases().
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import get_settings
from app.embeddings.embedder import EmbedderBackend
from app.embeddings.vector_store import VectorStore
from app.evaluation.retrieval_metrics import precision_at_k, recall_at_k, reciprocal_rank
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.graph.graph_expansion import expand_via_graph
from app.search.bm25_search import bm25_search
from app.search.grounding_gate import evaluate_context_sufficiency
from app.search.mmr import mmr_select
from app.search.reranker import Reranker
from app.search.rrf_fusion import reciprocal_rank_fusion
from app.search.vector_search import vector_search


@dataclass
class EvalCase:
    query: str
    relevant_chunk_ids: set[int]


@dataclass
class EvalCaseResult:
    query: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    gate_sufficient: bool
    gate_reason: str
    retrieved_chunk_ids: list[int] = field(default_factory=list)


@dataclass
class EvaluationReport:
    k: int
    results: list[EvalCaseResult] = field(default_factory=list)

    @property
    def mean_recall_at_k(self) -> float:
        return _mean(r.recall_at_k for r in self.results)

    @property
    def mean_precision_at_k(self) -> float:
        return _mean(r.precision_at_k for r in self.results)

    @property
    def mrr(self) -> float:
        return _mean(r.reciprocal_rank for r in self.results)

    @property
    def gate_refusal_rate(self) -> float:
        """Fraction of cases the grounding gate would refuse — high alongside
        low recall is the signature of "retrieval can't find the answer, and
        at least the gate is catching that" rather than silent hallucination."""
        return _mean(0.0 if r.gate_sufficient else 1.0 for r in self.results)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


async def evaluate_retrieval(
    db: Session,
    embedder: EmbedderBackend,
    vector_store: VectorStore,
    graph_builder: CooccurrenceGraphBuilder,
    extractor: SpacyEntityExtractor,
    cases: list[EvalCase],
    user_id: int,
    k: int = 5,
    reranker: Reranker | None = None,
) -> EvaluationReport:
    settings = get_settings()
    results: list[EvalCaseResult] = []

    for case in cases:
        bm25_hits, vector_hits, graph_hits = await asyncio.gather(
            asyncio.to_thread(bm25_search, db, case.query, settings.bm25_top_k, user_id),
            vector_search(embedder, vector_store, case.query, settings.vector_top_k, user_id),
            asyncio.to_thread(expand_via_graph, db, graph_builder, extractor, case.query, settings.graph_top_k, user_id),
        )
        fused = reciprocal_rank_fusion(
            {"bm25": bm25_hits, "vector": vector_hits, "graph": graph_hits},
            k=settings.rrf_k,
            top_n=settings.rrf_top_n,
        )
        if reranker is not None and fused:
            rerank_pool_size = len(fused) if settings.mmr_enabled else settings.rerank_top_k
            fused = await asyncio.to_thread(reranker.rerank, case.query, fused, rerank_pool_size)
        if settings.mmr_enabled and fused:
            candidate_embeddings = await embedder.embed([c.content for c in fused], input_type="search_document")
            fused = mmr_select(fused, candidate_embeddings, k=settings.mmr_top_k, lambda_param=settings.mmr_lambda)
        retrieved_ids = [c.chunk_id for c in fused]
        verdict = evaluate_context_sufficiency(case.query, fused)

        results.append(
            EvalCaseResult(
                query=case.query,
                recall_at_k=recall_at_k(retrieved_ids, case.relevant_chunk_ids, k),
                precision_at_k=precision_at_k(retrieved_ids, case.relevant_chunk_ids, k),
                reciprocal_rank=reciprocal_rank(retrieved_ids, case.relevant_chunk_ids),
                gate_sufficient=verdict.sufficient,
                gate_reason=verdict.reason,
                retrieved_chunk_ids=retrieved_ids,
            )
        )

    return EvaluationReport(k=k, results=results)


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text())
    return [EvalCase(query=item["query"], relevant_chunk_ids=set(item["relevant_chunk_ids"])) for item in data]
