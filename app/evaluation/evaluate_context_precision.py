
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import get_settings
from app.embeddings.embedder import EmbedderBackend
from app.embeddings.vector_store import VectorStore
from app.evaluation.retrieval_metrics import context_precision_at_k, context_recall
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.graph.graph_expansion import expand_via_graph
from app.llm.base import LLMBackend, LLMError
from app.search.bm25_search import bm25_search
from app.search.mmr import mmr_select
from app.search.reranker import Reranker
from app.search.rrf_fusion import reciprocal_rank_fusion
from app.search.vector_search import vector_search

logger = logging.getLogger(__name__)

# Source documents extracted from PDFs often carry citation-marker
# artifacts like "[cite: 10]" or "[cite: 11, 12]" — noise for a judge
# model trying to read the actual answer text, not part of its meaning.
_CITATION_MARKER = re.compile(r"\[cite:[^\]]*\]", re.IGNORECASE)


@dataclass
class GoldenCase:
    id: int | str
    question: str
    ground_truth: str


@dataclass
class ContextPrecisionResult:
    case_id: int | str
    question: str
    context_precision: float
    context_recall: float
    num_retrieved: int
    relevance_flags: list[bool] = field(default_factory=list)
    claim_flags: list[bool] = field(default_factory=list)


@dataclass
class ContextPrecisionReport:
    k: int
    results: list[ContextPrecisionResult] = field(default_factory=list)

    @property
    def mean_context_precision(self) -> float:
        values = [r.context_precision for r in self.results]
        return sum(values) / len(values) if values else 0.0

    @property
    def mean_context_recall(self) -> float:
        values = [r.context_recall for r in self.results]
        return sum(values) / len(values) if values else 0.0


_JUDGE_SYSTEM_PROMPT = (
    "You are evaluating a retrieval system for a question-answering pipeline. "
    "You will be given a QUESTION, the known correct ANSWER, and a numbered list "
    "of CONTEXT CHUNKS that were retrieved for that question.\n\n"
    "First, for each context chunk, decide whether it is relevant: whether it "
    "contains information that is actually useful for arriving at the ANSWER.\n\n"
    "Second, break the ANSWER down into its individual atomic factual claims. "
    "Each claim must state exactly one fact — if the ANSWER lists several "
    "distinct items (e.g. 'the system supports A, B, or C'), each item is its "
    "own separate claim, not one combined claim covering all of them. For each "
    "atomic claim you identify, decide whether it is attributable: whether the "
    "CONTEXT CHUNKS (collectively) actually support or contain that one fact.\n\n"
    "Respond in exactly this format, nothing else - no explanations, no claim "
    "text, only the verdicts:\n"
    "CONTEXT RELEVANCE:\n"
    "Chunk <n>: RELEVANT\n"
    "Chunk <n>: NOT_RELEVANT\n"
    "(one line per chunk, in the order given)\n\n"
    "CLAIM ATTRIBUTION:\n"
    "Claim 1: ATTRIBUTABLE\n"
    "Claim 2: NOT_ATTRIBUTABLE\n"
    "(one line per atomic claim you identified, numbered in the order you identified them)"
)


def _build_judge_prompt(question: str, ground_truth: str, chunk_texts: list[str]) -> str:
    numbered_chunks = "\n\n".join(f"Chunk {i}:\n{text}" for i, text in enumerate(chunk_texts, start=1))
    return f"QUESTION:\n{question}\n\nANSWER:\n{ground_truth}\n\nCONTEXT CHUNKS:\n{numbered_chunks}"


def _parse_verdicts(judge_output: str, num_chunks: int) -> list[bool]:
    
    pattern = re.compile(r"chunk\s*(\d+)\s*:\s*(not_relevant|relevant)", re.IGNORECASE)
    verdicts = [False] * num_chunks
    for match in pattern.finditer(judge_output):
        index = int(match.group(1)) - 1
        if 0 <= index < num_chunks:
            verdicts[index] = match.group(2).lower() == "relevant"
    return verdicts


def _parse_claim_verdicts(judge_output: str) -> list[bool]:
    
    pattern = re.compile(r"claim\s*(\d+)\s*:\s*(not_attributable|attributable)", re.IGNORECASE)
    verdicts_by_index: dict[int, bool] = {}
    for match in pattern.finditer(judge_output):
        index = int(match.group(1))
        if index >= 1:
            verdicts_by_index[index] = match.group(2).lower() == "attributable"
    if not verdicts_by_index:
        return []
    highest = max(verdicts_by_index)
    return [verdicts_by_index.get(i, False) for i in range(1, highest + 1)]


async def _collect(llm: LLMBackend, system_prompt: str, user_prompt: str) -> str:
    pieces = []
    async for piece in llm.generate_stream(system_prompt, user_prompt):
        pieces.append(piece)
    return "".join(pieces)


_RETRY_AFTER_SECONDS = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)
_MAX_JUDGE_RETRIES = 5
_DEFAULT_BACKOFF_SECONDS = 5.0


async def _collect_with_retry(llm: LLMBackend, system_prompt: str, user_prompt: str) -> str:
   
    last_exc: LLMError | None = None
    for attempt in range(_MAX_JUDGE_RETRIES + 1):
        try:
            return await _collect(llm, system_prompt, user_prompt)
        except LLMError as exc:
            last_exc = exc
            if attempt == _MAX_JUDGE_RETRIES:
                break
            match = _RETRY_AFTER_SECONDS.search(str(exc))
            wait_seconds = float(match.group(1)) + 0.5 if match else _DEFAULT_BACKOFF_SECONDS
            logger.info(
                "Context precision/recall judge call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, _MAX_JUDGE_RETRIES, wait_seconds, exc,
            )
            await asyncio.sleep(wait_seconds)
    assert last_exc is not None
    raise last_exc


async def evaluate_context_precision(
    db: Session,
    embedder: EmbedderBackend,
    vector_store: VectorStore,
    graph_builder: CooccurrenceGraphBuilder,
    extractor: SpacyEntityExtractor,
    llm: LLMBackend,
    cases: list[GoldenCase],
    user_id: int,
    k: int = 5,
    reranker: Reranker | None = None,
) -> ContextPrecisionReport:
    settings = get_settings()
    results: list[ContextPrecisionResult] = []

    for case in cases:
        bm25_hits, vector_hits, graph_hits = await asyncio.gather(
            asyncio.to_thread(bm25_search, db, case.question, settings.bm25_top_k, user_id),
            vector_search(embedder, vector_store, case.question, settings.vector_top_k, user_id),
            asyncio.to_thread(
                expand_via_graph, db, graph_builder, extractor, case.question, settings.graph_top_k, user_id
            ),
        )
        fused = reciprocal_rank_fusion(
            {"bm25": bm25_hits, "vector": vector_hits, "graph": graph_hits},
            k=settings.rrf_k,
            top_n=settings.rrf_top_n,
        )
        if reranker is not None and fused:
            rerank_pool_size = len(fused) if settings.mmr_enabled else settings.rerank_top_k
            fused = await asyncio.to_thread(reranker.rerank, case.question, fused, rerank_pool_size)
        if settings.mmr_enabled and fused:
            candidate_embeddings = await embedder.embed([c.content for c in fused], input_type="search_document")
            fused = mmr_select(fused, candidate_embeddings, k=settings.mmr_top_k, lambda_param=settings.mmr_lambda)

        retrieved = fused[:k]
        if not retrieved:
           
            results.append(
                ContextPrecisionResult(
                    case_id=case.id, question=case.question, context_precision=0.0, context_recall=0.0,
                    num_retrieved=0,
                )
            )
            continue

        chunk_texts = [c.content for c in retrieved]
        judge_prompt = _build_judge_prompt(case.question, case.ground_truth, chunk_texts)

        try:
            judge_output = await _collect_with_retry(llm, _JUDGE_SYSTEM_PROMPT, judge_prompt)
        except LLMError as exc:
            logger.warning("Context precision/recall judge call failed for case %r: %s", case.id, exc)
            results.append(
                ContextPrecisionResult(
                    case_id=case.id, question=case.question, context_precision=0.0, context_recall=0.0,
                    num_retrieved=len(retrieved),
                )
            )
            continue

        relevance_flags = _parse_verdicts(judge_output, len(chunk_texts))
        claim_flags = _parse_claim_verdicts(judge_output)
        precision_score = context_precision_at_k(relevance_flags)
        recall_score = context_recall(claim_flags)

        results.append(
            ContextPrecisionResult(
                case_id=case.id,
                question=case.question,
                context_precision=precision_score,
                context_recall=recall_score,
                num_retrieved=len(retrieved),
                relevance_flags=relevance_flags,
                claim_flags=claim_flags,
            )
        )

    return ContextPrecisionReport(k=k, results=results)


def load_golden_cases(path: str | Path) -> list[GoldenCase]:
    data = json.loads(Path(path).read_text())
    return [
        GoldenCase(
            id=item["id"],
            question=item["question"],
            ground_truth=_CITATION_MARKER.sub("", item["ground_truth"]).strip(),
        )
        for item in data
    ]
