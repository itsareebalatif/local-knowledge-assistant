"""kengine — command-line interface (SRS 5.1).

    kengine ingest <path> [--user-id N]
    kengine query "<question>"
    kengine evaluate <cases.json> [--k N]
    kengine evaluate-context-precision <golden.json> [--k N]

Builds real singletons per invocation (Ollama embedder/LLM, Chroma, spaCy,
the on-disk graph) and talks to the same service layer the FastAPI app
uses — no HTTP round trip, same code path either way.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import get_settings
from app.db.base import SessionLocal
from app.embeddings import get_embedder_backend
from app.embeddings.vector_store import ChromaVectorStore
from app.evaluation.evaluate_context_precision import evaluate_context_precision, load_golden_cases
from app.evaluation.evaluate_retrieval import evaluate_retrieval, load_eval_cases
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.ingestion.loader import load_local_file
from app.llm import get_llm_backend
from app.search.reranker import CrossEncoderReranker
from app.services.generation_service import generate_answer_stream
from app.services.pipeline_service import ingest_index_and_graph
from app.services.retrieval_service import retrieve_and_verify
from app.services.user_service import DEFAULT_LOCAL_USER_EMAIL as _DEFAULT_LOCAL_USER_EMAIL
from app.services.user_service import get_or_create_default_user as _get_or_create_default_user


async def _cmd_ingest(args: argparse.Namespace) -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        user_id = args.user_id if args.user_id is not None else _get_or_create_default_user(db)
        file_bytes, filename = await load_local_file(args.path)

        embedder = get_embedder_backend()
        vector_store = ChromaVectorStore()
        graph_builder = CooccurrenceGraphBuilder.load(settings.graph_store_path)
        extractor = SpacyEntityExtractor()

        outcome = await ingest_index_and_graph(
            db, user_id, file_bytes, filename, embedder, vector_store, graph_builder, extractor
        )

        if outcome.status == "ingested":
            graph_builder.save(settings.graph_store_path)
            print(
                f"Ingested {filename}: {outcome.total_chunks} chunks "
                f"({outcome.new_chunks} new, {outcome.reused_chunks} reused embeddings)"
            )
        elif outcome.status == "duplicate_file":
            print(f"Skipped {filename}: already ingested (document_id={outcome.document_id})")
        else:
            print(f"Failed to ingest {filename}: {outcome.error}", file=sys.stderr)
            sys.exit(1)
    finally:
        db.close()


async def _cmd_query(args: argparse.Namespace) -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        user_id = args.user_id if args.user_id is not None else _get_or_create_default_user(db)
        embedder = get_embedder_backend()
        vector_store = ChromaVectorStore()
        graph_builder = CooccurrenceGraphBuilder.load(settings.graph_store_path)
        extractor = SpacyEntityExtractor()
        llm = get_llm_backend()
        reranker = CrossEncoderReranker() if settings.rerank_enabled else None

        retrieval_outcome = await retrieve_and_verify(
            db, embedder, vector_store, graph_builder, extractor, args.question, user_id, reranker
        )

        if retrieval_outcome.status != "grounded":
            print(f"I don't have enough grounded information to answer that (reason: {retrieval_outcome.reason}).")
            return

        async for event in generate_answer_stream(db, llm, retrieval_outcome):
            if event["type"] == "token":
                print(event["text"], end="", flush=True)  # real-time stdout token printing (SRS 5.1)
            elif event["type"] == "error":
                print(f"\n[error] {event['message']}", file=sys.stderr)
            elif event["type"] == "done":
                print()
                if event["citations"]:
                    print("\nSources:")
                    for c in event["citations"]:
                        print(f"  [{c['marker']}] {c['file_name']} — {c['snippet']}")
    finally:
        db.close()


async def _cmd_evaluate(args: argparse.Namespace) -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        user_id = args.user_id if args.user_id is not None else _get_or_create_default_user(db)
        cases = load_eval_cases(args.cases_file)
        embedder = get_embedder_backend()
        vector_store = ChromaVectorStore()
        graph_builder = CooccurrenceGraphBuilder.load(settings.graph_store_path)
        extractor = SpacyEntityExtractor()
        reranker = CrossEncoderReranker() if settings.rerank_enabled else None

        report = await evaluate_retrieval(
            db, embedder, vector_store, graph_builder, extractor, cases, user_id, k=args.k, reranker=reranker
        )

        print(f"Retrieval evaluation — {len(report.results)} cases, k={report.k}")
        print(f"  Recall@{report.k}:    {report.mean_recall_at_k:.3f}")
        print(f"  Precision@{report.k}: {report.mean_precision_at_k:.3f}")
        print(f"  MRR:          {report.mrr:.3f}")
        print(f"  Gate refusal rate: {report.gate_refusal_rate:.3f}")
        print()
        for result in report.results:
            print(
                f"  [{'OK' if result.gate_sufficient else 'REFUSED'}] {result.query!r} — "
                f"recall={result.recall_at_k:.2f} precision={result.precision_at_k:.2f} rr={result.reciprocal_rank:.2f}"
            )
    finally:
        db.close()


async def _cmd_evaluate_context_precision(args: argparse.Namespace) -> None:
    settings = get_settings()
    db = SessionLocal()
    try:
        user_id = args.user_id if args.user_id is not None else _get_or_create_default_user(db)
        cases = load_golden_cases(args.golden_file)
        embedder = get_embedder_backend()
        vector_store = ChromaVectorStore()
        graph_builder = CooccurrenceGraphBuilder.load(settings.graph_store_path)
        extractor = SpacyEntityExtractor()
        # Judging can run on a different backend than live generation does —
        # EVAL_LLM_BACKEND overrides LLM_BACKEND for this call only, and
        # defaults to it (unchanged behavior) when left unset in .env.
        llm = get_llm_backend(settings.eval_llm_backend or settings.llm_backend)
        reranker = CrossEncoderReranker() if settings.rerank_enabled else None

        report = await evaluate_context_precision(
            db, embedder, vector_store, graph_builder, extractor, llm, cases, user_id, k=args.k, reranker=reranker
        )

        print(f"Context Precision / Context Recall evaluation — {len(report.results)} cases, k={report.k}")
        print(f"  Mean Context Precision@{report.k}: {report.mean_context_precision:.3f}")
        print(f"  Mean Context Recall@{report.k}:    {report.mean_context_recall:.3f}")
        print()
        for result in report.results:
            print(
                f"  [{result.case_id}] {result.question!r} — "
                f"context_precision={result.context_precision:.2f} "
                f"({sum(result.relevance_flags)}/{result.num_retrieved} chunks relevant), "
                f"context_recall={result.context_recall:.2f} "
                f"({sum(result.claim_flags)}/{len(result.claim_flags)} claims attributable)"
            )
    finally:
        db.close()


def _cmd_benchmark_latency(args: argparse.Namespace) -> None:
    import tempfile

    from app.evaluation.benchmark_latency import benchmark_vector_query_latency

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Inserting {args.n_chunks:,} synthetic vectors and running {args.n_queries} queries...")
        report = benchmark_vector_query_latency(tmp_dir, n_chunks=args.n_chunks, n_queries=args.n_queries)

    print(f"\nVector query latency — {report.n_chunks:,} chunks, {report.n_queries} queries")
    print(f"  Insert time: {report.insert_seconds:.1f}s")
    print(f"  Mean: {report.mean_ms:.1f}ms   p50: {report.p50_ms:.1f}ms   p95: {report.p95_ms:.1f}ms   "
          f"p99: {report.p99_ms:.1f}ms   max: {report.max_ms:.1f}ms")
    target_ms = 400
    worst = report.p99_ms
    verdict = "PASS" if worst < target_ms else "FAIL"
    print(f"  NFR-2 target: <{target_ms}ms — {verdict} (p99={worst:.1f}ms)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="kengine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a file into the knowledge base")
    ingest_parser.add_argument("path", help="Path to the file to ingest")
    ingest_parser.add_argument("--user-id", type=int, default=None)

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question")
    query_parser.add_argument("--user-id", type=int, default=None)

    evaluate_parser = subparsers.add_parser("evaluate", help="Benchmark retrieval quality against labeled cases")
    evaluate_parser.add_argument("cases_file", help="JSON file: [{query, relevant_chunk_ids}, ...]")
    evaluate_parser.add_argument("--k", type=int, default=5)
    evaluate_parser.add_argument("--user-id", type=int, default=None)

    context_precision_parser = subparsers.add_parser(
        "evaluate-context-precision", help="Score Context Precision against a {question, ground_truth} golden file"
    )
    context_precision_parser.add_argument("golden_file", help="JSON file: [{id, question, ground_truth}, ...]")
    context_precision_parser.add_argument("--k", type=int, default=5)
    context_precision_parser.add_argument("--user-id", type=int, default=None)

    latency_parser = subparsers.add_parser("benchmark-latency", help="Benchmark vector query latency at scale (NFR-2)")
    latency_parser.add_argument("--n-chunks", type=int, default=250_000)
    latency_parser.add_argument("--n-queries", type=int, default=50)

    args = parser.parse_args()
    if args.command == "ingest":
        asyncio.run(_cmd_ingest(args))
    elif args.command == "query":
        asyncio.run(_cmd_query(args))
    elif args.command == "evaluate":
        asyncio.run(_cmd_evaluate(args))
    elif args.command == "evaluate-context-precision":
        asyncio.run(_cmd_evaluate_context_precision(args))
    elif args.command == "benchmark-latency":
        _cmd_benchmark_latency(args)


if __name__ == "__main__":
    main()
