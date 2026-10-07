from __future__ import annotations

import json

from app.evaluation.evaluate_retrieval import EvalCase, evaluate_retrieval, load_eval_cases
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder


async def test_evaluate_retrieval_scores_a_findable_chunk_well(
    db_session, make_chunk, fake_embedder, fake_vector_store, user
):
    relevant_chunk = make_chunk("Ollama runs local language models directly on your own machine.")
    make_chunk("Completely unrelated content about gardening and soil pH.")

    fake_vector_store.add(
        ids=[f"chunk-{relevant_chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[relevant_chunk.content],
        metadatas=[{"doc_id": relevant_chunk.doc_id, "chunk_id": relevant_chunk.chunk_id, "user_id": user.user_id}],
    )

    cases = [EvalCase(query="What does Ollama run models on?", relevant_chunk_ids={relevant_chunk.chunk_id})]

    report = await evaluate_retrieval(
        db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(),
        cases, user.user_id, k=5,
    )

    assert report.k == 5
    assert len(report.results) == 1
    result = report.results[0]
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.gate_sufficient is True
    assert relevant_chunk.chunk_id in result.retrieved_chunk_ids

    assert report.mean_recall_at_k == 1.0
    assert report.mrr == 1.0
    assert report.gate_refusal_rate == 0.0


async def test_evaluate_retrieval_scores_zero_when_relevant_chunk_is_unreachable(
    db_session, make_chunk, fake_embedder, fake_vector_store, user
):
    make_chunk("Some content that exists but isn't the one we're looking for.")
    # fake_vector_store deliberately left empty, and the "relevant" id below
    # doesn't correspond to anything retrievable — simulates a genuine gap.
    cases = [EvalCase(query="completely different unmatched question", relevant_chunk_ids={999999})]

    report = await evaluate_retrieval(
        db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor(),
        cases, user.user_id, k=5,
    )

    result = report.results[0]
    assert result.recall_at_k == 0.0
    assert result.reciprocal_rank == 0.0
    assert report.mrr == 0.0
    # Gate should refuse too — nothing relevant was found, so it shouldn't say otherwise.
    assert result.gate_sufficient is False
    assert report.gate_refusal_rate == 1.0


def test_load_eval_cases_from_json(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"query": "what is Ollama?", "relevant_chunk_ids": [1, 2]}]))

    cases = load_eval_cases(path)

    assert len(cases) == 1
    assert cases[0].query == "what is Ollama?"
    assert cases[0].relevant_chunk_ids == {1, 2}
