from __future__ import annotations

import sys
import types

import pytest

from app.search.reranker import CrossEncoderReranker, _sigmoid
from app.search.types import CandidateChunk


def test_sigmoid_squashes_into_zero_one_range_and_is_monotonic():
    assert _sigmoid(0.0) == pytest.approx(0.5)
    # x=20 not 50: exp(-50) underflows to where 1+exp(-50) rounds to exactly 1.0 in float64.
    assert 0.0 < _sigmoid(-20.0) < 0.5
    assert 0.5 < _sigmoid(20.0) < 1.0
    assert _sigmoid(-1.0) < _sigmoid(0.0) < _sigmoid(1.0)


class _FakeCrossEncoder:
    # Scores each (query, text) pair by shared word count, no real model needed.
    def __init__(self, model_name: str):
        self.model_name = model_name

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return scores


@pytest.fixture
def stubbed_cross_encoder(monkeypatch):
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


def test_rerank_reorders_candidates_by_relevance_to_the_query(stubbed_cross_encoder):
    reranker = CrossEncoderReranker(model_name="fake-model")
    candidates = [
        CandidateChunk(chunk_id=1, doc_id=1, content="completely unrelated gardening tips", score=0.9, sources=["bm25"]),
        CandidateChunk(chunk_id=2, doc_id=1, content="ollama runs models locally", score=0.1, sources=["vector"]),
    ]

    reranked = reranker.rerank("where does ollama run models", candidates, top_k=5)

    assert reranked[0].chunk_id == 2
    assert reranked[0].score > reranked[1].score


def test_rerank_respects_top_k(stubbed_cross_encoder):
    reranker = CrossEncoderReranker(model_name="fake-model")
    candidates = [
        CandidateChunk(chunk_id=i, doc_id=1, content=f"chunk number {i}", score=0.0, sources=["bm25"]) for i in range(5)
    ]

    reranked = reranker.rerank("chunk", candidates, top_k=2)

    assert len(reranked) == 2


def test_rerank_returns_empty_list_for_no_candidates(stubbed_cross_encoder):
    reranker = CrossEncoderReranker(model_name="fake-model")
    assert reranker.rerank("anything", [], top_k=5) == []


def test_rerank_preserves_sources_metadata(stubbed_cross_encoder):
    reranker = CrossEncoderReranker(model_name="fake-model")
    candidates = [CandidateChunk(chunk_id=1, doc_id=1, content="ollama", score=0.0, sources=["bm25", "vector"])]

    reranked = reranker.rerank("ollama", candidates, top_k=5)

    assert reranked[0].sources == ["bm25", "vector"]
