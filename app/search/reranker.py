
from __future__ import annotations

import math
from typing import Protocol

from app.search.types import CandidateChunk


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[CandidateChunk], top_k: int) -> list[CandidateChunk]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:

    def __init__(self, model_name: str | None = None):
       
        from sentence_transformers import CrossEncoder

        from app.config import get_settings

        self._model = CrossEncoder(model_name or get_settings().rerank_model)

    def rerank(self, query: str, candidates: list[CandidateChunk], top_k: int) -> list[CandidateChunk]:
        if not candidates:
            return []

        pairs = [(query, c.content) for c in candidates]
        raw_scores = self._model.predict(pairs)

        reranked = [
            CandidateChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                content=c.content,
                
                score=_sigmoid(float(raw_score)),
                sources=c.sources,
            )
            for c, raw_score in zip(candidates, raw_scores)
        ]
        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked[:top_k]
