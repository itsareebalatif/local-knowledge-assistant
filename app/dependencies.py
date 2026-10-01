"""FastAPI dependencies for the shared, expensive-to-create resources: the
embedder's HTTP client config, the Chroma client, the spaCy pipeline, and
the in-memory NetworkX graph. All of these are built once in main.py's
lifespan and stashed on `app.state` — these functions just hand them to
route handlers via Depends(), so nothing gets reconstructed per-request
(spaCy alone would make every request noticeably slower if it did).
"""

from __future__ import annotations

from fastapi import Request

from app.db.base import get_db  # re-exported for convenience: routes import everything from here
from app.embeddings.embedder import OllamaEmbedder
from app.embeddings.vector_store import VectorStore
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.llm.base import LLMBackend

__all__ = [
    "get_db",
    "get_embedder",
    "get_vector_store",
    "get_graph_builder",
    "get_entity_extractor",
    "get_llm",
]


def get_embedder(request: Request) -> OllamaEmbedder:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_graph_builder(request: Request) -> CooccurrenceGraphBuilder:
    return request.app.state.graph_builder


def get_entity_extractor(request: Request) -> SpacyEntityExtractor:
    return request.app.state.extractor


def get_llm(request: Request) -> LLMBackend:
    return request.app.state.llm
