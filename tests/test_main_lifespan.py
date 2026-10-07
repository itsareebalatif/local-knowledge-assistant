from __future__ import annotations

import sys
import types

from app.config import get_settings
from app.embeddings.embedder import OllamaEmbedder
from app.embeddings.vector_store import ChromaVectorStore
from app.graph.entity_extractor import SpacyEntityExtractor
from app.llm.ollama_backend import OllamaLLM
from app.main import app, lifespan
from app.search.reranker import CrossEncoderReranker


async def test_lifespan_builds_every_shared_resource(tmp_path, monkeypatch):
    settings = get_settings()
    # Redirect disk-backed resources into tmp_path so this test never
    # touches the real project's data/ directory.
    monkeypatch.setattr(settings, "vector_store_path", str(tmp_path / "chroma"), raising=False)
    monkeypatch.setattr(settings, "graph_store_path", str(tmp_path / "graph.json"), raising=False)
    # This test asserts the lifespan wires up the LOCAL backends correctly —
    # that's a claim about the wiring code, not about whatever LLM_BACKEND /
    # EMBEDDING_BACKEND happen to be set to in the real .env right now, so
    # force both explicitly rather than inherit the environment's choice.
    monkeypatch.setattr(settings, "llm_backend", "local", raising=False)
    monkeypatch.setattr(settings, "embedding_backend", "ollama", raising=False)
    # Reranking disabled here specifically to avoid downloading a real
    # cross-encoder model in the test suite — the "reranker actually gets
    # built when enabled" claim is covered separately below, with the model
    # itself stubbed out.
    monkeypatch.setattr(settings, "rerank_enabled", False, raising=False)

    async with lifespan(app):
        assert isinstance(app.state.embedder, OllamaEmbedder)
        assert isinstance(app.state.vector_store, ChromaVectorStore)
        assert isinstance(app.state.extractor, SpacyEntityExtractor)
        assert isinstance(app.state.llm, OllamaLLM)
        assert app.state.graph_builder.graph.number_of_nodes() == 0  # no graph.json yet at this path
        assert app.state.reranker is None


async def test_lifespan_builds_a_reranker_when_enabled(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "vector_store_path", str(tmp_path / "chroma"), raising=False)
    monkeypatch.setattr(settings, "graph_store_path", str(tmp_path / "graph.json"), raising=False)
    monkeypatch.setattr(settings, "llm_backend", "local", raising=False)
    monkeypatch.setattr(settings, "embedding_backend", "ollama", raising=False)
    monkeypatch.setattr(settings, "rerank_enabled", True, raising=False)

    # Stub sentence_transformers so CrossEncoderReranker's lazy import
    # resolves without downloading a real model.
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = lambda model_name: object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    async with lifespan(app):
        assert isinstance(app.state.reranker, CrossEncoderReranker)
