"""FastAPI application entrypoint.

Builds the shared, expensive-to-create resources — the embedder's HTTP
client config, the Chroma client, the spaCy pipeline, the NetworkX graph
loaded from disk, and the LLM backend — once at startup via the lifespan
context manager, not per-request (spaCy alone would make every request
noticeably slower if it did).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.query_routes import router as query_router
from app.config import get_settings
from app.embeddings.embedder import OllamaEmbedder
from app.embeddings.vector_store import ChromaVectorStore
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.llm import get_llm_backend


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.embedder = OllamaEmbedder()
    app.state.vector_store = ChromaVectorStore()
    app.state.graph_builder = CooccurrenceGraphBuilder.load(settings.graph_store_path)
    app.state.extractor = SpacyEntityExtractor()
    app.state.llm = get_llm_backend()
    yield
    # Nothing to explicitly tear down: Chroma's PersistentClient and the
    # in-memory graph hold no open connections that need closing.


app = FastAPI(title="Personal Knowledge Engine", lifespan=lifespan)
app.include_router(query_router, prefix="/api")
