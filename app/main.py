from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.auth_routes import router as auth_router
from app.api.chunk_routes import router as chunk_router
from app.api.conversation_routes import router as conversation_router
from app.api.document_routes import router as document_router
from app.api.graph_routes import router as graph_router
from app.api.ingest_routes import router as ingest_router
from app.api.query_routes import router as query_router
from app.config import get_settings
from app.embeddings import get_embedder_backend
from app.embeddings.vector_store import ChromaVectorStore
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.llm import get_llm_backend
from app.notifications.resend_client import ResendEmailSender
from app.observability import get_langfuse
from app.search.reranker import CrossEncoderReranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.embedder = get_embedder_backend()
    app.state.vector_store = ChromaVectorStore()
    app.state.graph_builder = CooccurrenceGraphBuilder.load(settings.graph_store_path)
    app.state.extractor = SpacyEntityExtractor()
    app.state.llm = get_llm_backend()
    app.state.reranker = CrossEncoderReranker() if settings.rerank_enabled else None
    app.state.email_sender = ResendEmailSender()
    yield
    langfuse = get_langfuse()
    if langfuse is not None:
        langfuse.shutdown()


app = FastAPI(
    title="Personal Knowledge Engine",
    description=(
        "Local-first AI knowledge assistant: ingest documents, search them with a "
        "hybrid BM25 + vector + graph-expansion pipeline, and get grounded, cited "
        "answers. Interactive docs at /docs (Swagger UI) and /redoc."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(auth_router, prefix="/api")
app.include_router(ingest_router, prefix="/api")
app.include_router(query_router, prefix="/api")
app.include_router(graph_router, prefix="/api")
app.include_router(chunk_router, prefix="/api")
app.include_router(document_router, prefix="/api")
app.include_router(conversation_router, prefix="/api")

# Mounted under /ui, not "/", so it can't shadow /api/*, /docs, or /redoc.
_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="ui")
