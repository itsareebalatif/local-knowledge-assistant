from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

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
from app.llm.orchestration import build_rag_graph
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
    # Primary first, then each configured fallback (skipping the primary's
    # own name if it's listed again) — app.llm.orchestration tries these in
    # order within a single answer attempt, only advancing past one that
    # fails before producing any visible output.
    fallback_names = [n.strip() for n in settings.llm_fallback_backends.split(",") if n.strip()]
    app.state.llm_chain = [app.state.llm] + [
        get_llm_backend(name) for name in fallback_names if name != settings.llm_backend
    ]
    app.state.reranker = CrossEncoderReranker() if settings.rerank_enabled else None
    app.state.email_sender = ResendEmailSender()

    # Opened for the server's whole lifetime, closed on shutdown below —
    # this is what lets graph.aget_state_history(thread_id=conversation_id)
    # replay a past conversation's routing/retrieval/generation steps after
    # the fact, not just while Langfuse is watching it live. The store is
    # separate from the checkpointer: it's cross-thread (shared by every
    # conversation, not scoped to one), which is what the circuit breaker
    # in app.llm.orchestration needs — "is this backend down" isn't a fact
    # about any single conversation.
    async with (
        AsyncSqliteSaver.from_conn_string(settings.langgraph_checkpoint_path) as checkpointer,
        AsyncSqliteStore.from_conn_string(settings.langgraph_store_path) as store,
    ):
        await checkpointer.setup()
        await store.setup()
        app.state.rag_graph = build_rag_graph(checkpointer=checkpointer, store=store)
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


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    # The dashboard is mounted at /ui, not "/", so it can't shadow /api/*,
    # /docs, or /redoc (see the mount below) — but that means the bare root
    # URL had no route at all and 404'd, which is what a deployed app's
    # visitors actually land on first. Confirmed live: that looked like "the
    # site only shows Swagger," since /docs is the one thing that *did*
    # resolve. This sends a real visitor straight to the dashboard without
    # taking the path away from anything else.
    return RedirectResponse(url="/ui/")


# Mounted under /ui, not "/", so it can't shadow /api/*, /docs, or /redoc.
_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="ui")
