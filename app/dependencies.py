
from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.auth_service import get_user_by_token
from app.db.base import SessionLocal, get_db  # re-exported so routes import everything from here
from app.embeddings.embedder import EmbedderBackend
from app.embeddings.vector_store import VectorStore
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.llm.base import LLMBackend
from app.models import User
from app.notifications.resend_client import EmailSender
from app.search.reranker import Reranker

__all__ = [
    "get_current_user",
    "get_db",
    "get_email_sender",
    "get_embedder",
    "get_entity_extractor",
    "get_graph_builder",
    "get_llm",
    "get_reranker",
    "get_session_factory",
    "get_vector_store",
]


def get_session_factory() -> Callable[[], Session]:
    # For background tasks that outlive the request: they can't share the
    # request-scoped `get_db` session, but still need to land in the same
    # database, so tests override this (not a hardcoded SessionLocal import)
    # to point background work at their isolated per-test engine too.
    return SessionLocal


def get_embedder(request: Request) -> EmbedderBackend:
    return request.app.state.embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_graph_builder(request: Request) -> CooccurrenceGraphBuilder:
    return request.app.state.graph_builder


def get_entity_extractor(request: Request) -> SpacyEntityExtractor:
    return request.app.state.extractor


def get_llm(request: Request) -> LLMBackend:
    return request.app.state.llm


def get_reranker(request: Request) -> Reranker | None:
    return request.app.state.reranker


def get_email_sender(request: Request) -> EmailSender:
    return request.app.state.email_sender


# auto_error=False so a missing header falls through to our own 401 message.
_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or malformed Authorization header")

    user = get_user_by_token(db, credentials.credentials)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session token")
    return user
