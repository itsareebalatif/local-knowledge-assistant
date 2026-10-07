
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.auth_service import get_user_by_token
from app.db.base import get_db  # re-exported for convenience: routes import everything from here
from app.embeddings.embedder import EmbedderBackend
from app.embeddings.vector_store import VectorStore
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.llm.base import LLMBackend
from app.models import User
from app.search.reranker import Reranker

__all__ = [
    "get_db",
    "get_embedder",
    "get_vector_store",
    "get_graph_builder",
    "get_entity_extractor",
    "get_llm",
    "get_reranker",
    "get_current_user",
]


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


# A real FastAPI security scheme (not a plain Header param) — this is what
# makes Swagger UI show the lock-icon "Authorize" button at the top of the
# page: paste the token once there, it's attached to every request after,
# across every endpoint. auto_error=False so a missing header falls through
# to our own 401 with a clearer message instead of FastAPI's generic one.
_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Resolves the bearer token to the User it belongs to. Every route that
    reads or writes a specific user's data depends on this — never on a
    client-supplied user_id, which would let anyone claim to be anyone (see
    auth_routes.py for how the token gets issued)."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or malformed Authorization header")

    user = get_user_by_token(db, credentials.credentials)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session token")
    return user
