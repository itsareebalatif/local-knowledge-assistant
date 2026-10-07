from __future__ import annotations

from fastapi.testclient import TestClient

from app.dependencies import get_db
from app.main import app


def _override_db(db_session):
    def _get_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_db


def test_list_documents_returns_only_the_current_users_documents(db_session, registered_user, auth_headers, make_chunk):
    _override_db(db_session)
    try:
        make_chunk("first chunk of doc A", owner=registered_user.user)
        make_chunk("first chunk of doc B", owner=registered_user.user)

        client = TestClient(app)
        response = client.get("/api/documents", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["total_documents"] == 2
        assert body["total_chunks"] == 2
        assert {d["chunk_count"] for d in body["documents"]} == {1, 1}
        assert all("file_name" in d and "created_at" in d for d in body["documents"])
    finally:
        app.dependency_overrides.clear()


def test_list_documents_excludes_other_users_documents(db_session, user, registered_user, auth_headers, make_chunk):
    _override_db(db_session)
    try:
        make_chunk("belongs to the other user", owner=user)  # `user` fixture, not `registered_user`

        client = TestClient(app)
        response = client.get("/api/documents", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == {"documents": [], "total_documents": 0, "total_chunks": 0}
    finally:
        app.dependency_overrides.clear()


def test_list_documents_requires_authentication(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        assert client.get("/api/documents").status_code == 401
    finally:
        app.dependency_overrides.clear()
