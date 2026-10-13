from __future__ import annotations

from fastapi.testclient import TestClient

from app.dependencies import get_db, get_graph_builder, get_vector_store
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.main import app


def _override_db(db_session, vector_store=None, graph_builder=None):
    def _get_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_graph_builder] = lambda: graph_builder or CooccurrenceGraphBuilder()


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
        assert {d["chunk_count"] for d in body["documents"]} == {1}
        assert all("file_name" in d and "created_at" in d for d in body["documents"])
    finally:
        app.dependency_overrides.clear()


def test_list_documents_excludes_other_users_documents(db_session, user, registered_user, auth_headers, make_chunk):
    _override_db(db_session)
    try:
        make_chunk("belongs to the other user", owner=user)

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


def test_delete_document_removes_it_from_the_list(
    db_session, registered_user, auth_headers, make_chunk, fake_vector_store, tmp_path, monkeypatch
):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "graph_store_path", str(tmp_path / "graph.json"), raising=False)
    _override_db(db_session, vector_store=fake_vector_store)
    try:
        chunk = make_chunk("goes away soon", owner=registered_user.user)
        client = TestClient(app)

        response = client.delete(f"/api/documents/{chunk.doc_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == {"status": "deleted", "doc_id": chunk.doc_id}
        assert (tmp_path / "graph.json").exists()

        after = client.get("/api/documents", headers=auth_headers)
        assert after.json() == {"documents": [], "total_documents": 0, "total_chunks": 0}
    finally:
        app.dependency_overrides.clear()


def test_delete_document_404s_for_an_unknown_doc_id(db_session, auth_headers, fake_vector_store):
    _override_db(db_session, vector_store=fake_vector_store)
    try:
        client = TestClient(app)
        response = client.delete("/api/documents/999999", headers=auth_headers)
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_delete_document_404s_for_another_users_document(db_session, user, registered_user, auth_headers, make_chunk, fake_vector_store):
    _override_db(db_session, vector_store=fake_vector_store)
    try:
        other_chunk = make_chunk("belongs to someone else", owner=user)
        client = TestClient(app)

        response = client.delete(f"/api/documents/{other_chunk.doc_id}", headers=auth_headers)
        assert response.status_code == 404

        from app.models import Document

        assert db_session.get(Document, other_chunk.doc_id) is not None
    finally:
        app.dependency_overrides.clear()


def test_delete_document_requires_authentication(db_session, fake_vector_store):
    _override_db(db_session, vector_store=fake_vector_store)
    try:
        client = TestClient(app)
        assert client.delete("/api/documents/1").status_code == 401
    finally:
        app.dependency_overrides.clear()
