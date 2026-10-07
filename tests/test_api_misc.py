from __future__ import annotations

from fastapi.testclient import TestClient

from app.dependencies import get_db
from app.main import app


def _override_db(db_session):
    def _get_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_db


def test_get_chunk_returns_content_and_file_name(db_session, make_chunk, registered_user, auth_headers):
    chunk = make_chunk("Ollama runs language models locally.", owner=registered_user.user)
    _override_db(db_session)
    try:
        client = TestClient(app)
        response = client.get(f"/api/chunks/{chunk.chunk_id}", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["content"] == chunk.content
        assert body["file_name"].endswith(".md")
    finally:
        app.dependency_overrides.clear()


def test_get_chunk_404_for_missing_chunk(db_session, registered_user, auth_headers):
    _override_db(db_session)
    try:
        client = TestClient(app)
        response = client.get("/api/chunks/999999", headers=auth_headers)
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_get_chunk_404_for_another_users_chunk(db_session, make_chunk, user, registered_user, auth_headers):
    # The real ownership check: a chunk owned by the `user` fixture must not
    # come back for the `registered_user` fixture's token, even though the
    # chunk_id itself is perfectly valid.
    other_chunk = make_chunk("Someone else's private notes.", owner=user)
    _override_db(db_session)
    try:
        client = TestClient(app)
        response = client.get(f"/api/chunks/{other_chunk.chunk_id}", headers=auth_headers)
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_get_chunk_requires_authentication(db_session, make_chunk):
    chunk = make_chunk("Some content.")
    _override_db(db_session)
    try:
        client = TestClient(app)
        response = client.get(f"/api/chunks/{chunk.chunk_id}")  # no Authorization header
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_dashboard_static_page_is_served():
    client = TestClient(app)
    response = client.get("/ui/")
    assert response.status_code == 200
    assert "Personal Knowledge Engine" in response.text


def test_dashboard_assets_are_served():
    client = TestClient(app)
    js_response = client.get("/ui/app.js")
    css_response = client.get("/ui/style.css")
    assert js_response.status_code == 200
    assert css_response.status_code == 200
    assert "streamQuery" in js_response.text
