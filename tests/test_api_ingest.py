from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import get_db, get_embedder, get_entity_extractor, get_graph_builder, get_vector_store
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.main import app


def _override_all(db_session, embedder, vector_store, graph_builder, extractor):
    def _get_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_graph_builder] = lambda: graph_builder
    app.dependency_overrides[get_entity_extractor] = lambda: extractor


def test_ingest_endpoint_returns_ingested_status(
    db_session, fake_embedder, fake_vector_store, tmp_path, monkeypatch, registered_user, auth_headers
):
    graph_builder = CooccurrenceGraphBuilder()
    _override_all(db_session, fake_embedder, fake_vector_store, graph_builder, SpacyEntityExtractor())
    monkeypatch.setattr(get_settings(), "graph_store_path", str(tmp_path / "graph.json"), raising=False)

    try:
        client = TestClient(app)
        response = client.post(
            "/api/ingest",
            files={"file": ("notes.md", b"# Notes\n\nOllama runs language models locally.\n", "text/markdown")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ingested"
        assert body["total_chunks"] >= 1
        assert (tmp_path / "graph.json").exists()  # graph persisted to disk after a successful ingest
    finally:
        app.dependency_overrides.clear()


def test_ingest_endpoint_reports_unsupported_file_type(
    db_session, fake_embedder, fake_vector_store, registered_user, auth_headers
):
    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor())
    try:
        client = TestClient(app)
        response = client.post(
            "/api/ingest",
            files={"file": ("archive.zip", b"whatever", "application/zip")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "unsupported"
    finally:
        app.dependency_overrides.clear()


def test_ingest_requires_authentication(db_session, fake_embedder, fake_vector_store):
    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor())
    try:
        client = TestClient(app)
        response = client.post(
            "/api/ingest", files={"file": ("notes.md", b"hello", "text/markdown")}
        )  # no Authorization header
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()
