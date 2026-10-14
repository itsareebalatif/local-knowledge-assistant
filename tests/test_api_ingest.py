from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.dependencies import (
    get_db,
    get_embedder,
    get_entity_extractor,
    get_graph_builder,
    get_session_factory,
    get_vector_store,
)
from app.graph.entity_extractor import SpacyEntityExtractor
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.main import app


def _override_all(db_session, embedder, vector_store, graph_builder, extractor):
    def _get_db():
        yield db_session

    # The background task opens its own session (it must not share the
    # request-scoped one), but it still needs to land in this test's
    # isolated in-memory DB — so it gets a fresh sessionmaker bound to the
    # *same* engine/connection as db_session, not the real app engine.
    test_session_factory = sessionmaker(bind=db_session.get_bind(), autoflush=False, autocommit=False, future=True)

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_graph_builder] = lambda: graph_builder
    app.dependency_overrides[get_entity_extractor] = lambda: extractor
    app.dependency_overrides[get_session_factory] = lambda: test_session_factory


def test_ingest_endpoint_queues_a_job_and_completes_it_in_the_background(
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
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        assert response.json()["status"] == "processing"

        # TestClient runs the ASGI app (and its background tasks) fully
        # synchronously before returning, so the job is already finished.
        status_response = client.get(f"/api/ingest/{job_id}", headers=auth_headers)
        assert status_response.status_code == 200
        body = status_response.json()
        assert body["status"] == "ingested"
        assert body["total_chunks"] >= 1
        assert (tmp_path / "graph.json").exists()
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
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        status_response = client.get(f"/api/ingest/{job_id}", headers=auth_headers)
        assert status_response.json()["status"] == "unsupported"
    finally:
        app.dependency_overrides.clear()


def test_ingest_requires_authentication(db_session, fake_embedder, fake_vector_store):
    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor())
    try:
        client = TestClient(app)
        response = client.post("/api/ingest", files={"file": ("notes.md", b"hello", "text/markdown")})
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_ingest_job_status_requires_authentication(db_session, fake_embedder, fake_vector_store):
    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor())
    try:
        client = TestClient(app)
        response = client.get("/api/ingest/1")
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_ingest_job_status_is_scoped_to_its_owner(
    db_session, fake_embedder, fake_vector_store, registered_user, auth_headers, user
):
    # `user` fixture creates a second, different account than `registered_user`.
    from app.models import IngestJob

    other_users_job = IngestJob(user_id=user.user_id, filename="private.pdf", status="processing")
    db_session.add(other_users_job)
    db_session.commit()

    _override_all(db_session, fake_embedder, fake_vector_store, CooccurrenceGraphBuilder(), SpacyEntityExtractor())
    try:
        client = TestClient(app)
        response = client.get(f"/api/ingest/{other_users_job.job_id}", headers=auth_headers)
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()
