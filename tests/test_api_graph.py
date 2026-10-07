from __future__ import annotations

from fastapi.testclient import TestClient

from app.dependencies import get_db, get_graph_builder
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.main import app


def _override(db_session, graph_builder):
    def _get_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_graph_builder] = lambda: graph_builder


def test_get_graph_exports_adjacency_data(db_session, registered_user, auth_headers):
    graph_builder = CooccurrenceGraphBuilder()
    graph_builder.add_chunk_entities(1, [("Ollama", "Organization"), ("qwen", "Concept")])

    _override(db_session, graph_builder)
    try:
        client = TestClient(app)
        response = client.get("/api/graph", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert len(body["nodes"]) == 2
    finally:
        app.dependency_overrides.clear()


def test_get_graph_neighbors_returns_weighted_neighbors_sorted_descending(db_session, registered_user, auth_headers):
    graph_builder = CooccurrenceGraphBuilder()
    for _ in range(3):
        graph_builder.add_chunk_entities(1, [("Ollama", "Organization"), ("qwen", "Concept")])
    graph_builder.add_chunk_entities(2, [("Ollama", "Organization"), ("llama", "Concept")])

    _override(db_session, graph_builder)
    try:
        client = TestClient(app)
        response = client.get(
            "/api/graph/neighbors", params={"name": "Ollama", "type": "Organization"}, headers=auth_headers
        )
        assert response.status_code == 200
        body = response.json()
        assert body["entity"] == "Organization:Ollama"
        assert [n["name"] for n in body["neighbors"]] == ["qwen", "llama"]
        assert body["neighbors"][0]["weight"] == 3
    finally:
        app.dependency_overrides.clear()


def test_get_graph_neighbors_for_unknown_entity_returns_empty_list(db_session, registered_user, auth_headers):
    _override(db_session, CooccurrenceGraphBuilder())
    try:
        client = TestClient(app)
        response = client.get(
            "/api/graph/neighbors", params={"name": "Nonexistent", "type": "Concept"}, headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["neighbors"] == []
    finally:
        app.dependency_overrides.clear()


def test_get_graph_requires_authentication(db_session):
    _override(db_session, CooccurrenceGraphBuilder())
    try:
        client = TestClient(app)
        response = client.get("/api/graph")  # no Authorization header
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()
