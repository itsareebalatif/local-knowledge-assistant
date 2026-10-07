from __future__ import annotations

from types import SimpleNamespace

from app.dependencies import (
    get_db,
    get_embedder,
    get_entity_extractor,
    get_graph_builder,
    get_llm,
    get_reranker,
    get_vector_store,
)


def _fake_request(**state_attrs):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state_attrs)))


def test_each_getter_returns_the_matching_app_state_attribute():
    request = _fake_request(embedder="E", vector_store="V", graph_builder="G", extractor="X", llm="L", reranker="R")

    assert get_embedder(request) == "E"
    assert get_vector_store(request) == "V"
    assert get_graph_builder(request) == "G"
    assert get_entity_extractor(request) == "X"
    assert get_llm(request) == "L"
    assert get_reranker(request) == "R"


def test_get_db_is_re_exported_from_db_base():
    from app.db.base import get_db as real_get_db

    assert get_db is real_get_db
