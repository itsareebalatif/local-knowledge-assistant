from __future__ import annotations

import json
from types import SimpleNamespace

import app.cli as cli
from app.models import User


def _patch_common(monkeypatch, db_session, fake_embedder, fake_vector_store, graph_path):
    """Point every CLI-internal singleton constructor at test doubles, and
    SessionLocal at the test's in-memory db_session, so no CLI test ever
    touches a real Ollama server, a real Chroma directory, or the project's
    real sqlite file."""
    monkeypatch.setattr(cli, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(cli, "get_embedder_backend", lambda: fake_embedder)
    monkeypatch.setattr(cli, "ChromaVectorStore", lambda: fake_vector_store)

    settings = cli.get_settings()
    monkeypatch.setattr(settings, "graph_store_path", str(graph_path), raising=False)

    # db.close() is called in each command's `finally` — harmless no-op here
    # since the fixture's own teardown closes db_session; just stop it from
    # closing early mid-test if a command calls it more than once.
    monkeypatch.setattr(db_session, "close", lambda: None)


def _default_user(db_session) -> User:
    """The CLI's own default-local-user, not the `user` fixture — these are
    two different rows unless a test is careful to use this one. A chunk
    made via make_chunk(..., owner=_default_user(db_session)) is the one a
    CLI query/evaluate call (which also resolves this same user) can
    actually see, now that retrieval is scoped per-user."""
    user_id = cli._get_or_create_default_user(db_session)
    return db_session.get(User, user_id)


async def test_ingest_creates_default_local_user_on_first_use(
    tmp_path, monkeypatch, capsys, db_session, fake_embedder, fake_vector_store
):
    _patch_common(monkeypatch, db_session, fake_embedder, fake_vector_store, tmp_path / "graph.json")

    file_path = tmp_path / "notes.md"
    file_path.write_text("# Notes\n\nOllama runs language models locally.\n")

    await cli._cmd_ingest(SimpleNamespace(path=str(file_path), user_id=None))

    out = capsys.readouterr().out
    assert "Ingested notes.md" in out

    user = db_session.query(User).filter(User.email == cli._DEFAULT_LOCAL_USER_EMAIL).first()
    assert user is not None


async def test_ingest_reports_duplicate_without_reingesting(
    tmp_path, monkeypatch, capsys, db_session, fake_embedder, fake_vector_store
):
    _patch_common(monkeypatch, db_session, fake_embedder, fake_vector_store, tmp_path / "graph.json")
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Notes\n\nSome content here.\n")

    await cli._cmd_ingest(SimpleNamespace(path=str(file_path), user_id=None))
    capsys.readouterr()  # discard first call's output
    await cli._cmd_ingest(SimpleNamespace(path=str(file_path), user_id=None))

    out = capsys.readouterr().out
    assert "Skipped notes.md: already ingested" in out
    assert len(fake_embedder.calls) == 1  # second ingest never re-embedded


async def test_query_prints_streamed_tokens_and_sources(
    tmp_path, monkeypatch, capsys, db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm
):
    _patch_common(monkeypatch, db_session, fake_embedder, fake_vector_store, tmp_path / "graph.json")
    monkeypatch.setattr(cli, "get_llm_backend", lambda: fake_llm)

    owner = _default_user(db_session)
    chunk = make_chunk("Ollama runs language models locally on your own machine.", owner=owner)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": owner.user_id}],
    )
    fake_llm.pieces = ["Ollama ", "runs ", "locally."]

    await cli._cmd_query(SimpleNamespace(question="What does Ollama run models on?", user_id=None))

    out = capsys.readouterr().out
    assert "Ollama runs locally." in out
    assert "Sources:" in out


async def test_query_prints_refusal_message_without_calling_llm(
    tmp_path, monkeypatch, capsys, db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm
):
    _patch_common(monkeypatch, db_session, fake_embedder, fake_vector_store, tmp_path / "graph.json")
    monkeypatch.setattr(cli, "get_llm_backend", lambda: fake_llm)
    make_chunk("completely unrelated content about gardening", owner=_default_user(db_session))

    await cli._cmd_query(SimpleNamespace(question="quantum computing architecture", user_id=None))

    out = capsys.readouterr().out
    assert "don't have enough grounded information" in out
    assert fake_llm.calls == []


async def test_evaluate_prints_recall_precision_and_mrr(
    tmp_path, monkeypatch, capsys, db_session, make_chunk, fake_embedder, fake_vector_store
):
    _patch_common(monkeypatch, db_session, fake_embedder, fake_vector_store, tmp_path / "graph.json")

    owner = _default_user(db_session)
    chunk = make_chunk("Ollama runs language models locally on your own machine.", owner=owner)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": owner.user_id}],
    )

    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps([{"query": "What does Ollama run models on?", "relevant_chunk_ids": [chunk.chunk_id]}]))

    await cli._cmd_evaluate(SimpleNamespace(cases_file=str(cases_file), k=5, user_id=None))

    out = capsys.readouterr().out
    assert "Recall@5:    1.000" in out
    assert "MRR:          1.000" in out


async def test_evaluate_context_precision_prints_mean_score(
    tmp_path, monkeypatch, capsys, db_session, make_chunk, fake_embedder, fake_vector_store, fake_llm
):
    _patch_common(monkeypatch, db_session, fake_embedder, fake_vector_store, tmp_path / "graph.json")
    monkeypatch.setattr(cli, "get_llm_backend", lambda *a, **kw: fake_llm)

    owner = _default_user(db_session)
    chunk = make_chunk("Ollama runs language models locally on your own machine.", owner=owner)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": owner.user_id}],
    )
    fake_llm.pieces = ["Chunk 1: RELEVANT\n", "Claim 1: ATTRIBUTABLE\n"]

    golden_file = tmp_path / "golden.json"
    golden_file.write_text(
        json.dumps([{"id": 1, "question": "What does Ollama run models on?", "ground_truth": "Your own machine."}])
    )

    await cli._cmd_evaluate_context_precision(SimpleNamespace(golden_file=str(golden_file), k=5, user_id=None))

    out = capsys.readouterr().out
    assert "Mean Context Precision@5: 1.000" in out
    assert "Mean Context Recall@5:    1.000" in out


async def test_evaluate_context_precision_uses_eval_llm_backend_override_when_set(
    tmp_path, monkeypatch, capsys, db_session, fake_embedder, fake_vector_store, fake_llm
):
    _patch_common(monkeypatch, db_session, fake_embedder, fake_vector_store, tmp_path / "graph.json")
    requested_backends: list[str | None] = []
    monkeypatch.setattr(cli, "get_llm_backend", lambda name=None: (requested_backends.append(name), fake_llm)[1])

    settings = cli.get_settings()
    monkeypatch.setattr(settings, "llm_backend", "groq", raising=False)
    monkeypatch.setattr(settings, "eval_llm_backend", "gemini", raising=False)

    golden_file = tmp_path / "golden.json"
    golden_file.write_text(json.dumps([{"id": 1, "question": "anything", "ground_truth": "anything"}]))

    await cli._cmd_evaluate_context_precision(SimpleNamespace(golden_file=str(golden_file), k=5, user_id=None))

    assert requested_backends == ["gemini"]  # override wins over llm_backend ("groq")


async def test_evaluate_context_precision_falls_back_to_llm_backend_when_override_unset(
    tmp_path, monkeypatch, capsys, db_session, fake_embedder, fake_vector_store, fake_llm
):
    _patch_common(monkeypatch, db_session, fake_embedder, fake_vector_store, tmp_path / "graph.json")
    requested_backends: list[str | None] = []
    monkeypatch.setattr(cli, "get_llm_backend", lambda name=None: (requested_backends.append(name), fake_llm)[1])

    settings = cli.get_settings()
    monkeypatch.setattr(settings, "llm_backend", "local", raising=False)
    monkeypatch.setattr(settings, "eval_llm_backend", None, raising=False)

    golden_file = tmp_path / "golden.json"
    golden_file.write_text(json.dumps([{"id": 1, "question": "anything", "ground_truth": "anything"}]))

    await cli._cmd_evaluate_context_precision(SimpleNamespace(golden_file=str(golden_file), k=5, user_id=None))

    assert requested_backends == ["local"]  # no override -> falls back to llm_backend


async def test_ingest_prints_error_and_exits_nonzero_on_failure(
    tmp_path, monkeypatch, capsys, db_session, fake_embedder, fake_vector_store
):
    _patch_common(monkeypatch, db_session, fake_embedder, fake_vector_store, tmp_path / "graph.json")

    file_path = tmp_path / "archive.zip"  # unsupported extension -> ingest_index_and_graph fails cleanly
    file_path.write_bytes(b"not a real file type we parse")

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        await cli._cmd_ingest(SimpleNamespace(path=str(file_path), user_id=None))

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Failed to ingest archive.zip" in err


async def test_query_prints_error_event_to_stderr(
    tmp_path, monkeypatch, capsys, db_session, make_chunk, fake_embedder, fake_vector_store
):
    from app.llm.base import LLMError

    class FailingLLM:
        async def generate_stream(self, system_prompt, user_prompt):
            if True:
                raise LLMError("the model server is unreachable")
            yield ""  # pragma: no cover - unreachable; keeps this an async generator function

    _patch_common(monkeypatch, db_session, fake_embedder, fake_vector_store, tmp_path / "graph.json")
    monkeypatch.setattr(cli, "get_llm_backend", lambda: FailingLLM())

    owner = _default_user(db_session)
    chunk = make_chunk("Ollama runs language models locally.", owner=owner)
    fake_vector_store.add(
        ids=[f"chunk-{chunk.chunk_id}"],
        embeddings=[[1.0, 0.0]],
        documents=[chunk.content],
        metadatas=[{"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "user_id": owner.user_id}],
    )

    await cli._cmd_query(SimpleNamespace(question="What does Ollama run models on?", user_id=None))

    err = capsys.readouterr().err
    assert "the model server is unreachable" in err


def test_main_dispatches_ingest_query_and_evaluate(monkeypatch):
    calls = []

    async def fake_ingest(args):
        calls.append(("ingest", args.path, args.user_id))

    async def fake_query(args):
        calls.append(("query", args.question, args.user_id))

    async def fake_evaluate(args):
        calls.append(("evaluate", args.cases_file, args.k, args.user_id))

    monkeypatch.setattr(cli, "_cmd_ingest", fake_ingest)
    monkeypatch.setattr(cli, "_cmd_query", fake_query)
    monkeypatch.setattr(cli, "_cmd_evaluate", fake_evaluate)

    import sys

    monkeypatch.setattr(sys, "argv", ["kengine", "ingest", "notes.md"])
    cli.main()
    assert calls[-1] == ("ingest", "notes.md", None)

    monkeypatch.setattr(sys, "argv", ["kengine", "query", "what is Ollama?"])
    cli.main()
    assert calls[-1] == ("query", "what is Ollama?", None)

    monkeypatch.setattr(sys, "argv", ["kengine", "evaluate", "cases.json", "--k", "10"])
    cli.main()
    assert calls[-1] == ("evaluate", "cases.json", 10, None)


def test_cmd_benchmark_latency_prints_pass_verdict(capsys):
    _cmd_benchmark_latency = cli._cmd_benchmark_latency
    _cmd_benchmark_latency(SimpleNamespace(n_chunks=50, n_queries=3))
    out = capsys.readouterr().out
    assert "NFR-2 target" in out
