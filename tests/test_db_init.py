from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

import app.db.init_db as init_db_module


def test_init_db_creates_every_table_and_the_fts_index(tmp_path, monkeypatch):
    test_engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite3'}", future=True)
    monkeypatch.setattr(init_db_module, "engine", test_engine)

    init_db_module.init_db()

    inspector = inspect(test_engine)
    tables = set(inspector.get_table_names())
    expected = {
        "users",
        "user_auth",
        "documents",
        "metadata",
        "chunks",
        "chunk_entity_junction",
        "entity_nodes",
        "relation_edges",
    }
    assert expected <= tables

    with test_engine.connect() as conn:
        fts_tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
        ).fetchall()
    assert len(fts_tables) == 1


def test_init_db_is_safe_to_run_twice(tmp_path, monkeypatch):
    test_engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite3'}", future=True)
    monkeypatch.setattr(init_db_module, "engine", test_engine)

    init_db_module.init_db()
    init_db_module.init_db()  # must not raise — idempotent schema setup

    inspector = inspect(test_engine)
    assert "users" in inspector.get_table_names()


def test_init_db_adds_access_token_expires_at_to_a_pre_existing_user_auth_table(tmp_path, monkeypatch):
    """`create_all` never alters an existing table, so a database created
    before UserAuth.access_token_expires_at existed would otherwise be stuck
    without it forever. Simulates that by building the table by hand
    without the column, then checking init_db() adds it rather than
    silently leaving the old schema in place."""
    test_engine = create_engine(f"sqlite:///{tmp_path / 'test.sqlite3'}", future=True)
    with test_engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE user_auth ("
                "auth_id INTEGER PRIMARY KEY, user_id INTEGER, password_hash VARCHAR(255), "
                "refresh_token_hash VARCHAR(255), last_login_at DATETIME)"
            )
        )
    monkeypatch.setattr(init_db_module, "engine", test_engine)

    init_db_module.init_db()

    inspector = inspect(test_engine)
    columns = {col["name"] for col in inspector.get_columns("user_auth")}
    assert "access_token_expires_at" in columns
