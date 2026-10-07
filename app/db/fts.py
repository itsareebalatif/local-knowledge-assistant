
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

_CREATE_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='chunk_id'
);
"""

_CREATE_INSERT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS chunks_fts_after_insert
AFTER INSERT ON chunks
BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.chunk_id, new.content);
END;
"""

_CREATE_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS chunks_fts_after_update
AFTER UPDATE ON chunks
BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.chunk_id, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.chunk_id, new.content);
END;
"""

_CREATE_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS chunks_fts_after_delete
AFTER DELETE ON chunks
BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.chunk_id, old.content);
END;
"""


def create_fts_index(engine: Engine) -> None:
    """Create the chunks_fts virtual table and its sync triggers if they
    don't already exist. Safe to call every startup (init_db.py does)."""
    with engine.begin() as conn:
        conn.execute(text(_CREATE_FTS_TABLE))
        conn.execute(text(_CREATE_INSERT_TRIGGER))
        conn.execute(text(_CREATE_UPDATE_TRIGGER))
        conn.execute(text(_CREATE_DELETE_TRIGGER))
