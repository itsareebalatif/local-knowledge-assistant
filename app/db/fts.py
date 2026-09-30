"""FTS5 full-text index over Chunk.content, for BM25 keyword search (FR-3.3).

This is a raw-SQL virtual table, not a SQLAlchemy model — FTS5 virtual
tables don't map cleanly to the ORM, so it's created and queried with plain
SQL instead.

Uses SQLite's "external content table" pattern: `chunks_fts` stores no data
of its own, it indexes `chunks.content` by rowid (`chunk_id`). Three triggers
on `chunks` (AFTER INSERT/UPDATE/DELETE) keep the index in sync automatically
at the database level — app/services/ingest_service.py never needs to know
this index exists, and can't accidentally forget to update it.
"""

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
