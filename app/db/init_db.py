
from __future__ import annotations

import logging

from sqlalchemy import inspect, text

from app.db.base import Base, engine
from app.db.fts import create_fts_index

from app import models  # noqa: F401

logger = logging.getLogger(__name__)


def _add_missing_columns(bind) -> None:
    """`create_all` only creates missing tables, never alters an existing
    one — so a column added to a model after a database already exists
    (like UserAuth.access_token_expires_at) needs to be added by hand here.
    Checked against PRAGMA table_info so this stays safe to run every time,
    on both a fresh database and one that already has the column."""
    inspector = inspect(bind)
    if "user_auth" not in inspector.get_table_names():
        return  # create_all (just above) is about to create it with every column already
    existing_columns = {col["name"] for col in inspector.get_columns("user_auth")}
    if "access_token_expires_at" not in existing_columns:
        bind.execute(text("ALTER TABLE user_auth ADD COLUMN access_token_expires_at DATETIME"))
    if "refresh_hash" not in existing_columns:
        bind.execute(text("ALTER TABLE user_auth ADD COLUMN refresh_hash VARCHAR(255)"))
    if "refresh_expires_at" not in existing_columns:
        bind.execute(text("ALTER TABLE user_auth ADD COLUMN refresh_expires_at DATETIME"))


def init_db() -> None:
    logger.info("Creating database schema at %s", engine.url)
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        _add_missing_columns(conn)
    create_fts_index(engine)
    logger.info(
        "Schema ready: %s (+ chunks_fts)",
        ", ".join(sorted(Base.metadata.tables.keys())),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    init_db()
