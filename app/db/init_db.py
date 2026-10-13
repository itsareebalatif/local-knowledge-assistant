
from __future__ import annotations

import logging

from sqlalchemy import inspect, text

from app import models  # noqa: F401
from app.db.base import Base, engine
from app.db.fts import create_fts_index

logger = logging.getLogger(__name__)


def _add_missing_columns(bind) -> None:
    # create_all only creates missing tables, never alters existing ones.
    inspector = inspect(bind)
    if "user_auth" not in inspector.get_table_names():
        return
    existing_columns = {col["name"] for col in inspector.get_columns("user_auth")}
    if "access_token_expires_at" not in existing_columns:
        bind.execute(text("ALTER TABLE user_auth ADD COLUMN access_token_expires_at DATETIME"))
    if "refresh_hash" not in existing_columns:
        bind.execute(text("ALTER TABLE user_auth ADD COLUMN refresh_hash VARCHAR(255)"))
    if "refresh_expires_at" not in existing_columns:
        bind.execute(text("ALTER TABLE user_auth ADD COLUMN refresh_expires_at DATETIME"))
    if "reset_token_hash" not in existing_columns:
        bind.execute(text("ALTER TABLE user_auth ADD COLUMN reset_token_hash VARCHAR(255)"))
    if "reset_token_expires_at" not in existing_columns:
        bind.execute(text("ALTER TABLE user_auth ADD COLUMN reset_token_expires_at DATETIME"))


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
