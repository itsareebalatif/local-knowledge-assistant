"""SQLAlchemy engine/session setup.

SQLite is the default per the SRS (Section 5.2 / 6.4 NFR-9 — swappable
storage), but DATABASE_URL can point at Postgres/MySQL later without
touching the models.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

settings = get_settings()

# SQLite stores its file relative to the project root — make sure the parent
# directory exists before the engine tries to open it.
if settings.is_sqlite:
    db_path = settings.database_url.split("sqlite:///", 1)[-1]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

connect_args = {"check_same_thread": False} if settings.is_sqlite else {}

engine = create_engine(settings.database_url, connect_args=connect_args, future=True)


if settings.is_sqlite:

    @event.listens_for(Engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):  # noqa: ARG001
        # SQLite ignores FK constraints unless this pragma is set per-connection.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Shared declarative base for every ORM model in app/db/models.py."""


def get_db():
    """FastAPI dependency: yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
