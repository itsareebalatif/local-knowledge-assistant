"""Database initialization.

Creates every table declared under app/models/ (User, UserAuth, Document,
DocumentMetadata, Chunk, ChunkEntityJunction, EntityNode, RelationEdge) plus
their indexes and constraints.

Usage:
    python -m app.db.init_db
"""

from __future__ import annotations

import logging

from app.db.base import Base, engine

# Importing app.models registers every mapped class on Base.metadata —
# required even though nothing below references the names directly.
from app import models  # noqa: F401

logger = logging.getLogger(__name__)


def init_db() -> None:
    logger.info("Creating database schema at %s", engine.url)
    Base.metadata.create_all(bind=engine)
    logger.info(
        "Schema ready: %s",
        ", ".join(sorted(Base.metadata.tables.keys())),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    init_db()
