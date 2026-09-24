"""ORM models package — one table per module (SRS ERD, section 4.3).

Every model must be imported here so it registers on Base.metadata before
`app.db.init_db.init_db()` calls `create_all`, and so relationship() string
references (e.g. "RelationEdge.source_entity_id") can resolve.
"""

from app.models.chunk import Chunk
from app.models.chunk_entity_junction import ChunkEntityJunction
from app.models.document import Document
from app.models.entity_node import EntityNode
from app.models.metadata import DocumentMetadata
from app.models.relation_edge import RelationEdge
from app.models.user import User
from app.models.user_auth import UserAuth

__all__ = [
    "User",
    "UserAuth",
    "Document",
    "DocumentMetadata",
    "Chunk",
    "ChunkEntityJunction",
    "EntityNode",
    "RelationEdge",
]
