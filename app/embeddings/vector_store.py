"""Vector store abstraction (FR-3.2). ChromaVectorStore is the default,
concrete implementation; a FAISS backend can be added later behind the same
VectorStore interface without touching any caller (NFR-9: swappable vector
tools).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

# Chroma's telemetry module has a known version mismatch with the `posthog`
# package that logs a scary-looking (but harmless — already caught
# internally) ERROR on every call. anonymized_telemetry=False below stops it
# from actually sending anything; this just stops it from logging about it.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


class VectorStore(ABC):
    @abstractmethod
    def add(
        self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]
    ) -> None: ...

    @abstractmethod
    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]: ...


class ChromaVectorStore(VectorStore):
    """Local, on-disk Chroma collection.

    `anonymized_telemetry=False` is deliberate, not a default we happened to
    keep: Chroma phones home by default, which would break this project's
    zero-data-tracking requirement (NFR-5) if left on.
    """

    def __init__(self, path: str | None = None, collection_name: str | None = None):
        settings = get_settings()
        store_path = Path(path or settings.vector_store_path)
        store_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(store_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(collection_name or settings.vector_store_collection)

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        if not ids:
            return
        self._collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            {"id": ids[i], "document": documents[i], "distance": distances[i], "metadata": metadatas[i]}
            for i in range(len(ids))
        ]
