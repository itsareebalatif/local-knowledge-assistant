from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

# Silences a harmless ERROR from a known chromadb/posthog version mismatch.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


class VectorStore(ABC):
    @abstractmethod
    def add(
        self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]
    ) -> None: ...

    @abstractmethod
    def query(self, embedding: list[float], top_k: int = 5, where: dict | None = None) -> list[dict]: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...


class ChromaVectorStore(VectorStore):
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
        # upsert, not add: Chroma's add() silently keeps whatever's already
        # stored under a given id and drops the new write — no error, no
        # warning surfaced to the caller (only a log line). Confirmed live:
        # this id scheme is `chunk-{chunk_id}`, a plain SQL autoincrement
        # integer, which is not permanently unique — it gets reused whenever
        # the SQL database is reset/recreated independently of this (separate,
        # persistent) vector store. A reused id then silently kept serving
        # another user's stale content indefinitely. upsert() makes a write
        # always reflect what was actually just ingested.
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, embedding: list[float], top_k: int = 5, where: dict | None = None) -> list[dict]:
        result = self._collection.query(query_embeddings=[embedding], n_results=top_k, where=where)
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            {"id": ids[i], "document": documents[i], "distance": distances[i], "metadata": metadatas[i]}
            for i in range(len(ids))
        ]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._collection.delete(ids=ids)
