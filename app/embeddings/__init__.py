from app.config import get_settings
from app.embeddings.cohere_embedder import CohereEmbedder
from app.embeddings.embedder import EmbeddingError, OllamaEmbedder
from app.embeddings.vector_store import ChromaVectorStore, VectorStore


def get_embedder_backend(name: str | None = None):
    backend = (name or get_settings().embedding_backend).lower()
    if backend == "ollama":
        return OllamaEmbedder()
    if backend == "cohere":
        return CohereEmbedder()
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r} — expected 'ollama' or 'cohere'")


__all__ = ["ChromaVectorStore", "CohereEmbedder", "EmbeddingError", "OllamaEmbedder", "VectorStore", "get_embedder_backend"]
