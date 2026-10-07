from app.config import get_settings
from app.embeddings.cohere_embedder import CohereEmbedder
from app.embeddings.embedder import EmbeddingError, OllamaEmbedder
from app.embeddings.vector_store import ChromaVectorStore, VectorStore


def get_embedder_backend(name: str | None = None):
    """Picks Ollama (local, default) or Cohere (opt-in cloud) based on
    Settings.embedding_backend — the embedding-side equivalent of
    app.llm.get_llm_backend(), so the rest of the app depends only on the
    shared `embed(texts, input_type=...)` interface, never a provider."""
    backend = (name or get_settings().embedding_backend).lower()
    if backend == "ollama":
        return OllamaEmbedder()
    if backend == "cohere":
        return CohereEmbedder()
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r} — expected 'ollama' or 'cohere'")


__all__ = ["OllamaEmbedder", "CohereEmbedder", "EmbeddingError", "VectorStore", "ChromaVectorStore", "get_embedder_backend"]
