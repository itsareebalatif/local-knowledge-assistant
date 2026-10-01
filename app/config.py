"""Central application configuration.

Reads from environment variables / a local .env file (never committed — see
.env.example for the template). Every other module should pull settings from
here via `get_settings()` rather than reading `os.environ` directly, so the
whole app has one source of truth and stays easy to test.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"

    # --- Database ---
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'pke.sqlite3'}"

    # --- Vector store ---
    vector_store_backend: str = "chroma"
    vector_store_path: str = str(BASE_DIR / "data" / "chroma")
    vector_store_collection: str = "chunks"

    # --- Graph store ---
    graph_store_path: str = str(BASE_DIR / "data" / "graph.gpickle")

    # --- Local LLM (Ollama) ---
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout_seconds: float = 60.0
    embedding_model: str = "nomic-embed-text"

    # --- Optional cloud fallback (blank = fully local/offline) ---
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.1-8b-instant"
    groq_timeout_seconds: float = 60.0

    # --- Generation (Pipeline 2) ---
    llm_backend: str = "local"  # "local" (Ollama) | "groq"
    min_sentence_grounding_coverage: float = 0.3  # per-sentence keyword overlap floor, post-generation

    # --- Chunking ---
    chunk_token_size: int = 512

    # --- Retrieval: per-source candidate counts (before fusion) ---
    bm25_top_k: int = 5
    vector_top_k: int = 5
    graph_top_k: int = 5

    # --- Retrieval: Reciprocal Rank Fusion ---
    rrf_k: int = 60  # dampening constant from the original RRF paper
    rrf_top_n: int = 10  # candidates kept after fusion, before the grounding gate

    # --- Retrieval: Context Sufficiency / Grounding Gate ---
    # Starting points, not yet tuned against real query/candidate data.
    min_retrieval_score: float = 0.25
    min_keyword_coverage: float = 0.4
    min_grounding_candidates: int = 1

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — call this, don't instantiate Settings() directly."""
    return Settings()
