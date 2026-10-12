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
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-20b"
    groq_timeout_seconds: float = 60.0

    # --- Optional cloud LLM: Google AI Studio (Gemini), OpenAI-compatible endpoint ---
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_seconds: float = 60.0

    # --- Generation (Pipeline 2) ---
    llm_backend: str = "local"  # "local" (Ollama) | "groq" | "gemini"
    eval_llm_backend: str | None = None
    llm_temperature: float = 0.0
    llm_seed: int = 42
    min_sentence_grounding_coverage: float = 0.3
    # Rewrites a conversational follow-up ("what about his siblings?") into a
    # standalone query before retrieval, using the prior turn — skipped
    # automatically on a conversation's first message (nothing to resolve
    # against), so this only adds latency/cost on genuine follow-ups.
    query_rewrite_enabled: bool = True

    # --- Embeddings backend ---
    embedding_backend: str = "ollama"  # "ollama" (local) | "cohere"
    cohere_api_key: str | None = None
    cohere_base_url: str = "https://api.cohere.com/v2"
    cohere_embed_model: str = "embed-english-v3.0"
    cohere_timeout_seconds: float = 60.0

    # --- Notifications: transactional email via Resend ---
    resend_api_key: str | None = None
    email_from_address: str | None = None
    frontend_base_url: str = "http://localhost:8501"
    password_reset_token_expire_minutes: int = 30

    # --- Auth: token lifetimes ---
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30

    # --- Chunking ---
    chunk_token_size: int = 512

    # --- Retrieval: per-source candidate counts (before fusion) ---
    bm25_top_k: int = 5
    vector_top_k: int = 5
    graph_top_k: int = 5

    # --- Retrieval: Reciprocal Rank Fusion ---
    rrf_k: int = 60
    rrf_top_n: int = 10

    # --- Retrieval: Reranking ---
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5

    # --- Retrieval: MMR ---
    mmr_enabled: bool = True
    mmr_top_k: int = 5
    mmr_lambda: float = 0.7  # tuned live — 0.6 let an unrelated chunk win on diversity

    # --- Observability: Langfuse tracing (optional — leave blank to disable) ---
    langfuse_secret_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()
