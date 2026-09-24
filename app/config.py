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

    # --- Graph store ---
    graph_store_path: str = str(BASE_DIR / "data" / "graph.gpickle")

    # --- Local LLM (Ollama) ---
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    embedding_model: str = "nomic-embed-text"

    # --- Optional cloud fallback (blank = fully local/offline) ---
    groq_api_key: str | None = None
    gemini_api_key: str | None = None

    # --- Chunking ---
    chunk_token_size: int = 512

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — call this, don't instantiate Settings() directly."""
    return Settings()
