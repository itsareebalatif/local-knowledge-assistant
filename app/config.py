
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
    # Groq's exact available model IDs vary by account — before trusting any
    # hardcoded default (including this one), check what your key can
    # actually use: curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
    groq_model: str = "openai/gpt-oss-20b"
    groq_timeout_seconds: float = 60.0

    # --- Optional cloud LLM: Google AI Studio (Gemini), OpenAI-compatible endpoint ---
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    # gemini-2.0-flash and gemini-2.5-flash have both been retired by
    # Google; confirmed via a live 404 pointing at this replacement.
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_seconds: float = 60.0

    # --- Generation (Pipeline 2) ---
    llm_backend: str = "local"  # "local" (Ollama) | "groq" | "gemini"
    # The evaluation harness's LLM-as-judge call (context precision) can use
    # a different backend than live generation — e.g. Gemini's free tier for
    # judging, Groq or local Ollama for real user queries. Defaults to
    # whatever LLM_BACKEND already is, so leaving this unset changes nothing.
    eval_llm_backend: str | None = None
    min_sentence_grounding_coverage: float = 0.3  # per-sentence keyword overlap floor, post-generation

    # --- Embeddings backend ---
    embedding_backend: str = "ollama"  # "ollama" (local) | "cohere"
    cohere_api_key: str | None = None
    cohere_base_url: str = "https://api.cohere.com/v2"
    cohere_embed_model: str = "embed-english-v3.0"
    cohere_timeout_seconds: float = 60.0

    # --- Auth: token lifetimes ---
    # The access token is short-lived and sent on every request — if it
    # leaks, the window it's useful in is small. The refresh token is
    # long-lived, sent only to POST /api/auth/refresh, and is what actually
    # gets stored client-side for the long haul; it's rotated (a new one
    # issued, the old one invalidated) every time it's used, so a stolen
    # refresh token only works once before the legitimate client's next
    # refresh call reveals the theft (its own next refresh attempt fails).
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30

    # --- Chunking ---
    chunk_token_size: int = 512

    # --- Retrieval: per-source candidate counts (before fusion) ---
    bm25_top_k: int = 5
    vector_top_k: int = 5
    graph_top_k: int = 5

    # --- Retrieval: Reciprocal Rank Fusion ---
    rrf_k: int = 60  # dampening constant from the original RRF paper
    rrf_top_n: int = 10  # candidates kept after fusion, before reranking/the grounding gate

    # --- Retrieval: Reranking (second relevance pass over RRF's shortlist) ---
    # Free and fully local — no API key, see app/search/reranker.py. First
    # use downloads the model (~90MB) from Hugging Face and caches it;
    # after that, no network access is needed. Set to false to skip this
    # step entirely (faster startup, falls back to RRF's own ordering).
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Candidates kept after reranking, before the grounding gate — but only
    # when MMR (below) is disabled. When MMR is also enabled, it does the
    # actual final cut instead, so reranking just reorders without
    # truncating (MMR needs the fuller pool to have anything to pick a
    # diverse subset from).
    rerank_top_k: int = 5

    # --- Retrieval: MMR (final diversity-aware selection, replaces a plain
    # top-k cut) --- see app/search/mmr.py. Keeps two high-scoring but
    # near-duplicate chunks from both occupying the limited context budget.
    mmr_enabled: bool = True
    mmr_top_k: int = 5  # final candidates handed to the grounding gate
    mmr_lambda: float = 0.5  # 1.0 = pure relevance (same as plain top-k); 0.0 = pure diversity

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
