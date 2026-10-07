"""LLM backend factory — picks Ollama (local, default), Groq, or Gemini
(both opt-in cloud) based on Settings.llm_backend, so the rest of the app
depends only on the LLMBackend interface, never a specific provider.
"""

from __future__ import annotations

from app.config import get_settings
from app.llm.base import LLMBackend, LLMError
from app.llm.gemini_backend import GeminiLLM
from app.llm.groq_backend import GroqLLM
from app.llm.ollama_backend import OllamaLLM


def get_llm_backend(name: str | None = None) -> LLMBackend:
    backend = (name or get_settings().llm_backend).lower()
    if backend == "local":
        return OllamaLLM()
    if backend == "groq":
        return GroqLLM()
    if backend == "gemini":
        return GeminiLLM()
    raise ValueError(f"Unknown LLM_BACKEND: {backend!r} — expected 'local', 'groq', or 'gemini'")


__all__ = ["LLMBackend", "LLMError", "OllamaLLM", "GroqLLM", "GeminiLLM", "get_llm_backend"]
