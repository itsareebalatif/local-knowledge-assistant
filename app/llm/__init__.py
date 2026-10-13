"""LLM backend factory — every cloud provider is the same OpenAICompatibleLLM
implementation configured differently; only "local" (Ollama) is a genuinely
different class, because Ollama speaks a different wire protocol, not because
it needed separate code for the same protocol. The rest of the app depends
only on the LLMBackend interface, never a specific provider.
"""

from __future__ import annotations

from collections.abc import Callable

from app.config import get_settings
from app.llm.base import LLMBackend, LLMError
from app.llm.ollama_backend import OllamaLLM
from app.llm.openai_compatible_backend import OpenAICompatibleLLM


def _groq() -> OpenAICompatibleLLM:
    s = get_settings()
    return OpenAICompatibleLLM(
        api_key=s.groq_api_key,
        base_url=s.groq_base_url,
        model=s.groq_model,
        timeout=s.groq_timeout_seconds,
        provider_name="Groq",
        api_key_env_var="GROQ_API_KEY",
        send_seed=True,
        # openai/gpt-oss-20b is a reasoning model — see
        # OpenAICompatibleLLM.generate_stream for why this headroom and the
        # exhaustion check below both matter for it specifically.
        max_completion_tokens=2048,
        detect_reasoning_exhaustion=True,
    )


def _gemini() -> OpenAICompatibleLLM:
    s = get_settings()
    return OpenAICompatibleLLM(
        api_key=s.gemini_api_key,
        base_url=s.gemini_base_url,
        model=s.gemini_model,
        timeout=s.gemini_timeout_seconds,
        provider_name="Gemini",
        api_key_env_var="GEMINI_API_KEY",
        # Gemini's OpenAI-compatible endpoint rejects "seed" outright
        # (confirmed live: 400, 'Unknown name "seed"').
        send_seed=False,
    )


def _openai() -> OpenAICompatibleLLM:
    s = get_settings()
    return OpenAICompatibleLLM(
        api_key=s.openai_api_key,
        base_url=s.openai_base_url,
        model=s.openai_model,
        timeout=s.openai_timeout_seconds,
        provider_name="OpenAI",
        api_key_env_var="OPENAI_API_KEY",
        send_seed=True,
    )


# Every cloud entry here is the same OpenAICompatibleLLM implementation with
# different settings — adding a new OpenAI-compatible provider (Together,
# Fireworks, a self-hosted vLLM server, etc.) means adding one factory
# function above plus its *_api_key/*_base_url/*_model settings in
# app/config.py, never a new backend class.
_PROVIDERS: dict[str, Callable[[], LLMBackend]] = {
    "local": OllamaLLM,
    "groq": _groq,
    "gemini": _gemini,
    "openai": _openai,
}


def get_llm_backend(name: str | None = None) -> LLMBackend:
    backend = (name or get_settings().llm_backend).lower()
    factory = _PROVIDERS.get(backend)
    if factory is None:
        raise ValueError(f"Unknown LLM_BACKEND: {backend!r} — expected one of {sorted(_PROVIDERS)}")
    return factory()


__all__ = ["LLMBackend", "LLMError", "OllamaLLM", "OpenAICompatibleLLM", "get_llm_backend"]
