from __future__ import annotations

import json

import httpx

from app.config import get_settings
from app.llm.base import LLMBackend, LLMError


class OpenAICompatibleLLM(LLMBackend):
    """One backend implementation for every provider that speaks the OpenAI
    chat-completions wire format — Groq, Gemini's OpenAI-compatible endpoint,
    OpenAI itself, and most others (Together, Fireworks, etc. all use the same
    shape). Adding a new provider, or switching models on an existing one, is
    a configuration change in app/llm/__init__.py's provider registry or a
    .env value — never a new backend class or file.

    The handful of real differences between providers (whether `seed` is
    accepted, how much headroom a reasoning model needs, whether to watch for
    a reasoning model silently exhausting its token budget) are constructor
    parameters here, not separate code paths in separate files."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        provider_name: str = "the model provider",
        api_key_env_var: str = "API_KEY",
        send_seed: bool = True,
        max_completion_tokens: int | None = None,
        detect_reasoning_exhaustion: bool = False,
        transport: httpx.BaseTransport | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.provider_name = provider_name
        self.api_key_env_var = api_key_env_var
        self.send_seed = send_seed
        self.max_completion_tokens = max_completion_tokens
        self.detect_reasoning_exhaustion = detect_reasoning_exhaustion
        self._transport = transport
        self.last_usage: dict[str, int] | None = None

    async def generate_stream(self, system_prompt: str, user_prompt: str):
        if not self.api_key:
            raise LLMError(
                f"{self.api_key_env_var} is not set — add it to .env or switch LLM_BACKEND to a configured provider."
            )

        settings = get_settings()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "temperature": settings.llm_temperature,
        }
        if self.send_seed:
            payload["seed"] = settings.llm_seed
        if self.max_completion_tokens is not None:
            # Some models (reasoning models especially — see
            # detect_reasoning_exhaustion below) spend tokens on a hidden
            # "reasoning" pass before writing visible "content". A tight cap
            # lets that hidden pass consume the whole budget, leaving content
            # empty. This headroom is for that hidden pass, not just the
            # visible answer.
            payload["max_completion_tokens"] = self.max_completion_tokens

        headers = {"Authorization": f"Bearer {self.api_key}"}
        got_any_content = False
        finish_reason = None
        try:
            async with (
                httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client,
                client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
                ) as response,
            ):
                if response.status_code >= 400:
                    body = await response.aread()
                    raise LLMError(
                        f"{self.provider_name} returned {response.status_code}: {body.decode(errors='replace')}"
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:") :].strip()
                    if payload_str == "[DONE]":
                        break
                    data = json.loads(payload_str)
                    usage = data.get("usage")
                    if usage:
                        self.last_usage = {
                            "input": usage.get("prompt_tokens", 0),
                            "output": usage.get("completion_tokens", 0),
                        }
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    finish_reason = choices[0].get("finish_reason") or finish_reason
                    piece = choices[0].get("delta", {}).get("content", "")
                    if piece:
                        got_any_content = True
                        yield piece
        except httpx.ConnectError as exc:
            raise LLMError(f"Could not reach {self.provider_name} at {self.base_url}.") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"{self.provider_name} did not respond within {self.timeout}s.") from exc

        if self.detect_reasoning_exhaustion and not got_any_content and finish_reason == "length":
            raise LLMError(
                f"{self.provider_name}'s model ran out of room reasoning about this question before writing "
                "an answer. Try a shorter or more specific question."
            )
