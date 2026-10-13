from __future__ import annotations

import json

import httpx

from app.config import get_settings
from app.llm.base import LLMBackend, LLMError


class OllamaLLM(LLMBackend):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout if timeout is not None else settings.ollama_timeout_seconds
        self._transport = transport

    async def generate_stream(self, system_prompt: str, user_prompt: str):
        settings = get_settings()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            # Ollama nests temperature/seed under "options", unlike Groq/Gemini's top-level fields.
            "options": {"temperature": settings.llm_temperature, "seed": settings.llm_seed},
        }
        try:
            async with (
                httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client,
                client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response,
            ):
                if response.status_code >= 400:
                    body = await response.aread()
                    raise LLMError(
                        f"Ollama returned {response.status_code}: {body.decode(errors='replace')}. "
                        f"Is the model pulled? (`ollama pull {self.model}`)"
                    )
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done"):
                        break
        except httpx.ConnectError as exc:
            raise LLMError(f"Could not reach Ollama at {self.base_url}. Is it running? (`ollama serve`)") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama did not respond within {self.timeout}s.") from exc
