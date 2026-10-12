from __future__ import annotations

import json

import httpx

from app.config import get_settings
from app.llm.base import LLMBackend, LLMError


class GeminiLLM(LLMBackend):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.base_url = (base_url or settings.gemini_base_url).rstrip("/")
        self.model = model or settings.gemini_model
        self.timeout = timeout if timeout is not None else settings.gemini_timeout_seconds
        self._transport = transport
        self.last_usage: dict[str, int] | None = None

    async def generate_stream(self, system_prompt: str, user_prompt: str):
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set — add it to .env or switch to a different backend.")

        settings = get_settings()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            # No "seed": Gemini's OpenAI-compatible endpoint rejects it (400 Unknown name "seed").
            "temperature": settings.llm_temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
                ) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise LLMError(f"Gemini returned {response.status_code}: {body.decode(errors='replace')}")
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
                        piece = choices[0].get("delta", {}).get("content", "")
                        if piece:
                            yield piece
        except httpx.ConnectError as exc:
            raise LLMError(f"Could not reach Gemini at {self.base_url}.") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Gemini did not respond within {self.timeout}s.") from exc
