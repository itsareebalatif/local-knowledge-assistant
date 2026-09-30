
from __future__ import annotations

import httpx

from app.config import get_settings


class EmbeddingError(Exception):
    """Raised when Ollama can't be reached or returns something unusable —
    e.g. the model hasn't been pulled yet (`ollama pull nomic-embed-text`)."""


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.embedding_model
        self.timeout = timeout if timeout is not None else settings.ollama_timeout_seconds
        self._transport = transport  # test hook; None uses a real network transport

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": texts},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise EmbeddingError(
                f"Could not reach Ollama at {self.base_url}. Is it running? (`ollama serve`)"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Ollama returned {exc.response.status_code}: {exc.response.text}. "
                f"Is the model pulled? (`ollama pull {self.model}`)"
            ) from exc

        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise EmbeddingError(f"Unexpected response from Ollama embeddings endpoint: {data!r}")
        return embeddings
