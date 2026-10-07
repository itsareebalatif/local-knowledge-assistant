"""Cloud embedding backend via Cohere's Embed v2 API — the non-local
alternative to app.embeddings.embedder.OllamaEmbedder.

Picked over the LLM's own cloud fallback (Groq) specifically because Groq
doesn't offer an embeddings endpoint at all; Cohere does. Choosing this one
means chunk text leaves the machine at embed time, same privacy trade-off
already made for the Groq LLM backend — opt-in via EMBEDDING_BACKEND=cohere
in .env, never the default.

input_type is required by Cohere's API and actually matters for quality:
"search_document" at index time, "search_query" at query time, on the same
embedding space. The shared embedder interface threads this through from
both call sites (embedding_service.py for indexing, vector_search.py for
querying) rather than hardcoding one.
"""

from __future__ import annotations

import httpx

from app.config import get_settings
from app.embeddings.embedder import EmbeddingError


class CohereEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.cohere_api_key
        self.base_url = (base_url or settings.cohere_base_url).rstrip("/")
        self.model = model or settings.cohere_embed_model
        self.timeout = timeout if timeout is not None else settings.cohere_timeout_seconds
        self._transport = transport

    async def embed(self, texts: list[str], input_type: str = "search_document") -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise EmbeddingError("COHERE_API_KEY is not set — add it to .env or switch EMBEDDING_BACKEND=ollama.")

        payload = {"model": self.model, "texts": texts, "input_type": input_type}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                response = await client.post(f"{self.base_url}/embed", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise EmbeddingError(f"Could not reach Cohere at {self.base_url}.") from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(f"Cohere returned {exc.response.status_code}: {exc.response.text}") from exc

        # v2 nests float vectors under embeddings.float (see module docstring
        # — other formats like int8/binary are opt-in via embedding_types,
        # not requested here).
        embeddings = (data.get("embeddings") or {}).get("float")
        if not embeddings or len(embeddings) != len(texts):
            raise EmbeddingError(f"Unexpected response from Cohere embed endpoint: {data!r}")
        return embeddings
