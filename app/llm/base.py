"""Common interface every LLM backend implements (FR-4.5 Pipeline 2, FR-4.6).

Both backends stream: `generate_stream` is an async generator yielding text
pieces as they arrive, so app/services/generation_service.py can forward
them to an SSE endpoint token-by-token instead of waiting for a full
response (FR-4.6: real-time streaming, low time-to-first-token).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMError(Exception):
    """Raised when an LLM backend can't be reached or returns something
    unusable. Never leaks a raw connection traceback to the caller."""


class LLMBackend(ABC):
    @abstractmethod
    def generate_stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        """Yield the answer as it's generated, piece by piece."""
        raise NotImplementedError
