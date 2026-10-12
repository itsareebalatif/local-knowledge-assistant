from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMError(Exception):
    pass


class LLMBackend(ABC):
    @abstractmethod
    def generate_stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        raise NotImplementedError
