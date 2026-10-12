from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from app.ingestion.types import ParsedDocument


class BaseParser(ABC):
    file_type: str

    async def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        return await asyncio.to_thread(self.parse_sync, file_bytes, filename)

    @abstractmethod
    def parse_sync(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        # blocking; call via parse() so it runs off the event loop thread
        raise NotImplementedError
