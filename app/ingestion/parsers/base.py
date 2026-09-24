"""Base parser interface every format parser implements.

`parse` takes raw bytes rather than a filesystem path so the ingestion
package stays decoupled from where a file came from (local disk, a network
drive, a web download) — FR-1.1 lists all three as ingestion sources, but
sourcing bytes is out of scope for this module; see app/ingestion/loader.py
for the local-filesystem case.

Parsing libraries here (pypdf/pdfplumber, python-docx, BeautifulSoup) are all
synchronous/CPU-bound. `parse` is declared `async` because the ingestion
pipeline processes many files concurrently (FR-1.1: "read files
asynchronously") — each parser offloads its blocking work to a thread via
`asyncio.to_thread` so it never blocks the event loop, without pretending the
underlying C/Python parsing itself is non-blocking I/O.
"""

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
        """Blocking parse implementation. Never call directly from async code —
        go through `parse()` so it runs off the event loop thread."""
        raise NotImplementedError
