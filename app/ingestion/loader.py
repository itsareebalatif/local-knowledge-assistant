"""Async local-file loading. Network-drive paths work the same way here (a
mounted drive is still a filesystem path); a web-link source would be a
separate loader that also produces (bytes, filename) for `ingest_bytes`."""

from __future__ import annotations

from pathlib import Path

import aiofiles


async def load_local_file(path: str | Path) -> tuple[bytes, str]:
    path = Path(path)
    async with aiofiles.open(path, "rb") as f:
        data = await f.read()
    return data, path.name
