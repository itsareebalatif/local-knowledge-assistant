
from __future__ import annotations

import hashlib
from pathlib import Path

import aiofiles

_STREAM_CHUNK_SIZE = 1024 * 1024  # 1 MiB


async def load_local_file(path: str | Path) -> tuple[bytes, str]:
    path = Path(path)
    async with aiofiles.open(path, "rb") as f:
        data = await f.read()
    return data, path.name


async def hash_and_load_local_file(path: str | Path) -> tuple[str, bytes, str]:

    path = Path(path)
    hasher = hashlib.sha256()
    buffer = bytearray()
    async with aiofiles.open(path, "rb") as f:
        while True:
            block = await f.read(_STREAM_CHUNK_SIZE)
            if not block:
                break
            hasher.update(block)
            buffer.extend(block)
    return hasher.hexdigest(), bytes(buffer), path.name
