from __future__ import annotations

import hashlib

from app.ingestion.loader import hash_and_load_local_file, load_local_file


async def test_load_local_file_returns_bytes_and_filename(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"hello world")

    data, filename = await load_local_file(path)

    assert data == b"hello world"
    assert filename == "notes.txt"


async def test_hash_and_load_local_file_matches_separate_hash_and_read(tmp_path):
    path = tmp_path / "doc.md"
    content = b"# Title\n\nSome content here, repeated. " * 50  # bigger than one read, still < stream chunk size
    path.write_bytes(content)

    file_hash, data, filename = await hash_and_load_local_file(path)

    assert data == content
    assert filename == "doc.md"
    assert file_hash == hashlib.sha256(content).hexdigest()


async def test_hash_and_load_local_file_streams_across_multiple_chunks(tmp_path, monkeypatch):
    import app.ingestion.loader as loader_module

    # Force a tiny stream chunk size so a normal-sized file exercises the
    # while-loop's multi-read path, not just a single read() call.
    monkeypatch.setattr(loader_module, "_STREAM_CHUNK_SIZE", 16)
    path = tmp_path / "big.txt"
    content = b"x" * 1000
    path.write_bytes(content)

    file_hash, data, _ = await hash_and_load_local_file(path)

    assert data == content
    assert file_hash == hashlib.sha256(content).hexdigest()


async def test_hash_and_load_local_file_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")

    file_hash, data, _ = await hash_and_load_local_file(path)

    assert data == b""
    assert file_hash == hashlib.sha256(b"").hexdigest()
