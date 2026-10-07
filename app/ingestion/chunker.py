
from __future__ import annotations

import re

from app.ingestion.tokenizer import count_tokens
from app.ingestion.types import Block, ChunkDraft

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])")
_MAX_BREADCRUMB_DEPTH = 6


class HeaderAwareChunker:
    def __init__(self, max_tokens: int = 512):
        if max_tokens < 32:
            raise ValueError("max_tokens must be large enough to hold at least a sentence or two")
        self.max_tokens = max_tokens

    def chunk(self, blocks: list[Block]) -> list[ChunkDraft]:
        chunks: list[ChunkDraft] = []
        heading_stack: list[tuple[int, str]] = []  # (level, text), outermost first
        buffer_parts: list[str] = []
        buffer_tokens = 0

        def heading_path() -> list[str]:
            return [text for _, text in heading_stack[:_MAX_BREADCRUMB_DEPTH]]

        def flush() -> None:
            nonlocal buffer_parts, buffer_tokens
            if not buffer_parts:
                return
            content = "\n\n".join(buffer_parts)
            chunks.append(
                ChunkDraft(
                    chunk_index=len(chunks),
                    content=content,
                    token_count=count_tokens(content),
                    heading_path=heading_path(),
                )
            )
            buffer_parts = []
            buffer_tokens = 0

        for block in blocks:
            if block.type == "heading":
                # A new heading always starts a fresh chunk: mixing two
                # sections into one chunk would blur retrieval relevance.
                flush()
                while heading_stack and heading_stack[-1][0] >= block.level:
                    heading_stack.pop()
                heading_stack.append((block.level, block.text))

                # Seed the new section with its own heading text. Without
                # this, a heading followed by no paragraph/table/code block
                # (an outline or agenda made entirely of headings — "Day 1",
                # "Day 2", ...) never reaches flush() with anything in the
                # buffer, so the whole section silently vanishes: zero
                # chunks, zero warnings, looks like a successful ingest of
                # nothing. Bonus effect: every chunk now starts with its own
                # section heading, which helps BM25/keyword matching on
                # queries that name the section directly.
                heading_text = block.text.strip()
                if heading_text:
                    buffer_parts.append(heading_text)
                    buffer_tokens += count_tokens(heading_text)
                continue

            for piece in self._split_to_budget(block.text):
                piece_tokens = count_tokens(piece)
                if buffer_parts and buffer_tokens + piece_tokens > self.max_tokens:
                    flush()
                buffer_parts.append(piece)
                buffer_tokens += piece_tokens

        flush()
        return chunks

    def _split_to_budget(self, text: str) -> list[str]:

        if count_tokens(text) <= self.max_tokens:
            return [text]

        pieces: list[str] = []
        current = ""
        for sentence in _SENTENCE_SPLIT.split(text):
            candidate = f"{current} {sentence}".strip() if current else sentence
            if count_tokens(candidate) <= self.max_tokens:
                current = candidate
                continue
            if current:
                pieces.append(current)
            if count_tokens(sentence) <= self.max_tokens:
                current = sentence
            else:
                pieces.extend(self._hard_wrap(sentence))
                current = ""
        if current:
            pieces.append(current)
        return pieces or [text]

    def _hard_wrap(self, text: str) -> list[str]:
        words = text.split(" ")
        pieces: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if current and count_tokens(candidate) > self.max_tokens:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces
