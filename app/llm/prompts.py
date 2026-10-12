from __future__ import annotations

from app.search.types import CandidateChunk

SYSTEM_PROMPT = (
    "You are a careful knowledge assistant. Answer the user's question using ONLY the "
    "numbered context passages provided below — never use outside knowledge, and never "
    "guess. Every factual claim in your answer must be supported by at least one passage. "
    "Cite the passage you used immediately after each claim with a bracketed marker like "
    "[1] or [2], matching the passage numbers given. "
    "If the passages do not contain enough information to answer, say plainly that you "
    "don't know rather than filling the gap with anything not stated in the context."
)


def build_user_prompt(query: str, candidates: list[CandidateChunk]) -> tuple[str, dict[int, int]]:
    # Chunk numbering here must match app/llm/citations.py's marker numbering.
    marker_by_chunk_id = {c.chunk_id: i + 1 for i, c in enumerate(candidates)}
    context_blocks = "\n\n".join(f"[{i + 1}] {c.content}" for i, c in enumerate(candidates))
    prompt = f"Context passages:\n{context_blocks}\n\nQuestion: {query}\n\nAnswer:"
    return prompt, marker_by_chunk_id
