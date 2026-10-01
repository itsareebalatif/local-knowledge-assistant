from __future__ import annotations

from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.search.types import CandidateChunk


def test_system_prompt_is_fact_restricted():
    lowered = SYSTEM_PROMPT.lower()
    assert "only" in lowered
    assert "don't know" in lowered
    assert "[1]" in SYSTEM_PROMPT or "bracketed marker" in lowered


def test_build_user_prompt_numbers_candidates_and_maps_markers():
    candidates = [
        CandidateChunk(chunk_id=10, doc_id=1, content="first chunk", score=1.0),
        CandidateChunk(chunk_id=20, doc_id=1, content="second chunk", score=0.9),
    ]
    prompt, marker_by_chunk_id = build_user_prompt("what is X?", candidates)

    assert "[1] first chunk" in prompt
    assert "[2] second chunk" in prompt
    assert "what is X?" in prompt
    assert marker_by_chunk_id == {10: 1, 20: 2}


def test_build_user_prompt_with_no_candidates():
    prompt, marker_by_chunk_id = build_user_prompt("what is X?", [])
    assert marker_by_chunk_id == {}
    assert "what is X?" in prompt
