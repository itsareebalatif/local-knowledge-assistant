from __future__ import annotations

from app.llm.prompts import SYSTEM_PROMPT, build_system_prompt, build_user_prompt
from app.search.types import CandidateChunk


def test_system_prompt_is_fact_restricted():
    lowered = SYSTEM_PROMPT.lower()
    assert "only" in lowered
    assert "don't know" in lowered
    assert "[1]" in SYSTEM_PROMPT


def test_build_system_prompt_defaults_match_settings(monkeypatch):
    from app import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("ASSISTANT_NAME", "Study Buddy")
    config.get_settings.cache_clear()
    try:
        prompt = build_system_prompt()
        assert "You are Study Buddy," in prompt
    finally:
        config.get_settings.cache_clear()


def test_build_system_prompt_accepts_an_explicit_override():
    prompt = build_system_prompt(assistant_name="Acme Support Bot", citation_marker_style="(Source {n})")
    assert "You are Acme Support Bot," in prompt
    assert "(Source 1)" in prompt
    assert "(Source 2)" in prompt


def test_build_system_prompt_override_does_not_touch_global_settings():
    from app.config import get_settings

    build_system_prompt(assistant_name="Temporary Override Bot")
    assert get_settings().assistant_name != "Temporary Override Bot"


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
