from __future__ import annotations

import json

import httpx
import pytest

from app.llm import get_llm_backend
from app.llm.base import LLMError
from app.llm.gemini_backend import GeminiLLM
from app.llm.groq_backend import GroqLLM
from app.llm.ollama_backend import OllamaLLM


def _ollama_ndjson_handler(request: httpx.Request) -> httpx.Response:
    lines = [
        json.dumps({"message": {"content": "Hello "}, "done": False}),
        json.dumps({"message": {"content": "world."}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]
    return httpx.Response(200, text="\n".join(lines) + "\n")


async def test_ollama_streams_pieces_in_order():
    transport = httpx.MockTransport(_ollama_ndjson_handler)
    llm = OllamaLLM(base_url="http://fake-ollama", model="qwen2.5:3b", transport=transport)
    pieces = [p async for p in llm.generate_stream("sys", "user")]
    assert pieces == ["Hello ", "world."]


async def test_ollama_sends_temperature_and_seed_under_options(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "llm_temperature", 0.0, raising=False)
    monkeypatch.setattr(get_settings(), "llm_seed", 42, raising=False)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _ollama_ndjson_handler(request)

    transport = httpx.MockTransport(handler)
    llm = OllamaLLM(base_url="http://fake-ollama", transport=transport)
    [_ async for _ in llm.generate_stream("sys", "user")]
    assert captured["payload"]["options"] == {"temperature": 0.0, "seed": 42}


async def test_ollama_raises_clear_error_when_unreachable():
    # No mock transport, nothing listening on this port: a real connection
    # failure, proving the error path works without needing a live Ollama.
    llm = OllamaLLM(base_url="http://127.0.0.1:1", timeout=2.0)
    with pytest.raises(LLMError, match="Is it running"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


async def test_ollama_raises_clear_error_on_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated slow response")

    transport = httpx.MockTransport(handler)
    llm = OllamaLLM(base_url="http://fake-ollama", transport=transport)
    with pytest.raises(LLMError, match="did not respond"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


async def test_ollama_raises_on_bad_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    transport = httpx.MockTransport(handler)
    llm = OllamaLLM(base_url="http://fake-ollama", transport=transport)
    with pytest.raises(LLMError, match="ollama pull"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


def _groq_sse_handler(request: httpx.Request) -> httpx.Response:
    body = (
        'data: {"choices":[{"delta":{"content":"Hi "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"there."}}]}\n\n'
        "data: [DONE]\n\n"
    )
    return httpx.Response(200, text=body)


async def test_groq_streams_pieces_in_order():
    transport = httpx.MockTransport(_groq_sse_handler)
    llm = GroqLLM(api_key="test-key", base_url="http://fake-groq", transport=transport)
    pieces = [p async for p in llm.generate_stream("sys", "user")]
    assert pieces == ["Hi ", "there."]


def _groq_sse_handler_with_usage(request: httpx.Request) -> httpx.Response:
    # Confirmed against a real Groq response: the final chunk (empty delta,
    # finish_reason "stop") carries a top-level "usage" object even without
    # requesting stream_options.include_usage.
    body = (
        'data: {"choices":[{"delta":{"content":"Hi."}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":78,"completion_tokens":208,"total_tokens":286}}\n\n'
        "data: [DONE]\n\n"
    )
    return httpx.Response(200, text=body)


async def test_groq_captures_token_usage_from_the_final_chunk():
    transport = httpx.MockTransport(_groq_sse_handler_with_usage)
    llm = GroqLLM(api_key="test-key", base_url="http://fake-groq", transport=transport)
    assert llm.last_usage is None  # nothing parsed yet, before the stream has run
    [_ async for _ in llm.generate_stream("sys", "user")]
    assert llm.last_usage == {"input": 78, "output": 208}


async def test_groq_without_api_key_raises_immediately(monkeypatch):
    # Same reasoning as the Cohere equivalent: GroqLLM(api_key=None) falls
    # back to settings.groq_api_key, so this must force that setting blank
    # too or it'll silently pass using whatever real key is in .env.
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "groq_api_key", None, raising=False)
    llm = GroqLLM(api_key=None)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


async def test_groq_raises_on_bad_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    transport = httpx.MockTransport(handler)
    llm = GroqLLM(api_key="bad-key", base_url="http://fake-groq", transport=transport)
    with pytest.raises(LLMError, match="401"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


async def test_groq_sends_temperature_and_seed(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "llm_temperature", 0.0, raising=False)
    monkeypatch.setattr(get_settings(), "llm_seed", 42, raising=False)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _groq_sse_handler(request)

    transport = httpx.MockTransport(handler)
    llm = GroqLLM(api_key="test-key", base_url="http://fake-groq", transport=transport)
    [_ async for _ in llm.generate_stream("sys", "user")]
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["seed"] == 42


async def test_groq_raises_clear_error_on_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated slow response")

    transport = httpx.MockTransport(handler)
    llm = GroqLLM(api_key="test-key", base_url="http://fake-groq", transport=transport)
    with pytest.raises(LLMError, match="did not respond"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


async def test_groq_sends_a_max_completion_tokens_budget():
    # openai/gpt-oss-20b spends tokens on hidden "reasoning" before writing
    # "content" — confirmed live that a tight budget lets reasoning consume
    # the whole cap, leaving content empty. Needs real headroom, not just
    # whatever Groq's own unset default happens to be.
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _groq_sse_handler(request)

    transport = httpx.MockTransport(handler)
    llm = GroqLLM(api_key="test-key", base_url="http://fake-groq", transport=transport)
    [_ async for _ in llm.generate_stream("sys", "user")]
    assert captured["payload"]["max_completion_tokens"] == 2048


async def test_groq_raises_a_clear_error_when_reasoning_exhausts_the_token_budget():
    # Regression for a real, confirmed-live failure: a reasoning model can
    # hit finish_reason="length" while still mid-reasoning, so "content" never
    # gets written — the app must surface this as an error, not a silent
    # blank answer dressed up as a trivially "fully grounded" response.
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    transport = httpx.MockTransport(handler)
    llm = GroqLLM(api_key="test-key", base_url="http://fake-groq", transport=transport)
    with pytest.raises(LLMError, match="ran out of room"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


async def test_groq_does_not_raise_when_length_is_hit_after_real_content_was_written():
    # A normal answer can also legitimately hit the token cap (a long,
    # well-supported response) — only an EMPTY completion at finish_reason
    # "length" is the reasoning-starved failure case worth raising on.
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices":[{"delta":{"content":"A full answer."}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    transport = httpx.MockTransport(handler)
    llm = GroqLLM(api_key="test-key", base_url="http://fake-groq", transport=transport)
    pieces = [p async for p in llm.generate_stream("sys", "user")]
    assert pieces == ["A full answer."]


def _gemini_sse_handler(request: httpx.Request) -> httpx.Response:
    body = (
        'data: {"choices":[{"delta":{"content":"Hey "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"there."}}]}\n\n'
        "data: [DONE]\n\n"
    )
    return httpx.Response(200, text=body)


async def test_gemini_streams_pieces_in_order():
    transport = httpx.MockTransport(_gemini_sse_handler)
    llm = GeminiLLM(api_key="test-key", base_url="http://fake-gemini", transport=transport)
    pieces = [p async for p in llm.generate_stream("sys", "user")]
    assert pieces == ["Hey ", "there."]


async def test_gemini_sends_temperature_but_not_seed(monkeypatch):
    # Gemini's OpenAI-compatible endpoint rejects "seed" outright
    # (confirmed live: 400, "Unknown name \"seed\": Cannot find field") —
    # unlike Groq/Ollama, only temperature should ever be sent.
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "llm_temperature", 0.0, raising=False)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _gemini_sse_handler(request)

    transport = httpx.MockTransport(handler)
    llm = GeminiLLM(api_key="test-key", base_url="http://fake-gemini", transport=transport)
    [_ async for _ in llm.generate_stream("sys", "user")]
    assert captured["payload"]["temperature"] == 0.0
    assert "seed" not in captured["payload"]


async def test_gemini_without_api_key_raises_immediately(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "gemini_api_key", None, raising=False)
    llm = GeminiLLM(api_key=None)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


async def test_gemini_raises_on_bad_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    transport = httpx.MockTransport(handler)
    llm = GeminiLLM(api_key="bad-key", base_url="http://fake-gemini", transport=transport)
    with pytest.raises(LLMError, match="401"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


async def test_gemini_raises_clear_error_on_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated slow response")

    transport = httpx.MockTransport(handler)
    llm = GeminiLLM(api_key="test-key", base_url="http://fake-gemini", transport=transport)
    with pytest.raises(LLMError, match="did not respond"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


def test_factory_picks_ollama_for_local():
    assert isinstance(get_llm_backend("local"), OllamaLLM)


def test_factory_picks_groq():
    assert isinstance(get_llm_backend("groq"), GroqLLM)


def test_factory_picks_gemini():
    assert isinstance(get_llm_backend("gemini"), GeminiLLM)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown LLM_BACKEND"):
        get_llm_backend("something-else")
