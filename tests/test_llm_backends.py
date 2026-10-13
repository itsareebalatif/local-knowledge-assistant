from __future__ import annotations

import json

import httpx
import pytest

from app.llm import get_llm_backend
from app.llm.base import LLMError
from app.llm.ollama_backend import OllamaLLM
from app.llm.openai_compatible_backend import OpenAICompatibleLLM


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


def _sse_handler(request: httpx.Request) -> httpx.Response:
    body = (
        'data: {"choices":[{"delta":{"content":"Hi "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"there."}}]}\n\n'
        "data: [DONE]\n\n"
    )
    return httpx.Response(200, text=body)


def _make_llm(**overrides) -> OpenAICompatibleLLM:
    params = {
        "api_key": "test-key",
        "base_url": "http://fake-provider",
        "model": "test-model",
        "provider_name": "TestProvider",
        "api_key_env_var": "TEST_API_KEY",
    }
    params.update(overrides)
    return OpenAICompatibleLLM(**params)


# --- OpenAICompatibleLLM: shared behavior, provider-agnostic -----------------


async def test_openai_compatible_streams_pieces_in_order():
    transport = httpx.MockTransport(_sse_handler)
    llm = _make_llm(transport=transport)
    pieces = [p async for p in llm.generate_stream("sys", "user")]
    assert pieces == ["Hi ", "there."]


async def test_openai_compatible_without_api_key_raises_immediately_naming_its_env_var():
    llm = _make_llm(api_key=None, api_key_env_var="SOME_PROVIDER_API_KEY")
    with pytest.raises(LLMError, match="SOME_PROVIDER_API_KEY"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


async def test_openai_compatible_raises_on_bad_status_naming_the_provider():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    transport = httpx.MockTransport(handler)
    llm = _make_llm(provider_name="TestProvider", transport=transport)
    with pytest.raises(LLMError, match="TestProvider returned 401"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


async def test_openai_compatible_raises_clear_error_on_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated slow response")

    transport = httpx.MockTransport(handler)
    llm = _make_llm(transport=transport)
    with pytest.raises(LLMError, match="did not respond"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


async def test_openai_compatible_sends_temperature_and_seed_when_enabled(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "llm_temperature", 0.0, raising=False)
    monkeypatch.setattr(get_settings(), "llm_seed", 42, raising=False)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _sse_handler(request)

    transport = httpx.MockTransport(handler)
    llm = _make_llm(send_seed=True, transport=transport)
    [_ async for _ in llm.generate_stream("sys", "user")]
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["seed"] == 42


async def test_openai_compatible_omits_seed_when_disabled():
    # Gemini's OpenAI-compatible endpoint rejects "seed" outright
    # (confirmed live: 400, 'Unknown name "seed": Cannot find field').
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _sse_handler(request)

    transport = httpx.MockTransport(handler)
    llm = _make_llm(send_seed=False, transport=transport)
    [_ async for _ in llm.generate_stream("sys", "user")]
    assert "seed" not in captured["payload"]


async def test_openai_compatible_sends_max_completion_tokens_when_set():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _sse_handler(request)

    transport = httpx.MockTransport(handler)
    llm = _make_llm(max_completion_tokens=2048, transport=transport)
    [_ async for _ in llm.generate_stream("sys", "user")]
    assert captured["payload"]["max_completion_tokens"] == 2048


async def test_openai_compatible_omits_max_completion_tokens_when_unset():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return _sse_handler(request)

    transport = httpx.MockTransport(handler)
    llm = _make_llm(max_completion_tokens=None, transport=transport)
    [_ async for _ in llm.generate_stream("sys", "user")]
    assert "max_completion_tokens" not in captured["payload"]


async def test_openai_compatible_raises_a_clear_error_when_reasoning_exhausts_the_token_budget():
    # Regression for a real, confirmed-live failure: a reasoning model can
    # hit finish_reason="length" while still mid-reasoning, so "content" never
    # gets written — the app must surface this as an error, not a silent
    # blank answer dressed up as a trivially "fully grounded" response.
    def handler(request: httpx.Request) -> httpx.Response:
        body = 'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text=body)

    transport = httpx.MockTransport(handler)
    llm = _make_llm(detect_reasoning_exhaustion=True, transport=transport)
    with pytest.raises(LLMError, match="ran out of room"):
        async for _ in llm.generate_stream("sys", "user"):
            pass


async def test_openai_compatible_does_not_raise_when_length_is_hit_after_real_content_was_written():
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
    llm = _make_llm(detect_reasoning_exhaustion=True, transport=transport)
    pieces = [p async for p in llm.generate_stream("sys", "user")]
    assert pieces == ["A full answer."]


async def test_openai_compatible_does_not_detect_exhaustion_when_disabled():
    # A provider that isn't known to be a reasoning model shouldn't have this
    # check applied at all — an ordinary length-capped empty response (rare,
    # but possible) shouldn't be mislabeled as "reasoning exhaustion."
    def handler(request: httpx.Request) -> httpx.Response:
        body = 'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text=body)

    transport = httpx.MockTransport(handler)
    llm = _make_llm(detect_reasoning_exhaustion=False, transport=transport)
    pieces = [p async for p in llm.generate_stream("sys", "user")]
    assert pieces == []


# --- Provider registry: each provider is config on top of the one class ----


def test_factory_picks_ollama_for_local():
    assert isinstance(get_llm_backend("local"), OllamaLLM)


def test_factory_picks_groq_configured_as_a_reasoning_model_with_token_headroom():
    llm = get_llm_backend("groq")
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.provider_name == "Groq"
    assert llm.send_seed is True
    assert llm.max_completion_tokens == 2048
    assert llm.detect_reasoning_exhaustion is True


def test_factory_picks_gemini_configured_without_seed():
    llm = get_llm_backend("gemini")
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.provider_name == "Gemini"
    assert llm.send_seed is False
    assert llm.detect_reasoning_exhaustion is False


def test_factory_picks_openai():
    llm = get_llm_backend("openai")
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.provider_name == "OpenAI"
    assert llm.send_seed is True
    assert llm.detect_reasoning_exhaustion is False


def test_switching_model_is_a_settings_change_not_a_code_change(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "groq_model", "a-brand-new-groq-model", raising=False)
    assert get_llm_backend("groq").model == "a-brand-new-groq-model"


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown LLM_BACKEND"):
        get_llm_backend("something-else")
