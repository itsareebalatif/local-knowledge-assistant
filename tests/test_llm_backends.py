from __future__ import annotations

import json

import httpx
import pytest

from app.llm import get_llm_backend
from app.llm.base import LLMError
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


async def test_ollama_raises_clear_error_when_unreachable():
    # No mock transport, nothing listening on this port: a real connection
    # failure, proving the error path works without needing a live Ollama.
    llm = OllamaLLM(base_url="http://127.0.0.1:1", timeout=2.0)
    with pytest.raises(LLMError, match="Is it running"):
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


async def test_groq_without_api_key_raises_immediately():
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


def test_factory_picks_ollama_for_local():
    assert isinstance(get_llm_backend("local"), OllamaLLM)


def test_factory_picks_groq():
    assert isinstance(get_llm_backend("groq"), GroqLLM)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown LLM_BACKEND"):
        get_llm_backend("something-else")
