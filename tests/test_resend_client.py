from __future__ import annotations

import json

import httpx
import pytest

from app.notifications.resend_client import EmailError, ResendEmailSender


async def test_send_posts_the_expected_payload_and_headers():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "fake-email-id"})

    transport = httpx.MockTransport(handler)
    sender = ResendEmailSender(
        api_key="test-key", from_address="notifications@example.com", transport=transport
    )

    await sender.send("someone@example.com", "Subject here", "<p>body</p>")

    assert captured["url"] == "https://api.resend.com/emails"
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["payload"] == {
        "from": "notifications@example.com",
        "to": ["someone@example.com"],
        "subject": "Subject here",
        "html": "<p>body</p>",
    }


async def test_send_without_api_key_raises_immediately(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "resend_api_key", None, raising=False)
    sender = ResendEmailSender(api_key=None, from_address="notifications@example.com")
    with pytest.raises(EmailError, match="RESEND_API_KEY"):
        await sender.send("someone@example.com", "Subject", "<p>body</p>")


async def test_send_without_from_address_raises_immediately(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "email_from_address", None, raising=False)
    sender = ResendEmailSender(api_key="test-key", from_address=None)
    with pytest.raises(EmailError, match="EMAIL_FROM_ADDRESS"):
        await sender.send("someone@example.com", "Subject", "<p>body</p>")


async def test_send_raises_on_bad_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text="domain not verified")

    transport = httpx.MockTransport(handler)
    sender = ResendEmailSender(api_key="test-key", from_address="notifications@example.com", transport=transport)
    with pytest.raises(EmailError, match="422"):
        await sender.send("someone@example.com", "Subject", "<p>body</p>")


async def test_send_raises_clear_error_when_unreachable():
    sender = ResendEmailSender(
        api_key="test-key",
        from_address="notifications@example.com",
        base_url="http://127.0.0.1:1",
        timeout=2.0,
    )
    with pytest.raises(EmailError, match="Could not reach Resend"):
        await sender.send("someone@example.com", "Subject", "<p>body</p>")


async def test_send_raises_clear_error_on_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated slow response")

    transport = httpx.MockTransport(handler)
    sender = ResendEmailSender(api_key="test-key", from_address="notifications@example.com", transport=transport)
    with pytest.raises(EmailError, match="did not respond"):
        await sender.send("someone@example.com", "Subject", "<p>body</p>")
