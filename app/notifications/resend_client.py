from __future__ import annotations

from typing import Protocol

import httpx

from app.config import get_settings


class EmailError(Exception):
    pass


class EmailSender(Protocol):
    async def send(self, to: str, subject: str, html: str) -> None: ...


class ResendEmailSender:
    def __init__(
        self,
        api_key: str | None = None,
        from_address: str | None = None,
        base_url: str = "https://api.resend.com",
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.resend_api_key
        self.from_address = from_address or settings.email_from_address
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport  # test hook; None uses a real network transport

    async def send(self, to: str, subject: str, html: str) -> None:
        if not self.api_key:
            raise EmailError("RESEND_API_KEY is not set — add it to .env to enable email sending.")
        if not self.from_address:
            raise EmailError("EMAIL_FROM_ADDRESS is not set — add it to .env to enable email sending.")

        payload = {"from": self.from_address, "to": [to], "subject": subject, "html": html}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                response = await client.post(f"{self.base_url}/emails", json=payload, headers=headers)
                if response.status_code >= 400:
                    raise EmailError(f"Resend returned {response.status_code}: {response.text}")
        except httpx.ConnectError as exc:
            raise EmailError(f"Could not reach Resend at {self.base_url}.") from exc
        except httpx.TimeoutException as exc:
            raise EmailError(f"Resend did not respond within {self.timeout}s.") from exc
