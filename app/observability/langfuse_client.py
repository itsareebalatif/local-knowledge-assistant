"""Langfuse tracing client — a single, explicitly-configured instance built
from this project's own Settings, not from process environment variables.

Explicit credentials avoid the classic "Langfuse imported before .env was
loaded" ordering bug: this project's pydantic-settings Settings class reads
.env directly into typed fields and never populates os.environ, so the
SDK's own environment-variable auto-detection would otherwise find nothing
no matter when it's imported.

Optional end-to-end: if LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY aren't
set, get_langfuse() returns None and every call site treats that as
"tracing disabled" — the same None-means-off convention already used for
the optional reranker and MMR dependencies elsewhere in this app.
"""

from __future__ import annotations

import os
from functools import lru_cache

import certifi
from langfuse import Langfuse

from app.config import get_settings

# Works around a real, reproducible failure on this machine's Python.org
# install: its default OpenSSL cert path (.../etc/openssl/cert.pem) doesn't
# exist (the "Install Certificates.command" step was never run), so the
# span exporter's SSL verification fails with CERTIFICATE_VERIFY_FAILED.
# Other API calls in this app (Groq/Gemini/Cohere) go through httpx, which
# bundles its own certifi-based default and isn't affected; whatever HTTP
# client the Langfuse SDK's OTel exporter uses underneath apparently falls
# back to the broken system path instead. Setting SSL_CERT_FILE explicitly
# fixes it for any library that respects the standard OpenSSL env var,
# without touching the system Python installation.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())


@lru_cache
def get_langfuse() -> Langfuse | None:
    settings = get_settings()
    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
    )
