from __future__ import annotations

import os
from functools import lru_cache

import certifi
from langfuse import Langfuse

from app.config import get_settings

# Works around missing system OpenSSL certs on some Python.org installs (CERTIFICATE_VERIFY_FAILED)
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
