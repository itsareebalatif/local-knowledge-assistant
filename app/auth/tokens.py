"""Opaque random strings, stored hashed — the raw material for both the
access token and the refresh token (app/auth/auth_service.py decides the
expiry and column for each; this module just generates and hashes).

Not JWT: a JWT needs a signing secret to manage and gives you a stateless
token you can't revoke without a blocklist anyway. An opaque token plus a
`UserAuth` column does the same job more simply for a tool this size, and
revocation (logout, or refresh-token rotation) is just "overwrite the
column" rather than needing a blocklist.

The token itself is hashed with SHA-256 before being stored or compared —
fast on purpose, unlike password hashing. A 32-byte `secrets.token_urlsafe`
value already has enough entropy that a slow hash buys nothing, and this
comparison has to happen on every authenticated request (for the access
token) or every refresh call (for the refresh token), not just at login.
"""

from __future__ import annotations

import hashlib
import secrets


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
