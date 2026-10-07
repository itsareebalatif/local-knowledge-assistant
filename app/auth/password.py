"""Password hashing — bcrypt, deliberately slow.

bcrypt is used here and nowhere else in this codebase on purpose: a
password is low-frequency (checked once at login) and needs to resist
offline brute-forcing, which is exactly what bcrypt's deliberate slowness
and per-hash salt are for. Session tokens (app/auth/tokens.py) are the
opposite case — high-frequency, already high-entropy — and use a fast hash
instead. Don't reuse one for the other.
"""

from __future__ import annotations

import bcrypt


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # Malformed stored hash (shouldn't happen outside of corrupted data) —
        # fail closed, never raise past an auth check.
        return False
