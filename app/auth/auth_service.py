"""Registration, login, logout, and token refresh — the actual auth logic,
independent of the HTTP layer (app/api/auth_routes.py is a thin wrapper
over this, same pattern as every other service in app/services/).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.password import hash_password, verify_password
from app.auth.tokens import generate_token, hash_token
from app.config import get_settings
from app.models import User, UserAuth
from app.models._utils import utcnow


class EmailAlreadyRegisteredError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


class InvalidRefreshTokenError(Exception):
    pass


@dataclass
class AuthResult:
    user: User
    access_token: str  # raw tokens — shown to the caller exactly once, never stored raw
    access_token_expires_at: datetime.datetime
    refresh_token: str
    refresh_token_expires_at: datetime.datetime


def _as_aware_utc(value: datetime.datetime) -> datetime.datetime:
    # SQLite doesn't persist tzinfo — a value round-tripped through the DB
    # comes back naive even though it was stored as UTC-aware, so it has to
    # be re-labeled before comparing against utcnow().
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.timezone.utc)
    return value


def _issue_access_token(auth: UserAuth) -> tuple[str, datetime.datetime]:
    token = generate_token()
    expires_at = utcnow() + datetime.timedelta(minutes=get_settings().access_token_expire_minutes)
    auth.access_token_hash = hash_token(token)
    auth.access_token_expires_at = expires_at
    return token, expires_at


def _issue_refresh_token(auth: UserAuth) -> tuple[str, datetime.datetime]:
    token = generate_token()
    expires_at = utcnow() + datetime.timedelta(days=get_settings().refresh_token_expire_days)
    auth.refresh_token_hash = hash_token(token)
    auth.refresh_token_expires_at = expires_at
    return token, expires_at


def _issue_token_pair(auth: UserAuth) -> tuple[str, datetime.datetime, str, datetime.datetime]:
    access_token, access_expires_at = _issue_access_token(auth)
    refresh_token, refresh_expires_at = _issue_refresh_token(auth)
    return access_token, access_expires_at, refresh_token, refresh_expires_at


def register_user(db: Session, email: str, password: str, full_name: str) -> AuthResult:
    existing = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if existing is not None:
        raise EmailAlreadyRegisteredError(f"{email} is already registered")

    user = User(email=email, full_name=full_name, role="USER")
    db.add(user)
    db.flush()  # need user.user_id for the UserAuth row below

    auth = UserAuth(user_id=user.user_id, password_hash=hash_password(password))
    db.add(auth)
    db.flush()
    access_token, access_expires_at, refresh_token, refresh_expires_at = _issue_token_pair(auth)
    db.commit()

    return AuthResult(
        user=user,
        access_token=access_token,
        access_token_expires_at=access_expires_at,
        refresh_token=refresh_token,
        refresh_token_expires_at=refresh_expires_at,
    )


def login_user(db: Session, email: str, password: str) -> AuthResult:
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None or user.auth is None or not verify_password(password, user.auth.password_hash):
        # Deliberately the same error for "no such user" and "wrong password" —
        # distinguishing them lets an attacker enumerate registered emails.
        raise InvalidCredentialsError("Invalid email or password")
    if not user.is_active:
        raise InvalidCredentialsError("Invalid email or password")

    access_token, access_expires_at, refresh_token, refresh_expires_at = _issue_token_pair(user.auth)
    user.auth.last_login_at = utcnow()
    db.commit()

    return AuthResult(
        user=user,
        access_token=access_token,
        access_token_expires_at=access_expires_at,
        refresh_token=refresh_token,
        refresh_token_expires_at=refresh_expires_at,
    )


def refresh_access_token(db: Session, refresh_token: str) -> AuthResult:
    """Trades a still-valid refresh token for a brand-new token pair. The
    refresh token itself is rotated (the old one stops working the moment
    this call succeeds) — if someone else ever gets hold of it and uses it
    first, the legitimate client's own next refresh attempt will fail here
    (InvalidRefreshTokenError), which is the signal that it was stolen."""
    token_hash = hash_token(refresh_token)
    auth = db.execute(select(UserAuth).where(UserAuth.refresh_token_hash == token_hash)).scalar_one_or_none()
    if auth is None:
        raise InvalidRefreshTokenError("Invalid refresh token")
    if auth.refresh_token_expires_at is None or _as_aware_utc(auth.refresh_token_expires_at) < utcnow():
        raise InvalidRefreshTokenError("Refresh token has expired")
    if not auth.user.is_active:
        raise InvalidRefreshTokenError("Invalid refresh token")

    access_token, access_expires_at, new_refresh_token, refresh_expires_at = _issue_token_pair(auth)
    db.commit()

    return AuthResult(
        user=auth.user,
        access_token=access_token,
        access_token_expires_at=access_expires_at,
        refresh_token=new_refresh_token,
        refresh_token_expires_at=refresh_expires_at,
    )


def logout_user(db: Session, user: User) -> None:
    if user.auth is not None:
        user.auth.access_token_hash = None
        user.auth.access_token_expires_at = None
        user.auth.refresh_token_hash = None
        user.auth.refresh_token_expires_at = None
        db.commit()


def get_user_by_token(db: Session, token: str) -> User | None:
    token_hash = hash_token(token)
    auth = db.execute(select(UserAuth).where(UserAuth.access_token_hash == token_hash)).scalar_one_or_none()
    if auth is None:
        return None
    expires_at = auth.access_token_expires_at
    if expires_at is not None and _as_aware_utc(expires_at) < utcnow():
        return None  # expired — the client is expected to call /api/auth/refresh, not treat this as fatal
    return auth.user
