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


class InvalidResetTokenError(Exception):
    pass


@dataclass
class AuthResult:
    user: User
    access_token: str
    access_token_expires_at: datetime.datetime
    refresh_token: str
    refresh_token_expires_at: datetime.datetime


def _as_aware_utc(value: datetime.datetime) -> datetime.datetime:
    # SQLite doesn't persist tzinfo, so a stored UTC value comes back naive.
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
    db.flush()  # need user.user_id for the UserAuth row

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
        # Same error for both cases to avoid leaking which emails are registered.
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


def request_password_reset(db: Session, email: str) -> tuple[User, str] | None:
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None or user.auth is None or not user.is_active:
        return None

    token = generate_token()
    expires_at = utcnow() + datetime.timedelta(minutes=get_settings().password_reset_token_expire_minutes)
    user.auth.reset_token_hash = hash_token(token)
    user.auth.reset_token_expires_at = expires_at
    db.commit()
    return user, token


def reset_password(db: Session, token: str, new_password: str) -> User:
    token_hash = hash_token(token)
    auth = db.execute(select(UserAuth).where(UserAuth.reset_token_hash == token_hash)).scalar_one_or_none()
    if auth is None:
        raise InvalidResetTokenError("Invalid or already-used reset token")
    if auth.reset_token_expires_at is None or _as_aware_utc(auth.reset_token_expires_at) < utcnow():
        raise InvalidResetTokenError("Reset token has expired")
    if not auth.user.is_active:
        raise InvalidResetTokenError("Invalid or already-used reset token")

    auth.password_hash = hash_password(new_password)
    auth.reset_token_hash = None
    auth.reset_token_expires_at = None
    # A password reset invalidates existing sessions too — assume compromise.
    auth.access_token_hash = None
    auth.access_token_expires_at = None
    auth.refresh_token_hash = None
    auth.refresh_token_expires_at = None
    db.commit()
    return auth.user


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
        return None
    return auth.user
