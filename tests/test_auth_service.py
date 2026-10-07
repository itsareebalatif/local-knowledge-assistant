from __future__ import annotations

import datetime

import pytest

from app.auth.auth_service import (
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
    get_user_by_token,
    login_user,
    logout_user,
    refresh_access_token,
    register_user,
)
from app.auth.password import hash_password, verify_password
from app.auth.tokens import generate_token, hash_token
from app.models._utils import utcnow


def test_hash_password_produces_a_verifiable_but_different_string():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong password", hashed)


def test_hash_password_salts_so_two_hashes_of_the_same_password_differ():
    assert hash_password("same password") != hash_password("same password")


def test_verify_password_fails_closed_on_malformed_hash():
    assert verify_password("anything", "not a real bcrypt hash") is False


def test_generate_token_is_random_and_url_safe():
    a, b = generate_token(), generate_token()
    assert a != b
    assert len(a) > 20


def test_hash_token_is_deterministic_and_fast():
    token = generate_token()
    assert hash_token(token) == hash_token(token)
    assert hash_token(token) != hash_token(generate_token())


def test_register_user_creates_user_and_auth_row_with_working_token(db_session):
    result = register_user(db_session, "new@example.com", "correct horse battery staple", "New Person")

    assert result.user.email == "new@example.com"
    assert result.user.full_name == "New Person"
    assert result.user.auth is not None
    assert result.user.auth.password_hash != "correct horse battery staple"
    assert result.access_token_expires_at > utcnow()
    assert result.refresh_token_expires_at > result.access_token_expires_at

    found = get_user_by_token(db_session, result.access_token)
    assert found is not None
    assert found.user_id == result.user.user_id


def test_register_user_rejects_duplicate_email(db_session):
    register_user(db_session, "dup@example.com", "correct horse battery staple", "First")
    with pytest.raises(EmailAlreadyRegisteredError):
        register_user(db_session, "dup@example.com", "a different password entirely", "Second")


def test_login_user_with_correct_password_issues_a_new_token(db_session):
    registered = register_user(db_session, "login@example.com", "correct horse battery staple", "User")
    logged_in = login_user(db_session, "login@example.com", "correct horse battery staple")

    assert logged_in.user.user_id == registered.user.user_id
    assert logged_in.access_token != registered.access_token  # a fresh token, not the registration one reused
    assert get_user_by_token(db_session, logged_in.access_token) is not None


def test_login_user_rejects_wrong_password(db_session):
    register_user(db_session, "wrongpw@example.com", "correct horse battery staple", "User")
    with pytest.raises(InvalidCredentialsError):
        login_user(db_session, "wrongpw@example.com", "not the right password")


def test_login_user_rejects_unknown_email(db_session):
    with pytest.raises(InvalidCredentialsError):
        login_user(db_session, "nobody@example.com", "whatever-password")


def test_login_user_rejects_inactive_account(db_session):
    registered = register_user(db_session, "inactive@example.com", "correct horse battery staple", "User")
    registered.user.is_active = False
    db_session.commit()

    with pytest.raises(InvalidCredentialsError):
        login_user(db_session, "inactive@example.com", "correct horse battery staple")


def test_login_invalidates_the_previous_token(db_session):
    registered = register_user(db_session, "rotate@example.com", "correct horse battery staple", "User")
    assert get_user_by_token(db_session, registered.access_token) is not None

    login_user(db_session, "rotate@example.com", "correct horse battery staple")

    assert get_user_by_token(db_session, registered.access_token) is None  # old token no longer resolves


def test_logout_invalidates_the_token(db_session):
    registered = register_user(db_session, "logout@example.com", "correct horse battery staple", "User")
    assert get_user_by_token(db_session, registered.access_token) is not None

    logout_user(db_session, registered.user)

    assert get_user_by_token(db_session, registered.access_token) is None


def test_get_user_by_token_returns_none_for_unknown_token(db_session):
    assert get_user_by_token(db_session, "a-token-that-was-never-issued") is None


def test_get_user_by_token_returns_none_once_the_token_has_expired(db_session):
    registered = register_user(db_session, "expiring@example.com", "correct horse battery staple", "User")
    assert get_user_by_token(db_session, registered.access_token) is not None

    # Simulate time passing without needing a real sleep or a mocked clock —
    # rewind the stored expiry straight into the past instead.
    registered.user.auth.access_token_expires_at = utcnow() - datetime.timedelta(seconds=1)
    db_session.commit()

    assert get_user_by_token(db_session, registered.access_token) is None


def test_refresh_access_token_issues_a_brand_new_working_pair(db_session):
    registered = register_user(db_session, "refresher@example.com", "correct horse battery staple", "User")

    refreshed = refresh_access_token(db_session, registered.refresh_token)

    assert refreshed.user.user_id == registered.user.user_id
    assert refreshed.access_token != registered.access_token
    assert refreshed.refresh_token != registered.refresh_token
    assert get_user_by_token(db_session, refreshed.access_token) is not None


def test_refresh_access_token_rotates_the_refresh_token_invalidating_the_old_one(db_session):
    registered = register_user(db_session, "rotate-refresh@example.com", "correct horse battery staple", "User")

    refresh_access_token(db_session, registered.refresh_token)

    with pytest.raises(InvalidRefreshTokenError):
        refresh_access_token(db_session, registered.refresh_token)  # already used once — can't be replayed


def test_refresh_access_token_rejects_an_unknown_token(db_session):
    with pytest.raises(InvalidRefreshTokenError):
        refresh_access_token(db_session, "a-refresh-token-that-was-never-issued")


def test_refresh_access_token_rejects_an_expired_token(db_session):
    registered = register_user(db_session, "expired-refresh@example.com", "correct horse battery staple", "User")
    registered.user.auth.refresh_token_expires_at = utcnow() - datetime.timedelta(seconds=1)
    db_session.commit()

    with pytest.raises(InvalidRefreshTokenError):
        refresh_access_token(db_session, registered.refresh_token)


def test_refresh_access_token_rejects_an_inactive_user(db_session):
    registered = register_user(db_session, "inactive-refresh@example.com", "correct horse battery staple", "User")
    registered.user.is_active = False
    db_session.commit()

    with pytest.raises(InvalidRefreshTokenError):
        refresh_access_token(db_session, registered.refresh_token)


def test_logout_invalidates_both_the_access_and_refresh_tokens(db_session):
    registered = register_user(db_session, "logout-both@example.com", "correct horse battery staple", "User")

    logout_user(db_session, registered.user)

    assert get_user_by_token(db_session, registered.access_token) is None
    with pytest.raises(InvalidRefreshTokenError):
        refresh_access_token(db_session, registered.refresh_token)
