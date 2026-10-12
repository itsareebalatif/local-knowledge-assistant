from __future__ import annotations

import datetime

from fastapi.testclient import TestClient

from app.dependencies import get_db, get_email_sender, get_graph_builder
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.main import app
from app.models import User
from app.models._utils import utcnow
from tests.conftest import FakeEmailSender


def _override_db(db_session):
    def _get_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_graph_builder] = lambda: CooccurrenceGraphBuilder()
    app.dependency_overrides[get_email_sender] = lambda: FakeEmailSender()


def test_register_returns_a_usable_token(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/auth/register",
            json={"email": "new@example.com", "password": "correct horse battery staple", "full_name": "New User"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["email"] == "new@example.com"
        assert body["access_token"]

        me_check = client.get("/api/graph", headers={"Authorization": f"Bearer {body['access_token']}"})
        assert me_check.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_register_rejects_duplicate_email(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        payload = {"email": "dup@example.com", "password": "correct horse battery staple", "full_name": "Dup"}
        first = client.post("/api/auth/register", json=payload)
        assert first.status_code == 200

        second = client.post("/api/auth/register", json=payload)
        assert second.status_code == 409
    finally:
        app.dependency_overrides.clear()


def test_register_rejects_short_password(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/auth/register", json={"email": "short@example.com", "password": "123", "full_name": "Short"}
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_login_with_correct_password_succeeds(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        client.post(
            "/api/auth/register",
            json={"email": "login@example.com", "password": "correct horse battery staple", "full_name": "Login"},
        )
        response = client.post(
            "/api/auth/login", json={"email": "login@example.com", "password": "correct horse battery staple"}
        )
        assert response.status_code == 200
        assert response.json()["access_token"]
    finally:
        app.dependency_overrides.clear()


def test_login_with_wrong_password_is_rejected(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        client.post(
            "/api/auth/register",
            json={"email": "wrongpw@example.com", "password": "correct horse battery staple", "full_name": "User"},
        )
        response = client.post("/api/auth/login", json={"email": "wrongpw@example.com", "password": "not the password"})
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_login_with_unknown_email_is_rejected_same_as_wrong_password(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        response = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "whatever12345"})
        assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_login_issues_a_new_token_invalidating_the_old_one(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        payload = {"email": "rotate@example.com", "password": "correct horse battery staple", "full_name": "Rotate"}
        first = client.post("/api/auth/register", json=payload).json()
        second = client.post("/api/auth/login", json={"email": payload["email"], "password": payload["password"]}).json()

        assert first["access_token"] != second["access_token"]
        old_token_check = client.get("/api/graph", headers={"Authorization": f"Bearer {first['access_token']}"})
        assert old_token_check.status_code == 401  # only one active session per user
        new_token_check = client.get("/api/graph", headers={"Authorization": f"Bearer {second['access_token']}"})
        assert new_token_check.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_logout_invalidates_the_token(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        reg = client.post(
            "/api/auth/register",
            json={"email": "logout@example.com", "password": "correct horse battery staple", "full_name": "Logout"},
        ).json()
        headers = {"Authorization": f"Bearer {reg['access_token']}"}

        logout_response = client.post("/api/auth/logout", headers=headers)
        assert logout_response.status_code == 200

        after_logout = client.get("/api/graph", headers=headers)
        assert after_logout.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_protected_endpoint_rejects_an_expired_token(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        reg = client.post(
            "/api/auth/register",
            json={"email": "expiring@example.com", "password": "correct horse battery staple", "full_name": "Expiring"},
        ).json()
        headers = {"Authorization": f"Bearer {reg['access_token']}"}
        assert client.get("/api/graph", headers=headers).status_code == 200

        user = db_session.query(User).filter_by(email="expiring@example.com").one()
        user.auth.access_token_expires_at = utcnow() - datetime.timedelta(seconds=1)
        db_session.commit()

        assert client.get("/api/graph", headers=headers).status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_refresh_endpoint_issues_a_new_working_access_token(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        reg = client.post(
            "/api/auth/register",
            json={"email": "refresher@example.com", "password": "correct horse battery staple", "full_name": "Refresher"},
        ).json()

        refreshed = client.post("/api/auth/refresh", json={"refresh_token": reg["refresh_token"]})
        assert refreshed.status_code == 200
        body = refreshed.json()
        assert body["access_token"] != reg["access_token"]
        assert body["refresh_token"] != reg["refresh_token"]

        new_headers = {"Authorization": f"Bearer {body['access_token']}"}
        assert client.get("/api/graph", headers=new_headers).status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_refresh_endpoint_lets_an_expired_access_token_get_a_new_one_without_a_full_login(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        reg = client.post(
            "/api/auth/register",
            json={"email": "silent-renew@example.com", "password": "correct horse battery staple", "full_name": "Renew"},
        ).json()

        user = db_session.query(User).filter_by(email="silent-renew@example.com").one()
        user.auth.access_token_expires_at = utcnow() - datetime.timedelta(seconds=1)
        db_session.commit()

        old_headers = {"Authorization": f"Bearer {reg['access_token']}"}
        assert client.get("/api/graph", headers=old_headers).status_code == 401

        refreshed = client.post("/api/auth/refresh", json={"refresh_token": reg["refresh_token"]}).json()
        new_headers = {"Authorization": f"Bearer {refreshed['access_token']}"}
        assert client.get("/api/graph", headers=new_headers).status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_refresh_endpoint_rejects_an_unknown_or_already_used_token(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        reg = client.post(
            "/api/auth/register",
            json={"email": "one-shot@example.com", "password": "correct horse battery staple", "full_name": "One Shot"},
        ).json()

        first = client.post("/api/auth/refresh", json={"refresh_token": reg["refresh_token"]})
        assert first.status_code == 200

        replay = client.post("/api/auth/refresh", json={"refresh_token": reg["refresh_token"]})
        assert replay.status_code == 401

        made_up = client.post("/api/auth/refresh", json={"refresh_token": "totally-made-up"})
        assert made_up.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_logout_also_invalidates_the_refresh_token(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        reg = client.post(
            "/api/auth/register",
            json={"email": "logout-refresh@example.com", "password": "correct horse battery staple", "full_name": "LR"},
        ).json()
        headers = {"Authorization": f"Bearer {reg['access_token']}"}

        client.post("/api/auth/logout", headers=headers)

        after_logout = client.post("/api/auth/refresh", json={"refresh_token": reg["refresh_token"]})
        assert after_logout.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_protected_endpoint_rejects_missing_or_malformed_header(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        assert client.get("/api/graph").status_code == 401
        assert client.get("/api/graph", headers={"Authorization": "not-bearer-format"}).status_code == 401
        assert client.get("/api/graph", headers={"Authorization": "Bearer totally-made-up-token"}).status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_register_sends_a_welcome_email(db_session):
    sender = FakeEmailSender()
    _override_db(db_session)
    app.dependency_overrides[get_email_sender] = lambda: sender
    try:
        client = TestClient(app)
        client.post(
            "/api/auth/register",
            json={"email": "welcome@example.com", "password": "correct horse battery staple", "full_name": "Welcome"},
        )
        assert len(sender.calls) == 1
        to, subject, _html = sender.calls[0]
        assert to == "welcome@example.com"
        assert "Welcome" in subject or "Personal Knowledge Engine" in subject
    finally:
        app.dependency_overrides.clear()


def test_forgot_password_sends_a_reset_email_for_a_real_account(db_session):
    sender = FakeEmailSender()
    _override_db(db_session)
    app.dependency_overrides[get_email_sender] = lambda: sender
    try:
        client = TestClient(app)
        client.post(
            "/api/auth/register",
            json={"email": "forgot@example.com", "password": "correct horse battery staple", "full_name": "Forgot"},
        )
        sender.calls.clear()

        response = client.post("/api/auth/forgot-password", json={"email": "forgot@example.com"})
        assert response.status_code == 200
        assert len(sender.calls) == 1
        to, subject, html = sender.calls[0]
        assert to == "forgot@example.com"
        assert "reset-password?token=" in html
    finally:
        app.dependency_overrides.clear()


def test_forgot_password_gives_the_same_response_for_an_unknown_email(db_session):
    # Anti-enumeration: response must look the same whether or not the email exists.
    sender = FakeEmailSender()
    _override_db(db_session)
    app.dependency_overrides[get_email_sender] = lambda: sender
    try:
        client = TestClient(app)
        response = client.post("/api/auth/forgot-password", json={"email": "nobody-here@example.com"})
        assert response.status_code == 200
        assert sender.calls == []
    finally:
        app.dependency_overrides.clear()


def test_reset_password_with_a_valid_token_lets_the_user_log_in_with_the_new_password(db_session):
    sender = FakeEmailSender()
    _override_db(db_session)
    app.dependency_overrides[get_email_sender] = lambda: sender
    try:
        client = TestClient(app)
        client.post(
            "/api/auth/register",
            json={"email": "reset@example.com", "password": "old password here", "full_name": "Reset"},
        )
        client.post("/api/auth/forgot-password", json={"email": "reset@example.com"})
        _, _, html = sender.calls[-1]
        token = html.split("token=")[1].split('"')[0]

        reset_response = client.post(
            "/api/auth/reset-password", json={"token": token, "new_password": "brand new password"}
        )
        assert reset_response.status_code == 200

        old_login = client.post("/api/auth/login", json={"email": "reset@example.com", "password": "old password here"})
        assert old_login.status_code == 401

        new_login = client.post(
            "/api/auth/login", json={"email": "reset@example.com", "password": "brand new password"}
        )
        assert new_login.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_reset_password_rejects_an_unknown_token(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/auth/reset-password", json={"token": "totally-made-up", "new_password": "brand new password"}
        )
        assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_reset_password_token_is_single_use(db_session):
    sender = FakeEmailSender()
    _override_db(db_session)
    app.dependency_overrides[get_email_sender] = lambda: sender
    try:
        client = TestClient(app)
        client.post(
            "/api/auth/register",
            json={"email": "singleuse@example.com", "password": "old password here", "full_name": "Single"},
        )
        client.post("/api/auth/forgot-password", json={"email": "singleuse@example.com"})
        _, _, html = sender.calls[-1]
        token = html.split("token=")[1].split('"')[0]

        first = client.post("/api/auth/reset-password", json={"token": token, "new_password": "first new password"})
        assert first.status_code == 200

        replay = client.post("/api/auth/reset-password", json={"token": token, "new_password": "second new password"})
        assert replay.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_reset_password_invalidates_existing_sessions(db_session):
    sender = FakeEmailSender()
    _override_db(db_session)
    app.dependency_overrides[get_email_sender] = lambda: sender
    try:
        client = TestClient(app)
        reg = client.post(
            "/api/auth/register",
            json={"email": "forcelogout@example.com", "password": "old password here", "full_name": "Force"},
        ).json()
        old_access_token = reg["access_token"]

        client.post("/api/auth/forgot-password", json={"email": "forcelogout@example.com"})
        _, _, html = sender.calls[-1]
        token = html.split("token=")[1].split('"')[0]
        client.post("/api/auth/reset-password", json={"token": token, "new_password": "brand new password"})

        still_works = client.get("/api/graph", headers={"Authorization": f"Bearer {old_access_token}"})
        assert still_works.status_code == 401
    finally:
        app.dependency_overrides.clear()
