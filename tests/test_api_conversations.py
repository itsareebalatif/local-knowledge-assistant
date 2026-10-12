from __future__ import annotations

from fastapi.testclient import TestClient

from app.dependencies import get_db
from app.main import app


def _override_db(db_session):
    def _get_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_db


def _create_conversation(db_session, owner, title="First question"):
    from app.services.conversation_service import append_message, create_conversation

    conversation = create_conversation(db_session, owner.user_id, title)
    append_message(db_session, conversation, "user", title)
    append_message(db_session, conversation, "assistant", "An answer.", citations=[{"marker": 1}])
    return conversation


def test_list_conversations_returns_only_the_current_users_conversations(
    db_session, user, registered_user, auth_headers
):
    _override_db(db_session)
    try:
        _create_conversation(db_session, registered_user.user, "My question")
        _create_conversation(db_session, user, "Someone else's question")  # different owner

        client = TestClient(app)
        response = client.get("/api/conversations", headers=auth_headers)
        assert response.status_code == 200
        titles = [c["title"] for c in response.json()["conversations"]]
        assert titles == ["My question"]
    finally:
        app.dependency_overrides.clear()


def test_list_conversations_orders_most_recently_updated_first(db_session, registered_user, auth_headers):
    _override_db(db_session)
    try:
        from app.services.conversation_service import append_message

        older = _create_conversation(db_session, registered_user.user, "Older chat")
        newer = _create_conversation(db_session, registered_user.user, "Newer chat")
        append_message(db_session, older, "user", "a follow-up, bumping it back to the top")

        client = TestClient(app)
        response = client.get("/api/conversations", headers=auth_headers)
        titles = [c["title"] for c in response.json()["conversations"]]
        assert titles == ["Older chat", "Newer chat"]
    finally:
        app.dependency_overrides.clear()


def test_get_conversation_returns_its_full_message_history(db_session, registered_user, auth_headers):
    _override_db(db_session)
    try:
        conversation = _create_conversation(db_session, registered_user.user, "My question")

        client = TestClient(app)
        response = client.get(f"/api/conversations/{conversation.conversation_id}", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["title"] == "My question"
        assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
        assert body["messages"][1]["citations"] == [{"marker": 1}]
    finally:
        app.dependency_overrides.clear()


def test_get_conversation_404s_for_another_users_conversation(db_session, user, registered_user, auth_headers):
    _override_db(db_session)
    try:
        conversation = _create_conversation(db_session, user, "Not yours")  # `user`, not `registered_user`

        client = TestClient(app)
        response = client.get(f"/api/conversations/{conversation.conversation_id}", headers=auth_headers)
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_get_conversation_404s_for_an_unknown_id(db_session, auth_headers):
    _override_db(db_session)
    try:
        client = TestClient(app)
        response = client.get("/api/conversations/999999", headers=auth_headers)
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_delete_conversation_removes_it_and_its_messages(db_session, registered_user, auth_headers):
    _override_db(db_session)
    try:
        conversation = _create_conversation(db_session, registered_user.user, "Goes away")
        conversation_id = conversation.conversation_id

        client = TestClient(app)
        response = client.delete(f"/api/conversations/{conversation_id}", headers=auth_headers)
        assert response.status_code == 200

        after = client.get(f"/api/conversations/{conversation_id}", headers=auth_headers)
        assert after.status_code == 404

        from app.models import Message

        assert db_session.query(Message).filter_by(conversation_id=conversation_id).all() == []
    finally:
        app.dependency_overrides.clear()


def test_delete_conversation_404s_for_another_users_conversation(db_session, user, registered_user, auth_headers):
    _override_db(db_session)
    try:
        conversation = _create_conversation(db_session, user, "Not yours")

        client = TestClient(app)
        response = client.delete(f"/api/conversations/{conversation.conversation_id}", headers=auth_headers)
        assert response.status_code == 404

        from app.models import Conversation

        assert db_session.get(Conversation, conversation.conversation_id) is not None  # untouched
    finally:
        app.dependency_overrides.clear()


def test_conversation_endpoints_require_authentication(db_session):
    _override_db(db_session)
    try:
        client = TestClient(app)
        assert client.get("/api/conversations").status_code == 401
        assert client.get("/api/conversations/1").status_code == 401
        assert client.delete("/api/conversations/1").status_code == 401
    finally:
        app.dependency_overrides.clear()
