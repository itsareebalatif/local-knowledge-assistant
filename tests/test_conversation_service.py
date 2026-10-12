from __future__ import annotations

from app.services.conversation_service import (
    append_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
)


def test_create_conversation_titles_itself_from_the_first_question(db_session, user):
    conversation = create_conversation(db_session, user.user_id, "What does Ollama run models on?")
    assert conversation.title == "What does Ollama run models on?"


def test_create_conversation_truncates_a_long_first_question_for_the_title(db_session, user):
    long_question = "What " + "really " * 20 + "is the answer to this very long question?"
    conversation = create_conversation(db_session, user.user_id, long_question)
    assert len(conversation.title) <= 61  # 60 chars + the trailing ellipsis
    assert conversation.title.endswith("…")


def test_get_conversation_returns_none_for_another_users_conversation(db_session, user):
    from app.models import User

    other = User(email="other-conv@example.com", full_name="Other", role="USER")
    db_session.add(other)
    db_session.commit()

    conversation = create_conversation(db_session, other.user_id, "Not yours")

    assert get_conversation(db_session, user.user_id, conversation.conversation_id) is None


def test_list_conversations_only_returns_the_given_users_conversations(db_session, user):
    from app.models import User

    other = User(email="other-list@example.com", full_name="Other", role="USER")
    db_session.add(other)
    db_session.commit()

    mine = create_conversation(db_session, user.user_id, "Mine")
    create_conversation(db_session, other.user_id, "Not mine")

    results = list_conversations(db_session, user.user_id)
    assert [c.conversation_id for c in results] == [mine.conversation_id]


def test_append_message_bumps_the_conversations_updated_at(db_session, user):
    conversation = create_conversation(db_session, user.user_id, "Hello")
    original_updated_at = conversation.updated_at

    append_message(db_session, conversation, "assistant", "An answer.")

    assert conversation.updated_at >= original_updated_at


def test_append_message_stores_citations_and_grounding_as_is(db_session, user):
    conversation = create_conversation(db_session, user.user_id, "Hello")
    citations = [{"marker": 1, "file_name": "doc.pdf"}]
    grounding = {"overall_coverage": 0.9, "unsupported_sentences": []}

    message = append_message(db_session, conversation, "assistant", "An answer.", citations=citations, grounding=grounding)

    assert message.citations == citations
    assert message.grounding == grounding


def test_delete_conversation_removes_it(db_session, user):
    conversation = create_conversation(db_session, user.user_id, "Goes away")

    deleted = delete_conversation(db_session, user.user_id, conversation.conversation_id)

    assert deleted is True
    assert get_conversation(db_session, user.user_id, conversation.conversation_id) is None


def test_delete_conversation_returns_false_for_an_unknown_id(db_session, user):
    assert delete_conversation(db_session, user.user_id, 999999) is False


def test_delete_conversation_returns_false_for_another_users_conversation(db_session, user):
    from app.models import User

    other = User(email="other-delete@example.com", full_name="Other", role="USER")
    db_session.add(other)
    db_session.commit()

    conversation = create_conversation(db_session, other.user_id, "Not yours")

    assert delete_conversation(db_session, user.user_id, conversation.conversation_id) is False
