from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Conversation, Message
from app.models._utils import utcnow

_TITLE_MAX_LENGTH = 60


def _make_title(first_question: str) -> str:
    text = " ".join(first_question.split())
    if len(text) <= _TITLE_MAX_LENGTH:
        return text or "New chat"
    return text[:_TITLE_MAX_LENGTH].rsplit(" ", 1)[0] + "…"


def create_conversation(db: Session, user_id: int, first_question: str) -> Conversation:
    conversation = Conversation(user_id=user_id, title=_make_title(first_question))
    db.add(conversation)
    db.commit()
    return conversation


def get_conversation(db: Session, user_id: int, conversation_id: int) -> Conversation | None:
    return db.execute(
        select(Conversation).where(Conversation.conversation_id == conversation_id, Conversation.user_id == user_id)
    ).scalar_one_or_none()


def list_conversations(db: Session, user_id: int) -> list[Conversation]:
    return (
        db.execute(select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc()))
        .scalars()
        .all()
    )


def append_message(
    db: Session,
    conversation: Conversation,
    role: str,
    content: str,
    citations: list | None = None,
    grounding: dict | None = None,
) -> Message:
    message = Message(
        conversation_id=conversation.conversation_id,
        role=role,
        content=content,
        citations=citations,
        grounding=grounding,
    )
    db.add(message)
    conversation.updated_at = utcnow()
    db.commit()
    return message


def delete_conversation(db: Session, user_id: int, conversation_id: int) -> bool:
    conversation = get_conversation(db, user_id, conversation_id)
    if conversation is None:
        return False
    db.delete(conversation)
    db.commit()
    return True
