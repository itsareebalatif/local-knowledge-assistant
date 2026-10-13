from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.schemas import ConversationDetailResponse, ConversationListResponse, ConversationOut, MessageOut
from app.dependencies import get_current_user, get_db
from app.models import User
from app.services.conversation_service import delete_conversation, get_conversation, list_conversations

router = APIRouter()


@router.get(
    "/conversations", response_model=ConversationListResponse, tags=["conversations"], summary="List your past chats"
)
def list_conversations_route(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> ConversationListResponse:
    conversations = list_conversations(db, current_user.user_id)
    return ConversationListResponse(
        conversations=[
            ConversationOut(
                conversation_id=c.conversation_id, title=c.title, created_at=c.created_at, updated_at=c.updated_at
            )
            for c in conversations
        ]
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    tags=["conversations"],
    summary="Get one chat's full message history",
)
def get_conversation_route(
    conversation_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> ConversationDetailResponse:
    conversation = get_conversation(db, current_user.user_id, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    return ConversationDetailResponse(
        conversation_id=conversation.conversation_id,
        title=conversation.title,
        messages=[
            MessageOut(
                message_id=m.message_id,
                role=m.role,
                content=m.content,
                citations=m.citations,
                grounding=m.grounding,
                created_at=m.created_at,
            )
            for m in conversation.messages
        ],
    )


@router.delete(
    "/conversations/{conversation_id}", tags=["conversations"], summary="Delete a chat and all its messages"
)
def delete_conversation_route(
    conversation_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict:
    deleted = delete_conversation(db, current_user.user_id, conversation_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return {"status": "deleted", "conversation_id": conversation_id}
