from __future__ import annotations

import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    access_token_expires_at: datetime.datetime
    refresh_token: str
    refresh_token_expires_at: datetime.datetime
    user_id: int
    email: str


class RefreshRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8)


class QueryRequest(BaseModel):
    # No user_id: the caller is resolved from the auth header, not client input.
    query: str
    conversation_id: int | None = None


class IngestResponse(BaseModel):
    status: str  # "ingested" | "duplicate_file" | "unsupported" | "parse_failed"
    document_id: int | None = None
    total_chunks: int = 0
    new_chunks: int = 0
    reused_chunks: int = 0
    warnings: list[str] = []
    error: str | None = None


class CitationOut(BaseModel):
    marker: int
    chunk_id: int
    doc_id: int
    file_name: str
    source_url: str
    snippet: str


class GroundingOut(BaseModel):
    overall_coverage: float | None
    unsupported_sentences: list[str]


class QueryResponse(BaseModel):
    status: str  # "grounded" | "refused" | "error"
    reason: str
    answer: str = ""
    citations: list[CitationOut] = []
    grounding: GroundingOut | None = None
    conversation_id: int | None = None


class GraphNeighbor(BaseModel):
    name: str
    type: str
    weight: float


class GraphNeighborsResponse(BaseModel):
    entity: str
    neighbors: list[GraphNeighbor]


class ChunkOut(BaseModel):
    chunk_id: int
    doc_id: int
    file_name: str
    content: str


class DocumentOut(BaseModel):
    doc_id: int
    file_name: str
    file_type: str
    chunk_count: int
    created_at: datetime.datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]
    total_documents: int
    total_chunks: int


class ConversationOut(BaseModel):
    conversation_id: int
    title: str
    created_at: datetime.datetime
    updated_at: datetime.datetime


class ConversationListResponse(BaseModel):
    conversations: list[ConversationOut]


class MessageOut(BaseModel):
    message_id: int
    role: str
    content: str
    citations: list[dict] | None = None
    grounding: dict | None = None
    created_at: datetime.datetime


class ConversationDetailResponse(BaseModel):
    conversation_id: int
    title: str
    messages: list[MessageOut]
