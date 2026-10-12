from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.schemas import (
    AuthResponse,
    ForgotPasswordRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    ResetPasswordRequest,
)
from app.auth.auth_service import (
    AuthResult,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
    InvalidResetTokenError,
    login_user,
    logout_user,
    refresh_access_token,
    register_user,
    request_password_reset,
    reset_password,
)
from app.config import get_settings
from app.dependencies import get_current_user, get_db, get_email_sender
from app.models import User
from app.notifications.resend_client import EmailError, EmailSender
from app.notifications.templates import password_reset_email, welcome_email

logger = logging.getLogger(__name__)

router = APIRouter()


def _to_response(result: AuthResult) -> AuthResponse:
    return AuthResponse(
        access_token=result.access_token,
        access_token_expires_at=result.access_token_expires_at,
        refresh_token=result.refresh_token,
        refresh_token_expires_at=result.refresh_token_expires_at,
        user_id=result.user.user_id,
        email=result.user.email,
    )


@router.post("/auth/register", response_model=AuthResponse, tags=["auth"], summary="Create an account")
async def register(
    payload: RegisterRequest, db: Session = Depends(get_db), email_sender: EmailSender = Depends(get_email_sender)
) -> AuthResponse:
    try:
        result = register_user(db, payload.email, payload.password, payload.full_name)
    except EmailAlreadyRegisteredError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    try:
        subject, html = welcome_email(result.user.full_name)
        await email_sender.send(result.user.email, subject, html)
    except EmailError as exc:
        logger.warning("Welcome email failed for %r: %s", result.user.email, exc)

    return _to_response(result)


@router.post("/auth/login", response_model=AuthResponse, tags=["auth"], summary="Log in, get an access/refresh token pair")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    try:
        result = login_user(db, payload.email, payload.password)
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    return _to_response(result)


@router.post(
    "/auth/refresh",
    response_model=AuthResponse,
    tags=["auth"],
    summary="Trade a refresh token for a new access/refresh token pair",
)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)) -> AuthResponse:
    try:
        result = refresh_access_token(db, payload.refresh_token)
    except InvalidRefreshTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    return _to_response(result)


@router.post("/auth/logout", tags=["auth"], summary="Invalidate the current access and refresh tokens")
def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    logout_user(db, current_user)
    return {"status": "logged_out"}


@router.post(
    "/auth/forgot-password",
    tags=["auth"],
    summary="Request a password reset email",
)
async def forgot_password(
    payload: ForgotPasswordRequest, db: Session = Depends(get_db), email_sender: EmailSender = Depends(get_email_sender)
) -> dict:
    result = request_password_reset(db, payload.email)
    if result is not None:
        user, token = result
        reset_url = f"{get_settings().frontend_base_url}/reset-password?token={token}"
        try:
            subject, html = password_reset_email(user.full_name, reset_url)
            await email_sender.send(user.email, subject, html)
        except EmailError as exc:
            logger.warning("Password reset email failed for %r: %s", user.email, exc)

    return {
        "status": "ok",
        "message": (
            "If that email has an account, a reset link is on its way. "
            "New here? You can register instead."
        ),
    }


@router.post(
    "/auth/reset-password",
    tags=["auth"],
    summary="Set a new password using a reset token from the forgot-password email",
)
def reset_password_route(payload: ResetPasswordRequest, db: Session = Depends(get_db)) -> dict:
    try:
        reset_password(db, payload.token, payload.new_password)
    except InvalidResetTokenError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"status": "ok", "message": "Password has been reset. Please log in again."}
