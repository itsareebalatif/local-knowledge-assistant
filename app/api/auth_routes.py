"""Registration, login, logout, and token refresh.

Every other route in this app that touches a specific user's data depends
on `get_current_user` (app/dependencies.py), which only accepts an access
token issued here — there's no other way into the system. The refresh
token issued alongside it is never accepted by `get_current_user`; its only
job is proving itself to POST /auth/refresh to mint a new pair.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.schemas import AuthResponse, LoginRequest, RefreshRequest, RegisterRequest
from app.auth.auth_service import (
    AuthResult,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
    login_user,
    logout_user,
    refresh_access_token,
    register_user,
)
from app.dependencies import get_current_user, get_db
from app.models import User

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
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> AuthResponse:
    try:
        result = register_user(db, payload.email, payload.password, payload.full_name)
    except EmailAlreadyRegisteredError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
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
