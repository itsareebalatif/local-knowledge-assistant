
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class UserAuth(Base):
    __tablename__ = "user_auth"

    auth_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), unique=True, index=True, nullable=False
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    # Column name kept as "refresh_token_hash" (the name it was created
    # with) even though it now stores the access token's hash — renaming a
    # live SQLite column needs a migration for no real benefit. The actual
    # refresh token (below) needed a genuinely new column, hence the
    # deliberately different DB name "refresh_hash" for it.
    access_token_hash: Mapped[str | None] = mapped_column("refresh_token_hash", String(255), nullable=True)
    access_token_expires_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    refresh_token_hash: Mapped[str | None] = mapped_column("refresh_hash", String(255), nullable=True)
    refresh_token_expires_at: Mapped[datetime.datetime | None] = mapped_column(
        "refresh_expires_at", DateTime(timezone=True), nullable=True
    )
    last_login_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="auth")
