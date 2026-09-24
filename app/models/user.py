"""User model — account profile / identity. See SRS ERD (section 4.3)."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models._utils import utcnow

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.user_auth import UserAuth


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="USER")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    auth: Mapped["UserAuth"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")
    documents: Mapped[list["Document"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
