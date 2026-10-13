from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import User

DEFAULT_LOCAL_USER_EMAIL = "local@kengine"


def get_or_create_default_user(db: Session) -> int:
    user = db.execute(select(User).where(User.email == DEFAULT_LOCAL_USER_EMAIL)).scalar_one_or_none()
    if user is None:
        user = User(email=DEFAULT_LOCAL_USER_EMAIL, full_name="Local User", role="ADMIN")
        db.add(user)
        db.commit()
    return user.user_id
