"""Default local user — this is a personal, single-user-by-default tool.

Both the CLI and the dashboard need a user_id to attach ingested documents
to. Rather than making every caller invent one that means nothing yet, this
creates (or finds) one local user on first use. Multi-user support would
replace this with real auth later; nothing above the DB layer assumes it
can't be.
"""

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
