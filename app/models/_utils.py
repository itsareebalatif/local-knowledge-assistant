"""Small shared helpers for the model modules. Not a model itself."""

from __future__ import annotations

import datetime


def utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)
