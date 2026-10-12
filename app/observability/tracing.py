from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from app.observability.langfuse_client import get_langfuse


@contextmanager
def observe(name: str, as_type: str = "span", **kwargs: Any):
    langfuse = get_langfuse()
    if langfuse is None:
        yield None
        return
    with langfuse.start_as_current_observation(as_type=as_type, name=name, **kwargs) as obs:
        yield obs


def update(obs: Any, **kwargs: Any) -> None:
    if obs is not None:
        obs.update(**kwargs)
