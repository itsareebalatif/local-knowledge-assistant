"""Thin, None-safe wrappers around the Langfuse SDK's context-manager API.

Every call site in app/services/ wants the same thing: "create this
observation if tracing is enabled, otherwise just run the code." Rather
than repeating `if get_langfuse() is not None: ...` at every one of the
dozen or so spans/generations/retrievers across the retrieval and
generation pipelines, this module centralizes the None-check once, behind
the same convention already used for the optional reranker and MMR
dependencies elsewhere in this app: a disabled feature is represented by
None flowing through, not a different code path.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from app.observability.langfuse_client import get_langfuse


@contextmanager
def observe(name: str, as_type: str = "span", **kwargs: Any):
    """Yields the Langfuse observation (already the active OTel context, so
    anything created inside the `with` block nests under it automatically),
    or None if tracing is disabled — callers must guard `.update()` calls
    with `if obs is not None`, mirroring every other optional dependency in
    this codebase."""
    langfuse = get_langfuse()
    if langfuse is None:
        yield None
        return
    with langfuse.start_as_current_observation(as_type=as_type, name=name, **kwargs) as obs:
        yield obs


def update(obs: Any, **kwargs: Any) -> None:
    """`obs.update(**kwargs)`, but a no-op when obs is None (tracing
    disabled) — saves an `if obs is not None:` at every call site."""
    if obs is not None:
        obs.update(**kwargs)
