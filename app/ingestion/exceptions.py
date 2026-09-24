"""Ingestion-specific exceptions.

Parsers raise these; app.ingestion.pipeline is the only place that catches
them (turning a raised exception into a logged, non-fatal IngestResult).
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all ingestion-time failures."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when a file extension has no registered parser."""


class ParsingError(IngestionError):
    """Raised when a parser cannot make sense of a file's bytes (corrupt,
    password-protected, truncated, etc.)."""
