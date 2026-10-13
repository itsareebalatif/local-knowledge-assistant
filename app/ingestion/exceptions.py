from __future__ import annotations


class IngestionError(Exception):
    pass


class UnsupportedFileTypeError(IngestionError):
    pass


class ParsingError(IngestionError):
    pass
