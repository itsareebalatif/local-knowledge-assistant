"""Parser registry — maps a normalized file_type to a parser instance.

`file_type` values match Document.file_type in app/models/document.py
("PDF", "MD", "HTML", "DOCX", "TXT"), which is itself what the ERD in the
SRS specifies.
"""

from __future__ import annotations

from app.ingestion.exceptions import UnsupportedFileTypeError
from app.ingestion.parsers.base import BaseParser
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.html_parser import HtmlParser
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.parsers.txt_parser import TxtParser

EXTENSION_TO_FILE_TYPE: dict[str, str] = {
    ".pdf": "PDF",
    ".md": "MD",
    ".markdown": "MD",
    ".html": "HTML",
    ".htm": "HTML",
    ".docx": "DOCX",
    ".txt": "TXT",
}

_PARSERS: dict[str, BaseParser] = {
    "PDF": PdfParser(),
    "MD": MarkdownParser(),
    "HTML": HtmlParser(),
    "DOCX": DocxParser(),
    "TXT": TxtParser(),
}


def file_type_for_filename(filename: str) -> str:
    for ext, file_type in EXTENSION_TO_FILE_TYPE.items():
        if filename.lower().endswith(ext):
            return file_type
    raise UnsupportedFileTypeError(f"No parser registered for filename: {filename!r}")


def get_parser(file_type: str) -> BaseParser:
    try:
        return _PARSERS[file_type.upper()]
    except KeyError as exc:
        raise UnsupportedFileTypeError(f"No parser registered for file_type: {file_type!r}") from exc


__all__ = ["get_parser", "file_type_for_filename", "EXTENSION_TO_FILE_TYPE"]
