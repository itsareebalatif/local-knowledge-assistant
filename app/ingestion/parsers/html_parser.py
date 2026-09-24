"""HTML parser (BeautifulSoup + lxml).

Layout stripping: `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`,
`<aside>`, `<noscript>`, and comment nodes are removed before extraction, so
navigation chrome and boilerplate never reach the chunker.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Comment

from app.ingestion.cleaning import normalize_text
from app.ingestion.exceptions import ParsingError
from app.ingestion.parsers.base import BaseParser
from app.ingestion.types import Block, ParsedDocument

_NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript", "form", "svg"]
_HEADING_TAGS = [f"h{i}" for i in range(1, 7)]
_BLOCK_TAGS = [*_HEADING_TAGS, "p", "li", "blockquote", "pre", "table"]


class HtmlParser(BaseParser):
    file_type = "HTML"

    def parse_sync(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        try:
            soup = BeautifulSoup(file_bytes, "lxml")
        except Exception as exc:
            raise ParsingError(f"Could not parse HTML {filename}: {exc}") from exc

        for tag in soup(_NOISE_TAGS):
            tag.decompose()
        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            comment.extract()

        title = soup.title.get_text(strip=True) if soup.title else ""
        body = soup.body or soup

        blocks: list[Block] = []
        for tag in body.find_all(_BLOCK_TAGS):
            # Skip elements nested inside a block we already captured
            # (e.g. an <li> inside a <table>, or a heading inside a <li>) —
            # find_all() walks the whole tree so nested matches would
            # otherwise be emitted twice.
            if tag.find_parent(_BLOCK_TAGS) is not None:
                continue

            if tag.name in _HEADING_TAGS:
                text = normalize_text(tag.get_text(" ", strip=True))
                if text:
                    blocks.append(Block(type="heading", text=text, level=int(tag.name[1])))
            elif tag.name == "table":
                rows = [
                    " | ".join(cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"]))
                    for row in tag.find_all("tr")
                ]
                text = normalize_text("\n".join(r for r in rows if r.strip()))
                if text:
                    blocks.append(Block(type="table", text=text))
            elif tag.name == "pre":
                text = normalize_text(tag.get_text("\n", strip=False))
                if text.strip():
                    blocks.append(Block(type="code", text=text))
            else:  # p, li, blockquote
                text = normalize_text(tag.get_text(" ", strip=True))
                if text:
                    blocks.append(Block(type="paragraph", text=text))

        meta = {"source_format": "html"}
        if title:
            meta["title"] = title
        return ParsedDocument(blocks=blocks, meta=meta)
